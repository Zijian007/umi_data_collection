/**
 * x3_mono_slam.cc -- visual-only (MONOCULAR) driver for the UMI ORB-SLAM3 fork, for the KIWI reproduction.
 * Derived from Examples/Monocular-Inertial/gopro_slam.cc (GPL-3.0, ORB-SLAM3 authors / Cheng Chi).
 *
 * Differences from gopro_slam:
 *  - sensor = MONOCULAR, no IMU is read or fed (KIWI runs ORB-SLAM3 vision-only and fuses the IMU offline);
 *  - every processed frame is written to --output_frames_csv with the tracker's own pose at that time
 *    (frame_idx, t, state, ok, Twc as tx ty tz qx qy qz qw), so relocalizations are visible frame by frame;
 *  - --localization_only: after --load_map, switch to localization mode (no new keyframes, map frame fixed);
 *  - --frame_start / --frame_end restrict the processed range (frame time is still frame_idx / fps).
 *  - --reverse reads bounded blocks backwards for fixed-map localization, retaining original CSV indices/times.
 *  - synchronous offline mapping with 0.2 s minimum keyframe spacing (at the configured FPS);
 *    --async_mapping and --offline_keyframe_interval expose diagnostic alternatives.
 */
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <omp.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <System.h>
#include <CLI11.hpp>

using namespace std;

static_assert(UMI_KB4_FORWARD == 1, "The UMI driver requires the forward KB4 camera model");

namespace {

class FramePlayback {
 public:
  FramePlayback(cv::VideoCapture& video, int first_frame, int end_frame, bool is_reverse,
                cv::Size processing_size)
      : video_(video), first_frame_(first_frame), is_reverse_(is_reverse), block_start_(end_frame),
        block_capacity_(std::max(1, std::min(64, static_cast<int>(
            (64LL * 1024 * 1024) / (3LL * processing_size.width * processing_size.height))))) {}

  bool read(int frame_index, cv::Mat& image) {
    if (!is_reverse_) return video_.read(image);
    if (decoded_block_.empty() || frame_index < block_start_) {
      block_start_ = std::max(first_frame_, frame_index - block_capacity_ + 1);
      decoded_block_.clear();
      if (!video_.set(cv::CAP_PROP_POS_FRAMES, block_start_) ||
          std::abs(video_.get(cv::CAP_PROP_POS_FRAMES) - block_start_) > 0.5) {
        cerr << "Reverse seek failed at " << block_start_ << endl;
        return false;
      }
      for (int decode_index = block_start_; decode_index <= frame_index; ++decode_index) {
        cv::Mat decoded_image;
        if (!video_.read(decoded_image)) {
          cerr << "Reverse decode failed at " << decode_index << endl;
          return false;
        }
        decoded_block_.push_back(decoded_image);
      }
    }
    image = decoded_block_[frame_index - block_start_];
    return true;
  }

 private:
  cv::VideoCapture& video_;
  int first_frame_;
  bool is_reverse_;
  int block_start_;
  int block_capacity_;
  vector<cv::Mat> decoded_block_;
};

void write_frame_pose(std::ostream& output, int frame_index, double video_time_s, int tracking_state,
                      const std::pair<Sophus::SE3f, bool>& localization) {
  output << frame_index << "," << video_time_s << "," << tracking_state << "," << (localization.second ? 1 : 0);
  if (!localization.second) {
    output << ",,,,,,,\n";
    return;
  }
  const Sophus::SE3f camera_in_world = localization.first.inverse();
  const Eigen::Vector3f camera_position = camera_in_world.translation();
  const Eigen::Quaternionf camera_orientation = camera_in_world.unit_quaternion();
  output << "," << camera_position.x() << "," << camera_position.y() << "," << camera_position.z()
         << "," << camera_orientation.x() << "," << camera_orientation.y() << "," << camera_orientation.z()
         << "," << camera_orientation.w() << "\n";
}

}  // namespace

void signal_callback_handler(int signum) {
  cout << "x3_mono_slam caught signal " << signum << endl;
  exit(signum);
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--capabilities") {
    cout << "{\"schema_version\":1,\"kb4_geometry\":\"forward_v1\",\"maximum_ray_angle_degrees\":85,\"offline_sync\":true,\"reverse_localization\":true}\n";
    return 0;
  }
  signal(SIGINT, signal_callback_handler);
  CLI::App app{"X3 visual-only ORB-SLAM3"};

  std::string vocabulary = "../../Vocabulary/ORBvoc.txt";
  app.add_option("-v,--vocabulary", vocabulary)->capture_default_str();
  std::string setting;
  app.add_option("-s,--setting", setting)->required();
  std::string input_video;
  app.add_option("-i,--input_video", input_video)->required();
  std::string output_frames_csv;
  app.add_option("--output_frames_csv", output_frames_csv)->required();
  std::string output_trajectory_csv;
  app.add_option("-o,--output_trajectory_csv", output_trajectory_csv);
  std::string load_map;
  app.add_option("-l,--load_map", load_map);
  std::string save_map;
  app.add_option("--save_map", save_map);
  bool async_mapping = false;
  app.add_flag("--async_mapping", async_mapping, "Legacy threaded mapping (diagnostic only)");
  double offline_keyframe_interval_s = 0.2;
  app.add_option("--offline_keyframe_interval", offline_keyframe_interval_s, "Minimum offline keyframe spacing in video seconds [0,1]");
  bool localization_only = false;
  app.add_flag("--localization_only", localization_only);
  bool reverse = false;
  app.add_flag("--reverse", reverse, "Read frames backwards for fixed-map visual localization only");
  std::string mask_img_path;
  app.add_option("--mask_img", mask_img_path);
  int frame_start = 0;
  app.add_option("--frame_start", frame_start);
  int frame_end = -1;
  app.add_option("--frame_end", frame_end);
  int aruco_dict_id = cv::aruco::DICT_4X4_50;
  app.add_option("--aruco_dict_id", aruco_dict_id);
  int init_tag_id = 13;
  app.add_option("--init_tag_id", init_tag_id);
  float init_tag_size = 0.16;
  app.add_option("--init_tag_size", init_tag_size);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }
  if (reverse && (!localization_only || load_map.empty() || !save_map.empty() || !output_trajectory_csv.empty())) {
    cerr << "--reverse requires --load_map and --localization_only; map/trajectory export is not allowed" << endl;
    return 2;
  }
  cv::setNumThreads(async_mapping ? 4 : 1);
  // OpenCV can enable dynamic OpenMP teams at startup, collapsing the four
  // independent ORB pyramid workers to one even with an explicit team size.
  if (!async_mapping) omp_set_dynamic(0);

  cv::FileStorage fsSettings(setting, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    cerr << "Failed to open setting file at: " << setting << endl;
    return -1;
  }
  cv::Size img_size(fsSettings["Camera.width"], fsSettings["Camera.height"]);
  fsSettings.release();

  cv::Mat mask_img;
  if (!mask_img_path.empty()) {
    mask_img = cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);
    if (mask_img.size() != img_size) cv::resize(mask_img, mask_img, img_size);
  }

  cv::Ptr<cv::aruco::Dictionary> aruco_dict =
      cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(aruco_dict_id));
  ORB_SLAM3::System SLAM(vocabulary, setting, ORB_SLAM3::System::MONOCULAR, false, load_map, save_map,
                         aruco_dict, init_tag_id, init_tag_size, !async_mapping, offline_keyframe_interval_s);
  if (localization_only) {
    SLAM.ActivateLocalizationMode();
    cout << "localization_only: requested" << endl;
  }

  cv::VideoCapture cap(input_video, cv::CAP_FFMPEG);
  if (!cap.isOpened()) {
    cerr << "Error opening video " << input_video << endl;
    return -1;
  }
  int nImages = cap.get(cv::CAP_PROP_FRAME_COUNT);
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (frame_end < 0 || frame_end > nImages) frame_end = nImages;
  if (!std::isfinite(fps) || fps <= 0 || frame_start < 0 || frame_start >= frame_end) {
    cerr << "Invalid video FPS or frame range" << endl;
    return 2;
  }
  cout << "frames " << nImages << " fps " << setprecision(12) << fps << " range [" << frame_start << ", "
       << frame_end << ")" << endl;

  ofstream fcsv(output_frames_csv);
  fcsv << "frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw\n";
  fcsv << fixed << setprecision(9);

  int n_ok = 0, n_proc = 0;
  FramePlayback playback(cap, frame_start, frame_end, reverse, img_size);
  if (!reverse) {
    for (int i = 0; i < frame_start; ++i) if (!cap.grab()) return 2;
  }
  for (int step = 0; step < frame_end - frame_start; ++step) {
    const int frame_idx = reverse ? frame_end - 1 - step : frame_start + step;
    cv::Mat im, im_track;
    if (!playback.read(frame_idx, im)) {
      if (reverse) return 2;
      cout << "cap.read failed at " << frame_idx << endl;
      break;
    }
    im_track = im;
    if (im_track.size() != img_size) cv::resize(im_track, im_track, img_size);
    if (!mask_img.empty()) im_track.setTo(cv::Scalar(0, 0, 0), mask_img);

    double t = (double)frame_idx / fps;
    // The tracker requires increasing time. CSV retains original frame indices
    // and video times; the IMU is fused later on the original ascending clock.
    auto result = SLAM.LocalizeMonocular(im_track, reverse ? (double)step / fps : t);
    int state = SLAM.GetTrackingState();
    n_proc++;
    write_frame_pose(fcsv, frame_idx, t, state, result);
    if (result.second) n_ok++;
    if (frame_idx % 300 == 0) cout << "frame " << frame_idx << " state " << state << " ok " << n_ok << "/" << n_proc << endl;
  }
  fcsv.close();
  cout << "TRACKED " << n_ok << " / " << n_proc << endl;

  SLAM.Shutdown();
  if (!output_trajectory_csv.empty()) SLAM.SaveTrajectoryCSV(output_trajectory_csv);
  return 0;
}
