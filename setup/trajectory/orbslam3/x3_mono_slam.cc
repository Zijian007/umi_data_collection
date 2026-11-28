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
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <sys/stat.h>
#include <omp.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <System.h>
#include <KannalaBrandt8.h>
#include <sstream>
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

struct LocalizationPrior {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int frame;
  Sophus::SE3f camera_from_map;
  float max_translation;
};

using LocalizationPriors = std::vector<LocalizationPrior, Eigen::aligned_allocator<LocalizationPrior>>;

struct GyroRotation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Sophus::SO3f rotation;
  int segment = -1;
};

using GyroRotations = std::vector<GyroRotation, Eigen::aligned_allocator<GyroRotation>>;

bool read_gyro_rotations(const std::string& path, GyroRotations& rotations) {
  std::ifstream stream(path);
  if (!stream) return false;
  std::string line;
  int previous_frame = -1, previous_segment = -1;
  while (std::getline(stream, line)) {
    std::istringstream row(line);
    int frame, segment;
    float qx, qy, qz, qw;
    std::string extra;
    if (!(row >> frame >> segment >> qx >> qy >> qz >> qw) || (row >> extra) ||
        frame <= previous_frame || frame >= static_cast<int>(rotations.size()) ||
        segment < 0 || segment < previous_segment ||
        (segment == previous_segment && frame != previous_frame + 1)) return false;
    const Eigen::Quaternionf orientation(qw, qx, qy, qz);
    if (!orientation.coeffs().allFinite() || std::abs(orientation.squaredNorm() - 1.f) > 0.01f) return false;
    rotations[frame] = {Sophus::SO3f(orientation.normalized()), segment};
    previous_frame = frame;
    previous_segment = segment;
  }
  return stream.eof() && previous_frame >= 0;
}

bool read_localization_priors(const std::string& path, int first, int end, LocalizationPriors& priors) {
  std::ifstream stream(path);
  if (!stream) return false;
  std::string line;
  int previous = first - 1;
  while (std::getline(stream, line)) {
    std::istringstream row(line);
    int frame;
    float x, y, z, qx, qy, qz, qw, limit;
    std::string extra;
    if (!(row >> frame >> x >> y >> z >> qx >> qy >> qz >> qw >> limit) || (row >> extra) ||
        frame <= previous || frame >= end || !std::isfinite(limit) || limit <= 0.f) return false;
    Eigen::Vector3f position(x, y, z);
    Eigen::Quaternionf orientation(qw, qx, qy, qz);
    if (!position.allFinite() || !orientation.coeffs().allFinite() ||
        std::abs(orientation.squaredNorm() - 1.f) > 0.01f) return false;
    priors.push_back({frame, Sophus::SE3f(orientation.normalized(), position).inverse(), limit});
    previous = frame;
  }
  return stream.eof() && !priors.empty();
}

// Upper triangle of the pixel Hessian in right-local camera pose coordinates.
std::string camera_information(ORB_SLAM3::System& slam, ORB_SLAM3::KannalaBrandt8& camera,
                               const Sophus::SE3f& camera_from_map, double pyramid_scale) {
  const auto points = slam.GetTrackedMapPoints();
  const auto keys = slam.GetTrackedKeyPointsUn();
  if (points.size() != keys.size()) throw std::runtime_error("ORB feature/landmark count mismatch");
  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
  size_t count = 0;
  for (size_t i = 0; i < points.size(); ++i) {
    auto* point = points[i];
    if (!point || point->isBad() || !point->Observations()) continue;
    const Eigen::Vector3d q = (camera_from_map * point->GetWorldPos()).cast<double>();
    if (!q.allFinite() || q.z() <= 0.) continue;
    const double sigma = std::pow(pyramid_scale, keys[i].octave);
    const Eigen::Vector2d residual = camera.project(q) - Eigen::Vector2d(keys[i].pt.x, keys[i].pt.y);
    if (!residual.allFinite() || residual.squaredNorm() >= 5.991 * sigma * sigma) continue;
    Eigen::Matrix<double, 3, 6> motion;
    motion.leftCols<3>() = Sophus::SO3d::hat(q);
    motion.rightCols<3>() = -Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 2, 6> jac = camera.projectJac(q) * motion / sigma;
    if (!jac.allFinite()) continue;
    information.noalias() += jac.transpose() * jac;
    ++count;
  }
  if (count < 6) return "";
  std::ostringstream field;
  field << std::setprecision(16);
  for (int i = 0; i < 6; ++i)
    for (int j = i; j < 6; ++j) field << (i || j ? " " : "") << information(i, j);
  return field.str();
}

void write_frame_pose(std::ostream& output, int frame_index, double video_time_s, int tracking_state,
                      const std::pair<Sophus::SE3f, bool>& localization) {
  output << frame_index << "," << video_time_s << "," << tracking_state << "," << (localization.second ? 1 : 0);
  if (!localization.second) {
    output << ",,,,,,,";
    return;
  }
  const Sophus::SE3f camera_in_world = localization.first.inverse();
  const Eigen::Vector3f camera_position = camera_in_world.translation();
  const Eigen::Quaternionf camera_orientation = camera_in_world.unit_quaternion();
  output << "," << camera_position.x() << "," << camera_position.y() << "," << camera_position.z()
         << "," << camera_orientation.x() << "," << camera_orientation.y() << "," << camera_orientation.z()
         << "," << camera_orientation.w();
}

}  // namespace

void signal_callback_handler(int signum) {
  cout << "x3_mono_slam caught signal " << signum << endl;
  exit(signum);
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--capabilities") {
    cout << "{\"schema_version\":1,\"kb4_geometry\":\"forward_v1\",\"maximum_ray_angle_degrees\":85,\"offline_sync\":true,\"reverse_localization\":true,\"warm_reverse_localization\":true,\"inprocess_localization\":true,\"localization_pose_prior\":true,\"gyro_aided_mapping\":true}\n";
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
  int information_stride = 0;
  app.add_option("--front_information_stride", information_stride, "Export KB4 landmark information at this interval");
  std::string output_trajectory_csv;
  app.add_option("-o,--output_trajectory_csv", output_trajectory_csv);
  std::string load_map;
  app.add_option("-l,--load_map", load_map);
  std::string save_map;
  app.add_option("--save_map", save_map);
  std::string localize_after_mapping_dir;
  app.add_option("--localize_after_mapping_dir", localize_after_mapping_dir,
                 "After mapping finishes, localize all surviving maps in-process using cached features into loc_m<idx>/frames.csv");
  bool async_mapping = false;
  app.add_flag("--async_mapping", async_mapping, "Legacy threaded mapping (diagnostic only)");
  double offline_keyframe_interval_s = 0.2;
  app.add_option("--offline_keyframe_interval", offline_keyframe_interval_s, "Minimum offline keyframe spacing in video seconds [0,1]");
  bool localization_only = false;
  app.add_flag("--localization_only", localization_only);
  std::string localization_pose_priors;
  app.add_option("--localization_pose_priors", localization_pose_priors,
                 "Per-frame camera pose hints in the loaded map frame");
  std::string mapping_gyro_rotations;
  app.add_option("--mapping_gyro_rotations", mapping_gyro_rotations,
                 "Per-frame integrated camera rotations for forward offline mapping");
  bool reverse = false;
  app.add_flag("--reverse", reverse, "Read frames backwards for fixed-map visual localization only");
  bool warm_reverse = false;
  app.add_flag("--warm_reverse", warm_reverse, "Track forward through the turnaround frame before --reverse");
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
  if (warm_reverse && !reverse) {
    cerr << "--warm_reverse requires --reverse" << endl;
    return 2;
  }
  if (!localization_pose_priors.empty() && (!localization_only || load_map.empty() || reverse)) {
    cerr << "--localization_pose_priors requires forward --localization_only with --load_map" << endl;
    return 2;
  }
  if (!mapping_gyro_rotations.empty() && (localization_only || async_mapping || !load_map.empty())) {
    cerr << "--mapping_gyro_rotations requires forward offline mapping of a new atlas" << endl;
    return 2;
  }
  if (!localize_after_mapping_dir.empty() && (localization_only || reverse || !load_map.empty())) {
    cerr << "--localize_after_mapping_dir cannot be combined with --localization_only, --reverse, or --load_map" << endl;
    return 2;
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
  std::unique_ptr<ORB_SLAM3::KannalaBrandt8> information_camera;
  double pyramid_scale = 1.;
  if (information_stride) {
    if (information_stride < 0 || (!localization_only && localize_after_mapping_dir.empty())) return 2;
    std::vector<float> parameters;
    for (const char* key : {"fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"})
      parameters.push_back((float)fsSettings[std::string("Camera1.") + key]);
    pyramid_scale = (double)fsSettings["ORBextractor.scaleFactor"];
    if (!(pyramid_scale > 0. && parameters[0] > 0. && parameters[1] > 0.)) return 2;
    information_camera.reset(new ORB_SLAM3::KannalaBrandt8(parameters));
  }
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
  LocalizationPriors priors;
  if (!localization_pose_priors.empty() && !read_localization_priors(localization_pose_priors, frame_start, frame_end, priors)) {
    cerr << "Invalid localization pose priors: " << localization_pose_priors << endl;
    return 2;
  }
  size_t next_prior = 0;
  GyroRotations gyro_rotations;
  if (!mapping_gyro_rotations.empty()) {
    gyro_rotations.resize(nImages);
    if (!read_gyro_rotations(mapping_gyro_rotations, gyro_rotations)) {
      cerr << "Invalid mapping gyro rotations: " << mapping_gyro_rotations << endl;
      return 2;
    }
  }

  // The pivot is frame_end, just outside the requested reverse CSV interval.
  // Keep its tracker state, then process frame_end-1 .. frame_start normally.
  double reverse_clock_offset = 0.;
  if (warm_reverse) {
    if (frame_end >= nImages) {
      cerr << "Warm reverse needs a turnaround frame inside the video" << endl;
      return 2;
    }
    for (int i = 0; i <= frame_end; ++i) {
      cv::Mat image;
      if (!cap.read(image)) return 2;
      if (image.size() != img_size) cv::resize(image, image, img_size);
      if (!mask_img.empty()) image.setTo(cv::Scalar(0, 0, 0), mask_img);
      SLAM.LocalizeMonocular(image, static_cast<double>(i) / fps);
    }
    reverse_clock_offset = (frame_end + 1.) / fps;
  }

  const bool export_pass1_info = localization_only && information_stride > 0;
  ofstream fcsv(output_frames_csv);
  fcsv << "frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw" << (export_pass1_info ? ",camera_information" : "") << "\n";
  fcsv << fixed << setprecision(9);

  const bool cache_for_inprocess = !localize_after_mapping_dir.empty();
  std::vector<std::pair<int, double>> cached_frames;
  if (cache_for_inprocess) {
    SLAM.EnableMonoFeatureCache(true);
    cached_frames.reserve(frame_end - frame_start);
  }

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
    if (!priors.empty()) {
      if (next_prior < priors.size() && priors[next_prior].frame == frame_idx) {
        const auto& prior = priors[next_prior++];
        SLAM.SetLocalizationPosePrior(prior.camera_from_map, prior.max_translation);
      } else {
        SLAM.SetLocalizationPosePrior(Sophus::SE3f(), 0.f);
      }
    }
    if (!gyro_rotations.empty()) {
      const auto& sample = gyro_rotations[frame_idx];
      SLAM.SetMappingGyroRotation(sample.rotation, sample.segment);
    }
    auto result = SLAM.LocalizeMonocular(im_track, reverse ? reverse_clock_offset + (double)step / fps : t);
    int state = SLAM.GetTrackingState();
    n_proc++;
    if (cache_for_inprocess) cached_frames.emplace_back(frame_idx, t);
    write_frame_pose(fcsv, frame_idx, t, state, result);
    if (export_pass1_info) {
      fcsv << ",";
      if (result.second && state == 2 && (frame_idx % information_stride == 0 || frame_idx == frame_end - 1))
        fcsv << camera_information(SLAM, *information_camera, result.first, pyramid_scale);
    }
    fcsv << "\n";
    if (result.second) n_ok++;
    if (frame_idx % 300 == 0) cout << "frame " << frame_idx << " state " << state << " ok " << n_ok << "/" << n_proc << endl;
  }
  fcsv.close();
  cout << "TRACKED " << n_ok << " / " << n_proc << endl;
  if (!gyro_rotations.empty()) SLAM.ReportMappingGyroTracking();
  if (!priors.empty()) {
    const auto counts = SLAM.GetLocalizationPosePriorCounts();
    cout << "POSE_PRIOR available " << next_prior << " attempted " << counts.first
         << " accepted " << counts.second << endl;
  }

  if (cache_for_inprocess) {
    if (!gyro_rotations.empty()) SLAM.SetMappingGyroRotation(Sophus::SO3f(), -1);
    SLAM.FinalizeMappingForLocalization();
    const size_t n_maps = SLAM.AtlasMapCount();
    if (n_maps == 0) {
      cerr << "No surviving maps in atlas for in-process localization" << endl;
      return 5;
    }
    if (mkdir(localize_after_mapping_dir.c_str(), 0755) != 0 && errno != EEXIST) {
      cerr << "Failed to create directory: " << localize_after_mapping_dir << endl;
      return 1;
    }
    for (size_t map_index = 0; map_index < n_maps; ++map_index) {
      const std::string loc_dir = localize_after_mapping_dir + "/loc_m" + std::to_string(map_index);
      if (mkdir(loc_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        cerr << "Failed to create directory: " << loc_dir << endl;
        return 1;
      }
      const std::string loc_csv = loc_dir + "/frames.csv";
      ofstream lcsv(loc_csv);
      if (!lcsv.is_open()) {
        cerr << "Failed to open localization output CSV: " << loc_csv << endl;
        return 1;
      }
      lcsv << "frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw" << (information_stride ? ",camera_information" : "") << "\n";
      lcsv << fixed << setprecision(9);
      if (!SLAM.BeginInProcessLocalization(map_index)) {
        cerr << "Failed to begin in-process localization for map " << map_index << endl;
        return 1;
      }
      int loc_ok = 0, loc_proc = 0;
      for (size_t i = 0; i < cached_frames.size(); ++i) {
        const int frame_idx = cached_frames[i].first;
        const double t = cached_frames[i].second;
        auto result = SLAM.LocalizeCachedMonocular(i);
        int state = SLAM.GetTrackingState();
        ++loc_proc;
        write_frame_pose(lcsv, frame_idx, t, state, result);
        if (information_stride) {
          lcsv << ",";
          if (result.second && state == 2 && (frame_idx % information_stride == 0 || frame_idx == frame_end - 1))
            lcsv << camera_information(SLAM, *information_camera, result.first, pyramid_scale);
        }
        lcsv << "\n";
        if (result.second) ++loc_ok;
      }
      lcsv.close();
      if (!lcsv.good()) {
        cerr << "Failed writing localization CSV: " << loc_csv << endl;
        return 1;
      }
      cout << "LOCALIZED_MAP " << map_index << " TRACKED " << loc_ok << " / " << loc_proc << endl;
    }
    SLAM.EnableMonoFeatureCache(false);
  }

  SLAM.Shutdown();
  if (!output_trajectory_csv.empty()) SLAM.SaveTrajectoryCSV(output_trajectory_csv);
  return 0;
}
