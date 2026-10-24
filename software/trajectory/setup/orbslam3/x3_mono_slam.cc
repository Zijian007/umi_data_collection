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
 */
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <signal.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <System.h>
#include <CLI11.hpp>

using namespace std;

void signal_callback_handler(int signum) {
  cout << "x3_mono_slam caught signal " << signum << endl;
  exit(signum);
}

int main(int argc, char **argv) {
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
  bool localization_only = false;
  app.add_flag("--localization_only", localization_only);
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
  cv::setNumThreads(4);

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
                         aruco_dict, init_tag_id, init_tag_size);
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
  cout << "frames " << nImages << " fps " << setprecision(12) << fps << " range [" << frame_start << ", "
       << frame_end << ")" << endl;

  ofstream fcsv(output_frames_csv);
  fcsv << "frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw\n";
  fcsv << fixed << setprecision(9);

  int n_ok = 0, n_proc = 0;
  for (int frame_idx = 0; frame_idx < frame_end; frame_idx++) {
    if (frame_idx < frame_start) {
      if (!cap.grab()) break;
      continue;
    }
    cv::Mat im, im_track;
    if (!cap.read(im)) {
      cout << "cap.read failed at " << frame_idx << endl;
      break;
    }
    im_track = im;
    if (im_track.size() != img_size) cv::resize(im_track, im_track, img_size);
    if (!mask_img.empty()) im_track.setTo(cv::Scalar(0, 0, 0), mask_img);

    double t = (double)frame_idx / fps;
    auto result = SLAM.LocalizeMonocular(im_track, t);
    int state = SLAM.GetTrackingState();
    n_proc++;
    fcsv << frame_idx << "," << t << "," << state << "," << (result.second ? 1 : 0);
    if (result.second) {
      n_ok++;
      Sophus::SE3f Twc = result.first.inverse();
      Eigen::Vector3f p = Twc.translation();
      Eigen::Quaternionf q = Twc.unit_quaternion();
      fcsv << "," << p.x() << "," << p.y() << "," << p.z() << "," << q.x() << "," << q.y() << "," << q.z() << ","
           << q.w() << "\n";
    } else {
      fcsv << ",,,,,,,\n";
    }
    if (frame_idx % 300 == 0) cout << "frame " << frame_idx << " state " << state << " ok " << n_ok << "/" << n_proc << endl;
  }
  fcsv.close();
  cout << "TRACKED " << n_ok << " / " << n_proc << endl;

  SLAM.Shutdown();
  if (!output_trajectory_csv.empty()) SLAM.SaveTrajectoryCSV(output_trajectory_csv);
  return 0;
}
