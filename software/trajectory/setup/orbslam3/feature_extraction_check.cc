// Check scheduling-independent ORB output and the optional frame tag detector.
#include "Frame.h"
#include "ORBextractor.h"
#include "CameraModels/Pinhole.h"
#include <omp.h>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void sameFeatures(const std::vector<cv::KeyPoint>& expected,
                  const std::vector<cv::KeyPoint>& actual,
                  const cv::Mat& expectedDescriptors, const cv::Mat& actualDescriptors) {
    require(expected.size() == actual.size(), "ORB feature count changed");
    for (size_t index = 0; index < expected.size(); ++index) {
        const auto& reference = expected[index];
        const auto& candidate = actual[index];
        require(reference.pt == candidate.pt && reference.size == candidate.size &&
                reference.angle == candidate.angle && reference.response == candidate.response &&
                reference.octave == candidate.octave && reference.class_id == candidate.class_id,
                "ORB feature value or ordering changed");
    }
    require(expectedDescriptors.size() == actualDescriptors.size(), "ORB descriptor shape changed");
    if (!expectedDescriptors.empty())
        require(cv::norm(expectedDescriptors, actualDescriptors, cv::NORM_INF) == 0.,
                "ORB descriptor bytes changed");
}

void checkParallelFeatures() {
    omp_set_dynamic(0);
    for (int sample = 0; sample < 24; ++sample) {
        cv::Mat image(480 + 16 * (sample % 3), 640 + 16 * (sample % 5), CV_8U);
        cv::RNG pixels(100 + sample);
        if (sample == 0) image.setTo(127);
        else pixels.fill(image, cv::RNG::UNIFORM, 0, 256);
        ORB_SLAM3::ORBextractor extractor(1500, 1.2f, 8, 20, 7);
        // Exercise all-overlap, partial-overlap and no-overlap descriptor ordering.
        std::vector<int> overlap = sample % 3 == 0 ? std::vector<int>{0, image.cols}
            : sample % 3 == 1 ? std::vector<int>{100, 400} : std::vector<int>{-1, -1};
        std::vector<cv::KeyPoint> serialKeys, parallelKeys;
        cv::Mat serialDescriptors, parallelDescriptors;
        omp_set_max_active_levels(0);
        const int serialMono = extractor(image, cv::noArray(), serialKeys, serialDescriptors, overlap);
        omp_set_max_active_levels(1);
        const int parallelMono = extractor(image, cv::noArray(), parallelKeys, parallelDescriptors, overlap);
        require(serialMono == parallelMono, "ORB overlap boundary changed");
        sameFeatures(serialKeys, parallelKeys, serialDescriptors, parallelDescriptors);
    }
}

void checkOptionalTags() {
    const auto dictionary = cv::makePtr<cv::aruco::Dictionary>(
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50));
    cv::Mat image(480, 640, CV_8U, cv::Scalar(255)), marker;
    cv::aruco::generateImageMarker(*dictionary, 13, 160, marker);
    marker.copyTo(image(cv::Rect(240, 160, 160, 160)));
    ORB_SLAM3::Pinhole camera(std::vector<float>{500.f, 500.f, 320.f, 240.f});
    ORB_SLAM3::ORBextractor extractor(1000, 1.2f, 8, 20, 7);
    ORB_SLAM3::ORBVocabulary vocabulary;
    cv::Mat distortion = cv::Mat::zeros(5, 1, CV_32F);
    ORB_SLAM3::Frame initial(image, 0., &extractor, &vocabulary, &camera, distortion, 0.f, 0.f);
    ORB_SLAM3::Frame tracking(image, .1, &extractor, &vocabulary, &camera, distortion, 0.f, 0.f,
                             nullptr, ORB_SLAM3::IMU::Calib(), nullptr);
    ORB_SLAM3::Frame restarted(image, .2, &extractor, &vocabulary, &camera, distortion, 0.f, 0.f,
                              nullptr, ORB_SLAM3::IMU::Calib(), dictionary);
    require(initial.markerIds == std::vector<int>{13}, "default frame tag detection missing");
    require(tracking.markerIds.empty() && tracking.markerCorners.empty() &&
            tracking.rejectedCandidates.empty(), "disabled detector retained tags");
    require(restarted.markerIds == initial.markerIds && restarted.markerCorners == initial.markerCorners,
            "tag detection did not resume");
    sameFeatures(initial.mvKeys, tracking.mvKeys, initial.mDescriptors, tracking.mDescriptors);
    sameFeatures(initial.mvKeys, restarted.mvKeys, initial.mDescriptors, restarted.mDescriptors);
}
}

int main() {
    cv::setNumThreads(1);
    checkParallelFeatures();
    checkOptionalTags();
    std::cout << "feature extraction: serial/parallel ordering, descriptor bytes, optional tags: PASS\n";
}
