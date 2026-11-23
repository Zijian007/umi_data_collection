// Linked-engine smoke test: offline requests must not wait for absent workers.
#include "System.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    cv::setNumThreads(1);
    cv::FileStorage settings(argv[2], cv::FileStorage::READ);
    if (!settings.isOpened()) throw std::runtime_error("missing test settings");
    const int width = settings["Camera.width"], height = settings["Camera.height"];
    if (width <= 0 || height <= 0) throw std::runtime_error("invalid test dimensions");
    const auto dictionary = cv::makePtr<cv::aruco::Dictionary>(
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50));

    // Unsupported combinations must fail immediately, before loading vocabulary.
    for (int mode = 0; mode < 2; ++mode) {
        bool rejected = false;
        try {
            ORB_SLAM3::System invalid("", "", mode == 0 ? ORB_SLAM3::System::STEREO
                : ORB_SLAM3::System::MONOCULAR, mode == 1, "", "", dictionary, 13, .024, true);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("unsupported offline mode accepted");
    }

    cv::Mat blank = cv::Mat::zeros(height, width, CV_8UC3);
    ORB_SLAM3::System system(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR,
                            false, "", "", dictionary, 13, .024, true);
    system.TrackMonocular(blank, 0.);
    if (system.LocalizeMonocular(blank, .1).second)
        throw std::runtime_error("blank frame tracked");
    system.ActivateLocalizationMode();
    system.LocalizeMonocular(blank, .2);
    system.DeactivateLocalizationMode();
    system.LocalizeMonocular(blank, .3);
    system.ResetActiveMap();
    system.LocalizeMonocular(blank, .4);
    system.Reset();
    system.LocalizeMonocular(blank, .5);
    system.Shutdown();
    system.Shutdown();
    std::cout << "offline lifecycle: pause/resume/active reset/full reset/idempotent shutdown: PASS\n";
}
