// Read-only ORB-SLAM3 atlas exporter for the local UMI viewer (GPL-3.0).
// Uses the same archive classes and library as SLAM, without starting SLAM threads.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <boost/archive/binary_iarchive.hpp>
#include "Atlas.h"
#include "KeyFrameDatabase.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: umi_export_atlas <map.osa> <ORBvoc.txt> <output.json>\n";
        return 2;
    }
    try {
        ORB_SLAM3::ORBVocabulary vocabulary;
        if (!vocabulary.loadFromTextFile(argv[2])) throw std::runtime_error("cannot load ORB vocabulary");
        ORB_SLAM3::KeyFrameDatabase database(vocabulary);
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open atlas");
        boost::archive::binary_iarchive archive(input);
        std::string vocabulary_name, vocabulary_checksum;
        ORB_SLAM3::Atlas* loaded = nullptr;
        archive >> vocabulary_name >> vocabulary_checksum >> loaded;
        std::unique_ptr<ORB_SLAM3::Atlas> atlas(loaded);
        if (!atlas) throw std::runtime_error("empty atlas");
        atlas->SetKeyFrameDababase(&database);
        atlas->SetORBVocabulary(&vocabulary);
        atlas->PostLoad();

        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("cannot create export");
        output << std::setprecision(9) << "{\"schema_version\":1,\"maps\":[";
        auto maps = atlas->GetAllMaps();  // Same ascending map-ID order as the localization driver.
        for (size_t index = 0; index < maps.size(); ++index) {
            if (index) output << ',';
            auto* map = maps[index];
            output << "{\"index\":" << index << ",\"id\":" << map->GetId() << ",\"points\":[";
            auto points = map->GetAllMapPoints();
            std::sort(points.begin(), points.end(), [](auto* a, auto* b) { return a->mnId < b->mnId; });
            bool first = true;
            for (auto* point : points) {
                if (!point || point->isBad()) continue;
                auto p = point->GetWorldPos();
                if (!p.allFinite()) continue;
                if (!first) output << ',';
                first = false;
                output << '[' << p.x() << ',' << p.y() << ',' << p.z() << ']';
            }
            output << "],\"keyframes\":[";
            auto keyframes = map->GetAllKeyFrames();
            std::sort(keyframes.begin(), keyframes.end(), [](auto* a, auto* b) { return a->mnId < b->mnId; });
            first = true;
            for (auto* keyframe : keyframes) {
                if (!keyframe || keyframe->isBad()) continue;
                auto pose = keyframe->GetPoseInverse();
                auto p = pose.translation();
                Eigen::Quaternionf q(pose.rotationMatrix());
                if (!p.allFinite() || !q.coeffs().allFinite() || !std::isfinite(keyframe->mTimeStamp)) continue;
                if (!first) output << ',';
                first = false;
                output << "{\"id\":" << keyframe->mnId << ",\"time\":" << keyframe->mTimeStamp
                       << ",\"position\":[" << p.x() << ',' << p.y() << ',' << p.z()
                       << "],\"quaternion\":[" << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << "]}";
            }
            output << "]}";
        }
        output << "]}\n";
        if (!output) throw std::runtime_error("failed to write export");
    } catch (const std::exception& error) {
        std::cerr << "Atlas export failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
