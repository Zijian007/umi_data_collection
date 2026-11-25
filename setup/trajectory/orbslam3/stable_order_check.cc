// Exercise real ORB containers with deliberately opposing address and ID order.
#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "Map.h"
#include "MapPoint.h"
#include <array>
#include <iostream>
#include <stdexcept>

// The neighbor wins the first query, then shares too few words to be
// scored on the second. Its old score must not nominate it again.
static void checkRelocalizationScores() {
    ORB_SLAM3::ORBVocabulary vocabulary(2, 1);
    vocabulary.create({{cv::Mat::zeros(1, 32, CV_8U), cv::Mat::ones(1, 32, CV_8U)}});
    ORB_SLAM3::KeyFrameDatabase database(vocabulary);
    ORB_SLAM3::Map map(0);
    ORB_SLAM3::KeyFrame candidate, neighbor;
    candidate.mnId = 1;
    neighbor.mnId = 2;
    candidate.UpdateMap(&map);
    neighbor.UpdateMap(&map);
    candidate.AddConnection(&neighbor, 20);
    candidate.mBowVec[0] = 0.1;
    candidate.mBowVec[1] = 0.9;
    neighbor.mBowVec[0] = 1.0;
    database.add(&candidate);
    database.add(&neighbor);

    ORB_SLAM3::Frame query;
    query.mnId = 1;
    query.mBowVec[0] = 1.0;
    auto found = database.DetectRelocalizationCandidates(&query, &map);
    if (found.size() != 1 || found.front() != &neighbor)
        throw std::runtime_error("scored covisible neighbor should win the first query");
    ++query.mnId;
    query.mBowVec[0] = query.mBowVec[1] = 0.5;
    found = database.DetectRelocalizationCandidates(&query, &map);
    if (found.size() != 1 || found.front() != &candidate)
        throw std::runtime_error("relocalization reused a neighbor score from the previous query");
}

int main() {
    ORB_SLAM3::KeyFrame root;
    std::array<ORB_SLAM3::KeyFrame, 2> neighbors;
    root.mnId = 0;
    neighbors[0].mnId = 1;
    neighbors[1].mnId = 2;
    root.AddConnection(&neighbors[0], 20);
    root.AddConnection(&neighbors[1], 20);
    if (root.GetBestCovisibilityKeyFrames(1).front()->mnId != 1)
        throw std::runtime_error("covisibility tie depends on pointer address");
    root.AddConnection(&neighbors[1], 30);
    if (root.GetBestCovisibilityKeyFrames(1).front()->mnId != 2)
        throw std::runtime_error("weight must take precedence over ID");

    neighbors[0].mnId = 5;
    neighbors[1].mnId = 4;
    std::array<ORB_SLAM3::MapPoint, 2> points;
    points[0].mnId = 8;
    points[1].mnId = 7;
    ORB_SLAM3::Map map(0);
    for (auto& keyframe : neighbors) map.AddKeyFrame(&keyframe);
    for (auto& point : points) map.AddMapPoint(&point);
    if (map.GetAllKeyFrames().front()->mnId != 4 || map.GetAllMapPoints().front()->mnId != 7)
        throw std::runtime_error("map traversal depends on pointer address");
    checkRelocalizationScores();
    std::cout << "relocalization scores: PASS\n";
    std::cout << "stable order: covisibility ties and map traversal: PASS\n";
}
