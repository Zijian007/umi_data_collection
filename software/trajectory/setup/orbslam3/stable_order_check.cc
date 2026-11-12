// Exercise real ORB containers with deliberately opposing address and ID order.
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include <array>
#include <iostream>
#include <stdexcept>

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
    std::cout << "stable order: covisibility ties and map traversal: PASS\n";
}
