// Tests the actual linked ORB camera, including its existing forward triangulator.
#include "KannalaBrandt8.h"
#include "kb4_bearing.h"
#include <iostream>
#include <stdexcept>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    ORB_SLAM3::KannalaBrandt8 camera(std::vector<float>{450,451,360,359,.3f,-.01f,.001f,0});
    for (double degrees : {0.,35.,70.,84.,86.,95.,110.}) {
        const double t=degrees*umi_kb4::pi/180.;
        Eigen::Vector3f ray(std::sin(t),0,std::cos(t));
        auto pixel=camera.project(ray);
        auto inverse=camera.unproject(cv::Point2f(pixel[0],pixel[1]));
        if (degrees > 85) {
            require(!std::isfinite(inverse.x), "backward/outside-domain ray was not rejected");
        } else {
            Eigen::Vector3f reconstructed(inverse.x,inverse.y,inverse.z);
            require((reconstructed.normalized()-ray).norm()<2e-6, "actual ORB inverse direction mismatch");
            require((camera.project(reconstructed)-pixel).norm()<.003, "actual ORB pixel roundtrip mismatch");
        }
    }
    Eigen::Vector3f world(3,.1,1), baseline(.3,0,0), recovered;
    auto p1=camera.project(world);
    Eigen::Vector3f right=world-baseline;
    auto p2=camera.project(right);
    cv::KeyPoint a(cv::Point2f(p1[0],p1[1]),1), b(cv::Point2f(p2[0],p2[1]),1);
    float depth=camera.TriangulateMatches(&camera,a,b,Eigen::Matrix3f::Identity(),baseline,1,1,recovered);
    require(depth>0 && (recovered-world).norm()<1e-4, "forward triangulation changed scale/geometry");
    std::cout << "actual ORB KB4: unclipped forward rays, rejected >85 deg rays, triangulation: PASS\n";
}
