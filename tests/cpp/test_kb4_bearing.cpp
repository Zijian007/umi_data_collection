// Standalone numerical test: no ORB/OpenCV dependency or production binary writes.
#include "kb4_bearing.h"
#include <iostream>
#include <stdexcept>

static void require(bool value) {
    if (!value) throw std::runtime_error("KB4 bearing regression");
}

int main() {
    const std::array<double, 8> p = {{450., 451., 720., 719., .3, -.01, .001, 0.}};
    for (double degrees : {0., 35., 70., 85., 89., 90., 95., 110., 140.}) {
        const double t = degrees*umi_kb4::pi/180.;
        const double t2=t*t;
        const double rd=t*(1+t2*(p[4]+t2*(p[5]+t2*(p[6]+t2*p[7]))));
        for (double azimuth : {0., .7, 2.5, 4.9}) {
            auto b=umi_kb4::bearing(p[2]+p[0]*rd*std::cos(azimuth),
                                   p[3]+p[1]*rd*std::sin(azimuth), p);
            require(std::abs(b[0]-std::sin(t)*std::cos(azimuth)) < 1e-10);
            require(std::abs(b[1]-std::sin(t)*std::sin(azimuth)) < 1e-10);
            require(std::abs(b[2]-std::cos(t)) < 1e-10);
        }
    }
    auto invalid=p; invalid[0]=0;
    require(std::isnan(umi_kb4::bearing(0,0,invalid)[0]));
    for (size_t i=0; i<p.size(); ++i) {
        invalid=p; invalid[i]=std::numeric_limits<double>::quiet_NaN();
        require(std::isnan(umi_kb4::bearing(720,719,invalid)[0]));
    }
    std::cout << "spherical KB4 bearing: 0..140 degrees, no radius clipping: PASS\n";
}
