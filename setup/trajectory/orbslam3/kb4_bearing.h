// Spherical KB4 inverse for the experimental X3 forward-domain ORB build.
// Pixel radii are not angles and must not be clipped to pi/2.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace umi_kb4 {
constexpr double pi = 3.14159265358979323846;
constexpr double minimum_z = 0.08715574274765817;  // cos(85 degrees)

inline std::array<double, 3> bearing(double u, double v, const std::array<double, 8>& p) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double value : p) if (!std::isfinite(value)) return {{nan, nan, nan}};
    if (!(p[0] > 0 && p[1] > 0)) return {{nan, nan, nan}};
    const double x = (u-p[2])/p[0], y = (v-p[3])/p[1];
    const double rd = std::hypot(x, y);
    if (!std::isfinite(rd)) return {{nan, nan, nan}};
    if (rd < 1e-12) return {{0., 0., 1.}};
    double theta = std::min(rd, pi/2);
    auto radial = [&](double t) {
        double t2 = t*t;
        return t*(1+t2*(p[4]+t2*(p[5]+t2*(p[6]+t2*p[7]))));
    };
    auto derivative = [&](double t) {
        double t2 = t*t;
        return 1+t2*(3*p[4]+t2*(5*p[5]+t2*(7*p[6]+t2*9*p[7])));
    };
    for (int i=0; i<40; ++i) {
        double d = derivative(theta);
        if (!(d > 0) || !std::isfinite(d)) return {{nan, nan, nan}};
        double step = (radial(theta)-rd)/d;
        theta = std::max(0., std::min(pi-1e-6, theta-std::max(-.25, std::min(.25, step))));
        if (std::abs(step) < 1e-12) break;
    }
    if (!(derivative(theta) > 0) || !std::isfinite(theta) ||
        std::abs(radial(theta)-rd)*std::max(p[0], p[1]) > 1e-4)
        return {{nan, nan, nan}};
    const double s = std::sin(theta)/rd;
    return {{x*s, y*s, std::cos(theta)}};
}
}
