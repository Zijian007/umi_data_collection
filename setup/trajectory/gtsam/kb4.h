#pragma once

#include <gtsam/base/Matrix.h>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace umi {
using Intrinsics = Eigen::Matrix<double, 8, 1>;

// Matches Kb4Model.project and project_jacobian, including the optical-axis limit.
inline gtsam::Vector2 project(const gtsam::Vector3& p, const Intrinsics& k, gtsam::Matrix23* derivative) {
  const double r = std::hypot(p.x(), p.y()), z = p.z();
  if (derivative && r < 1e-8 && z <= 1e-12)
    throw std::invalid_argument("KB4 projection is not differentiable on the reverse optical axis");
  const double theta = std::atan2(r, z), t2 = theta * theta;
  const double rd = theta * (1 + t2 * (k[4] + t2 * (k[5] + t2 * (k[6] + t2 * k[7]))));
  const double scale = r > 1e-12 ? rd / r : 0.0;
  if (derivative) {
    if (r < 1e-8) {
      derivative->setZero();
      (*derivative)(0, 0) = k[0] / z;
      (*derivative)(1, 1) = k[1] / z;
      derivative->col(2) = -k.head<2>().cwiseProduct(p.head<2>()) / (z * z);
    } else {
      const double slope = 1 + t2 * (3*k[4] + t2 * (5*k[5] + t2 * (7*k[6] + t2*9*k[7])));
      const double d2 = r*r + z*z;
      const gtsam::Vector2 radial = p.head<2>() / r;
      derivative->leftCols<2>() = scale * gtsam::Matrix2::Identity() +
          (slope * z / d2 - scale) * radial * radial.transpose();
      derivative->col(2) = (-slope / d2) * p.head<2>();
      derivative->row(0) *= k[0];
      derivative->row(1) *= k[1];
    }
  }
  if (p.norm() < 1e-12 || (r < 1e-12 && z < 0))
    return gtsam::Vector2::Constant(std::numeric_limits<double>::quiet_NaN());
  return k.segment<2>(2) + scale * k.head<2>().cwiseProduct(p.head<2>());
}
}  // namespace umi
