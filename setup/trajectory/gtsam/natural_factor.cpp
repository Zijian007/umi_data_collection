// Four-observation KB4 factor for the fixed-camera poster VIO graph.
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "kb4.h"

void bind_poster_factors(pybind11::module_& m);

namespace {
using namespace gtsam;
using umi::Intrinsics;
using umi::project;

class NaturalTrackFactor final : public NoiseModelFactor {
 public:
  NaturalTrackFactor(const SharedNoiseModel& noise, const KeyVector& keys,
                     const Matrix4& extrinsic, const Intrinsics& intrinsics, const Matrix& pixels)
      : NoiseModelFactor(noise, keys), camera_R_imu_(extrinsic.topLeftCorner<3, 3>()),
        camera_t_imu_(extrinsic.topRightCorner<3, 1>()), intrinsics_(intrinsics), pixels_(pixels) {
    if (pixels.cols() != 2 || pixels.rows() < 1 || keys.size() != size_t(pixels.rows() + 1) ||
        !noise || noise->dim() != size_t(2 * pixels.rows()))
      throw std::invalid_argument("Invalid natural track factor dimensions");
  }

  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<NaturalTrackFactor>(*this);
  }

  Vector unwhitenedError(const Values& state, OptionalMatrixVecType H = nullptr) const override {
    const auto n = pixels_.rows();
    const Point3 world_point = state.at<Point3>(keys().back());
    Vector residual = Vector::Zero(2 * n);
    if (H) {
      for (Eigen::Index i = 0; i < n; ++i) (*H)[i].setZero(2 * n, 6);
      (*H)[n].setZero(2 * n, 3);
    }
    for (Eigen::Index i = 0; i < n; ++i) {
      const Pose3 pose = state.at<Pose3>(keys()[i]);
      const Matrix3 imu_R_world = pose.rotation().matrix().transpose();
      const Vector3 in_imu = imu_R_world * (world_point - pose.translation());
      const Vector3 in_camera = camera_R_imu_ * in_imu + camera_t_imu_;
      if (!in_camera.allFinite() || in_camera.z() <= 1e-3) continue;
      Matrix23 projection;
      residual.segment<2>(2*i) = project(in_camera, intrinsics_, H ? &projection : nullptr)
                                  - pixels_.row(i).transpose();
      if (H) {
        const Matrix23 J = projection * camera_R_imu_;
        (*H)[i].block<2, 3>(2*i, 0) = J * skewSymmetric(in_imu);
        (*H)[i].block<2, 3>(2*i, 3) = -J;
        (*H)[n].block<2, 3>(2*i, 0) = J * imu_R_world;
      }
    }
    return residual;
  }

 private:
  Matrix3 camera_R_imu_;
  Vector3 camera_t_imu_;
  Intrinsics intrinsics_;
  Matrix pixels_;
};
}  // namespace

PYBIND11_MODULE(_umi_gtsam_factors, m) {
  pybind11::module_::import("gtsam");
  bind_poster_factors(m);
  pybind11::class_<NaturalTrackFactor, gtsam::NoiseModelFactor, std::shared_ptr<NaturalTrackFactor>>(
      m, "NaturalTrackFactor")
      .def(pybind11::init<const gtsam::SharedNoiseModel&, const gtsam::KeyVector&,
                          const gtsam::Matrix4&, const Intrinsics&, const gtsam::Matrix&>());
}
