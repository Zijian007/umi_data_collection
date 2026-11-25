// Rear-camera poster constraints in the independent visual-inertial graph.
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>

#include "kb4.h"

namespace {
using namespace gtsam;
using umi::Intrinsics;

class PosterCornerFactor final : public NoiseModelFactor {
 public:
  PosterCornerFactor(const SharedNoiseModel& noise, const KeyVector& keys, const Matrix4& extrinsic,
                     const Intrinsics& intrinsics, const Matrix& points, const Matrix& pixels)
      : NoiseModelFactor(noise, keys), extrinsic_(extrinsic), intrinsics_(intrinsics),
        points_(points), pixels_(pixels) {
    if ((keys.size() != 2 && keys.size() != 3) || points.cols() != 3 || pixels.cols() != 2 ||
        points.rows() != pixels.rows() || !noise || noise->dim() != size_t(2 * points.rows()))
      throw std::invalid_argument("Invalid poster corner factor dimensions");
  }

  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<PosterCornerFactor>(*this);
  }

  Vector unwhitenedError(const Values& state, OptionalMatrixVecType H = nullptr) const override {
    const Pose3 body = state.at<Pose3>(keys()[0]), board = state.at<Pose3>(keys()[1]);
    const Pose3 imu_from_board = body.between(board);
    const bool refine = keys().size() == 3;
    Matrix3 correction = Matrix3::Identity();
    if (refine) correction = state.at<Rot3>(keys()[2]).matrix();
    const Matrix3 base = extrinsic_.topLeftCorner<3, 3>();
    const Matrix3 rotation = correction * base;
    Vector residual(2 * points_.rows());
    if (H) {
      (*H)[0].resize(residual.size(), 6);
      (*H)[1].resize(residual.size(), 6);
      if (refine) (*H)[2].resize(residual.size(), 3);
    }
    for (Eigen::Index i = 0; i < points_.rows(); ++i) {
      const Vector3 corner = points_.row(i).transpose();
      const Vector3 in_imu = imu_from_board.transformFrom(corner);
      const Vector3 in_camera = rotation * in_imu + extrinsic_.topRightCorner<3, 1>();
      Matrix23 projection;
      residual.segment<2>(2*i) = umi::project(in_camera, intrinsics_, H ? &projection : nullptr)
                                - pixels_.row(i).transpose();
      if (H) {
        const Matrix23 J = projection * rotation;
        const Matrix23 board_J = J * imu_from_board.rotation().matrix();
        (*H)[0].block<2, 3>(2*i, 0) = J * skewSymmetric(in_imu);
        (*H)[0].block<2, 3>(2*i, 3) = -J;
        (*H)[1].block<2, 3>(2*i, 0) = -board_J * skewSymmetric(corner);
        (*H)[1].block<2, 3>(2*i, 3) = board_J;
        if (refine)
          (*H)[2].block<2, 3>(2*i, 0) = -projection * correction * skewSymmetric(base * in_imu);
      }
    }
    return residual;
  }

 private:
  Matrix4 extrinsic_;
  Intrinsics intrinsics_;
  Matrix points_, pixels_;
};

class PosterDepthFactor final : public NoiseModelFactor {
 public:
  PosterDepthFactor(const SharedNoiseModel& noise, const KeyVector& keys, const Matrix4& extrinsic,
                    const Matrix& points, double min_depth, double tau, double sigma)
      : NoiseModelFactor(noise, keys), depth_axis_(extrinsic.block<1, 3>(2, 0)),
        offset_(extrinsic(2, 3)), points_(points), min_depth_(min_depth), tau_(tau), sigma_(sigma) {
    if (keys.size() != 2 || points.cols() != 3 || !noise || noise->dim() != size_t(points.rows()) ||
        !(tau > 0 && sigma > 0))
      throw std::invalid_argument("Invalid poster depth factor dimensions");
  }

  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<PosterDepthFactor>(*this);
  }

  Vector unwhitenedError(const Values& state, OptionalMatrixVecType H = nullptr) const override {
    const Pose3 body = state.at<Pose3>(keys()[0]), board = state.at<Pose3>(keys()[1]);
    const Pose3 imu_from_board = body.between(board);
    Vector residual = Vector::Zero(points_.rows());
    if (H) {
      (*H)[0].setZero(points_.rows(), 6);
      (*H)[1].setZero(points_.rows(), 6);
    }
    for (Eigen::Index i = 0; i < points_.rows(); ++i) {
      const Vector3 corner = points_.row(i).transpose();
      const Vector3 in_imu = imu_from_board.transformFrom(corner);
      const double depth = depth_axis_.dot(in_imu) + offset_;
      const double deficit = (min_depth_ - depth) / tau_;
      if (!(deficit > -8.0)) continue;
      const double softplus = std::max(deficit, 0.0) + std::log1p(std::exp(-std::abs(deficit)));
      residual[i] = (tau_ / sigma_) * softplus;
      if (H) {
        const double derivative = -1.0 / ((1.0 + std::exp(-deficit)) * sigma_);
        const Eigen::RowVector3d J = derivative * depth_axis_;
        const Eigen::RowVector3d board_J = J * imu_from_board.rotation().matrix();
        (*H)[0].block<1, 3>(i, 0) = J * skewSymmetric(in_imu);
        (*H)[0].block<1, 3>(i, 3) = -J;
        (*H)[1].block<1, 3>(i, 0) = -board_J * skewSymmetric(corner);
        (*H)[1].block<1, 3>(i, 3) = board_J;
      }
    }
    return residual;
  }

 private:
  Eigen::RowVector3d depth_axis_;
  double offset_;
  Matrix points_;
  double min_depth_, tau_, sigma_;
};
}  // namespace

void bind_poster_factors(pybind11::module_& m) {
  pybind11::class_<PosterCornerFactor, gtsam::NoiseModelFactor, std::shared_ptr<PosterCornerFactor>>(
      m, "PosterCornerFactor")
      .def(pybind11::init<const gtsam::SharedNoiseModel&, const gtsam::KeyVector&, const gtsam::Matrix4&,
                          const Intrinsics&, const gtsam::Matrix&, const gtsam::Matrix&>());
  pybind11::class_<PosterDepthFactor, gtsam::NoiseModelFactor, std::shared_ptr<PosterDepthFactor>>(
      m, "PosterDepthFactor")
      .def(pybind11::init<const gtsam::SharedNoiseModel&, const gtsam::KeyVector&, const gtsam::Matrix4&,
                          const gtsam::Matrix&, double, double, double>());
}
