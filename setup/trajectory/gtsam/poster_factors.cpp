// Rear-camera poster constraints for map-based and visual-inertial graphs.
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
                     const Intrinsics& intrinsics, const Matrix& points, const Matrix& pixels,
                     const Vector& sigmas = Vector(), double huber = 0.0)
      : NoiseModelFactor(noise, keys), extrinsic_(extrinsic), intrinsics_(intrinsics),
        points_(points), pixels_(pixels), sigmas_(sigmas), huber_(huber) {
    if (sigmas.size() && (points.rows() % 4 || sigmas.size() != 2 * points.rows() ||
                          !sigmas.allFinite() || sigmas.minCoeff() <= 0 || !std::isfinite(huber) || huber <= 0))
      throw std::invalid_argument("Invalid per-tag Huber parameters");
    if ((keys.size() != 2 && keys.size() != 3) || points.cols() != 3 || pixels.cols() != 2 ||
        points.rows() != pixels.rows() || !noise || noise->dim() != size_t(2 * points.rows() + sigmas.size() / 8))
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
    const Eigen::Index tag_count = sigmas_.size() / 8;
    Vector residual(2 * points_.rows() + tag_count);
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
    // Each tag has its own Huber loss on its eight whitened corner coordinates.
    // The extra residual supplies the cost offset. Its IRLS Jacobian is zero.
    for (Eigen::Index tag = 0; tag < tag_count; ++tag) {
      const Eigen::Index row = 8 * tag;
      const auto sigma = sigmas_.segment<8>(row);
      auto pixels = residual.segment<8>(row);
      pixels.array() /= sigma.array();
      const double norm = pixels.norm();
      const double sqrt_weight = std::sqrt(huber_ / std::max(norm, huber_));
      pixels *= sqrt_weight;
      residual[2 * points_.rows() + tag] = std::sqrt(huber_ * std::max(norm - huber_, 0.0));
      if (H) {
        for (Matrix& jacobian : *H) {
          jacobian.middleRows<8>(row).array().colwise() *= sqrt_weight / sigma.array();
          jacobian.row(2 * points_.rows() + tag).setZero();
        }
      }
    }
    return residual;
  }

 private:
  Matrix4 extrinsic_;
  Intrinsics intrinsics_;
  Matrix points_, pixels_;
  Vector sigmas_;
  double huber_;
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

Matrix3 right_jacobian_inverse(const Vector3& phi) {
  const double theta = phi.norm();
  const Matrix3 K = skewSymmetric(phi);
  if (theta < 1e-6) {
    return Matrix3::Identity() + 0.5 * K;
  }
  const double coeff = 1.0 / (theta * theta) - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
  return Matrix3::Identity() + 0.5 * K + coeff * (K * K);
}

class FrontMapFactor final : public NoiseModelFactor {
 public:
  FrontMapFactor(const SharedNoiseModel& noise, const KeyVector& keys, const Matrix3& measured_rotation,
                 const Vector3& measured_position, const Vector3& lever)
      : NoiseModelFactor(noise, keys),
        measured_rotation_(measured_rotation),
        measured_position_(measured_position),
        lever_(lever) {
    if (keys.size() != 3 || !noise || noise->dim() != 6)
      throw std::invalid_argument("Invalid front map factor dimensions");
  }

  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<FrontMapFactor>(*this);
  }

  Vector unwhitenedError(const Values& state, OptionalMatrixVecType H = nullptr) const override {
    const Pose3 body = state.at<Pose3>(keys()[0]);
    const Pose3 placement = state.at<Pose3>(keys()[1]);
    const Vector scale_vec = state.at<Vector>(keys()[2]);
    if (scale_vec.size() != 1) throw std::invalid_argument("FrontMapFactor scale must be 1D");
    const double scale = std::exp(scale_vec[0]);
    const Matrix3 R_map = placement.rotation().matrix();
    const Matrix3 R_body = body.rotation().matrix();
    const Matrix3 relative = (R_map * measured_rotation_).transpose() * R_body;
    const Vector3 angle = Rot3::Logmap(Rot3(relative));
    const Vector3 center = body.translation() + R_body * lever_;
    const Vector3 position = (R_map.transpose() * (center - placement.translation())) / scale;
    Vector residual(6);
    residual.head<3>() = angle;
    residual.tail<3>() = position - measured_position_;
    if (H) {
      const Matrix3 angular = right_jacobian_inverse(angle);
      const Matrix3 linear = (R_map.transpose() * R_body) / scale;
      Matrix& body_jac = (*H)[0];
      body_jac.setZero(6, 6);
      body_jac.topLeftCorner<3, 3>() = angular;
      body_jac.bottomLeftCorner<3, 3>() = -linear * skewSymmetric(lever_);
      body_jac.bottomRightCorner<3, 3>() = linear;
      Matrix& map_jac = (*H)[1];
      map_jac.setZero(6, 6);
      map_jac.topLeftCorner<3, 3>() = -angular * relative.transpose() * measured_rotation_.transpose();
      map_jac.bottomLeftCorner<3, 3>() = skewSymmetric(position);
      map_jac.bottomRightCorner<3, 3>() = -Matrix3::Identity() / scale;
      Matrix& scale_jac = (*H)[2];
      scale_jac.setZero(6, 1);
      scale_jac.bottomRows<3>() = -position;
    }
    return residual;
  }

 private:
  Matrix3 measured_rotation_;
  Vector3 measured_position_;
  Vector3 lever_;
};
}  // namespace

void bind_poster_factors(pybind11::module_& m) {
  pybind11::class_<PosterCornerFactor, gtsam::NoiseModelFactor, std::shared_ptr<PosterCornerFactor>>(
      m, "PosterCornerFactor")
      .def(pybind11::init<const gtsam::SharedNoiseModel&, const gtsam::KeyVector&, const gtsam::Matrix4&,
                          const Intrinsics&, const gtsam::Matrix&, const gtsam::Matrix&>())
      .def_static("per_tag_huber", [](const gtsam::KeyVector& keys, const gtsam::Matrix4& ext,
                                      const Intrinsics& k, const gtsam::Matrix& points, const gtsam::Matrix& pixels,
                                      const gtsam::Vector& sigmas, double huber) {
        if (!sigmas.size()) throw std::invalid_argument("Missing per-tag corner sigmas");
        auto noise = gtsam::noiseModel::Unit::Create(sigmas.size() + sigmas.size() / 8);
        return std::make_shared<PosterCornerFactor>(noise, keys, ext, k, points, pixels, sigmas, huber);
      });
  pybind11::class_<PosterDepthFactor, gtsam::NoiseModelFactor, std::shared_ptr<PosterDepthFactor>>(
      m, "PosterDepthFactor")
      .def(pybind11::init<const gtsam::SharedNoiseModel&, const gtsam::KeyVector&, const gtsam::Matrix4&,
                          const gtsam::Matrix&, double, double, double>());
  pybind11::class_<FrontMapFactor, gtsam::NoiseModelFactor, std::shared_ptr<FrontMapFactor>>(
      m, "FrontMapFactor")
      .def(pybind11::init<const gtsam::SharedNoiseModel&, const gtsam::KeyVector&, const gtsam::Matrix3&,
                          const gtsam::Vector3&, const gtsam::Vector3&>());
}
