# Specialist reference and local adaptations

The initialization, feature geometry/reassociation, batch triangulation,
warm-started cull/reoptimization, and keyframe/high-rate separation in this
directory reference the local **specialist** implementation, commit
`da29719cdbaa366df0599984068ad4553d9cc86d`, Copyright (c) 2026 Hybrid Robotics,
licensed under MIT. Its complete license is retained in
[LICENSE.specialist](LICENSE.specialist).

Reference paths under `src/specialist/vi_ba/`:

- `tag_detection.py`, `geometry.py`, `single_camera.py`: planar PnP hypotheses,
  static-anchor consensus and staged camera preparation.
- `features.py`, `tracks.py`, `init.py`: calibrated essential filtering, bounded
  revisit candidates and descriptors, track caps and batched triangulation.
- `reference_solve.py`, `landmark_cull.py`: short initial optimization, residual
  rejection and warm continuation.
- `compose.py`: a per-camera rigid base-tag Pose3 variable with a soft seed
  prior, self corner reprojection, and cross-camera corner factors connecting
  observer, owner and owner mount. UMI uses analytic KB4 Jacobians, its own
  time-node expansion and cumulative base-observation cull budgets.
- `highrate.py`, `build.py`: keyframe states and IMU-rate reconstruction.

Local implementations use KB4 and the known rigid ArUco board rather than MEI
and independent AprilTag poses. Board consensus does not average moving camera
poses; ambiguous/non-majority evidence is rejected. Warm continuation retains
physical state times and cumulative rejection budgets. High-rate recovery uses
fixed-endpoint conditional inertial optimization, rather than the reference's
distributed endpoint-error correction. Label/natural-pixel holdouts and frozen
input/publication contracts are UMI additions. No dependency on the specialist
package, its GPU kernels or recording format is introduced.
