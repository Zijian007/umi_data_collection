# Numerical method provenance

The scene metric alignment, front-to-rear placement, and per-arm fixed-frame
fusion in `umi/estimation/fixed_map/geometry.py`,
`umi/estimation/fixed_map/fusion.py`, and
`umi/estimation/inertial.py` adapt ideas and implementation structure
from **KIWI: Kinematic Interface for the Wild**, public reference release 0.1.0,
commit `22860a30dc1c2e9e5f9a2e55c4955a4fba84646e`:

- <https://github.com/TToTMooN/KIWI_software/blob/22860a30dc1c2e9e5f9a2e55c4955a4fba84646e/kiwi/track/metric_align.py>
- <https://github.com/TToTMooN/KIWI_software/blob/22860a30dc1c2e9e5f9a2e55c4955a4fba84646e/kiwi/track/scene_frame.py>
- <https://github.com/TToTMooN/KIWI_software/blob/22860a30dc1c2e9e5f9a2e55c4955a4fba84646e/kiwi/state/fusion.py>
- <https://github.com/TToTMooN/KIWI_software/blob/22860a30dc1c2e9e5f9a2e55c4955a4fba84646e/kiwi/sync/audio.py>

Authors: Benjamin C. Yang, Weiying Wang, Shenggao Li, Keming Yan,
Sasha Wilkinson, Zelin Wang, Yip Fun Yeung, and Lingfeng Sun.
Upstream is licensed under Apache-2.0; a copy is in [LICENSE](LICENSE).

`umi/perception/audio/kiwi_audio.py` adapts KIWI's 48 kHz signed 16-bit decoder,
spectral landmarks, offset voting, GCC-PHAT refinement, and drift validation.
UMI checks matching camera models from INSV metadata and matching recording
modes before applying KIWI's zero A/V correction for a matched pair. It
converts the estimate to UMI's left-referenced task clock.

Local adaptations add exact IMU interval boundaries, observability and residual
gates, parallel/antiparallel gravity handling, explicit scene/calibration versions,
configuration-driven noise, and integration with the X3 timestamp convention.
These files are modified adaptations, not unmodified upstream KIWI code.
Shared-map scale estimation and motion-segment screening now call the common
solver in `umi/estimation/fusion/alignment.py`; the three-node system in
`geometry.py` is retained only as an observability check. Shared maps fit one
scale, gravity vector and accelerometer bias across accepted segments, rather
than adopting independent per-segment map placements.
The separate per-recording implementation is in `umi/estimation/fusion/solver.py`;
it runs in a worker process through `umi/workflows/segment/fusion_runner.py`.

The task-local poster-VIO implementation additionally references specialist
commit `da29719cdbaa366df0599984068ad4553d9cc86d` (MIT, Hybrid Robotics).
The reference paths, local adaptations and complete MIT notice are retained in
[poster_vio/NOTICE.md](../../umi/estimation/poster_vio/NOTICE.md).
