# Numerical method provenance

The scene metric alignment, front-to-rear placement, and per-arm fixed-frame
fusion in `geometry.py` and `fusion.py` adapt ideas and implementation structure
from **KIWI: Kinematic Interface for the Wild**, public reference release 0.1.0,
commit `22860a30dc1c2e9e5f9a2e55c4955a4fba84646e`:

- <https://github.com/TToTMooN/KIWI_software/blob/22860a30dc1c2e9e5f9a2e55c4955a4fba84646e/kiwi/track/metric_align.py>
- <https://github.com/TToTMooN/KIWI_software/blob/22860a30dc1c2e9e5f9a2e55c4955a4fba84646e/kiwi/track/scene_frame.py>
- <https://github.com/TToTMooN/KIWI_software/blob/22860a30dc1c2e9e5f9a2e55c4955a4fba84646e/kiwi/state/fusion.py>

Authors: Benjamin C. Yang, Weiying Wang, Shenggao Li, Keming Yan,
Sasha Wilkinson, Zelin Wang, Yip Fun Yeung, and Lingfeng Sun.
Upstream is licensed under Apache-2.0; a copy is in [LICENSE](LICENSE).

Local adaptations add exact IMU interval boundaries, observability and residual
gates, parallel/antiparallel gravity handling, explicit scene/calibration versions,
configuration-driven noise, and integration with the X3 timestamp convention.
These files are modified adaptations, not unmodified upstream KIWI code.
The existing `trajectory/algorithms/kiwi_fuse3s.py` remains the separate legacy
per-recording implementation.
