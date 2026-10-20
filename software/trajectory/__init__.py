"""``.insv`` to a gripper trajectory.

``x3_extract`` writes the EuRoC timestamps and IMU. ``run_kiwi_segment.sh`` builds the lens video,
the map and the fusion. ``postprocess`` writes ``traj_tcp.txt``. ``cli.run_recording`` is that
sequence for one file; ``./umi-traj`` is the command.

The picture they share is ``frame``. A Basalt ``calibration.json`` becomes the bundle the runner
reads in ``calibration`` (this package). Solving that json is ``software.calibration``.
"""
