"""Two commands, one recording layout.

``umi-traj`` (``software.trajectory``) reads an X3 360 4K60 ``.insv`` and writes a gripper trajectory.
``umi-calib`` (``software.calibration``) solves a Basalt camera-IMU calibration. The trajectory
turns that file into a bundle with ``software.trajectory.calibration_assets``.

X3 bytes are read through ``software.camera.x3`` (container, IMU, stream geometry). The frame those
readers must agree on is ``software.camera.x3.frame``.
"""
