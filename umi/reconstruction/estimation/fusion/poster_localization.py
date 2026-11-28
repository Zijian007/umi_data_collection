"""Convert metric posterior-camera initialization into hints in each fixed ORB map."""

import numpy as np
from scipy.spatial.transform import Rotation

from umi.core.geometry import imu_from_camera
from umi.reconstruction.estimation.fusion.poster_initialization import fit_front_placement


def write_map_pose_priors(maps, seed, front_camera, frame_rate_hz, parameters, directory):
    body = seed.trajectory.T_world_imu
    imu_from_front = imu_from_camera(front_camera)
    camera = body @ imu_from_front
    camera_from_imu = np.linalg.inv(imu_from_front)
    supported = seed.supported_frames()
    supported_indices = np.flatnonzero(supported)
    paths, evidence = {}, []
    for orb_map in maps:
        index = orb_map['index']
        matched = {}
        for keyframe in orb_map['keyframes']:
            exposure = keyframe['time'] * frame_rate_hz
            frame = round(exposure)
            if 0 <= frame < len(body) and supported[frame] and abs(frame - exposure) <= .02:
                matched[frame] = keyframe
        frames = np.array(sorted(matched), dtype=int)
        audit = dict(map_index=index, common_keyframes=len(frames), prior_frames=0)
        evidence.append(audit)
        if len(frames) < max(parameters.minimum_common_frames, 6):
            audit['status'] = 'insufficient_keyframe_overlap'
            continue
        centers = np.asarray([matched[frame]['position'] for frame in frames])
        rotations = Rotation.from_quat([matched[frame]['quaternion'] for frame in frames]).as_matrix()
        train = np.arange(len(frames)) % 4 != 2
        placement, scale_std, _, _ = fit_front_placement(
            centers[train], rotations[train], body[frames[train]],
            camera_from_imu[:3, :3], camera_from_imu[:3, 3], parameters)
        scale, rotation, translation = placement['s'], placement['R'], placement['t']
        check = camera[frames[~train]]
        position_error = scale * (centers[~train] @ rotation.T) + translation - check[:, :3, 3]
        rms = float(np.sqrt(np.mean(np.sum(position_error**2, axis=1))))
        angle = Rotation.from_matrix(check[:, :3, :3].transpose(0, 2, 1) @ rotation @ rotations[~train]).magnitude()
        angle_deg = float(np.rad2deg(np.median(angle)))
        audit.update(scale=scale, relative_scale_std=scale_std, held_out_keyframes=len(check),
                     held_out_position_rms_m=rms, held_out_rotation_median_deg=angle_deg)
        if (scale_std is None or scale_std > parameters.maximum_relative_scale_std
                or rms > parameters.other_lens_rms_m or angle_deg > parameters.other_lens_rotation_deg):
            audit['status'] = 'placement_rejected'
            continue
        # Map units apply to camera centers only. The physical IMU lever arm was
        # applied above in meters, before converting the camera center.
        poses = camera[supported_indices]
        positions = (poses[:, :3, 3] - translation) @ rotation / scale
        quaternions = Rotation.from_matrix(rotation.T @ poses[:, :3, :3]).as_quat()
        path = directory / f'm{index}.txt'
        limit = np.full((len(positions), 1), .25 / scale)
        np.savetxt(path, np.column_stack((supported_indices, positions, quaternions, limit)),
                   fmt=['%d'] + ['%.9g'] * 8)
        paths[index] = path
        audit.update(status='accepted', prior_frames=len(positions))
    return paths, evidence
