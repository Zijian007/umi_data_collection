"""场景任务报告与运行清单（report.txt / episode.json）写入。"""

from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np

from software.infrastructure import provenance as artifact_identity
from software.camera.geometry import describe_tcp_frame
from software.scene.asset_identity import artifact_hashes
from software.scene.models import Episode, ArmTrajectory
from software.scene.result_io import is_local_scene
from software.scene.pipeline import synchronization
from software.infrastructure.file_store import write_json


def write_episode_result(episode: Episode, tracks: dict[str, ArmTrajectory], directory: Path, *,
                  common_interval_s: list[float], scene_record: dict, input_identity: dict,
                  video_offsets_s: dict[str, float], elapsed_s: float, map_path: Path | None,
                  gripper_width: dict | None = None, tag_distance: dict | None = None) -> None:
    """写任务报告与资产清单；不更新 results 指针或人工验收状态。"""
    local = is_local_scene(scene_record)
    if local and (len(tracks) != 1 or map_path is not None):
        raise ValueError('局部回退只允许单手且不能引用共享地图')
    sync_record = episode.synchronization
    sync_lines = [
        f'同步：{sync_record["method"]}；offsets_s={synchronization.format_offsets(video_offsets_s)}',
        f'同步确认：{sync_record["confirmed"]}；依据：{sync_record["note"]}',
    ]
    if sync_record.get('raw_audio_offset_s') is not None:
        sync_lines.append(
            '同步原始音频偏移：'
            f'{synchronization.format_offsets(sync_record["raw_audio_offset_s"])}')
    if sync_record.get('peak_ratio') is not None:
        sync_lines.append(f'同步音频峰值比：{float(sync_record["peak_ratio"]):.2f}')
    if sync_record.get('video_offset_status') is not None:
        sync_lines.append(f'同步视频时间轴：{sync_record["video_offset_status"]}')
    text = (f'场景任务：{episode.root.name}\n采集批次：{episode.session.name}\n坐标：{scene_record["frame_id"]}\n'
        f'操作指令：{episode.instruction or "未填写"}\n'
        f'手：{", ".join(tracks)}\n共同有效时间：{common_interval_s[0]:.6f}–{common_interval_s[1]:.6f} s\n'
        + '\n'.join(sync_lines) + '\n'
        + (f'夹爪宽度：{", ".join(gripper_width)}\n' if gripper_width else '')
        + (f'夹爪标签中心三维间距：{", ".join(tag_distance)}\n' if tag_distance else '')
        + '每手 trajectory.csv 保存 TCP 位姿及已配置的测量；缺失测量不代表闭合。\n'
        + ('独立回退：仅本次录像的局部米制坐标，不与其他任务/场景共享。\n'
           if local else '轨迹共享米制坐标；左右手分别融合。\n')
        + '实际误差：未测量；TCP 使用 CAD 名义几何。\n')
    if local:
        text += f'共享场景失败原因：{scene_record["fallback"]["shared_failure"]}\n'
    # 预检先独立落盘，保证算法失败时也有证据；成功后收入唯一运行清单。
    preflight_path = directory / 'preflight_report.json'
    preflight = json.loads(preflight_path.read_text()) if preflight_path.is_file() else None
    text += '\n原片检查：' + ('跳过' if preflight is None else 'PASS' if preflight['passed'] else 'FAIL') + '\n'
    if preflight is not None:
        for arm, video in preflight.get('videos', {}).items():
            for check in video['checks']:
                if check['status'] != 'PASS':
                    text += f'{arm} [{check["status"]}] {check["name"]}：{check["message"]}\n'
    for arm, track in tracks.items():
        arm_metadata = track.metadata
        text += '\n' + '\n'.join([
            f'{arm} {"独立局部" if local else "场景"}轨迹', f'相机：{input_identity["cameras"][arm]["serial"]}',
            'TCP 自身坐标：' + describe_tcp_frame(track.tcp_frame),
            f'视频时间偏移：{video_offsets_s[arm]:+.9f} s（t_episode = t_video + offset）',
            f'时长：{arm_metadata["duration_s"]:.2f} s；TCP 路程：{arm_metadata["path_length_m"]:.3f} m',
            f'地图位置约束：{arm_metadata["constraint_fraction"]["full"]:.1%}',
            f'最长无视觉：{arm_metadata["fusion"]["maximum_visual_gap_s"]:.3f} s',
            ('独立融合诊断保存在 independent/debug。' if local else
             f'视觉一致性 RMS：{arm_metadata["fusion"]["visual_consistency_rms_m"] * 1000:.2f} mm（内部指标）'),
        ]) + '\n'
        poster = arm_metadata['fusion'].get('environment_poster')
        if poster is not None:
            text += f'环境海报：{poster["status"]}；真实轨迹误差未测量。\n'
            if 'final_reprojection_rms_px' in poster:
                text += f'海报重投影 RMS：{poster["final_reprojection_rms_px"]:.3f} px（内部指标）\n'
            metrics = (poster.get('final_reprojection_metrics') or poster.get('reprojection_metrics')
                       or poster.get('initial_reprojection_metrics'))
            if metrics:
                text += (f'海报角点残差：median {metrics.get("median_px", float("nan")):.3f} px；'
                         f'P95 {metrics.get("p95_px", float("nan")):.3f} px；'
                         f'帧数 {len(metrics.get("frame_rms_px", []))}；'
                         f'ID 数 {len(metrics.get("id_rms_px", {}))}\n')
            if 'reason' in poster:
                text += f'海报诊断原因：{poster["reason"]}\n'
        for lens, recovery in arm_metadata.get('localization', {}).items():
            text += (f'{lens} 双向定位：正向 {recovery["forward_frames"]} 帧，'
                     f'反向补入 {recovery["added_frames"]} 帧；{recovery["status"]}\n')
        if arm in (tag_distance or {}):
            text += f'夹爪标签中心间距有效比例：{tag_distance[arm]["valid_fraction"]:.1%}；单位 mm\n'
    (directory / 'report.txt').write_text(text, encoding='utf-8')
    assets = artifact_hashes(directory, (*tracks, 'devices'))
    for name in ('report.txt', 'bimanual.csv', 'run_report.json'):
        if (directory / name).is_file():
            assets[name] = artifact_identity.file_hash(directory / name)
    if (directory/'fallback.json').exists():
        assets['fallback.json'] = artifact_identity.file_hash(directory/'fallback.json')
    primary_times = tracks.get('left', next(iter(tracks.values()))).episode_times_s
    metadata = dict(kind='umi_local_episode' if local else 'umi_scene_episode', schema_version=4,
        status='local_ok' if local else 'ok', name=episode.root.name, session=episode.session.name,
        kit=dict(id=episode.session.kit['id'], arms={arm: episode.session.kit['arms'][arm] for arm in tracks}),
        scene={key: value for key, value in scene_record.items() if key != 'session'},
        arms={arm: arm for arm in tracks}, common_interval_s=common_interval_s,
        rows=int(np.count_nonzero((primary_times >= common_interval_s[0]) & (primary_times <= common_interval_s[1]))),
        instruction=episode.instruction, synchronization=episode.synchronization,
        inputs=input_identity, assets=assets, elapsed_s=elapsed_s, actual_error='not_measured',
        tcp_frames={arm: track.tcp_frame for arm, track in tracks.items()},
        arm_results={arm: track.metadata for arm, track in tracks.items()}, preflight=preflight,
        map=None if local else dict(
            path=os.path.relpath(map_path, directory),
            sha256=artifact_identity.file_hash(map_path),
            map_epoch=(episode.map_selection or {}).get('epoch'),
            map_version=episode.map_version,
            selection=episode.map_selection,
        ),
        map_selection=None if local else episode.map_selection)
    if gripper_width:
        metadata['gripper_width'] = gripper_width
    if tag_distance:
        metadata['tag_distance'] = tag_distance
    if episode.scene.overrides is not None:
        metadata['runtime_overrides'] = episode.scene.overrides.snapshot()
    write_json(directory / 'episode.json', metadata)
    preflight_path.unlink(missing_ok=True)

