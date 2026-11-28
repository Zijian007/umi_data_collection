"""外部引擎的文件/日志协议替身；阶段编排和定位合并运行真实项目代码。"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from umi.reconstruction.estimation.constraint_quality import scale_consistency

REAL_RUN = subprocess.run


class FakeEngines:
    def __init__(self, root: Path, *, frames: int = 100):
        self.root, self.frames = root, frames
        self.calls = []
        self.bad_tries = set()
        self.scale_changes = {}
        self.failures = {}
        self.map_returncode = 0
        self.map_count = 1
        self.orb = root / 'orb'
        for name in ('Examples/Monocular-Inertial/x3_mono_slam', 'Vocabulary/ORBvoc.txt', 'lib/libORB_SLAM3.so'):
            path = self.orb / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fake engine')
        self.environment = dict(UMI_PY=sys.executable, UMI_PY_TOOLS=sys.executable,
                                ORB3_DIR=str(self.orb), ORB3_VOCAB=str(self.orb / 'Vocabulary/ORBvoc.txt'))

    def fuse(self, request, **kwargs):
        """Stand-in for ``fusion_runner.execute``; writes the files the real worker produces."""
        from umi.reconstruction.segment import fusion_runner
        directory = request.output
        self.calls.append(('fuse', request, kwargs))
        if self.failures.get('fuse', 0):
            raise fusion_runner.FusionFailed('NO_USABLE_LENS')
        directory.mkdir()
        number = int(directory.parent.name.removeprefix('try_'))
        chains = [[[0, self.frames-1], 'full']]
        if number in self.bad_tries:
            chains = [[[0, self.frames//2-1], 'full'], [[self.frames//2, self.frames-1], 'full']]
        (directory / 'fuse_info.json').write_text(json.dumps(dict(
            scale_consistency=scale_consistency([dict(lens='front', chain=0, sid=0,
                scale_initial=1., scale_final=self.scale_changes.get(number, 1.))]),
            frames=self.frames, init_lens='front', chains=dict(front=chains))))
        (directory / 'traj_tum.txt').write_text(''.join(
            f'{1+k*1001/30000:.9f} {k*.01:.6f} 0 0 0 0 0 1\n' for k in range(self.frames)))

    def patches(self, run=None, fuse=None):
        """Patch the external-engine and fusion-worker boundaries together."""
        from unittest.mock import patch
        from umi.perception import runtime
        from umi.reconstruction.segment import fusion_runner
        return (patch.object(runtime.subprocess, 'run', side_effect=run or self),
                patch.object(fusion_runner, 'execute', side_effect=fuse or self.fuse))

    def write_dataset(self, output: Path) -> None:
        cam, imu = output / 'mav0/cam0', output / 'mav0/imu0'
        cam.mkdir(parents=True, exist_ok=True)
        imu.mkdir(parents=True, exist_ok=True)
        (cam / 'data.csv').write_text('#timestamp [ns],filename\n' + ''.join(
            f'{round((1+k*1001/30000)*1e9)},{k}.png\n' for k in range(self.frames)))
        (imu / 'data.csv').write_text('#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n1000000000,0,0,0,0,0,9.81\n5000000000,0,0,0,0,0,9.81\n')

    def __call__(self, command, **kwargs):
        command = list(map(str, command))
        if command[-1:] == ['--capabilities'] and Path(command[0]).name == 'x3_mono_slam':
            return SimpleNamespace(returncode=0, stdout='{}')
        output = kwargs['stdout']
        if 'umi.perception.tools.make_lens_video' in command:
            pos = command.index('umi.perception.tools.make_lens_video')
            target = Path(command[pos + 3])
            self.calls.append(('video', command, kwargs))
            code = self.failures.get('video', 0)
            if not code:
                target.write_bytes(b'lens video')
            return SimpleNamespace(returncode=code)
        if './x3_mono_slam' in command:
            kind = 'localize' if '--localization_only' in command else 'map'
            frames = Path(command[command.index('--output_frames_csv') + 1])
            self.calls.append((kind, command, kwargs))
            code = self.failures.get(kind, 0)
            if code:
                output.write(f'{kind} failed\n')
                return SimpleNamespace(returncode=code)
            if '--save_map' in command:
                Path(command[command.index('--save_map') + 1]).write_bytes(b'map archive')
            else:
                for index in range(self.map_count):
                    output.write(f'atlas_map index {index} id {index} keyframes {10+index} mappoints 20\n')
                output.write('atlas_map selected index ' + kwargs['env']['ORB_MAP_INDEX'] + '\n')
            frames.write_text('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw\n' + ''.join(
                f'{k},{k*1001/30000:.9f},2,1,{k*.01:.6f},0,0,0,0,0,1\n' for k in range(self.frames)))
            output.write(f'TRACKED {self.frames} / {self.frames}\n')
            return SimpleNamespace(returncode=self.map_returncode if kind == 'map' else 0)
        if 'umi.perception.tools.merge_frames' in command:
            self.calls.append(('merge', command, kwargs))
            if self.failures.get('merge'):
                return SimpleNamespace(returncode=self.failures['merge'])
            return REAL_RUN(command, **kwargs)
        raise AssertionError(f'unexpected external command: {command}')
