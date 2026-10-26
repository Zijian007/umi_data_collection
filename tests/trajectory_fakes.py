"""外部引擎的文件/日志协议替身；阶段编排和定位合并运行真实项目代码。"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REAL_RUN = subprocess.run


class FakeEngines:
    def __init__(self, root: Path, *, frames: int = 100):
        self.root, self.frames = root, frames
        self.calls = []
        self.bad_tries = set()
        self.failures = {}
        self.map_returncode = 0
        self.map_count = 1
        self.orb = root / 'orb'
        for name in ('Examples/Monocular-Inertial/x3_mono_slam', 'Vocabulary/ORBvoc.txt', 'lib/libORB_SLAM3.so'):
            path = self.orb / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fake engine')
        self.environment = dict(KIWI_PY=sys.executable, KIWI_PY_TOOLS=sys.executable,
                                ORB3_DIR=str(self.orb), ORB3_VOCAB=str(self.orb / 'Vocabulary/ORBvoc.txt'))

    def write_dataset(self, output: Path) -> None:
        cam, imu = output / 'mav0/cam0', output / 'mav0/imu0'
        cam.mkdir(parents=True, exist_ok=True)
        imu.mkdir(parents=True, exist_ok=True)
        (cam / 'data.csv').write_text('#timestamp [ns],filename\n' + ''.join(
            f'{round((1+k*1001/30000)*1e9)},{k}.png\n' for k in range(self.frames)))
        (imu / 'data.csv').write_text('#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n1000000000,0,0,0,0,0,9.81\n5000000000,0,0,0,0,0,9.81\n')

    def __call__(self, command, **kwargs):
        command = list(map(str, command))
        output = kwargs['stdout']
        if any(Path(arg).name == 'make_lens_video.py' for arg in command):
            pos = next(i for i, arg in enumerate(command) if Path(arg).name == 'make_lens_video.py')
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
        if any(Path(arg).name == 'merge_frames.py' for arg in command):
            self.calls.append(('merge', command, kwargs))
            if self.failures.get('merge'):
                return SimpleNamespace(returncode=self.failures['merge'])
            return REAL_RUN(command, **kwargs)
        if any(Path(arg).name in ('fusion_cli.py', 'fusion.py', 'kiwi_fuse3s.py') for arg in command):
            pos = next(i for i, arg in enumerate(command) if Path(arg).name in ('fusion_cli.py', 'fusion.py', 'kiwi_fuse3s.py'))
            directory = Path(command[pos + 6])
            self.calls.append(('fuse', command, kwargs))
            code = self.failures.get('fuse', 0)
            if code:
                output.write('NO_USABLE_LENS\n')
                return SimpleNamespace(returncode=code)
            directory.mkdir()
            number = int(directory.parent.name.removeprefix('try_'))
            chains = [[[0, self.frames-1], 'full']]
            if number in self.bad_tries:
                chains = [[[0, self.frames//2-1], 'full'], [[self.frames//2, self.frames-1], 'full']]
            (directory / 'fuse_info.json').write_text(json.dumps(dict(
                frames=self.frames, init_lens='front', chains=dict(front=chains))))
            (directory / 'traj_tum.txt').write_text(''.join(
                f'{1+k*1001/30000:.9f} {k*.01:.6f} 0 0 0 0 0 1\n' for k in range(self.frames)))
            return SimpleNamespace(returncode=0)
        raise AssertionError(f'unexpected external command: {command}')
