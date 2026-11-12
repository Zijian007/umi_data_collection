"""INSV v3 factory lens parameters and MEI projection, independent of SLAM.

Field order and projection follow Insta360's offset specification. The same MEI
equations are used by specialist_camera. For the supported X3 installation,
metadata lens1 is the working lens (project front, specialist rear); lens0 is
the environment lens (project rear, specialist front).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from software.camera.x3.insv_reader import read_metadata, metadata_summary


FIELDS = ('xi fx fy cx cy yaw pitch roll tx ty tz k1 k2 k3 p1 p2 '
          'width height lens_type').split()
WORKING_LENS_INDEX = 1
ENVIRONMENT_LENS_INDEX = 0
LENS_ASSIGNMENT = {'working': WORKING_LENS_INDEX, 'environment': ENVIRONMENT_LENS_INDEX}


@dataclass(frozen=True)
class Mei:
    fx: float
    fy: float
    cx: float
    cy: float
    xi: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float

    def project(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, float).reshape(-1, 3)
        length = np.linalg.norm(p, axis=1)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            sphere = p / length[:, None]
            valid = sphere[:, 2] > -(self.xi if self.xi <= 1 else 1 / self.xi)
            xy = sphere[:, :2] / (sphere[:, 2:3] + self.xi)
            xd, yd = self._distort(xy[:, 0], xy[:, 1])
            uv = np.column_stack((self.fx * xd + self.cx, self.fy * yd + self.cy))
        uv[~valid | ~np.isfinite(uv).all(axis=1)] = np.nan
        return uv

    def _distort(self, x, y):
        r2 = x*x + y*y
        radial = 1 + self.k1*r2 + self.k2*r2*r2 + self.k3*r2**3
        return (x*radial + 2*self.p1*x*y + self.p2*(r2 + 2*x*x),
                y*radial + 2*self.p2*x*y + self.p1*(r2 + 2*y*y))

    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        uv = np.asarray(pixels, float).reshape(-1, 2)
        xd, yd = ((uv - [self.cx, self.cy]) / [self.fx, self.fy]).T
        x, y = xd.copy(), yd.copy()
        # Newton inversion of radial + tangential distortion. Validate the result
        # by forward projection; a finite iterate alone is not convergence.
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            for _ in range(30):
                r2 = x*x + y*y
                radial = 1 + self.k1*r2 + self.k2*r2*r2 + self.k3*r2**3
                slope = self.k1 + 2*self.k2*r2 + 3*self.k3*r2*r2
                a = radial + 2*x*x*slope + 2*self.p1*y + 6*self.p2*x
                b = 2*x*y*slope + 2*self.p1*x + 2*self.p2*y
                d = radial + 2*y*y*slope + 6*self.p1*y + 2*self.p2*x
                px, py = self._distort(x, y)
                ex, ey = px-xd, py-yd
                determinant = a*d-b*b
                x -= np.clip((d*ex-b*ey)/determinant, -.25, .25)
                y -= np.clip((a*ey-b*ex)/determinant, -.25, .25)
            r2 = x*x+y*y
            discriminant = 1 + (1-self.xi*self.xi)*r2
            factor = (self.xi + np.sqrt(discriminant))/(1+r2)
            rays = np.column_stack((factor*x, factor*y, factor-self.xi))
            rays /= np.linalg.norm(rays, axis=1)[:, None]
            error = np.linalg.norm(self.project(rays)-uv, axis=1)
        rays[~np.isfinite(error) | (error > .01)] = np.nan
        return rays


def parse_offset(raw: str) -> dict:
    if not isinstance(raw, str):
        raise ValueError('INSV 缺少字符串形式的 offset_v3')
    try:
        values = np.asarray([float(v) for v in raw.split('_')], float)
    except ValueError as error:
        raise ValueError('offset_v3 包含非数值字段') from error
    if values.shape != (40,) or not np.isfinite(values).all() or values[0] != 2:
        raise ValueError('仅支持双镜头 v3 offset（40 个有限数值）；不推断 v6 格式')
    lenses = []
    for index, begin in enumerate((1, 20)):
        block = dict(zip(FIELDS, values[begin:begin+19].tolist()))
        if (block['fx'] <= 0 or block['fy'] <= 0 or block['xi'] < 0
                or any(block[k] <= 0 or not block[k].is_integer() for k in ('width', 'height'))
                or block['width'] != 2*block['height']):
            raise ValueError(f'lens{index} 的焦距、ξ 或双镜头标定尺寸无效')
        for key in ('width', 'height', 'lens_type'):
            if not block[key].is_integer():
                raise ValueError(f'lens{index} 的 {key} 必须为整数')
            block[key] = int(block[key])
        lenses.append(dict(index=index,
                           role='working' if index == WORKING_LENS_INDEX else 'environment', **block))
    if (lenses[0]['width'], lenses[0]['height']) != (lenses[1]['width'], lenses[1]['height']):
        raise ValueError('两组工厂参数的标定分辨率不一致')
    return dict(source='insv_factory', model='mei', format='offset_v3', raw=raw,
                flag=float(values[-1]), extrinsics_convention='raw yaw/pitch/roll in degrees; translation as stored',
                lenses=lenses, mapping_status='fixed', lens_assignment=dict(LENS_ASSIGNMENT),
                assignment_basis='specialist offset_v3 block order and rear-lens gripper detection; '
                                 'adopted for the current X3 installation')


def read_factory(path: Path) -> dict:
    metadata = metadata_summary(read_metadata(path))
    if metadata.get('model') != 'Insta360 X3':
        raise ValueError('工厂参数对照目前仅验证 X3 的 3840×1920 录像模式')
    return dict(parse_offset(metadata.get('offset_v3')), serial=metadata['serial'],
                firmware=metadata['firmware'], camera_model=metadata['model'])


def working_from_environment(native: dict) -> np.ndarray:
    """Environment (index0) → working (index1), metres.

    Follows specialist.record.offset_v3: Rz(yaw) Ry(pitch) Rx(roll),
    t=-R*[tx,ty,tz]. Index0 gains a body-z half-turn in rotation only;
    its translation stays in the original body frame. Body is not the IMU.
    """
    from scipy.spatial.transform import Rotation
    from software.camera.geometry import transform, validate_transform

    body_from_lens = {}
    for lens in native['lenses']:
        rotation = Rotation.from_euler('ZYX', [lens['yaw'], lens['pitch'], lens['roll']], degrees=True).as_matrix()
        translation = -rotation @ np.array([lens['tx'], lens['ty'], lens['tz']])
        if lens['index'] == ENVIRONMENT_LENS_INDEX:
            rotation = np.diag([-1., -1., 1.]) @ rotation
        body_from_lens[lens['index']] = validate_transform(transform(rotation, translation))
    return validate_transform(np.linalg.inv(body_from_lens[WORKING_LENS_INDEX])
                              @ body_from_lens[ENVIRONMENT_LENS_INDEX])


def lens_model(lens: dict, size: int) -> Mei:
    """Factory plane → nominal 1920 lens → centred 1440/1920 view.

The factory-to-recording step follows specialist's continuous-coordinate scaling;
no additional video resize occurs here. It is a recorded mapping hypothesis.
"""
    if size not in (1440, 1920):
        raise ValueError('工厂对照只支持 1440 和 1920 视野')
    scale = 1920 / (lens['width']/2)
    margin = (1920-size)/2
    return Mei(**{k: lens[k] for k in ('xi', 'k1', 'k2', 'k3', 'p1', 'p2')},
               fx=lens['fx']*scale, fy=lens['fy']*scale,
               cx=(lens['cx']-lens['index']*lens['width']/2)*scale-margin,
               cy=lens['cy']*scale-margin)


@dataclass(frozen=True)
class Kb4Model:
    """Spherical KB4: atan2 projection and unit-bearing inversion beyond 180° FOV."""
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    k4: float

    def project(self, points):
        p = np.asarray(points, float).reshape(-1, 3)
        radius = np.linalg.norm(p[:, :2], axis=1)
        theta = np.arctan2(radius, p[:, 2])
        t2 = theta*theta
        rd = theta*(1+self.k1*t2+self.k2*t2**2+self.k3*t2**3+self.k4*t2**4)
        scale = np.divide(rd, radius, out=np.zeros_like(rd), where=radius > 1e-12)
        uv = p[:, :2]*scale[:, None]*[self.fx, self.fy]+[self.cx, self.cy]
        uv[(np.linalg.norm(p, axis=1) < 1e-12) | ((radius < 1e-12) & (p[:, 2] < 0))] = np.nan
        return uv

    def project_jacobian(self, points):
        """返回逐点 2×3 解析导数；与球面 project() 使用相同模型。"""
        p = np.asarray(points, float).reshape(-1, 3)
        result = np.empty((len(p), 2, 3))
        for i, (x, y, z) in enumerate(p):
            radius = np.hypot(x, y)
            if radius < 1e-8:
                if z <= 1e-12:
                    raise ValueError('KB4 投影在原点或反向光轴不可导')
                result[i] = [[self.fx/z, 0., -self.fx*x/z**2],
                             [0., self.fy/z, -self.fy*y/z**2]]
                continue
            theta = np.arctan2(radius, z)
            t2 = theta*theta
            rd = theta*(1+self.k1*t2+self.k2*t2**2+self.k3*t2**3+self.k4*t2**4)
            slope = 1+3*self.k1*t2+5*self.k2*t2**2+7*self.k3*t2**3+9*self.k4*t2**4
            distance2 = radius*radius+z*z
            radial_derivative = (slope*z/distance2*radius-rd)/radius**2
            xy = np.array([x, y])
            result[i, :, :2] = rd/radius*np.eye(2) + radial_derivative/radius*np.outer(xy, xy)
            result[i, :, 2] = -slope/distance2*xy
            result[i] *= np.array([self.fx, self.fy])[:, None]
        return result

    def unproject(self, pixels):
        uv = np.asarray(pixels, float).reshape(-1, 2)
        xy = (uv-[self.cx, self.cy])/[self.fx, self.fy]
        rd = np.linalg.norm(xy, axis=1)
        theta = np.minimum(rd, np.pi/2)
        for _ in range(30):
            t2 = theta*theta
            value = theta*(1+self.k1*t2+self.k2*t2**2+self.k3*t2**3+self.k4*t2**4)
            derivative = 1+3*self.k1*t2+5*self.k2*t2**2+7*self.k3*t2**3+9*self.k4*t2**4
            step = np.divide(value-rd, derivative, out=np.full_like(rd, np.nan), where=derivative > 0)
            theta = np.clip(theta-np.clip(step, -.25, .25), 0, np.pi-1e-6)
        scale = np.divide(np.sin(theta), rd, out=np.ones_like(rd), where=rd > 1e-12)
        rays = np.column_stack((xy*scale[:, None], np.cos(theta)))
        error = np.linalg.norm(self.project(rays)-uv, axis=1)
        t2 = theta*theta
        derivative = 1+3*self.k1*t2+5*self.k2*t2**2+7*self.k3*t2**3+9*self.k4*t2**4
        rays[~np.isfinite(error) | (error > .01) | (derivative <= 0)] = np.nan
        return rays

    def full_view(self):
        return replace(self, cx=self.cx+240, cy=self.cy+240)
