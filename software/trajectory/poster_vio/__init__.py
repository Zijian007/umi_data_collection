"""Task-local poster visual-inertial bundle adjustment."""

from .models import PosterVioObservation, PosterVioQuality, PosterVioRequest, PosterVioResult
from .contract import validate_request, validate_result
from .quality import PosterVioPolicy, assess_observations
from .solver import ArmSolveInput, NaturalCamera, solve_joint_poster_imu
from .initialization import select_observations, seed_poses

__all__ = [
    'PosterVioObservation',
    'PosterVioQuality',
    'PosterVioRequest',
    'PosterVioResult',
    'validate_request',
    'validate_result',
    'PosterVioPolicy',
    'assess_observations',
    'ArmSolveInput',
    'NaturalCamera',
    'solve_joint_poster_imu',
    'select_observations',
    'seed_poses',
]
