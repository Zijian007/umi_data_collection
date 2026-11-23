"""Task-local poster visual-inertial bundle adjustment."""

from software.trajectory.algorithms.poster_vio.models import PosterVioObservation, PosterVioQuality, PosterVioProblem, PosterVioResult
from software.trajectory.algorithms.poster_vio.models import PosterVioPolicy, assess_observations, validate_problem, validate_result
from software.trajectory.algorithms.poster_vio.solver import ArmSolveInput, NaturalCamera, solve_joint_poster_imu
from software.trajectory.algorithms.poster_vio.initialization import select_observations, seed_poses

__all__ = [
    'PosterVioObservation',
    'PosterVioQuality',
    'PosterVioProblem',
    'PosterVioResult',
    'validate_problem',
    'validate_result',
    'PosterVioPolicy',
    'assess_observations',
    'ArmSolveInput',
    'NaturalCamera',
    'solve_joint_poster_imu',
    'select_observations',
    'seed_poses',
]
