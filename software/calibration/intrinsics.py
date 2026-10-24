"""A: solve the fish-eye intrinsics and measure recording coverage."""

from __future__ import annotations

import sys
from pathlib import Path

from software.calibration.algorithms import aprilgrid_detect, board_pnp
from software.calibration.models import IntrinsicsResult, StageContext, StageError
from software.calibration import runtime


@runtime.stage_errors("A 内参")
def solve_intrinsics(dataset: Path, context: StageContext) -> IntrinsicsResult:
    """Input: A EuRoC data. Output: Basalt intrinsics and quality measurements."""
    runtime.console.write("A 段：求解相机内参", step=(2, 6))
    solver_output = context.paths.diagnostics / "basalt_cam"
    solver_output.mkdir()
    if runtime.run_logged([sys.executable, runtime.TOOLS / "calib_folder_to_bag.py", "--euroc", dataset, "--out", context.paths.work / "A.bag"], context.paths.logs / "A_to_bag.log") != 0:
        raise StageError("A 内参", "A 段转 bag 失败，见 diagnostics/logs/A_to_bag.log")
    return_code = runtime.run_logged(["nice", "-n", "10", *context.cpu_prefix, *runtime.basalt_command("basalt_calibrate"), "--dataset-path", context.paths.work / "A.bag", "--dataset-type", "bag",
                     "--aprilgrid", context.paths.board, "--result-path", solver_output, "--cache-name", "cam-A", "--cam-types", "kb4", "--no-gui"],
                    context.paths.logs / "calibrate_cam.log")
    (context.paths.work / "A.bag").unlink(missing_ok=True)
    if return_code != 0 or not (solver_output / "calibration.json").is_file():
        raise StageError("A 内参", "内参标定失败，见 diagnostics/logs/calibrate_cam.log（多半是 A 段里标定板太小、太糊或太少）")
    solver_metrics = runtime.last_reprojection(context.paths.logs / "calibrate_cam.log")
    kb4 = board_pnp.read_kb4(solver_output / "calibration.json")
    board_metrics = board_pnp.measure_board_coverage(board_pnp.detect_all(dataset, aprilgrid_detect.GridDetector, step=context.parameters.board.intrinsics_step, workers=context.workers, up=context.parameters.board.detection_upscale), kb4, context.grid, parameters=context.parameters.board)
    return IntrinsicsResult(solver_output / "calibration.json", kb4, solver_metrics, board_metrics)
