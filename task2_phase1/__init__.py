"""Task 2 Phase 1: 单帧障碍物识别"""
from .pipeline import PIPELINE_VERSION, process_frame
from .result_types import FrameResult, ObjectResult
from .io_utils import load_depth, load_intrinsics
from .visualize import visualize_result
from .chair_cv import detect_chair
from .basket_detector import detect_basket
from .trash_cv import detect_trash
from .model import Phase1Model

__all__ = [
    "PIPELINE_VERSION", "process_frame", "ObjectResult", "FrameResult",
    "load_depth", "load_intrinsics",
    "visualize_result",
    "detect_chair",
    "detect_basket",
    "detect_trash",
    "Phase1Model",
]
