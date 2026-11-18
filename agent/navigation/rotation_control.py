"""Feedback-assisted relative rotation using the chassis move_by API."""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, Mapping, Optional

from .data.motion_log import (
    json_safe,
    map_pose_from_response,
    normalize_radians,
    utc_now,
)


_MIN_ROTATION_RAD = math.radians(1.0)
_ACTION_ID_PATTERN = re.compile(r"\bAction\s+([0-9A-Za-z-]+)")


def _is_retryable_action_error(exc: BaseException) -> bool:
    return getattr(getattr(exc, "code", None), "value", None) in {1001, 2001}


def _cancel_timed_out_action(controller: Any, exc: BaseException) -> None:
    if getattr(getattr(exc, "code", None), "value", None) != 1001:
        return
    match = _ACTION_ID_PATTERN.search(str(exc))
    if match is None:
        raise RuntimeError("timed out chassis rotation has no action_id")
    controller.cancel_action([match.group(1)])
    print(
        f"[NavigationGraph] cancelled timed out action {match.group(1)}",
        flush=True,
    )


def rotate_by_with_feedback(
    controller: Any,
    rotation_rad: float,
    *,
    angular_speed: float = 0.0,
    initial_map_pose: Optional[Mapping[str, float]] = None,
    on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    retry_action_failed: bool = False,
    suppress_repeated_action_failed: bool = False,
) -> Dict[str, Any]:
    """Rotate relatively, measure map yaw error, then issue one correction."""
    requested = float(rotation_rad)
    if not math.isfinite(requested) or not -math.pi <= requested <= math.pi:
        raise ValueError("rotation_rad must be finite and within [-pi, pi]")

    before = (
        dict(initial_map_pose)
        if initial_map_pose is not None
        else map_pose_from_response(controller.get_pose())
    )
    target_yaw = normalize_radians(float(before["yaw_rad"]) + requested)
    report: Dict[str, Any] = {
        "method": "move_by_with_pose_feedback",
        "map_pose_before": before,
        "target_yaw_rad": target_yaw,
        "action_failed_count": 0,
        "first_rotation": {
            "status": "running",
            "started_at": utc_now(),
            "requested_rotation_rad": requested,
        },
    }

    if abs(requested) < _MIN_ROTATION_RAD:
        print(
            "[NavigationGraph] skip rotation "
            f"{math.degrees(requested):.2f} deg (<1 deg)",
            flush=True,
        )
        report["first_rotation"].update(
            {
                "status": "skipped",
                "finished_at": utc_now(),
                "reason": "requested rotation is below chassis 1 degree minimum",
            }
        )
        report["correction_rotation"] = {
            "status": "skipped",
            "finished_at": utc_now(),
            "reason": "first rotation was skipped",
        }
        report["map_pose_after"] = before
        report["final_yaw_error_rad"] = requested
        report["finished_at"] = utc_now()
        if on_update is not None:
            on_update(report)
        return report

    def notify() -> None:
        if on_update is not None:
            on_update(report)

    notify()
    first = report["first_rotation"]
    try:
        first_result = controller.move_by(
            direction=requested,
            distance=0,
            move_mode=1,
            v=0.0,
            w=float(angular_speed),
            wait=True,
        )
        after_first = map_pose_from_response(controller.get_pose())
        correction = normalize_radians(target_yaw - after_first["yaw_rad"])
        first.update(
            {
                "status": "completed",
                "finished_at": utc_now(),
                "controller_result": json_safe(first_result),
                "map_pose_after": after_first,
                "yaw_error_rad": correction,
            }
        )
    except Exception as exc:
        if not retry_action_failed or not _is_retryable_action_error(exc):
            first.update(
                {"status": "failed", "finished_at": utc_now(), "error": repr(exc)}
            )
            notify()
            raise
        _cancel_timed_out_action(controller, exc)
        after_first = map_pose_from_response(controller.get_pose())
        correction = normalize_radians(target_yaw - after_first["yaw_rad"])
        print(
            "[NavigationGraph] retry rotation after action interruption "
            f"remaining={math.degrees(correction):.2f} deg",
            flush=True,
        )
        report["action_failed_count"] += 1
        first.update(
            {
                "status": "reported_failed",
                "finished_at": utc_now(),
                "error": repr(exc),
                "map_pose_after": after_first,
                "yaw_error_rad": correction,
            }
        )
    notify()

    correction_log: Dict[str, Any] = {
        "requested_rotation_rad": correction,
        "started_at": utc_now(),
    }
    report["correction_rotation"] = correction_log
    if abs(correction) < _MIN_ROTATION_RAD:
        correction_log.update(
            {
                "status": "skipped",
                "finished_at": utc_now(),
                "reason": "yaw error is below chassis 1 degree minimum",
            }
        )
        final_pose = after_first
    else:
        correction_log["status"] = "running"
        notify()
        try:
            correction_result = controller.move_by(
                direction=correction,
                distance=0,
                move_mode=1,
                v=0.0,
                w=float(angular_speed),
                wait=True,
            )
            final_pose = map_pose_from_response(controller.get_pose())
            correction_log.update(
                {
                    "status": "completed",
                    "finished_at": utc_now(),
                    "controller_result": json_safe(correction_result),
                    "map_pose_after": final_pose,
                }
            )
        except Exception as exc:
            if _is_retryable_action_error(exc):
                _cancel_timed_out_action(controller, exc)
            if (
                not suppress_repeated_action_failed
                or not _is_retryable_action_error(exc)
            ):
                correction_log.update(
                    {
                        "status": "failed",
                        "finished_at": utc_now(),
                        "error": repr(exc),
                    }
                )
                notify()
                raise
            final_pose = map_pose_from_response(controller.get_pose())
            report["action_failed_count"] += 1
            correction_log.update(
                {
                    "status": "reported_failed",
                    "finished_at": utc_now(),
                    "error": repr(exc),
                    "map_pose_after": final_pose,
                    "yaw_error_rad": normalize_radians(
                        target_yaw - final_pose["yaw_rad"]
                    ),
                }
            )

    report["map_pose_after"] = final_pose
    report["final_yaw_error_rad"] = normalize_radians(target_yaw - final_pose["yaw_rad"])
    report["within_tolerance"] = abs(report["final_yaw_error_rad"]) < _MIN_ROTATION_RAD
    report["finished_at"] = utc_now()
    notify()
    return report
