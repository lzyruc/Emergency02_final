"""REST-only Knewbot chassis controller for TienYi 2.0 Pro.

This module implements the HTTP APIs documented in ``晓悟底盘API.md``.  It
does not import ROS 2, ``rclpy``, ``knewbots_msgs`` or other ROS message
packages.
"""

from __future__ import annotations

import math
import time
from enum import IntEnum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import requests

#from xrocs.utils.logger.logger_loader import logger


MOVE_WITH_THETA = 0x00000001


class ChassisErrorCode(IntEnum):
    """Error categories raised by the REST controller."""

    NAV_TIMEOUT = 1001
    REQUEST_TIMEOUT = 1002
    INVALID_REQUEST = 1003
    ACTION_FAILED = 2001
    ACTION_CANCELLED = 2003
    CONNECTION_ERROR = 3001
    HTTP_ERROR = 3002
    INTERNAL_ERROR = 9001


class ChassisNavError(RuntimeError):
    """Unified exception for validation, HTTP and action errors."""

    def __init__(
        self,
        code: ChassisErrorCode,
        message: str,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.cause = cause
        super().__init__(f"[{code.value}] {message}")


class ActionHandle:
    """A REST action that can be polled, waited for, or cancelled."""

    _SUCCESS_STATES = {"success", "succeeded", "successed", "completed", "done"}
    _FAILED_STATES = {"failed", "error", "rejected"}
    _CANCELLED_STATES = {"cancelled", "canceled"}

    def __init__(self, controller: "KnewbotChassisController", action_id: Union[int, str]) -> None:
        self.action_id = action_id
        self._controller = controller
        self._done = False
        self._last_status: Optional[Dict[str, Any]] = None

    @property
    def is_done(self) -> bool:
        """Whether a terminal action state has been observed."""
        return self._done

    @property
    def last_status(self) -> Optional[Dict[str, Any]]:
        """Return the most recently polled status response."""
        return self._last_status

    @staticmethod
    def _state(data: Mapping[str, Any]) -> Union[str, int, None]:
        state = data.get("status")
        if state is None and isinstance(data.get("state"), Mapping):
            state = data["state"].get("status")
        return state.lower() if isinstance(state, str) else state

    def poll(self) -> Dict[str, Any]:
        """Fetch and return the current action status once."""
        status = self._controller.get_action_status(self.action_id)
        self._last_status = status
        state = self._state(status)
        if state in self._SUCCESS_STATES or state == 4:
            self._done = True
        elif state in self._FAILED_STATES or state in {5, 6}:
            self._done = True
        elif state in self._CANCELLED_STATES:
            self._done = True
        return status

    def wait(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Block until the action reaches a terminal state."""
        timeout = self._controller.nav_timeout if timeout is None else timeout
        if timeout <= 0:
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "timeout must be greater than 0")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.poll()
            state = self._state(status)
            if state in self._SUCCESS_STATES or state == 4:
                return status
            if state in self._CANCELLED_STATES:
                raise ChassisNavError(
                    ChassisErrorCode.ACTION_CANCELLED,
                    f"Action {self.action_id} was cancelled",
                )
            if state in self._FAILED_STATES or state in {5, 6}:
                raise ChassisNavError(
                    ChassisErrorCode.ACTION_FAILED,
                    f"Action {self.action_id} failed (status={state!r})",
                )
            time.sleep(self._controller.status_check_interval)

        raise ChassisNavError(
            ChassisErrorCode.NAV_TIMEOUT,
            f"Action {self.action_id} timed out after {timeout} seconds",
        )

    def cancel(self) -> Any:
        """Cancel this action through the REST cancellation endpoint."""
        result = self._controller.cancel_action([self.action_id])
        self._done = True
        return result


class KnewbotChassisController:
    """Control a Knewbot chassis exclusively through its REST API.

    Example configuration::

        {
            "host": "192.168.11.10",
            "rest_port": 9090,
            "nav_timeout": 30,
            "request_timeout": 3,
            "status_check_interval": 0.3,
        }

    ``base_url`` may be supplied instead of ``host`` and ``rest_port``.  A
    station-style outer configuration containing a ``knewbot`` dictionary is
    also accepted.
    """

    def __init__(
        self,
        chassis_config: Optional[Dict[str, Any]] = None,
        *,
        session: Optional[requests.Session] = None,
    ) -> None:
        outer_cfg = chassis_config or {}
        cfg = outer_cfg.get("knewbot", outer_cfg)
        if not isinstance(cfg, Mapping):
            raise TypeError("chassis_config['knewbot'] must be a mapping")

        self.cfg_dict = dict(cfg)
        self._host = str(cfg.get("host", "192.168.11.10"))
        self._rest_port = int(cfg.get("rest_port", 9090))
        self._nav_timeout = float(cfg.get("nav_timeout", cfg.get("navigation_timeout", 30)))
        self._request_timeout = float(cfg.get("request_timeout", 3))
        self._status_check_interval = float(cfg.get("status_check_interval", 0.3))

        if self._nav_timeout <= 0 or self._request_timeout <= 0:
            raise ValueError("nav_timeout and request_timeout must be greater than 0")
        if self._status_check_interval <= 0:
            raise ValueError("status_check_interval must be greater than 0")

        configured_url = cfg.get("base_url")
        if configured_url:
            self._base_url = str(configured_url).rstrip("/")
        else:
            host = self._host.rstrip("/")
            if host.startswith(("http://", "https://")):
                self._base_url = host
            else:
                self._base_url = f"http://{host}:{self._rest_port}"

        self._session = session or requests.Session()
        self._owns_session = session is None
        self._session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
        self._closed = False
        self._waypoints: Dict[str, Dict[str, float]] = {}

        #logger.info(f"REST-only Knewbot chassis controller initialized at {self._base_url}.")

    @property
    def nav_timeout(self) -> float:
        """Default action timeout in seconds."""
        return self._nav_timeout

    @property
    def status_check_interval(self) -> float:
        """Action polling interval in seconds."""
        return self._status_check_interval

    @property
    def base_url(self) -> str:
        """REST server base URL."""
        return self._base_url

    @property
    def nav(self) -> "KnewbotChassisController":
        """Compatibility property; REST methods are implemented by this object."""
        return self

    def close(self) -> None:
        """Close the owned HTTP session."""
        if self._closed:
            return
        if self._owns_session:
            self._session.close()
        self._closed = True
        #logger.info("REST-only Knewbot chassis controller closed.")

    def __enter__(self) -> "KnewbotChassisController":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Any] = None,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        if self._closed:
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "controller is closed")

        request_timeout = self._request_timeout if timeout is None else timeout
        try:
            response = self._session.request(
                method,
                f"{self._base_url}{path}",
                json=payload,
                params=params,
                timeout=request_timeout,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise ChassisNavError(
                ChassisErrorCode.REQUEST_TIMEOUT,
                f"Request timed out: {method} {path}",
                exc,
            ) from exc
        except requests.ConnectionError as exc:
            raise ChassisNavError(
                ChassisErrorCode.CONNECTION_ERROR,
                f"Cannot connect to {self._base_url}",
                exc,
            ) from exc
        except requests.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else "unknown"
            detail = response.text[:300] if response is not None else str(exc)
            code = ChassisErrorCode.INVALID_REQUEST if status_code == 400 else ChassisErrorCode.HTTP_ERROR
            raise ChassisNavError(code, f"REST {status_code}: {detail}", exc) from exc
        except requests.RequestException as exc:
            raise ChassisNavError(ChassisErrorCode.HTTP_ERROR, f"REST request failed: {exc}", exc) from exc

        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    def _get(self, path: str, **kwargs: Any) -> Any:
        return self._request("GET", path, **kwargs)

    def _post(self, path: str, payload: Optional[Any] = None, **kwargs: Any) -> Any:
        return self._request("POST", path, payload=payload, **kwargs)

    def _put(self, path: str, payload: Optional[Any] = None, **kwargs: Any) -> Any:
        return self._request("PUT", path, payload=payload, **kwargs)

    def _delete(self, path: str, **kwargs: Any) -> Any:
        return self._request("DELETE", path, **kwargs)

    @staticmethod
    def _require_finite(name: str, value: Union[int, float]) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, f"{name} must be finite")
        return result

    @classmethod
    def _pose(cls, x: float, y: float, yaw: float) -> Dict[str, float]:
        yaw_value = cls._require_finite("yaw", yaw)
        if not -math.pi <= yaw_value <= math.pi:
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "yaw must be within [-pi, pi]")
        return {
            "x": cls._require_finite("x", x),
            "y": cls._require_finite("y", y),
            "yaw": yaw_value,
        }

    @classmethod
    def _pose_from_mapping(cls, name: str, pose: Mapping[str, float]) -> Dict[str, float]:
        missing = [key for key in ("x", "y", "yaw") if key not in pose]
        if missing:
            raise ChassisNavError(
                ChassisErrorCode.INVALID_REQUEST,
                f"{name} is missing fields: {', '.join(missing)}",
            )
        return cls._pose(pose["x"], pose["y"], pose["yaw"])

    def _action_result(self, response: Any, wait: bool) -> Union[ActionHandle, Dict[str, Any]]:
        if not isinstance(response, Mapping) or "action_id" not in response:
            raise ChassisNavError(
                ChassisErrorCode.INTERNAL_ERROR,
                f"Action response does not contain action_id: {response!r}",
            )
        handle = ActionHandle(self, response["action_id"])
        return handle.wait(self._nav_timeout) if wait else handle

    # System APIs

    def get_robot_info(self) -> Dict[str, Any]:
        """Get manufacturer, model, device ID and version information."""
        return self._get("/api/core/system/v1/robot/info")

    def get_power_status(self) -> Dict[str, Any]:
        """Get battery percentage, charging, docking and sleep state."""
        return self._get("/api/core/system/v1/power/status")

    def get_robot_health(self) -> Dict[str, Any]:
        """Get chassis warnings, errors, fatal status and raw error codes."""
        return self._get("/api/core/system/v1/robot/health")

    def get_battery_pack(self) -> Dict[str, Any]:
        """Get battery pack current and temperature readings."""
        return self._get("/api/core/system/v1/battery/pack")

    def get_speed_params(self) -> Dict[str, Any]:
        """Get maximum moving and angular speeds."""
        return self._get("/api/core/system/v1/parameter")

    def set_speed_params(
        self,
        max_moving_speed: Optional[float] = None,
        max_angular_speed: Optional[float] = None,
    ) -> Any:
        """Set either or both chassis speed limits."""
        payload: Dict[str, float] = {}
        if max_moving_speed is not None:
            payload["max_moving_speed"] = self._require_finite("max_moving_speed", max_moving_speed)
        if max_angular_speed is not None:
            payload["max_angular_speed"] = self._require_finite("max_angular_speed", max_angular_speed)
        if not payload:
            raise ChassisNavError(
                ChassisErrorCode.INVALID_REQUEST,
                "at least one speed limit must be provided",
            )
        return self._put("/api/core/system/v1/parameter", payload)

    def set_sensor_ignore_zone(
        self,
        center: Mapping[str, float],
        half_extents: Mapping[str, float],
    ) -> Any:
        """Set the upper-body volume ignored by chassis obstacle sensing."""
        for field_name, value in (("center", center), ("half_extents", half_extents)):
            if any(axis not in value for axis in ("x", "y", "z")):
                raise ChassisNavError(
                    ChassisErrorCode.INVALID_REQUEST,
                    f"{field_name} must contain x, y and z",
                )
        payload = {
            "center": {axis: self._require_finite(f"center.{axis}", center[axis]) for axis in ("x", "y", "z")},
            "half_extents": {
                axis: self._require_finite(f"half_extents.{axis}", half_extents[axis])
                for axis in ("x", "y", "z")
            },
        }
        if any(value < 0 for value in payload["half_extents"].values()):
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "half_extents values cannot be negative")
        return self._put("/api/core/system/v1/sensor/ignore_zone", payload)

    # Platform and statistics APIs

    def get_timestamp(self) -> Dict[str, Any]:
        """Get the chassis system timestamp."""
        return self._get("/api/platform/v1/timestamp")

    def get_odometry(self) -> Dict[str, Any]:
        """Get total travelled distance in metres."""
        return self._get("/api/core/statistics/v1/odometry")

    def get_runtime(self) -> Dict[str, Any]:
        """Get total runtime in seconds."""
        return self._get("/api/core/statistics/v1/runtime")

    # SLAM APIs

    def get_mapping_enabled(self) -> Dict[str, Any]:
        """Get mapping mode: 0=stopped, 1=running, 2=paused."""
        return self._get("/api/core/slam/v1/mapping/enable")

    def set_mapping_mode(self, mode: int, maptype: int = 1, save_path: str = "") -> Any:
        """Set mapping mode according to the REST API."""
        if mode not in (0, 1, 2):
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "mode must be 0, 1 or 2")
        if maptype not in (0, 1, 2):
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "maptype must be 0, 1 or 2")
        payload = {"mode": mode, "maptype": maptype, "save_path": save_path}
        return self._put("/api/core/slam/v1/mapping/enable", payload)

    def set_mapping_enabled(self, enable: bool) -> Any:
        """Start mapping when true, or pause mapping when false."""
        return self.set_mapping_mode(1 if enable else 2, maptype=1)

    def set_map_update(self, enabled: bool, maptype: int = 1, save_path: str = "") -> Any:
        """Compatibility helper: start mapping or stop and save it."""
        return self.set_mapping_mode(1 if enabled else 0, maptype=maptype, save_path=save_path)

    def start_localization(self, map_name: str, slam_type: int = 1) -> Any:
        """Start QR-code (0) or laser (1) localization on a map."""
        if slam_type not in (0, 1):
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "slam_type must be 0 or 1")
        if not map_name:
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "map_name cannot be empty")
        return self._put(
            "/api/core/slam/v1/start_localization",
            {"slam_type": slam_type, "map_name": map_name},
        )

    def get_current_map(self) -> Dict[str, Any]:
        """Get the currently localized map name."""
        return self._get("/api/core/slam/v1/current_map")

    def get_localization_status(self) -> Dict[str, Any]:
        """Get localization type and status code."""
        return self._get("/api/core/slam/v1/localization/status")

    def recover_localization(
        self,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
        max_time_ms: Optional[int] = None,
        area: Optional[Mapping[str, float]] = None,
        wait: bool = False,
    ) -> Union[ActionHandle, Dict[str, Any]]:
        """Request relocalization, optionally constrained to a search area."""
        payload: Dict[str, Any] = {"pose": self._pose(x, y, yaw)}
        if max_time_ms is not None:
            if max_time_ms <= 0:
                raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "max_time_ms must be positive")
            payload["max_time_ms"] = int(max_time_ms)
        if area is not None:
            if any(key not in area for key in ("x", "y", "w", "h")):
                raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "area must contain x, y, w and h")
            payload["has_search_area"] = True
            payload["area"] = {
                key: self._require_finite(f"area.{key}", area[key]) for key in ("x", "y", "w", "h")
            }
        else:
            payload["has_search_area"] = False
        response = self._put("/api/core/slam/v1/localization/recover", payload)
        return self._action_result(response, wait)

    # Navigation action APIs

    def get_action_status(self, action_id: Union[int, str]) -> Dict[str, Any]:
        """Get the state of a previously issued action."""
        return self._get(f"/api/core/motion/v1/actions/{action_id}")

    def move_by(
        self,
        direction: float,
        distance: int,
        move_mode: int = 0,
        v: float = 0.0,
        w: float = 0.0,
        wait: bool = True,
    ) -> Union[ActionHandle, Dict[str, Any]]:
        """Move in a relative direction or rotate in place through REST."""
        direction = self._require_finite("direction", direction)
        if not -math.pi <= direction <= math.pi:
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "direction must be within [-pi, pi]")
        if move_mode not in (0, 1):
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "move_mode must be 0 or 1")
        if not isinstance(distance, int) or isinstance(distance, bool) or distance < 0:
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "distance must be a non-negative integer in mm")
        if move_mode == 0 and not 20 <= distance <= 3000:
            raise ChassisNavError(
                ChassisErrorCode.INVALID_REQUEST,
                f"translation distance must be within [20, 3000] mm, got {distance}",
            )
        if move_mode == 1:
            if distance != 0:
                raise ChassisNavError(
                    ChassisErrorCode.INVALID_REQUEST,
                    "rotation requires distance=0",
                )
            if abs(direction) < math.radians(1.0):
                raise ChassisNavError(
                    ChassisErrorCode.INVALID_REQUEST,
                    f"rotation requires |angle|>=1 degree, got {math.degrees(direction):.3f} degrees",
                )
        payload = {
            "move_mode": move_mode,
            "direction": direction,
            "distance": distance,
            "v": self._require_finite("v", v),
            "w": self._require_finite("w", w),
        }
        response = self._put("/api/core/navigation/v1/actions/moveby", payload)
        return self._action_result(response, wait)

    def move_to(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        flags: int = MOVE_WITH_THETA,
        wait: bool = True,
    ) -> Union[ActionHandle, Dict[str, Any]]:
        """Navigate to a map pose through REST."""
        payload = {"pose": self._pose(x, y, yaw), "flags": int(flags)}
        response = self._put("/api/core/navigation/v1/actions/moveto", payload)
        return self._action_result(response, wait)

    def move_to_poi(
        self,
        poi_name: str,
        flags: int = MOVE_WITH_THETA,
        wait: bool = True,
    ) -> Union[ActionHandle, Dict[str, Any]]:
        """Navigate to a named POI through REST."""
        if not poi_name:
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "poi_name cannot be empty")
        response = self._put(
            "/api/core/navigation/v1/actions/moveto_poi",
            {"poi_name": poi_name, "flags": int(flags)},
        )
        return self._action_result(response, wait)

    def move_to_locations(
        self,
        track_id: int,
        start_position: Mapping[str, float],
        target_position: Mapping[str, float],
        uptrack_enabled: bool = True,
        wait: bool = True,
    ) -> Union[ActionHandle, Dict[str, Any]]:
        """Navigate through a track using explicit start and target poses."""
        payload = {
            "track_id": int(track_id),
            "start_position": self._pose_from_mapping("start_position", start_position),
            "target_position": self._pose_from_mapping("target_position", target_position),
            "uptrack_enabled": bool(uptrack_enabled),
        }
        response = self._put("/api/core/navigation/v1/actions/moveto_locations", payload)
        return self._action_result(response, wait)

    def move_to_locations_poi(
        self,
        track_id: int,
        poi_name: str,
        uptrack_enabled: bool = True,
        wait: bool = True,
    ) -> Union[ActionHandle, Dict[str, Any]]:
        """Navigate through a track to a named POI."""
        if not poi_name:
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "poi_name cannot be empty")
        response = self._put(
            "/api/core/navigation/v1/actions/moveto_locations_poi",
            {"track_id": int(track_id), "poi_name": poi_name, "uptrack_enabled": bool(uptrack_enabled)},
        )
        return self._action_result(response, wait)

    def go_home(
        self,
        enable: bool = True,
        dock_id: int = 0,
        wait: bool = True,
    ) -> Union[ActionHandle, Dict[str, Any], None]:
        """Navigate to a charging dock; ``enable=False`` cancels actions."""
        if not enable:
            return self.cancel_action()
        response = self._put("/api/core/navigation/v1/actions/go_home", {"id": int(dock_id)})
        return self._action_result(response, wait)

    def cancel_action(self, action_list: Optional[Sequence[Union[int, str]]] = None) -> Any:
        """Cancel selected actions, or all actions when the list is empty."""
        return self._put(
            "/api/core/navigation/v1/cancel_action",
            {"action_list": list(action_list) if action_list is not None else []},
        )

    def get_pose(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get current x, y, yaw and localization confidence."""
        return self._get("/api/core/navigation/v1/pose", timeout=timeout)

    def get_plan_path(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get the current move action's global planned path."""
        return self._get("/api/core/navigation/v1/global_plan_path", timeout=timeout)

    def get_robot_basic_state(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Compose basic state from the documented REST status endpoints."""
        return {
            "power": self._get("/api/core/system/v1/power/status", timeout=timeout),
            "mapping": self._get("/api/core/slam/v1/mapping/enable", timeout=timeout),
            "localization": self._get("/api/core/slam/v1/localization/status", timeout=timeout),
        }

    # POI APIs

    def get_pois(self, map_name: Optional[str] = None, point_name: Optional[str] = None) -> Dict[str, Any]:
        """Get POIs, optionally filtered by map and point name."""
        params = {key: value for key, value in {"map_name": map_name, "point_name": point_name}.items() if value}
        return self._get("/api/core/navigation/v1/poi", params=params)

    def create_poi(
        self,
        point_name: str,
        x: float,
        y: float,
        yaw: float = 0.0,
        map_name: Optional[str] = None,
    ) -> Any:
        """Create a POI on the current or specified map."""
        payload: Dict[str, Any] = {"point_name": point_name, **self._pose(x, y, yaw)}
        if map_name:
            payload["map_name"] = map_name
        return self._post("/api/core/navigation/v1/poi", payload)

    def delete_poi(self, point_name: str, map_name: Optional[str] = None) -> Any:
        """Delete a POI by name."""
        params = {"point_name": point_name}
        if map_name:
            params["map_name"] = map_name
        return self._delete("/api/core/navigation/v1/poi", params=params)

    # Track APIs

    def create_track(
        self,
        track_id: int,
        track_name: str,
        waypoints: Sequence[Mapping[str, Any]],
        map_name: Optional[str] = None,
    ) -> Any:
        """Create a virtual navigation track."""
        payload: Dict[str, Any] = {
            "track_id": int(track_id),
            "track_name": track_name,
            "waypoints": [dict(waypoint) for waypoint in waypoints],
        }
        if map_name:
            payload["map_name"] = map_name
        return self._post("/api/core/navigation/v1/track", payload)

    def update_track(
        self,
        track_id: int,
        track_name: str,
        waypoints: Sequence[Mapping[str, Any]],
        map_name: Optional[str] = None,
    ) -> Any:
        """Update an existing virtual navigation track."""
        payload: Dict[str, Any] = {
            "track_id": int(track_id),
            "track_name": track_name,
            "waypoints": [dict(waypoint) for waypoint in waypoints],
        }
        if map_name:
            payload["map_name"] = map_name
        return self._put(f"/api/core/navigation/v1/track/{track_id}", payload)

    def get_tracks(
        self,
        map_name: Optional[str] = None,
        track_id: Optional[int] = None,
        track_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get one or all virtual tracks on a map."""
        params: Dict[str, Any] = {}
        if map_name:
            params["map_name"] = map_name
        if track_id is not None:
            params["track_id"] = int(track_id)
        if track_name:
            params["track_name"] = track_name
        return self._get("/api/core/navigation/v1/track", params=params)

    def delete_track(
        self,
        map_name: Optional[str] = None,
        track_id: Optional[int] = None,
        track_name: Optional[str] = None,
    ) -> Any:
        """Delete a virtual track by ID or name."""
        if track_id is None and not track_name:
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "track_id or track_name is required")
        params: Dict[str, Any] = {}
        if map_name:
            params["map_name"] = map_name
        if track_id is not None:
            params["track_id"] = int(track_id)
        if track_name:
            params["track_name"] = track_name
        return self._delete("/api/core/navigation/v1/track", params=params)

    # Forbidden-area APIs

    @staticmethod
    def _area_payload(
        map_name: str,
        area_id: int,
        area_name: str,
        area_config: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        points = [list(point) for point in area_config]
        if len(points) < 2 or any(len(point) != 2 for point in points):
            raise ChassisNavError(
                ChassisErrorCode.INVALID_REQUEST,
                "area_config must contain at least two [x, y] points",
            )
        return {
            "map_name": map_name,
            "area_id": int(area_id),
            "area_name": area_name,
            "area_config": points,
        }

    def create_forbidden_area(
        self,
        map_name: str,
        area_id: int,
        area_name: str,
        area_config: Sequence[Sequence[float]],
    ) -> Any:
        """Create a virtual wall or forbidden polygon."""
        return self._post(
            "/api/core/navigation/v1/forbidden_area",
            self._area_payload(map_name, area_id, area_name, area_config),
        )

    def update_forbidden_area(
        self,
        map_name: str,
        area_id: int,
        area_name: str,
        area_config: Sequence[Sequence[float]],
    ) -> Any:
        """Update a virtual wall or forbidden polygon."""
        return self._put(
            "/api/core/navigation/v1/forbidden_area",
            self._area_payload(map_name, area_id, area_name, area_config),
        )

    def delete_forbidden_area(
        self,
        map_name: str,
        area_id: Optional[int] = None,
        area_name: Optional[str] = None,
    ) -> Any:
        """Delete a forbidden area by ID or name."""
        if area_id is None and not area_name:
            raise ChassisNavError(ChassisErrorCode.INVALID_REQUEST, "area_id or area_name is required")
        params: Dict[str, Any] = {"map_name": map_name}
        if area_id is not None:
            params["area_id"] = int(area_id)
        if area_name:
            params["area_name"] = area_name
        return self._delete("/api/core/navigation/v1/forbidden_area", params=params)

    # Local convenience methods

    def set_waypoint(self, name: str, x: float, y: float, yaw: float = 0.0) -> None:
        """Store an in-process waypoint without changing the chassis map."""
        self._waypoints[name] = self._pose(x, y, yaw)

    def move_to_waypoint(self, name: str, wait: bool = True) -> Union[ActionHandle, Dict[str, Any]]:
        """Navigate to an in-process waypoint."""
        try:
            waypoint = self._waypoints[name]
        except KeyError as exc:
            raise ChassisNavError(
                ChassisErrorCode.INVALID_REQUEST,
                f"unknown waypoint {name!r}; available={list(self._waypoints)}",
            ) from exc
        return self.move_to(**waypoint, wait=wait)

    def save_current_as_waypoint(self, name: str) -> Dict[str, Any]:
        """Save the current REST pose as an in-process waypoint."""
        pose = self.get_pose()
        self.set_waypoint(name, pose["x"], pose["y"], pose["yaw"])
        return dict(self._waypoints[name])

    def save_current_as(self, name: str) -> Dict[str, Any]:
        """Alias of :meth:`save_current_as_waypoint`."""
        return self.save_current_as_waypoint(name)

    def is_healthy(self) -> bool:
        """Return true when the REST health response has no error or fatal."""
        health = self.get_robot_health()
        return not bool(health.get("hasError", False) or health.get("hasFatal", False))

    def is_localized(self, timeout: Optional[float] = None) -> bool:
        """Return true when the documented localization status is normal (0)."""
        status = self._get("/api/core/slam/v1/localization/status", timeout=timeout)
        return status.get("slam_status") == 0

    def ensure_localized(self, retries: int = 1, wait_sec: float = 5.0) -> bool:
        """Retry REST relocalization when localization is not normal."""
        if self.is_localized():
            return True
        for _ in range(retries):
            self.recover_localization(wait=False)
            time.sleep(wait_sec)
            if self.is_localized():
                return True
        return False

    def safe_navigate(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        flags: int = MOVE_WITH_THETA,
        recover_retries: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Run health/localization checks before a blocking move."""
        if not self.is_healthy() or not self.ensure_localized(recover_retries):
            return None
        result = self.move_to(x, y, yaw, flags=flags, wait=True)
        return result if isinstance(result, dict) else None

    def run_mapping(self, duration_sec: float, maptype: int = 1, save_path: str = "") -> None:
        """Run mapping for a fixed duration, then stop and save it."""
        self.set_map_update(True, maptype=maptype)
        try:
            time.sleep(duration_sec)
        finally:
            self.set_map_update(False, maptype=maptype, save_path=save_path)

    def shuttle(self, wp_a: str, wp_b: str, loops: int = 1, pause_sec: float = 2.0) -> None:
        """Shuttle between two in-process waypoints."""
        self.patrol([wp_a, wp_b], loops=loops, pause_sec=pause_sec)

    def patrol(
        self,
        waypoints: Sequence[Union[str, Tuple[float, float, float]]],
        loops: int = 1,
        pause_sec: float = 2.0,
    ) -> None:
        """Patrol in-process waypoint names or explicit pose tuples."""
        for _ in range(loops):
            for waypoint in waypoints:
                if isinstance(waypoint, str):
                    self.move_to_waypoint(waypoint, wait=True)
                else:
                    self.move_to(*waypoint, wait=True)
                time.sleep(pause_sec)


KnewbotChassisHttpController = KnewbotChassisController


__all__ = [
    "ActionHandle",
    "ChassisErrorCode",
    "ChassisNavError",
    "KnewbotChassisController",
    "KnewbotChassisHttpController",
    "MOVE_WITH_THETA",
]
