#!/usr/bin/env python3
"""Verify that scene-derived graphs use only explicit connections."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from agent.navigation import NavigationGraph, NavigationGraphError
from agent.navigation.scene_adapter import convert_scene


SOURCE_SCENE = (
    Path(__file__).resolve().parents[3]
    / "0803/Emergency/agent/navigation/data/logs/emergency.navigation.scene.json"
)
CONFIG_ROOT = Path(__file__).resolve().parents[3] / "0803/Emergency/config"


class SceneGraphDerivationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.scene_path = Path(self.temporary_directory.name) / SOURCE_SCENE.name
        shutil.copy2(SOURCE_SCENE, self.scene_path)
        self.route_path, self.graph_path = convert_scene(self.scene_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_graph_edges_equal_explicit_scene_connections(self) -> None:
        scene = json.loads(self.scene_path.read_text(encoding="utf-8"))
        graph = json.loads(self.graph_path.read_text(encoding="utf-8"))
        names = {node["id"]: node["name"] for node in scene["robots"]}
        expected = {
            (names[connection["fromId"]], names[connection["toId"]])
            for connection in scene["connections"]
        }
        expected |= {(target, source) for source, target in expected}
        actual = {(edge["from"], edge["to"]) for edge in graph["edges"]}

        self.assertEqual(actual, expected)
        self.assertTrue(
            all(len(connection.get("actions", [])) == 1 for connection in scene["connections"])
        )
        self.assertTrue(all(len(edge["actions"]) == 1 for edge in graph["edges"]))
        self.assertIn(("h270_track_end", "h270_transfer_point"), actual)
        self.assertIn(("h270_transfer_point", "h270_track_end"), actual)
        self.assertNotIn(("h270_window1_action", "h270_transfer_point"), actual)
        self.assertNotIn(("h270_transfer_point", "h270_window1_action"), actual)

    def test_scene_matches_authoritative_config(self) -> None:
        scene = json.loads(SOURCE_SCENE.read_text(encoding="utf-8"))
        route = json.loads(
            (CONFIG_ROOT / "emergency.navigation.json").read_text(encoding="utf-8")
        )
        config_graph = json.loads(
            (CONFIG_ROOT / "emergency.navigation.graph.json").read_text(
                encoding="utf-8"
            )
        )
        derived_graph = json.loads(self.graph_path.read_text(encoding="utf-8"))
        scene_nodes = {
            node["name"]: (node["id"], node["x"], node["y"], node["angle"])
            for node in scene["robots"]
        }
        route_nodes = {
            point["name"]: (
                point["source_id"],
                point["logical_pose"]["x_m"],
                point["logical_pose"]["y_m"],
                point["logical_pose"]["theta_deg"],
            )
            for point in route["points"]
        }
        config_edges = {
            (edge["from"], edge["to"]): edge["actions"]
            for edge in config_graph["edges"]
        }
        derived_edges = {
            (edge["from"], edge["to"]): edge["actions"]
            for edge in derived_graph["edges"]
        }

        self.assertEqual(scene_nodes, route_nodes)
        self.assertEqual(derived_edges, config_edges)
        self.assertEqual(scene["nextRobotId"], 23)

    def test_scene_derived_route_rejects_sequential_rebuild(self) -> None:
        with self.assertRaisesRegex(
            NavigationGraphError,
            "sequential route rebuild is disabled",
        ):
            NavigationGraph(self.route_path, rebuild_graph=True)

    def test_scene_derived_route_requires_generated_graph(self) -> None:
        self.graph_path.unlink()

        with self.assertRaisesRegex(
            NavigationGraphError,
            "sequential route rebuild is disabled",
        ):
            NavigationGraph(self.route_path)


if __name__ == "__main__":
    unittest.main()
