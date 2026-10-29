import unittest
from unittest.mock import Mock, patch

import glfw
import numpy as np

from engine.application import Application
from engine.game.player import Player
from engine.world.voxel_world import (
    _FACE_DATA,
    BLOCK_SAND,
    HOTBAR_BLOCKS,
    TEXTURE_DIRT,
    TEXTURE_GRASS_SIDE,
    TEXTURE_GRASS_TOP,
    TEXTURE_SAND,
    TEXTURE_STONE,
    VoxelWorld,
)


class VoxelWorldTests(unittest.TestCase):
    def test_world_mesh_contains_only_exposed_faces(self) -> None:
        world = VoxelWorld(size=8)
        mesh = world.build_mesh()

        self.assertEqual(world.blocks.shape, (8, 8, 8))
        self.assertEqual(mesh.shape, (1152, 7))
        self.assertTrue(world.is_solid(0, 0, 0))
        self.assertTrue(world.is_solid(0, 1, 0))
        self.assertFalse(world.is_solid(-1, 0, 0))
        self.assertFalse(world.is_solid(0, 2, 0))

    def test_grass_mesh_uses_top_side_and_dirt_textures(self) -> None:
        world = VoxelWorld(size=8)
        mesh = world.build_mesh()

        self.assertIn(TEXTURE_GRASS_TOP, mesh[:, 5])
        self.assertIn(TEXTURE_GRASS_SIDE, mesh[:, 5])
        self.assertIn(TEXTURE_DIRT, mesh[:, 5])

    def test_side_face_uvs_keep_texture_upright_on_each_side(self) -> None:
        for normal, expected_u in (
            ((1, 0, 0), (0.0, 0.0, 1.0, 1.0)),
            ((-1, 0, 0), (0.0, 1.0, 1.0, 0.0)),
            ((0, 0, 1), (0.0, 1.0, 1.0, 0.0)),
            ((0, 0, -1), (0.0, 0.0, 1.0, 1.0)),
        ):
            corners = next(face[1] for face in _FACE_DATA if face[0] == normal)
            uvs = VoxelWorld._face_uvs(normal, corners)
            vertical_values = tuple(
                uv[1] for corner, uv in zip(corners, uvs) if corner[1] == 0
            )
            self.assertEqual(vertical_values, (0.0, 0.0))
            self.assertEqual(
                tuple(uv[1] for corner, uv in zip(corners, uvs) if corner[1] == 1),
                (1.0, 1.0),
            )
            self.assertEqual(tuple(uv[0] for uv in uvs), expected_u)

    def test_block_mesh_uses_material_specific_texture_layers(self) -> None:
        world = VoxelWorld(size=8)
        world.set_block((3, 3, 3), BLOCK_SAND)
        world.set_block((5, 3, 3), 3)
        mesh = world.build_mesh()

        self.assertIn(TEXTURE_SAND, mesh[:, 5])
        self.assertIn(TEXTURE_STONE, mesh[:, 5])

    def test_raycast_returns_hit_and_adjacent_empty_block(self) -> None:
        world = VoxelWorld(size=8)
        world.set_block((3, 1, 5), 0)
        origin = np.array((3.5, 1.5, 5.5))
        direction = np.array((0.0, 0.0, -1.0))

        hit = world.raycast(origin, direction, max_distance=4.0)

        self.assertEqual(hit, ((3, 1, 4), (3, 1, 5)))
        np.testing.assert_array_equal(direction, (0.0, 0.0, -1.0))

    def test_raycast_respects_reach_and_placeable_world_height(self) -> None:
        world = VoxelWorld(size=8)
        world.set_block((3, 1, 5), 0)
        origin = np.array((3.5, 1.5, 5.5))

        self.assertIsNone(world.raycast(origin, (0.0, 0.0, -1.0), 0.4))
        self.assertTrue(world.set_block((3, 2, 5), 2))
        self.assertTrue(world.is_solid(3, 2, 5))

    def test_set_block_rejects_out_of_bounds_and_unchanged_values(self) -> None:
        world = VoxelWorld(size=8)

        self.assertFalse(world.set_block((-1, 2, 0), 2))
        self.assertFalse(world.set_block((3, 2, 3), 0))
        self.assertTrue(world.set_block((3, 2, 3), 2))
        self.assertFalse(world.set_block((3, 2, 3), 2))

    def test_world_rejects_too_small_size(self) -> None:
        with self.assertRaises(ValueError):
            VoxelWorld(size=3)


class PlayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = VoxelWorld(size=24)
        self.player = Player(12.0, 2.0, 12.0)

    def update_with_keys(self, keys: set[int], count: int = 1) -> None:
        def get_key(window: object, key: int) -> int:
            del window
            return glfw.PRESS if key in keys else glfw.RELEASE

        with patch("engine.game.player.glfw.get_key", side_effect=get_key):
            for _ in range(count):
                self.player.update(None, self.world, 1 / 60)

    def test_player_moves_and_stays_on_ground(self) -> None:
        self.update_with_keys(set(), count=3)
        self.assertTrue(self.player._grounded)

        initial_z = self.player.position[2]
        self.update_with_keys({glfw.KEY_W}, count=30)

        self.assertLess(self.player.position[2], initial_z - 2.0)
        self.assertAlmostEqual(self.player.position[1], 2.0)

    def test_player_stops_against_solid_block(self) -> None:
        self.player.position[0] = 12.5
        self.world.blocks[11, 2, 12] = 1
        self.update_with_keys(set(), count=3)
        self.update_with_keys({glfw.KEY_A}, count=30)

        self.assertGreaterEqual(self.player.position[0], 12.3 - 1e-6)

    def test_player_cannot_leave_world_bounds(self) -> None:
        self.player.position[0] = 0.5
        self.update_with_keys(set(), count=3)
        self.update_with_keys({glfw.KEY_A}, count=30)

        self.assertGreaterEqual(self.player.position[0], Player.RADIUS - 1e-6)

    def test_player_can_jump_and_land(self) -> None:
        self.update_with_keys(set(), count=3)
        self.update_with_keys({glfw.KEY_SPACE})

        self.assertGreater(self.player.position[1], 2.0)
        self.update_with_keys(set(), count=120)
        self.assertTrue(self.player._grounded)
        self.assertAlmostEqual(self.player.position[1], 2.0)

    def test_mouse_look_clamps_pitch(self) -> None:
        self.player.look(100.0, -10000.0)

        self.assertEqual(self.player.yaw, -80.0)
        self.assertEqual(self.player.pitch, 89.0)
        self.assertTrue(np.isfinite(self.player.view_matrix()).all())

    def test_player_detects_blocks_overlapping_its_body(self) -> None:
        self.assertTrue(self.player.intersects_block((12, 3, 12)))
        self.assertFalse(self.player.intersects_block((12, 4, 12)))


class HotbarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Application()
        self.app.world = VoxelWorld(size=24)
        self.app.player = Player(12.0, 2.0, 12.0)
        self.app.player.pitch = -35.0
        self.app.renderer = Mock()

    def test_number_keys_and_mouse_wheel_select_hotbar_slots(self) -> None:
        self.app._on_key(None, glfw.KEY_3, 0, glfw.PRESS, 0)
        self.assertEqual(self.app.selected_hotbar_slot, 2)

        self.app._on_scroll(None, 0.0, 1.0)
        self.assertEqual(self.app.selected_hotbar_slot, 1)

        self.app._on_scroll(None, 0.0, -1.0)
        self.assertEqual(self.app.selected_hotbar_slot, 2)

    def test_selected_block_is_placed_and_can_be_broken(self) -> None:
        player = self.app.player
        world = self.app.world
        renderer = self.app.renderer
        assert player is not None and world is not None

        target = world.raycast(player.eye_position(), player.forward_direction(), 6.0)
        self.assertIsNotNone(target)
        assert target is not None
        hit_block, adjacent_block = target
        self.assertIsNotNone(adjacent_block)
        assert adjacent_block is not None

        self.app._on_key(None, glfw.KEY_4, 0, glfw.PRESS, 0)
        self.app._on_mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)
        self.assertEqual(world.blocks[adjacent_block], BLOCK_SAND)
        renderer.update_mesh.assert_called_once()

        self.app._on_mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.RELEASE, 0)
        self.app._on_mouse_button(None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
        self.assertFalse(world.is_solid(*adjacent_block))
        self.assertEqual(renderer.update_mesh.call_count, 2)
        self.assertTrue(world.is_solid(*hit_block))

    def test_empty_hotbar_slot_does_not_place_a_block(self) -> None:
        player = self.app.player
        world = self.app.world
        assert player is not None and world is not None
        target = world.raycast(player.eye_position(), player.forward_direction(), 6.0)
        self.assertIsNotNone(target)
        assert target is not None
        _, adjacent_block = target
        self.assertIsNotNone(adjacent_block)
        assert adjacent_block is not None

        empty_slot_key = glfw.KEY_1 + HOTBAR_BLOCKS.index(None)
        self.app._on_key(None, empty_slot_key, 0, glfw.PRESS, 0)
        self.app._on_mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)

        self.assertFalse(world.is_solid(*adjacent_block))
        self.app.renderer.update_mesh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
