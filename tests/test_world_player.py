import unittest
from unittest.mock import Mock, patch

import glfw
import numpy as np

from engine.application import Application
from engine.game.player import Player
from engine.graphics.hud import Hud
from engine.graphics.renderer import Renderer
from engine.world.voxel_world import (
    _FACE_DATA,
    BLOCK_DIRT,
    BLOCK_GRASS,
    BLOCK_LEAF,
    BLOCK_SAND,
    BLOCK_STONE_BRICKS,
    BLOCK_STONE_COBBLE,
    BLOCK_STONE,
    BLOCK_WOOD,
    BLOCK_WOOD_BOARD,
    DIRT_LAYER_DEPTH,
    HOTBAR_BLOCKS,
    TEXTURE_DIRT,
    TEXTURE_GRASS_SIDE,
    TEXTURE_GRASS_TOP,
    TEXTURE_LEAF,
    TEXTURE_SAND,
    TEXTURE_STONE,
    TEXTURE_STONE_BRICKS,
    TEXTURE_STONE_COBBLE,
    TEXTURE_WOOD_SIDE,
    TEXTURE_WOOD_TOP,
    TEXTURE_WOOD_BOARD,
    VoxelWorld,
)


class VoxelWorldTests(unittest.TestCase):
    def test_default_world_dimensions_are_one_hundred_blocks(self) -> None:
        world = VoxelWorld()

        self.assertEqual(world.blocks.shape, (100, 100, 100))
        self.assertGreater(int(world.surface_heights.max()), int(world.surface_heights.min()))
        self.assertEqual(
            world.blocks[50, world.surface_height_at(50, 50), 50], BLOCK_GRASS
        )
        self.assertEqual(world.blocks[50, 0, 50], BLOCK_STONE)
        self.assertFalse(world.is_solid(100, 1, 99))

    def test_surface_has_grass_dirt_layers_and_deep_stone(self) -> None:
        world = VoxelWorld()
        x = z = world.size // 2
        surface = world.surface_height_at(x, z)

        self.assertEqual(world.blocks[x, surface, z], BLOCK_GRASS)
        for depth in range(1, DIRT_LAYER_DEPTH + 1):
            self.assertEqual(world.blocks[x, surface - depth, z], BLOCK_DIRT)
        self.assertEqual(
            world.blocks[x, surface - DIRT_LAYER_DEPTH - 1, z], BLOCK_STONE
        )

    def test_surface_generation_is_deterministic(self) -> None:
        first_world = VoxelWorld()
        second_world = VoxelWorld()

        np.testing.assert_array_equal(
            first_world.surface_heights, second_world.surface_heights
        )

    def test_surface_height_rejects_out_of_bounds_coordinates(self) -> None:
        world = VoxelWorld(size=8)

        with self.assertRaises(ValueError):
            world.surface_height_at(8, 0)

    def test_world_mesh_contains_only_exposed_faces(self) -> None:
        world = VoxelWorld(size=8)
        mesh = world.build_mesh()

        self.assertEqual(world.blocks.shape, (8, 8, 8))
        self.assertGreater(len(mesh), 0)
        self.assertEqual(mesh.shape[1], 10)
        self.assertEqual(len(mesh) % 6, 0)
        self.assertTrue(world.is_solid(0, 0, 0))
        self.assertFalse(world.is_solid(-1, 0, 0))
        surface = world.surface_height_at(0, 0)
        self.assertFalse(world.is_solid(0, surface + 1, 0))
        normals = mesh[:, 7:10]
        self.assertTrue(np.all(np.isin(normals, (-1.0, 0.0, 1.0))))
        self.assertEqual(len(np.unique(normals, axis=0)), 6)

    def test_cached_chunk_mesh_updates_blocks_across_chunk_boundaries(self) -> None:
        world = VoxelWorld(size=32)
        world.build_mesh()
        world.set_block((15, 4, 8), BLOCK_STONE)
        world.set_block((16, 4, 8), BLOCK_STONE)

        updated_mesh = world.build_mesh()
        expected_world = VoxelWorld(size=32)
        expected_world.set_block((15, 4, 8), BLOCK_STONE)
        expected_world.set_block((16, 4, 8), BLOCK_STONE)
        expected_mesh = expected_world.build_mesh()

        np.testing.assert_array_equal(updated_mesh, expected_mesh)

        world.set_block((16, 4, 8), 0)
        updated_after_remove = world.build_mesh()
        expected_world.set_block((16, 4, 8), 0)
        np.testing.assert_array_equal(
            updated_after_remove, expected_world.build_mesh()
        )

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

    def test_new_hotbar_blocks_use_their_texture_layers(self) -> None:
        world = VoxelWorld(size=8)
        blocks_and_layers = (
            (BLOCK_LEAF, TEXTURE_LEAF),
            (BLOCK_STONE_BRICKS, TEXTURE_STONE_BRICKS),
            (BLOCK_STONE_COBBLE, TEXTURE_STONE_COBBLE),
        )
        for index, (block_type, texture_layer) in enumerate(blocks_and_layers):
            world.set_block((2 + index * 2, 4, 3), block_type)

        wood_position = (7, 4, 3)
        world.set_block(wood_position, BLOCK_WOOD)
        mesh = world.build_mesh()

        for block_type, texture_layer in blocks_and_layers:
            self.assertIn(block_type, HOTBAR_BLOCKS)
            self.assertIn(texture_layer, mesh[:, 5])
        self.assertIn(BLOCK_WOOD, HOTBAR_BLOCKS)
        wood_vertices = mesh[
            (mesh[:, 0] >= wood_position[0])
            & (mesh[:, 0] <= wood_position[0] + 1)
            & (mesh[:, 1] >= wood_position[1])
            & (mesh[:, 1] <= wood_position[1] + 1)
            & (mesh[:, 2] >= wood_position[2])
            & (mesh[:, 2] <= wood_position[2] + 1)
        ]
        self.assertIn(TEXTURE_WOOD_SIDE, wood_vertices[:, 5])
        self.assertIn(TEXTURE_WOOD_TOP, wood_vertices[:, 5])

    def test_wood_board_is_available_in_hotbar_and_uses_its_texture(self) -> None:
        world = VoxelWorld(size=8)
        world.set_block((3, 4, 3), BLOCK_WOOD_BOARD)

        self.assertEqual(HOTBAR_BLOCKS[8], BLOCK_WOOD_BOARD)
        self.assertIn(TEXTURE_WOOD_BOARD, world.build_mesh()[:, 5])

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
        surface = max(world.surface_height_at(3, 4), world.surface_height_at(3, 5))
        target_y = surface + 1
        world.set_block((3, target_y, 4), BLOCK_STONE)
        world.set_block((3, target_y, 5), 0)
        origin = np.array((3.5, target_y + 0.5, 5.5))

        self.assertIsNone(world.raycast(origin, (0.0, 0.0, -1.0), 0.4))
        self.assertTrue(world.set_block((3, target_y, 5), BLOCK_GRASS))
        self.assertTrue(world.is_solid(3, target_y, 5))

    def test_set_block_rejects_out_of_bounds_and_unchanged_values(self) -> None:
        world = VoxelWorld(size=8)
        empty_y = world.surface_height_at(3, 3) + 1

        self.assertFalse(world.set_block((-1, 2, 0), 2))
        self.assertFalse(world.set_block((3, empty_y, 3), 0))
        self.assertTrue(world.set_block((3, empty_y, 3), 2))
        self.assertFalse(world.set_block((3, empty_y, 3), 2))

    def test_world_rejects_too_small_size(self) -> None:
        with self.assertRaises(ValueError):
            VoxelWorld(size=3)


class PlayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = VoxelWorld(size=24)
        spawn_y = self.world.surface_height_at(12, 12) + 1.0
        self.player = Player(12.5, spawn_y, 12.5)

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
        self.assertAlmostEqual(
            self.player.position[1],
            self.world.surface_height_at(12, 12) + 1.0,
        )

    def test_player_stops_against_solid_block(self) -> None:
        self.player.position[0] = 12.5
        player_floor = int(self.player.position[1])
        self.world.blocks[11, player_floor, 12] = 1
        self.update_with_keys(set(), count=3)
        self.update_with_keys({glfw.KEY_A}, count=30)

        self.assertGreaterEqual(self.player.position[0], 12.3 - 1e-6)

    def test_player_cannot_leave_world_bounds(self) -> None:
        self.player.position[0] = 0.5
        self.update_with_keys(set(), count=3)
        self.update_with_keys({glfw.KEY_A}, count=30)

        self.assertGreaterEqual(self.player.position[0], Player.RADIUS - 1e-6)

    def test_player_can_jump_and_land(self) -> None:
        ground_height = self.player.position[1]
        self.update_with_keys(set(), count=3)
        self.update_with_keys({glfw.KEY_SPACE})

        self.assertGreater(self.player.position[1], ground_height)
        self.update_with_keys(set(), count=120)
        self.assertTrue(self.player._grounded)
        self.assertAlmostEqual(self.player.position[1], ground_height)

    def test_mouse_look_clamps_pitch(self) -> None:
        self.player.look(100.0, -10000.0)

        self.assertEqual(self.player.yaw, -80.0)
        self.assertEqual(self.player.pitch, 89.0)
        self.assertTrue(np.isfinite(self.player.view_matrix()).all())

    def test_player_detects_blocks_overlapping_its_body(self) -> None:
        feet_y = int(self.player.position[1])
        self.assertTrue(self.player.intersects_block((12, feet_y, 12)))
        self.assertFalse(self.player.intersects_block((12, feet_y + 2, 12)))


class HotbarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Application()
        self.app.world = VoxelWorld(size=24)
        spawn_y = self.app.world.surface_height_at(12, 12) + 1.0
        self.app.player = Player(12.5, spawn_y, 12.5)
        self.app.player.pitch = -35.0
        self.app.renderer = Mock()

    def test_number_keys_and_mouse_wheel_select_hotbar_slots(self) -> None:
        self.app._on_key(None, glfw.KEY_3, 0, glfw.PRESS, 0)
        self.assertEqual(self.app.selected_hotbar_slot, 2)

        self.app._on_scroll(None, 0.0, 1.0)
        self.assertEqual(self.app.selected_hotbar_slot, 1)

        self.app._on_scroll(None, 0.0, -1.0)
        self.assertEqual(self.app.selected_hotbar_slot, 2)

    def test_f3_toggles_performance_overlay(self) -> None:
        self.assertFalse(self.app.show_performance_overlay)

        self.app._on_key(None, glfw.KEY_F3, 0, glfw.PRESS, 0)
        self.assertTrue(self.app.show_performance_overlay)

        self.app._on_key(None, glfw.KEY_F3, 0, glfw.PRESS, 0)
        self.assertFalse(self.app.show_performance_overlay)

    def test_f4_toggles_chunk_grid(self) -> None:
        self.assertFalse(self.app.show_chunk_grid)

        self.app._on_key(None, glfw.KEY_F4, 0, glfw.PRESS, 0)
        self.assertTrue(self.app.show_chunk_grid)

        self.app._on_key(None, glfw.KEY_F4, 0, glfw.PRESS, 0)
        self.assertFalse(self.app.show_chunk_grid)

    def test_chunk_grid_has_lines_at_chunk_and_world_boundaries(self) -> None:
        surface_heights = np.zeros((10, 10), dtype=np.int16)
        vertices = Renderer._build_chunk_grid_vertices(surface_heights)

        self.assertEqual(vertices.shape, (120, 3))
        np.testing.assert_allclose(vertices[::2, 1], np.full(60, 1.02))
        boundary_coordinates = (0.0, 8.0, 10.0)
        vertices_per_boundary = 4 * len(surface_heights)
        for index, coordinate in enumerate(boundary_coordinates):
            boundary_vertices = vertices[
                index * vertices_per_boundary:
                (index + 1) * vertices_per_boundary
            ]
            vertical_vertices = boundary_vertices[:2 * len(surface_heights)]
            horizontal_vertices = boundary_vertices[2 * len(surface_heights):]
            self.assertEqual(set(vertical_vertices[:, 0]), {coordinate})
            self.assertEqual(set(horizontal_vertices[:, 2]), {coordinate})

    def test_chunk_grid_follows_terrain_height(self) -> None:
        surface_heights = np.arange(100, dtype=np.int16).reshape(10, 10)

        vertices = Renderer._build_chunk_grid_vertices(surface_heights)

        self.assertGreater(float(vertices[:, 1].max()), float(vertices[:, 1].min()))
        self.assertAlmostEqual(float(vertices[:, 1].min()), 1.02)

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

    @patch.object(Player, "intersects_block", return_value=False)
    def test_ninth_hotbar_slot_places_wood_board(
        self, _intersects_block: Mock
    ) -> None:
        player = self.app.player
        world = self.app.world
        assert player is not None and world is not None
        adjacent_block = (10, 2, 10)
        world.raycast = Mock(return_value=((10, 1, 10), adjacent_block))
        self.app._on_key(None, glfw.KEY_9, 0, glfw.PRESS, 0)
        self.app._on_mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)

        self.assertEqual(world.blocks[adjacent_block], BLOCK_WOOD_BOARD)
        self.app.renderer.update_mesh.assert_called_once()

    @patch.object(Player, "intersects_block", return_value=False)
    def test_placing_grass_on_grass_turns_lower_block_into_dirt(
        self, _intersects_block: Mock
    ) -> None:
        world = self.app.world
        assert world is not None
        hit_block = (10, 0, 10)
        placed_block = (10, 2, 10)
        lower_block = (10, 1, 10)
        world.blocks[lower_block] = BLOCK_GRASS
        world.raycast = Mock(return_value=(hit_block, placed_block))

        self.app._on_mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)

        self.assertEqual(world.blocks[placed_block], BLOCK_GRASS)
        self.assertEqual(world.blocks[lower_block], BLOCK_DIRT)
        self.app.renderer.update_mesh.assert_called_once()

    @patch.object(Player, "intersects_block", return_value=False)
    def test_placing_non_grass_on_grass_turns_lower_block_into_dirt(
        self, _intersects_block: Mock
    ) -> None:
        world = self.app.world
        assert world is not None
        hit_block = (10, 0, 10)
        placed_block = (10, 2, 10)
        lower_block = (10, 1, 10)
        world.blocks[lower_block] = BLOCK_GRASS
        world.raycast = Mock(return_value=(hit_block, placed_block))
        self.app.selected_hotbar_slot = HOTBAR_BLOCKS.index(BLOCK_SAND)

        self.app._on_mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)

        self.assertEqual(world.blocks[placed_block], BLOCK_SAND)
        self.assertEqual(world.blocks[lower_block], BLOCK_DIRT)
        self.app.renderer.update_mesh.assert_called_once()


class PauseMenuTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Application()
        self.app.window = Mock()

    def test_pause_menu_buttons_are_ordered_top_to_bottom(self) -> None:
        button_bounds = Hud._menu_button_bounds(1280, 720)

        self.assertGreater(button_bounds[0][1], button_bounds[1][1])
        self.assertGreater(button_bounds[1][1], button_bounds[2][1])

    @patch("engine.application.glfw.set_cursor_pos")
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    @patch("engine.application.glfw.set_input_mode")
    def test_escape_opens_and_closes_pause_menu(
        self,
        set_input_mode: Mock,
        _framebuffer_size: Mock,
        _window_size: Mock,
        set_cursor_pos: Mock,
    ) -> None:
        self.app._on_key(None, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)

        self.assertTrue(self.app.menu_open)
        set_input_mode.assert_called_once_with(
            self.app.window, glfw.CURSOR, glfw.CURSOR_NORMAL
        )
        continue_x, continue_y, continue_width, continue_height = (
            Hud._menu_button_bounds(1280, 720)[0]
        )
        set_cursor_pos.assert_called_once_with(
            self.app.window,
            continue_x + continue_width / 2,
            720 - continue_y - continue_height / 2,
        )

        self.app._on_key(None, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)

        self.assertFalse(self.app.menu_open)
        set_input_mode.assert_called_with(
            self.app.window, glfw.CURSOR, glfw.CURSOR_DISABLED
        )

    @patch("engine.application.glfw.get_cursor_pos", return_value=(640.0, 360.0))
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    def test_settings_button_opens_page_and_escape_returns_to_menu(
        self,
        _framebuffer_size: Mock,
        _window_size: Mock,
        _cursor_pos: Mock,
    ) -> None:
        self.app.menu_open = True
        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0
        )

        self.assertTrue(self.app.menu_open)
        self.assertTrue(self.app.settings_open)

        self.app._on_key(None, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)
        self.assertTrue(self.app.menu_open)
        self.assertFalse(self.app.settings_open)

    @patch("engine.application.glfw.get_cursor_pos")
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    def test_texture_setting_click_updates_renderer(
        self,
        _framebuffer_size: Mock,
        _window_size: Mock,
        get_cursor_pos: Mock,
    ) -> None:
        button_x, button_y, button_width, button_height = (
            Hud._settings_choice_bounds(1280, 720, 0)[2]
        )
        get_cursor_pos.return_value = (
            button_x + button_width / 2,
            720 - button_y - button_height / 2,
        )
        self.app.menu_open = True
        self.app.settings_open = True
        self.app.renderer = Mock()

        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0
        )

        self.assertEqual(self.app.texture_quality, 2)
        self.app.renderer.set_texture_quality.assert_called_once_with(2)

    @patch("engine.application.glfw.get_cursor_pos")
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    def test_shadow_setting_can_disable_shadows(
        self,
        _framebuffer_size: Mock,
        _window_size: Mock,
        get_cursor_pos: Mock,
    ) -> None:
        button_x, button_y, button_width, button_height = (
            Hud._settings_choice_bounds(1280, 720, 1)[0]
        )
        get_cursor_pos.return_value = (
            button_x + button_width / 2,
            720 - button_y - button_height / 2,
        )
        self.app.menu_open = True
        self.app.settings_open = True
        self.app.renderer = Mock()

        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0
        )

        self.assertEqual(self.app.shadow_quality, 0)
        self.app.renderer.set_shadow_quality.assert_called_once_with(0)

    @patch("engine.application.glfw.get_cursor_pos")
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    def test_color_grade_setting_updates_renderer(
        self,
        _framebuffer_size: Mock,
        _window_size: Mock,
        get_cursor_pos: Mock,
    ) -> None:
        button_x, button_y, button_width, button_height = (
            Hud._settings_choice_bounds(1280, 720, 2)[3]
        )
        get_cursor_pos.return_value = (
            button_x + button_width / 2,
            720 - button_y - button_height / 2,
        )
        self.app.menu_open = True
        self.app.settings_open = True
        self.app.renderer = Mock()

        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0
        )

        self.assertEqual(self.app.color_grade, 3)
        self.app.renderer.set_color_grade.assert_called_once_with(3)

    def test_color_grade_settings_cover_every_preset_without_back_overlap(self) -> None:
        presets = Hud._settings_choice_bounds(1280, 720, 2)
        self.assertEqual(len(presets), 4)
        for option, bounds in enumerate(presets):
            self.assertEqual(
                Hud.settings_option_at(
                    1280,
                    720,
                    bounds[0] + bounds[2] / 2,
                    bounds[1] + bounds[3] / 2,
                ),
                (2, option),
            )

        back_bounds = Hud._settings_back_bounds(1280, 720)
        self.assertEqual(
            Hud.settings_option_at(
                1280,
                720,
                back_bounds[0] + back_bounds[2] / 2,
                back_bounds[1] + back_bounds[3] / 2,
            ),
            (4, 0),
        )

        compact_back = Hud._settings_back_bounds(640, 360)
        compact_last_preset = Hud._settings_choice_bounds(640, 360, 2)[-1]
        self.assertLess(
            compact_back[1] + compact_back[3], compact_last_preset[1]
        )

    @patch("engine.application.glfw.get_cursor_pos")
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    def test_visual_effects_setting_can_disable_effects(
        self,
        _framebuffer_size: Mock,
        _window_size: Mock,
        get_cursor_pos: Mock,
    ) -> None:
        button_x, button_y, button_width, button_height = (
            Hud._settings_choice_bounds(1280, 720, 3)[0]
        )
        get_cursor_pos.return_value = (
            button_x + button_width / 2,
            720 - button_y - button_height / 2,
        )
        self.app.menu_open = True
        self.app.settings_open = True
        self.app.renderer = Mock()

        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0
        )

        self.assertFalse(self.app.effects_enabled)
        self.app.renderer.set_effects_enabled.assert_called_once_with(False)

    def test_effects_settings_row_maps_on_and_off(self) -> None:
        choices = Hud._settings_choice_bounds(1280, 720, 3)
        self.assertEqual(len(choices), 2)
        for option, bounds in enumerate(choices):
            self.assertEqual(
                Hud.settings_option_at(
                    1280,
                    720,
                    bounds[0] + bounds[2] / 2,
                    bounds[1] + bounds[3] / 2,
                ),
                (3, option),
            )

    def test_color_grading_updates_sky_clear_color(self) -> None:
        clear_colors = [
            self.app._clear_color_for_grade(grade) for grade in range(4)
        ]
        self.assertEqual(clear_colors[0], (0.04, 0.07, 0.14))
        self.assertEqual(len(set(clear_colors)), 4)

    def test_settings_page_maps_quality_choices(self) -> None:
        texture_bounds = Hud._settings_choice_bounds(1280, 720, 0)[1]
        shadow_bounds = Hud._settings_choice_bounds(1280, 720, 1)[0]

        self.assertEqual(
            Hud.settings_option_at(
                1280,
                720,
                texture_bounds[0] + texture_bounds[2] / 2,
                texture_bounds[1] + texture_bounds[3] / 2,
            ),
            (0, 1),
        )
        self.assertEqual(
            Hud.settings_option_at(
                1280,
                720,
                shadow_bounds[0] + shadow_bounds[2] / 2,
                shadow_bounds[1] + shadow_bounds[3] / 2,
            ),
            (1, 0),
        )

    @patch("engine.application.glfw.get_cursor_pos", return_value=(640.0, 306.0))
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    @patch("engine.application.glfw.set_input_mode")
    def test_resume_button_closes_pause_menu(
        self,
        _set_input_mode: Mock,
        _framebuffer_size: Mock,
        _window_size: Mock,
        _cursor_pos: Mock,
    ) -> None:
        self.app.menu_open = True
        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0
        )

        self.assertFalse(self.app.menu_open)

    @patch("engine.application.glfw.get_cursor_pos", return_value=(640.0, 414.0))
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    @patch("engine.application.glfw.set_window_should_close")
    def test_exit_button_requests_window_close(
        self,
        set_window_should_close: Mock,
        _framebuffer_size: Mock,
        _window_size: Mock,
        _cursor_pos: Mock,
    ) -> None:
        self.app.menu_open = True
        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0
        )

        set_window_should_close.assert_called_once_with(self.app.window, True)


if __name__ == "__main__":
    unittest.main()
