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
    BLOCK_GLOWSTONE,
    BLOCK_LANTERN,
    BLOCK_LEAF,
    BLOCK_SAND,
    BLOCK_STONE_BRICKS,
    BLOCK_STONE_COBBLE,
    BLOCK_STONE,
    BLOCK_WOOD,
    BLOCK_WOOD_BOARD,
    BLOCK_TYPES,
    DIRT_LAYER_DEPTH,
    GLOWSTONE_LIGHT_LEVEL,
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
    TEXTURE_LANTERN,
    VoxelWorld,
)


class VoxelWorldTests(unittest.TestCase):
    def test_default_world_dimensions_are_one_thousand_by_one_thousand(self) -> None:
        world = VoxelWorld()

        self.assertEqual(world.blocks.shape, (1000, 100, 1000))
        self.assertGreater(int(world.surface_heights.max()), int(world.surface_heights.min()))
        self.assertEqual(
            world.blocks[500, world.surface_height_at(500, 500), 500], BLOCK_GRASS
        )
        self.assertEqual(world.blocks[500, 0, 500], BLOCK_STONE)
        self.assertFalse(world.is_solid(1000, 1, 999))

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
        self.assertGreaterEqual(
            int(first_world.surface_heights.max() - first_world.surface_heights.min()),
            40,
        )
        self.assertGreater(float(first_world.surface_heights.std()), 4.0)

    def test_surface_height_rejects_out_of_bounds_coordinates(self) -> None:
        world = VoxelWorld(size=8)

        with self.assertRaises(ValueError):
            world.surface_height_at(8, 0)

    def test_world_mesh_contains_only_exposed_faces(self) -> None:
        world = VoxelWorld(size=8)
        mesh = world.build_mesh()

        self.assertEqual(world.blocks.shape, (8, 8, 8))
        self.assertGreater(len(mesh), 0)
        self.assertEqual(mesh.shape[1], 12)
        self.assertEqual(len(mesh) % 6, 0)
        self.assertTrue(world.is_solid(0, 0, 0))
        self.assertFalse(world.is_solid(-1, 0, 0))
        surface = world.surface_height_at(0, 0)
        self.assertFalse(world.is_solid(0, surface + 1, 0))
        normals = mesh[:, 9:12]
        self.assertTrue(np.all(np.isin(normals, (-1.0, 0.0, 1.0))))
        self.assertEqual(len(np.unique(normals, axis=0)), 6)
        triangles = mesh.reshape(-1, 3, 12)
        edges = triangles[:, 1:3, :3] - triangles[:, :1, :3]
        winding = np.cross(edges[:, 0], edges[:, 1])
        self.assertTrue(np.all(np.sum(winding * triangles[:, 0, 9:12], axis=1) > 0))

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

    def test_glowstone_light_decreases_over_empty_blocks(self) -> None:
        world = VoxelWorld(size=16)
        y = int(world.surface_heights.max()) + 3
        source = (8, y, 8)
        world.set_block(source, BLOCK_GLOWSTONE)
        world.set_block((7, y, 8), BLOCK_STONE)

        mesh = world.build_mesh()

        self.assertEqual(int(world._block_light_levels[source]), GLOWSTONE_LIGHT_LEVEL)
        self.assertEqual(int(world._block_light_levels[7, y, 8]), 0)
        self.assertEqual(int(world._block_light_levels[9, y, 8]), 11)
        self.assertEqual(int(world._block_light_levels[10, y, 8]), 10)
        source_vertices = mesh[
            (mesh[:, 0] >= source[0])
            & (mesh[:, 0] <= source[0] + 1)
            & (mesh[:, 1] >= source[1])
            & (mesh[:, 1] <= source[1] + 1)
            & (mesh[:, 2] >= source[2])
            & (mesh[:, 2] <= source[2] + 1)
        ]
        self.assertLess(float(source_vertices[:, 7].max()), 12 / 15)
        self.assertGreater(float(source_vertices[:, 7].min()), 0.0)
        self.assertLess(
            float(source_vertices[:, 7].min()),
            float(source_vertices[:, 7].max()),
        )

        world.set_block(source, 0)
        world.build_mesh()
        self.assertEqual(int(world._block_light_levels[9, y, 8]), 0)

    def test_lantern_uses_its_texture_and_emits_light(self) -> None:
        world = VoxelWorld(size=16)
        y = int(world.surface_heights.max()) + 3
        source = (8, y, 8)
        world.set_block(source, BLOCK_LANTERN)

        mesh = world.build_mesh()

        self.assertIn(BLOCK_LANTERN, BLOCK_TYPES)
        self.assertNotIn(BLOCK_LANTERN, HOTBAR_BLOCKS)
        self.assertIn(TEXTURE_LANTERN, mesh[:, 5])
        self.assertEqual(
            int(world._block_light_levels[source]), GLOWSTONE_LIGHT_LEVEL
        )
        self.assertEqual(int(world._block_light_levels[9, y, 8]), 11)

    def test_sunlight_reaches_open_sky_and_fades_under_a_roof(self) -> None:
        world = VoxelWorld(size=8)
        world.blocks.fill(0)
        world.blocks[:, 5, :] = BLOCK_STONE
        world.blocks[3, 5, 3] = 0

        mesh = world.build_mesh()

        self.assertEqual(int(world._sky_light_levels[0, 6, 0]), 15)
        self.assertEqual(int(world._sky_light_levels[3, 4, 3]), 15)
        self.assertEqual(int(world._sky_light_levels[4, 4, 3]), 14)
        self.assertEqual(int(world._sky_light_levels[5, 4, 3]), 13)
        self.assertEqual(int(world._sky_light_levels[6, 4, 3]), 12)

        roof_vertices = mesh[
            (mesh[:, 0] >= 0)
            & (mesh[:, 0] <= 1)
            & (mesh[:, 1] == 6)
            & (mesh[:, 2] >= 0)
            & (mesh[:, 2] <= 1)
            & (mesh[:, 9] == 0)
            & (mesh[:, 10] == 1)
            & (mesh[:, 11] == 0)
        ]
        self.assertGreaterEqual(len(roof_vertices), 6)
        np.testing.assert_array_equal(
            roof_vertices[:, 8], np.ones(len(roof_vertices))
        )

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

    def test_block_count_tracks_added_replaced_and_removed_blocks(self) -> None:
        world = VoxelWorld(size=8)
        initial_count = int(np.count_nonzero(world.blocks))
        self.assertEqual(world.block_count, initial_count)

        empty_y = world.surface_height_at(3, 3) + 1
        self.assertTrue(world.set_block((3, empty_y, 3), BLOCK_GRASS))
        self.assertEqual(world.block_count, initial_count + 1)
        self.assertTrue(world.set_block((3, empty_y, 3), BLOCK_STONE))
        self.assertEqual(world.block_count, initial_count + 1)
        self.assertTrue(world.set_block((3, empty_y, 3), 0))
        self.assertEqual(world.block_count, initial_count)

        original_block = int(world.blocks[0, 0, 0])
        self.assertTrue(world.set_block((0, 0, 0), 0))
        self.assertEqual(world.block_count, initial_count - 1)
        self.assertTrue(world.set_block((0, 0, 0), original_block))
        self.assertEqual(world.block_count, initial_count)

    def test_world_rejects_too_small_size(self) -> None:
        with self.assertRaises(ValueError):
            VoxelWorld(size=3)

    def test_world_height_can_be_set_independently_of_surface_dimensions(self) -> None:
        world = VoxelWorld(size=24, height=12)

        self.assertEqual(world.blocks.shape, (24, 12, 24))
        self.assertLess(int(world.surface_heights.max()), world.height)

    def test_mesh_streaming_only_builds_chunks_near_requested_position(self) -> None:
        world = VoxelWorld(size=64, height=16)

        world.build_mesh((8.0, 8.0, 8.0), render_distance=8)
        first_loaded_chunks = set(world._mesh_cache)
        first_lighting_bounds = world._lighting_bounds
        world.build_mesh((56.0, 8.0, 56.0), render_distance=8)

        self.assertTrue(first_loaded_chunks)
        self.assertTrue(set(world._mesh_cache).isdisjoint(first_loaded_chunks))
        self.assertNotEqual(world._lighting_bounds, first_lighting_bounds)
        self.assertLess(
            world._lighting_bounds[1] - world._lighting_bounds[0], world.size
        )
        self.assertLess(
            world._lighting_bounds[3] - world._lighting_bounds[2], world.size
        )

    def test_mesh_streaming_includes_new_blocks_in_previously_empty_chunks(self) -> None:
        world = VoxelWorld(size=16, height=64)
        elevated_chunk = (1, 7, 1)

        self.assertTrue(world.set_block((8, 60, 8), BLOCK_STONE))
        mesh = world.build_mesh((8.0, 60.0, 8.0), render_distance=8)

        self.assertIn(elevated_chunk, world._active_mesh_chunks)
        self.assertIn(elevated_chunk, world._mesh_cache)
        self.assertGreater(len(mesh), 0)

        self.assertTrue(world.set_block((8, 60, 8), 0))
        world.build_mesh((8.0, 60.0, 8.0), render_distance=8)
        self.assertNotIn(elevated_chunk, world._active_mesh_chunks)

    def test_fully_enclosed_solid_chunks_do_not_generate_meshes(self) -> None:
        world = VoxelWorld(size=24, height=24)
        world.blocks.fill(BLOCK_STONE)

        self.assertEqual(len(world._build_chunk_mesh((1, 1, 1))), 0)

        world.blocks[7, 8, 8] = 0
        self.assertGreater(len(world._build_chunk_mesh((1, 1, 1))), 0)


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

    def test_zero_selects_the_tenth_glowstone_slot(self) -> None:
        self.app._on_key(None, glfw.KEY_0, 0, glfw.PRESS, 0)

        self.assertEqual(self.app.selected_hotbar_slot, 9)
        self.assertEqual(HOTBAR_BLOCKS[self.app.selected_hotbar_slot], BLOCK_GLOWSTONE)

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
    def test_tenth_hotbar_slot_places_glowstone(
        self, _intersects_block: Mock
    ) -> None:
        world = self.app.world
        assert world is not None
        adjacent_block = (10, 2, 10)
        world.raycast = Mock(return_value=((10, 1, 10), adjacent_block))
        self.app._on_key(None, glfw.KEY_0, 0, glfw.PRESS, 0)
        self.app._on_mouse_button(None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)

        self.assertEqual(world.blocks[adjacent_block], BLOCK_GLOWSTONE)
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


class InventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Application()
        self.app.window = Mock()
        set_input_mode = patch("engine.application.glfw.set_input_mode")
        set_input_mode.start()
        self.addCleanup(set_input_mode.stop)

    def test_i_opens_inventory_and_escape_closes_it(self) -> None:
        self.app._on_key(None, glfw.KEY_I, 0, glfw.PRESS, 0)
        self.assertTrue(self.app.inventory_open)
        self.assertFalse(self.app.menu_open)

        self.app._on_key(None, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)
        self.assertFalse(self.app.inventory_open)
        self.assertFalse(self.app.menu_open)

    def test_number_keys_switch_inventory_tabs_without_changing_hotbar(self) -> None:
        self.app._on_key(None, glfw.KEY_I, 0, glfw.PRESS, 0)
        self.app._on_key(None, glfw.KEY_2, 0, glfw.PRESS, 0)
        self.assertEqual(self.app.inventory_tab, 1)
        self.assertEqual(self.app.selected_hotbar_slot, 0)

        self.app._on_key(None, glfw.KEY_1, 0, glfw.PRESS, 0)
        self.assertEqual(self.app.inventory_tab, 0)

    def test_inventory_tabs_are_mouse_selectable(self) -> None:
        for expected_tab, bounds in enumerate(Hud._inventory_tab_bounds(1280, 720)):
            x, y, width, height = bounds
            self.assertEqual(
                Hud.inventory_tab_at(1280, 720, x + width / 2, y + height / 2),
                expected_tab,
            )
        self.assertIsNone(Hud.inventory_tab_at(1280, 720, 0, 0))

    def test_inventory_tabs_render_with_all_block_labels(self) -> None:
        with patch.object(Hud, "_rectangle"):
            Hud.draw_inventory(1280, 720, 0)
            Hud.draw_inventory(1280, 720, 1)
            Hud.draw_inventory(640, 360, 1)

    @patch("engine.application.glfw.get_cursor_pos")
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    def test_dragging_hotbar_block_swaps_slots(
        self,
        _framebuffer_size: Mock,
        _window_size: Mock,
        cursor_position: Mock,
    ) -> None:
        self.app._open_inventory()
        original_blocks = self.app.hotbar_blocks.copy()
        slot_bounds = Hud._inventory_hotbar_slot_bounds(1280, 720)
        source = slot_bounds[0]
        destination = slot_bounds[3]
        cursor_position.return_value = (
            source[0] + source[2] // 2,
            720 - source[1] - source[3] // 2,
        )
        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0
        )

        cursor_position.return_value = (
            destination[0] + destination[2] // 2,
            720 - destination[1] - destination[3] // 2,
        )
        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0
        )

        self.assertEqual(self.app.hotbar_blocks[0], original_blocks[3])
        self.assertEqual(self.app.hotbar_blocks[3], original_blocks[0])
        self.assertEqual(self.app.selected_hotbar_slot, 3)
        self.assertIsNone(self.app._inventory_drag_source)
        self.assertIsNone(self.app._inventory_drag_block)

    def test_inventory_hotbar_bounds_match_all_block_slots(self) -> None:
        bounds = Hud._inventory_hotbar_slot_bounds(1280, 720)

        self.assertEqual(len(bounds), len(HOTBAR_BLOCKS))
        for index, slot in enumerate(bounds):
            x, y, width, height = slot
            self.assertEqual(
                Hud.inventory_hotbar_slot_at(
                    1280, 720, x + width // 2, y + height // 2
                ),
                index,
            )

    def test_lantern_is_in_the_block_catalog(self) -> None:
        self.assertEqual(BLOCK_TYPES[-1], BLOCK_LANTERN)
        self.assertNotIn(BLOCK_LANTERN, HOTBAR_BLOCKS)
        self.assertEqual(Hud._INVENTORY_BLOCK_NAMES[-1], "ФОНАРЬ")

    def test_game_hud_draws_only_current_hotbar_blocks(self) -> None:
        drawn_blocks: list[int] = []
        with (
            patch.object(Hud, "_rectangle"),
            patch.object(
                Hud,
                "_draw_block_texture_icon",
                side_effect=lambda _x, _y, _size, _scale, block: drawn_blocks.append(
                    block
                ),
            ),
        ):
            Hud._draw_hotbar(0, 0, 20, 2, 1, 0, list(HOTBAR_BLOCKS))

        self.assertEqual(drawn_blocks, list(HOTBAR_BLOCKS))
        self.assertNotIn(BLOCK_LANTERN, drawn_blocks)

    def test_block_icon_draws_a_crisp_eight_by_eight_texture_sample(self) -> None:
        rectangles: list[tuple[int, int, int, int, tuple[float, float, float]]] = []
        with patch.object(
            Hud, "_rectangle", side_effect=lambda *rectangle: rectangles.append(rectangle)
        ):
            Hud._draw_block_texture_icon(0, 0, 20, 1, BLOCK_LANTERN)

        self.assertEqual(len(rectangles), 64)
        self.assertTrue(all(rectangle[2:4] == (2, 2) for rectangle in rectangles))
        self.assertGreater(len({rectangle[4] for rectangle in rectangles}), 1)

    def test_inventory_slots_use_small_block_texture_icons(self) -> None:
        with (
            patch.object(Hud, "_rectangle"),
            patch.object(Hud, "_draw_menu_text"),
            patch.object(Hud, "_draw_block_texture_icon") as draw_icon,
        ):
            Hud.draw_inventory(
                1280,
                720,
                0,
                HOTBAR_BLOCKS,
                0,
                [BLOCK_LANTERN] + [None] * 26,
            )
            inventory_icons = [
                call.args[-1] for call in draw_icon.call_args_list
            ]
            self.assertIn(BLOCK_LANTERN, inventory_icons)

            draw_icon.reset_mock()
            Hud.draw_inventory(1280, 720, 1, HOTBAR_BLOCKS)
            catalog_icons = [
                call.args[-1] for call in draw_icon.call_args_list
            ]

        self.assertEqual(catalog_icons, [*BLOCK_TYPES, *HOTBAR_BLOCKS])

    @patch("engine.application.glfw.get_cursor_pos")
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    def test_lantern_can_be_dragged_from_catalog_to_hotbar(
        self,
        _framebuffer_size: Mock,
        _window_size: Mock,
        cursor_position: Mock,
    ) -> None:
        self.app._open_inventory()
        self.app.inventory_tab = 1
        catalog_slot = Hud._inventory_storage_slot_bounds(1280, 720)[10]
        hotbar_slot = Hud._inventory_hotbar_slot_bounds(1280, 720)[9]

        cursor_position.return_value = (
            catalog_slot[0] + catalog_slot[2] // 2,
            720 - catalog_slot[1] - catalog_slot[3] // 2,
        )
        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0
        )
        cursor_position.return_value = (
            hotbar_slot[0] + hotbar_slot[2] // 2,
            720 - hotbar_slot[1] - hotbar_slot[3] // 2,
        )
        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0
        )

        self.assertEqual(self.app.hotbar_blocks[9], BLOCK_LANTERN)
        self.assertEqual(self.app.selected_hotbar_slot, 9)
        self.assertEqual(self.app._inventory_carried_block, BLOCK_GLOWSTONE)

    @patch("engine.application.glfw.get_cursor_pos")
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    def test_catalog_block_can_be_carried_into_inventory(
        self,
        _framebuffer_size: Mock,
        _window_size: Mock,
        cursor_position: Mock,
    ) -> None:
        self.app._open_inventory()
        self.app.inventory_tab = 1

        def click_at(x: int, y: int) -> None:
            cursor_position.return_value = (x, 720 - y)
            self.app._on_mouse_button(
                self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0
            )
            self.app._on_mouse_button(
                self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0
            )

        catalog_x, catalog_y, catalog_width, catalog_height = (
            Hud._inventory_storage_slot_bounds(1280, 720)[3]
        )
        click_at(
            catalog_x + catalog_width // 2,
            catalog_y + catalog_height // 2,
        )
        self.assertEqual(self.app._inventory_carried_block, BLOCK_SAND)

        tab_x, tab_y, tab_width, tab_height = Hud._inventory_tab_bounds(1280, 720)[0]
        click_at(tab_x + tab_width // 2, tab_y + tab_height // 2)
        self.assertEqual(self.app.inventory_tab, 0)

        slot_x, slot_y, slot_width, slot_height = (
            Hud._inventory_storage_slot_bounds(1280, 720)[0]
        )
        click_at(slot_x + slot_width // 2, slot_y + slot_height // 2)
        self.assertEqual(self.app.inventory_slots[0], BLOCK_SAND)
        self.assertIsNone(self.app._inventory_carried_block)

    @patch("engine.application.glfw.get_cursor_pos")
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    def test_catalog_block_can_be_dragged_directly_to_hotbar(
        self,
        _framebuffer_size: Mock,
        _window_size: Mock,
        cursor_position: Mock,
    ) -> None:
        self.app._open_inventory()
        self.app.inventory_tab = 1

        catalog_slot = Hud._inventory_storage_slot_bounds(1280, 720)[3]
        hotbar_slot = Hud._inventory_hotbar_slot_bounds(1280, 720)[6]
        cursor_position.return_value = (
            catalog_slot[0] + catalog_slot[2] // 2,
            720 - catalog_slot[1] - catalog_slot[3] // 2,
        )
        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0
        )

        cursor_position.return_value = (
            hotbar_slot[0] + hotbar_slot[2] // 2,
            720 - hotbar_slot[1] - hotbar_slot[3] // 2,
        )
        self.app._on_mouse_button(
            self.app.window, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0
        )

        self.assertEqual(self.app.hotbar_blocks[6], BLOCK_SAND)
        self.assertEqual(self.app.selected_hotbar_slot, 6)
        self.assertEqual(self.app._inventory_carried_block, HOTBAR_BLOCKS[6])


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
            Hud._settings_choice_bounds(1280, 720, 3)[3]
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
        presets = Hud._settings_choice_bounds(1280, 720, 3)
        self.assertEqual(len(presets), 4)
        for option, bounds in enumerate(presets):
            self.assertEqual(
                Hud.settings_option_at(
                    1280,
                    720,
                    bounds[0] + bounds[2] / 2,
                    bounds[1] + bounds[3] / 2,
                ),
                (3, option),
            )

        back_bounds = Hud._settings_back_bounds(1280, 720)
        self.assertEqual(
            Hud.settings_option_at(
                1280,
                720,
                back_bounds[0] + back_bounds[2] / 2,
                back_bounds[1] + back_bounds[3] / 2,
            ),
            (5, 0),
        )

        compact_back = Hud._settings_back_bounds(640, 360)
        compact_last_row = Hud._settings_choice_bounds(640, 360, 4)[0]
        self.assertLess(
            compact_back[1] + compact_back[3], compact_last_row[1]
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
            Hud._settings_choice_bounds(1280, 720, 4)[0]
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
        choices = Hud._settings_choice_bounds(1280, 720, 4)
        self.assertEqual(len(choices), 2)
        for option, bounds in enumerate(choices):
            self.assertEqual(
                Hud.settings_option_at(
                    1280,
                    720,
                    bounds[0] + bounds[2] / 2,
                    bounds[1] + bounds[3] / 2,
                ),
                (4, option),
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

    @patch("engine.application.glfw.get_cursor_pos")
    @patch("engine.application.glfw.get_window_size", return_value=(1280, 720))
    @patch("engine.application.glfw.get_framebuffer_size", return_value=(1280, 720))
    def test_block_lighting_setting_is_independent_from_shadows(
        self,
        _framebuffer_size: Mock,
        _window_size: Mock,
        get_cursor_pos: Mock,
    ) -> None:
        button_x, button_y, button_width, button_height = (
            Hud._settings_choice_bounds(1280, 720, 2)[0]
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

        self.assertFalse(self.app.block_lighting_enabled)
        self.assertEqual(self.app.shadow_quality, 1)
        self.app.renderer.set_block_lighting_enabled.assert_called_once_with(False)
        self.app.renderer.set_shadow_quality.assert_not_called()

    def test_block_lighting_settings_row_maps_on_and_off(self) -> None:
        choices = Hud._settings_choice_bounds(1280, 720, 2)
        self.assertEqual(len(choices), 2)
        for option, bounds in enumerate(choices):
            self.assertEqual(
                Hud.settings_option_at(
                    1280,
                    720,
                    bounds[0] + bounds[2] / 2,
                    bounds[1] + bounds[3] / 2,
                ),
                (2, option),
            )

    def test_settings_page_draws_block_lighting_toggle(self) -> None:
        with (
            patch.object(Hud, "_rectangle"),
            patch.object(Hud, "_draw_menu_text") as draw_menu_text,
        ):
            Hud._draw_settings(1280, 720, 1, 1, True, 0, True)

        self.assertIn(
            "БЛОЧНЫЙ СВЕТ",
            tuple(call.args[0] for call in draw_menu_text.call_args_list),
        )
        with patch.object(Hud, "_rectangle"):
            Hud._draw_menu_text("БЛОЧНЫЙ СВЕТ", 640, 100, 1, (1.0, 1.0, 1.0))

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
