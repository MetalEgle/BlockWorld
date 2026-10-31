from collections import deque

import numpy as np

BlockPosition = tuple[int, int, int]
BlockType = int
ChunkPosition = tuple[int, int, int]
CHUNK_SIZE = 8
MESH_RENDER_DISTANCE = 80
TEXTURE_GRASS_TOP = 0
TEXTURE_GRASS_SIDE = 1
TEXTURE_DIRT = 2
TEXTURE_STONE = 3
TEXTURE_SAND = 4
TEXTURE_LEAF = 5
TEXTURE_STONE_BRICKS = 6
TEXTURE_STONE_COBBLE = 7
TEXTURE_WOOD_SIDE = 8
TEXTURE_WOOD_TOP = 9
TEXTURE_WOOD_BOARD = 10
TEXTURE_LANTERN = 11

DIRT_LAYER_DEPTH = 3
MAX_LIGHT_LEVEL = 15
GLOWSTONE_LIGHT_LEVEL = 12

BLOCK_DIRT = 1
BLOCK_GRASS = 2
BLOCK_STONE = 3
BLOCK_SAND = 4
BLOCK_LEAF = 5
BLOCK_STONE_BRICKS = 6
BLOCK_STONE_COBBLE = 7
BLOCK_WOOD = 8
BLOCK_WOOD_BOARD = 9
BLOCK_GLOWSTONE = 10
BLOCK_LANTERN = 11
BLOCK_COLORS: dict[BlockType, tuple[float, float, float]] = {
    BLOCK_DIRT: (0.50, 0.32, 0.18),
    BLOCK_GRASS: (0.35, 0.72, 0.29),
    BLOCK_STONE: (0.52, 0.55, 0.57),
    BLOCK_SAND: (0.78, 0.68, 0.39),
    BLOCK_LEAF: (0.25, 0.58, 0.22),
    BLOCK_STONE_BRICKS: (0.54, 0.56, 0.58),
    BLOCK_STONE_COBBLE: (0.48, 0.50, 0.52),
    BLOCK_WOOD: (0.58, 0.39, 0.20),
    BLOCK_WOOD_BOARD: (0.68, 0.48, 0.27),
    BLOCK_GLOWSTONE: (0.96, 0.72, 0.30),
    BLOCK_LANTERN: (1.0, 0.72, 0.28),
}
BLOCK_TYPES: tuple[BlockType, ...] = (
    BLOCK_GRASS,
    BLOCK_DIRT,
    BLOCK_STONE,
    BLOCK_SAND,
    BLOCK_LEAF,
    BLOCK_STONE_BRICKS,
    BLOCK_STONE_COBBLE,
    BLOCK_WOOD,
    BLOCK_WOOD_BOARD,
    BLOCK_GLOWSTONE,
    BLOCK_LANTERN,
)
HOTBAR_BLOCKS: tuple[BlockType | None, ...] = BLOCK_TYPES[:-1]

_FACE_DATA = (
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)), 1.0),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), 0.58),
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)), 0.76),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), 0.68),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), 0.72),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), 0.64),
)
_LIGHT_NEIGHBORS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


class VoxelWorld:
    """A terrain-filled voxel world stored as a NumPy block array."""

    def __init__(self, size: int = 1000, height: int | None = None) -> None:
        if size < 4:
            raise ValueError("Размер мира должен быть не меньше четырёх блоков.")
        if height is None:
            height = min(size, 100)
        if height < 4:
            raise ValueError("Высота мира должна быть не меньше четырёх блоков.")

        self.size = size
        self.height = height
        self.surface_heights = self._generate_surface_heights(size, height)
        self.blocks = np.zeros((size, height, size), dtype=np.uint8)
        for y in range(int(self.surface_heights.max()) + 1):
            depth = self.surface_heights - y
            layer = np.where(
                depth == 0,
                BLOCK_GRASS,
                np.where(depth <= DIRT_LAYER_DEPTH, BLOCK_DIRT, BLOCK_STONE),
            )
            self.blocks[:, y, :] = np.where(depth >= 0, layer, 0)

        self._block_count = int(
            np.sum(self.surface_heights, dtype=np.int64) + size * size
        )
        self.chunk_count = (size + CHUNK_SIZE - 1) // CHUNK_SIZE
        self.vertical_chunk_count = (height + CHUNK_SIZE - 1) // CHUNK_SIZE
        chunk_starts = np.arange(0, size, CHUNK_SIZE)
        chunk_surface_heights = np.maximum.reduceat(
            np.maximum.reduceat(self.surface_heights, chunk_starts, axis=0),
            chunk_starts,
            axis=1,
        )
        self._non_empty_chunks = {
            (int(chunk_x), chunk_y, int(chunk_z))
            for chunk_y in range(self.vertical_chunk_count)
            for chunk_x, chunk_z in np.argwhere(
                chunk_surface_heights >= chunk_y * CHUNK_SIZE
            )
        }
        self._mesh_cache: dict[ChunkPosition, np.ndarray] = {}
        self._dirty_mesh_chunks: set[ChunkPosition] = set()
        self._active_mesh_chunks: set[ChunkPosition] | None = None
        self._lighting_bounds: tuple[int, int, int, int] | None = None
        self._block_light_levels = np.zeros((0, height, 0), dtype=np.uint8)
        self._sky_light_levels = np.zeros((0, height, 0), dtype=np.uint8)
        self._block_lighting_dirty = True

    @staticmethod
    def _generate_surface_heights(size: int, height: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        continental = VoxelWorld._smooth_value_noise(
            size, max(10, size // 4), rng
        )
        rolling_hills = (
            0.55 * VoxelWorld._smooth_value_noise(size, max(8, size // 7), rng)
            + 0.30 * VoxelWorld._smooth_value_noise(size, max(4, size // 22), rng)
            + 0.15 * VoxelWorld._smooth_value_noise(size, max(2, size // 55), rng)
        )
        mountain_regions = VoxelWorld._smooth_value_noise(
            size, max(12, size // 3), rng
        )
        mountain_mask = np.clip((mountain_regions + 0.05) / 0.75, 0.0, 1.0)
        mountain_mask = mountain_mask * mountain_mask * (3.0 - 2.0 * mountain_mask)
        ridge_noise = VoxelWorld._smooth_value_noise(
            size, max(3, size // 20), rng
        )
        ridges = 1.0 - np.abs(ridge_noise)
        fine_detail = VoxelWorld._smooth_value_noise(
            size, max(2, size // 50), rng
        )

        base_height = max(2, height // 4)
        terrain_scale = min(1.0, size / 128.0)
        heights = (
            base_height
            + terrain_scale
            * (
                continental * max(2, height * 0.09)
                + rolling_hills * max(2, height * 0.12)
                + mountain_mask
                * (0.25 + 0.75 * ridges**2)
                * max(2, height * 0.30)
                + fine_detail * max(1, height * 0.025)
            )
        )
        heights = np.rint(heights).astype(np.int16)
        return np.clip(heights, 1, height - 2)

    @staticmethod
    def _smooth_value_noise(
        size: int, spacing: int, rng: np.random.Generator
    ) -> np.ndarray:
        lattice_size = (size - 1 + spacing - 1) // spacing + 1
        lattice = rng.uniform(-1.0, 1.0, (lattice_size, lattice_size))
        coordinates = np.arange(size, dtype=np.float64) / spacing
        lower = coordinates.astype(np.intp)
        upper = np.minimum(lower + 1, lattice_size - 1)
        blend = coordinates - lower
        blend = blend * blend * (3.0 - 2.0 * blend)

        lower_x = lattice[lower[:, None], lower[None, :]]
        upper_x = lattice[upper[:, None], lower[None, :]]
        lower_z = lattice[lower[:, None], upper[None, :]]
        upper_z = lattice[upper[:, None], upper[None, :]]
        blend_x = blend[:, None]
        blend_z = blend[None, :]
        near_z = lower_x + (upper_x - lower_x) * blend_x
        far_z = lower_z + (upper_z - lower_z) * blend_x
        return near_z + (far_z - near_z) * blend_z

    def surface_height_at(self, x: int, z: int) -> int:
        if not (0 <= x < self.size and 0 <= z < self.size):
            raise ValueError(f"Координаты поверхности вне мира: {(x, z)}.")
        return int(self.surface_heights[x, z])

    def is_solid(self, x: int, y: int, z: int) -> bool:
        if not (
            0 <= x < self.size
            and 0 <= y < self.height
            and 0 <= z < self.size
        ):
            return False
        return bool(self.blocks[x, y, z])

    @property
    def block_count(self) -> int:
        return self._block_count

    def set_block(self, position: BlockPosition, block_type: int) -> bool:
        x, y, z = position
        if not (
            0 <= x < self.size
            and 0 <= y < self.height
            and 0 <= z < self.size
        ):
            return False
        if block_type != 0 and block_type not in BLOCK_COLORS:
            raise ValueError(f"Неизвестный тип блока: {block_type}.")
        if self.blocks[x, y, z] == block_type:
            return False

        if self.blocks[x, y, z] == 0:
            self._block_count += 1
        elif block_type == 0:
            self._block_count -= 1
        self.blocks[x, y, z] = block_type
        chunk = tuple(coordinate // CHUNK_SIZE for coordinate in position)
        self._refresh_chunk_occupancy(chunk)
        self._invalidate_mesh_chunks(position)
        self._block_lighting_dirty = True
        return True

    def _refresh_chunk_occupancy(self, chunk: ChunkPosition) -> None:
        chunk_start = tuple(coordinate * CHUNK_SIZE for coordinate in chunk)
        chunk_end = tuple(
            min(start + CHUNK_SIZE, limit)
            for start, limit in zip(chunk_start, (self.size, self.height, self.size))
        )
        chunk_blocks = self.blocks[
            chunk_start[0]:chunk_end[0],
            chunk_start[1]:chunk_end[1],
            chunk_start[2]:chunk_end[2],
        ]
        if np.any(chunk_blocks):
            self._non_empty_chunks.add(chunk)
        else:
            self._non_empty_chunks.discard(chunk)

    def _invalidate_mesh_chunks(self, position: BlockPosition) -> None:
        chunk = tuple(coordinate // CHUNK_SIZE for coordinate in position)
        world_limits = (self.size, self.height, self.size)
        affected_chunks = {chunk}
        for axis in range(3):
            for direction in (-1, 1):
                neighbor = list(position)
                neighbor[axis] += direction
                if all(
                    0 <= coordinate < world_limits[index]
                    for index, coordinate in enumerate(neighbor)
                ):
                    neighbor_chunk = tuple(
                        coordinate // CHUNK_SIZE for coordinate in neighbor
                    )
                    affected_chunks.add(neighbor_chunk)

        if self._active_mesh_chunks is None:
            self._dirty_mesh_chunks.update(affected_chunks)
        else:
            self._dirty_mesh_chunks.update(
                affected_chunks & self._active_mesh_chunks
            )

    def raycast(
        self, origin: np.ndarray, direction: np.ndarray, max_distance: float
    ) -> tuple[BlockPosition, BlockPosition] | None:
        if max_distance <= 0.0:
            return None

        ray = np.array(direction, dtype=np.float64, copy=True)
        ray_length = np.linalg.norm(ray)
        if ray_length == 0.0:
            raise ValueError("Направление луча не может быть нулевым.")
        ray /= ray_length

        cell = np.floor(origin).astype(np.int64)
        current = (int(cell[0]), int(cell[1]), int(cell[2]))
        if self.is_solid(*current):
            return current, current

        steps = np.sign(ray).astype(np.int64)
        t_delta = np.full(3, np.inf, dtype=np.float64)
        nonzero_axes = ray != 0.0
        t_delta[nonzero_axes] = 1.0 / np.abs(ray[nonzero_axes])
        next_boundary = cell + (steps > 0)
        t_max = np.full(3, np.inf, dtype=np.float64)
        t_max[nonzero_axes] = (
            next_boundary[nonzero_axes] - origin[nonzero_axes]
        ) / ray[nonzero_axes]

        distance = 0.0
        while distance <= max_distance:
            axis = int(np.argmin(t_max))
            distance = float(t_max[axis])
            if distance > max_distance:
                break

            previous = current
            cell[axis] += steps[axis]
            t_max[axis] += t_delta[axis]
            current = (int(cell[0]), int(cell[1]), int(cell[2]))
            if self.is_solid(*current):
                return current, previous

        return None

    def build_mesh(
        self,
        center_position: tuple[float, float, float] | None = None,
        render_distance: int = MESH_RENDER_DISTANCE,
    ) -> np.ndarray:
        """Return visible faces as position, UV, texture layer, and shading data."""
        if render_distance < 0:
            raise ValueError("Дальность отрисовки не может быть отрицательной.")

        if center_position is None:
            visible_chunks = set(self._non_empty_chunks)
            lighting_bounds = (0, self.size, 0, self.size)
        else:
            center_x = int(center_position[0]) // CHUNK_SIZE
            center_z = int(center_position[2]) // CHUNK_SIZE
            chunk_radius = render_distance // CHUNK_SIZE + 1
            min_x = max(0, center_x - chunk_radius)
            max_x = min(self.chunk_count, center_x + chunk_radius + 1)
            min_z = max(0, center_z - chunk_radius)
            max_z = min(self.chunk_count, center_z + chunk_radius + 1)
            visible_chunks = {
                chunk
                for chunk in self._non_empty_chunks
                if min_x <= chunk[0] < max_x and min_z <= chunk[2] < max_z
            }
            lighting_padding = GLOWSTONE_LIGHT_LEVEL
            lighting_bounds = (
                max(0, min_x * CHUNK_SIZE - lighting_padding),
                min(self.size, max_x * CHUNK_SIZE + lighting_padding),
                max(0, min_z * CHUNK_SIZE - lighting_padding),
                min(self.size, max_z * CHUNK_SIZE + lighting_padding),
            )

        previous_chunks = self._active_mesh_chunks
        if previous_chunks is not None:
            self._dirty_mesh_chunks.update(visible_chunks - previous_chunks)
            for chunk in previous_chunks - visible_chunks:
                self._mesh_cache.pop(chunk, None)
            self._dirty_mesh_chunks.intersection_update(visible_chunks)
        else:
            self._dirty_mesh_chunks.update(visible_chunks)
        self._active_mesh_chunks = visible_chunks

        self._update_block_lighting(lighting_bounds)
        for chunk in sorted(self._dirty_mesh_chunks):
            chunk_mesh = self._build_chunk_mesh(chunk)
            if len(chunk_mesh):
                self._mesh_cache[chunk] = chunk_mesh
            else:
                self._mesh_cache.pop(chunk, None)
        self._dirty_mesh_chunks.clear()

        meshes = [
            self._mesh_cache[chunk]
            for chunk in sorted(visible_chunks)
            if chunk in self._mesh_cache and len(self._mesh_cache[chunk])
        ]
        if not meshes:
            return np.empty((0, 12), dtype=np.float32)
        return np.concatenate(meshes)

    def _update_block_lighting(
        self, bounds: tuple[int, int, int, int]
    ) -> None:
        if not self._block_lighting_dirty and bounds == self._lighting_bounds:
            return

        x_start, x_end, z_start, z_end = bounds
        region_blocks = self.blocks[x_start:x_end, :, z_start:z_end]
        light_levels = np.zeros_like(region_blocks)
        pending: deque[BlockPosition] = deque()
        light_sources = (region_blocks == BLOCK_GLOWSTONE) | (
            region_blocks == BLOCK_LANTERN
        )
        for source in np.argwhere(light_sources):
            position = (int(source[0]), int(source[1]), int(source[2]))
            light_levels[position] = GLOWSTONE_LIGHT_LEVEL
            pending.append(position)

        self._spread_light(light_levels, pending, region_blocks)

        y_levels = np.arange(self.height, dtype=np.int16)[None, :, None]
        highest_solid = np.max(
            np.where(region_blocks != 0, y_levels, -1), axis=1
        )
        direct_sunlight = (
            (y_levels > highest_solid[:, None, :]) & (region_blocks == 0)
        )
        sky_light_levels = np.zeros_like(region_blocks)
        sky_light_levels[direct_sunlight] = MAX_LIGHT_LEVEL

        adjacent_to_sunlight = np.zeros_like(direct_sunlight)
        adjacent_to_sunlight[1:, :, :] |= direct_sunlight[:-1, :, :]
        adjacent_to_sunlight[:-1, :, :] |= direct_sunlight[1:, :, :]
        adjacent_to_sunlight[:, 1:, :] |= direct_sunlight[:, :-1, :]
        adjacent_to_sunlight[:, :-1, :] |= direct_sunlight[:, 1:, :]
        adjacent_to_sunlight[:, :, 1:] |= direct_sunlight[:, :, :-1]
        adjacent_to_sunlight[:, :, :-1] |= direct_sunlight[:, :, 1:]
        sky_frontier = (
            adjacent_to_sunlight & (region_blocks == 0) & ~direct_sunlight
        )
        sky_pending: deque[BlockPosition] = deque()
        for position_array in np.argwhere(sky_frontier):
            position = (
                int(position_array[0]),
                int(position_array[1]),
                int(position_array[2]),
            )
            sky_light_levels[position] = MAX_LIGHT_LEVEL - 1
            sky_pending.append(position)
        self._spread_light(sky_light_levels, sky_pending, region_blocks)

        changed_lighting = np.zeros_like(region_blocks, dtype=bool)
        if self._lighting_bounds is not None:
            old_x_start, old_x_end, old_z_start, old_z_end = self._lighting_bounds
            overlap_x_start = max(x_start, old_x_start)
            overlap_x_end = min(x_end, old_x_end)
            overlap_z_start = max(z_start, old_z_start)
            overlap_z_end = min(z_end, old_z_end)
            if overlap_x_start < overlap_x_end and overlap_z_start < overlap_z_end:
                new_overlap = (
                    slice(overlap_x_start - x_start, overlap_x_end - x_start),
                    slice(None),
                    slice(overlap_z_start - z_start, overlap_z_end - z_start),
                )
                old_overlap = (
                    slice(overlap_x_start - old_x_start, overlap_x_end - old_x_start),
                    slice(None),
                    slice(overlap_z_start - old_z_start, overlap_z_end - old_z_start),
                )
                changed_lighting[new_overlap] = (
                    (light_levels[new_overlap] != self._block_light_levels[old_overlap])
                    | (
                        sky_light_levels[new_overlap]
                        != self._sky_light_levels[old_overlap]
                    )
                )

        adjacent_to_block = region_blocks != 0
        adjacent_to_block[1:, :, :] |= region_blocks[:-1, :, :] != 0
        adjacent_to_block[:-1, :, :] |= region_blocks[1:, :, :] != 0
        adjacent_to_block[:, 1:, :] |= region_blocks[:, :-1, :] != 0
        adjacent_to_block[:, :-1, :] |= region_blocks[:, 1:, :] != 0
        adjacent_to_block[:, :, 1:] |= region_blocks[:, :, :-1] != 0
        adjacent_to_block[:, :, :-1] |= region_blocks[:, :, 1:] != 0
        changed_positions = np.argwhere(changed_lighting & adjacent_to_block)
        self._block_light_levels = light_levels
        self._sky_light_levels = sky_light_levels
        self._lighting_bounds = bounds
        self._block_lighting_dirty = False
        for position in changed_positions:
            self._invalidate_mesh_chunks(
                (
                    int(position[0]) + x_start,
                    int(position[1]),
                    int(position[2]) + z_start,
                )
            )

    def _spread_light(
        self,
        light_levels: np.ndarray,
        pending: deque[BlockPosition],
        region_blocks: np.ndarray,
    ) -> None:
        while pending:
            x, y, z = pending.popleft()
            next_level = int(light_levels[x, y, z]) - 1
            if next_level <= 0:
                continue
            for dx, dy, dz in _LIGHT_NEIGHBORS:
                neighbor = (x + dx, y + dy, z + dz)
                nx, ny, nz = neighbor
                if (
                    0 <= nx < light_levels.shape[0]
                    and 0 <= ny < light_levels.shape[1]
                    and 0 <= nz < light_levels.shape[2]
                    and region_blocks[neighbor] == 0
                    and light_levels[neighbor] < next_level
                ):
                    light_levels[neighbor] = next_level
                    pending.append(neighbor)

    def _build_chunk_mesh(self, chunk: ChunkPosition) -> np.ndarray:
        chunk_start = tuple(coordinate * CHUNK_SIZE for coordinate in chunk)
        chunk_end = tuple(
            min(start + CHUNK_SIZE, limit)
            for start, limit in zip(chunk_start, (self.size, self.height, self.size))
        )
        chunk_blocks = self.blocks[
            chunk_start[0]:chunk_end[0],
            chunk_start[1]:chunk_end[1],
            chunk_start[2]:chunk_end[2],
        ]
        if not np.any(chunk_blocks):
            return np.empty((0, 12), dtype=np.float32)
        chunk_is_enclosed = bool(np.all(chunk_blocks))
        world_limits = (self.size, self.height, self.size)
        if chunk_is_enclosed:
            for axis in range(3):
                if (
                    chunk_start[axis] == 0
                    or chunk_end[axis] == world_limits[axis]
                ):
                    chunk_is_enclosed = False
                    break
                for plane in (chunk_start[axis] - 1, chunk_end[axis]):
                    neighbor_slice: list[slice | int] = [
                        slice(chunk_start[index], chunk_end[index])
                        for index in range(3)
                    ]
                    neighbor_slice[axis] = plane
                    if not np.all(self.blocks[tuple(neighbor_slice)]):
                        chunk_is_enclosed = False
                        break
                if not chunk_is_enclosed:
                    break
        if chunk_is_enclosed:
            return np.empty((0, 12), dtype=np.float32)
        vertices: list[
            tuple[
                float, float, float, float, float, float, float, float, float, float,
                float, float,
            ]
        ] = []
        corner_lighting_cache: dict[
            tuple[int, int, int], tuple[float, float]
        ] = {}

        for (dx, dy, dz), corners, brightness in _FACE_DATA:
            axis = next(index for index, direction in enumerate((dx, dy, dz)) if direction)
            neighbor_solid = np.zeros(chunk_blocks.shape, dtype=bool)
            current_slice: list[slice | int] = [
                slice(None),
                slice(None),
                slice(None),
            ]
            neighbor_slice = current_slice.copy()
            if (dx, dy, dz)[axis] > 0:
                current_slice[axis] = slice(None, -1)
                neighbor_slice[axis] = slice(1, None)
                neighbor_solid[tuple(current_slice)] = (
                    chunk_blocks[tuple(neighbor_slice)] != 0
                )
                if chunk_end[axis] < world_limits[axis]:
                    current_slice = [slice(None), slice(None), slice(None)]
                    neighbor_slice = [
                        slice(chunk_start[index], chunk_end[index])
                        for index in range(3)
                    ]
                    current_slice[axis] = -1
                    neighbor_slice[axis] = chunk_end[axis]
                    neighbor_solid[tuple(current_slice)] = (
                        self.blocks[tuple(neighbor_slice)] != 0
                    )
            else:
                current_slice[axis] = slice(1, None)
                neighbor_slice[axis] = slice(None, -1)
                neighbor_solid[tuple(current_slice)] = (
                    chunk_blocks[tuple(neighbor_slice)] != 0
                )
                if chunk_start[axis] > 0:
                    current_slice = [slice(None), slice(None), slice(None)]
                    neighbor_slice = [
                        slice(chunk_start[index], chunk_end[index])
                        for index in range(3)
                    ]
                    current_slice[axis] = 0
                    neighbor_slice[axis] = chunk_start[axis] - 1
                    neighbor_solid[tuple(current_slice)] = (
                        self.blocks[tuple(neighbor_slice)] != 0
                    )

            for local_x, local_y, local_z in np.argwhere(
                (chunk_blocks != 0) & ~neighbor_solid
            ):
                x = int(local_x) + chunk_start[0]
                y = int(local_y) + chunk_start[1]
                z = int(local_z) + chunk_start[2]
                block_type = int(chunk_blocks[local_x, local_y, local_z])
                texture_layer = self._texture_layer(block_type, (dx, dy, dz))
                points = [
                    (x + corner[0], y + corner[1], z + corner[2])
                    for corner in corners
                ]
                corner_lighting = []
                for point in points:
                    lighting = corner_lighting_cache.get(point)
                    if lighting is None:
                        lighting = self._sample_corner_lighting(point)
                        corner_lighting_cache[point] = lighting
                    corner_lighting.append(lighting)
                uvs = self._face_uvs((dx, dy, dz), corners)
                for index in (0, 1, 2, 0, 2, 3):
                    point = points[index]
                    block_light, sky_light = corner_lighting[index]
                    vertices.append(
                        (
                            *point,
                            *uvs[index],
                            float(texture_layer),
                            brightness,
                            block_light,
                            sky_light,
                            float(dx),
                            float(dy),
                            float(dz),
                        )
                    )

        return np.asarray(vertices, dtype=np.float32).reshape(-1, 12)

    def _sample_corner_lighting(
        self, position: tuple[int, int, int]
    ) -> tuple[float, float]:
        totals = [0, 0]
        sample_count = 0
        lighting_bounds = self._lighting_bounds
        if lighting_bounds is None:
            return 0.0, 0.0
        x_start, x_end, z_start, z_end = lighting_bounds
        for dx in (-1, 0):
            for dy in (-1, 0):
                for dz in (-1, 0):
                    world_cell = (
                        position[0] + dx,
                        position[1] + dy,
                        position[2] + dz,
                    )
                    if not (
                        x_start <= world_cell[0] < x_end
                        and 0 <= world_cell[1] < self.height
                        and z_start <= world_cell[2] < z_end
                    ):
                        continue
                    cell = (
                        world_cell[0] - x_start,
                        world_cell[1],
                        world_cell[2] - z_start,
                    )
                    if self.blocks[cell] not in (
                        0,
                        BLOCK_GLOWSTONE,
                        BLOCK_LANTERN,
                    ):
                        continue
                    totals[0] += int(self._block_light_levels[cell])
                    totals[1] += int(self._sky_light_levels[cell])
                    sample_count += 1

        if sample_count == 0:
            return 0.0, 0.0
        return (
            totals[0] / (sample_count * MAX_LIGHT_LEVEL),
            totals[1] / (sample_count * MAX_LIGHT_LEVEL),
        )

    @staticmethod
    def _face_uvs(
        normal: tuple[int, int, int],
        corners: tuple[tuple[int, int, int], ...],
    ) -> tuple[tuple[float, float], ...]:
        if normal[1] != 0:
            return tuple((float(x), float(z)) for x, _, z in corners)

        horizontal_axis = 2 if normal[0] != 0 else 0
        return tuple((float(corner[horizontal_axis]), float(corner[1])) for corner in corners)

    @staticmethod
    def _texture_layer(block_type: BlockType, normal: tuple[int, int, int]) -> int:
        if block_type == BLOCK_GRASS:
            if normal == (0, 1, 0):
                return TEXTURE_GRASS_TOP
            if normal == (0, -1, 0):
                return TEXTURE_DIRT
            return TEXTURE_GRASS_SIDE
        if block_type == BLOCK_DIRT:
            return TEXTURE_DIRT
        if block_type == BLOCK_STONE:
            return TEXTURE_STONE
        if block_type == BLOCK_SAND:
            return TEXTURE_SAND
        if block_type == BLOCK_LEAF:
            return TEXTURE_LEAF
        if block_type == BLOCK_STONE_BRICKS:
            return TEXTURE_STONE_BRICKS
        if block_type == BLOCK_STONE_COBBLE:
            return TEXTURE_STONE_COBBLE
        if block_type == BLOCK_WOOD:
            return TEXTURE_WOOD_TOP if normal[1] != 0 else TEXTURE_WOOD_SIDE
        if block_type == BLOCK_WOOD_BOARD:
            return TEXTURE_WOOD_BOARD
        if block_type == BLOCK_GLOWSTONE:
            return TEXTURE_SAND
        if block_type == BLOCK_LANTERN:
            return TEXTURE_LANTERN
        raise ValueError(f"Неизвестный тип блока: {block_type}.")
