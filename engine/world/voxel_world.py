import numpy as np

BlockPosition = tuple[int, int, int]
BlockType = int
ChunkPosition = tuple[int, int, int]
CHUNK_SIZE = 8
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

DIRT_LAYER_DEPTH = 3

BLOCK_DIRT = 1
BLOCK_GRASS = 2
BLOCK_STONE = 3
BLOCK_SAND = 4
BLOCK_LEAF = 5
BLOCK_STONE_BRICKS = 6
BLOCK_STONE_COBBLE = 7
BLOCK_WOOD = 8
BLOCK_WOOD_BOARD = 9
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
}
HOTBAR_BLOCKS: tuple[BlockType | None, ...] = (
    BLOCK_GRASS,
    BLOCK_DIRT,
    BLOCK_STONE,
    BLOCK_SAND,
    BLOCK_LEAF,
    BLOCK_STONE_BRICKS,
    BLOCK_STONE_COBBLE,
    BLOCK_WOOD,
    BLOCK_WOOD_BOARD,
)

_FACE_DATA = (
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)), 1.0),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), 0.58),
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)), 0.76),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), 0.68),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), 0.72),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), 0.64),
)

class VoxelWorld:
    """A terrain-filled voxel world stored as a NumPy block array."""

    def __init__(self, size: int = 100) -> None:
        if size < 4:
            raise ValueError("Размер мира должен быть не меньше четырёх блоков.")

        self.size = size
        self.surface_heights = self._generate_surface_heights(size)
        height_levels = np.arange(size, dtype=np.int16)[None, :, None]
        depth_below_surface = self.surface_heights[:, None, :] - height_levels
        block_layers = np.where(
            depth_below_surface == 0,
            BLOCK_GRASS,
            np.where(depth_below_surface <= DIRT_LAYER_DEPTH, BLOCK_DIRT, BLOCK_STONE),
        )
        self.blocks = np.where(
            depth_below_surface >= 0, block_layers, 0
        ).astype(np.uint8)

        chunk_count = (size + CHUNK_SIZE - 1) // CHUNK_SIZE
        vertical_chunk_count = (
            int(self.surface_heights.max()) // CHUNK_SIZE + 1
        )
        self._mesh_cache: dict[ChunkPosition, np.ndarray] = {}
        self._dirty_mesh_chunks = {
            (x, y, z)
            for x in range(chunk_count)
            for y in range(vertical_chunk_count)
            for z in range(chunk_count)
        }

    @staticmethod
    def _generate_surface_heights(size: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        noise = (
            0.65 * VoxelWorld._smooth_value_noise(size, max(8, size // 6), rng)
            + 0.25 * VoxelWorld._smooth_value_noise(size, max(4, size // 16), rng)
            + 0.10 * VoxelWorld._smooth_value_noise(size, max(2, size // 40), rng)
        )
        base_height = max(2, size // 4)
        hill_amplitude = max(1, size // 14)
        heights = np.rint(base_height + noise * hill_amplitude).astype(np.int16)
        return np.clip(heights, 1, size - 2)

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
            and 0 <= y < self.blocks.shape[1]
            and 0 <= z < self.size
        ):
            return False
        return bool(self.blocks[x, y, z])

    def set_block(self, position: BlockPosition, block_type: int) -> bool:
        x, y, z = position
        if not (
            0 <= x < self.size
            and 0 <= y < self.blocks.shape[1]
            and 0 <= z < self.size
        ):
            return False
        if block_type != 0 and block_type not in BLOCK_COLORS:
            raise ValueError(f"Неизвестный тип блока: {block_type}.")
        if self.blocks[x, y, z] == block_type:
            return False

        self.blocks[x, y, z] = block_type
        self._invalidate_mesh_chunks((x, y, z))
        return True

    def _invalidate_mesh_chunks(self, position: BlockPosition) -> None:
        chunk = tuple(coordinate // CHUNK_SIZE for coordinate in position)
        affected_chunks = {chunk}
        for axis in range(3):
            for direction in (-1, 1):
                neighbor = list(position)
                neighbor[axis] += direction
                if all(0 <= coordinate < self.size for coordinate in neighbor):
                    neighbor_chunk = tuple(
                        coordinate // CHUNK_SIZE for coordinate in neighbor
                    )
                    affected_chunks.add(neighbor_chunk)

        self._dirty_mesh_chunks.update(affected_chunks)

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

    def build_mesh(self) -> np.ndarray:
        """Return visible faces as position, UV, texture layer, and shading data."""
        for chunk in sorted(self._dirty_mesh_chunks):
            chunk_mesh = self._build_chunk_mesh(chunk)
            if len(chunk_mesh):
                self._mesh_cache[chunk] = chunk_mesh
            else:
                self._mesh_cache.pop(chunk, None)
        self._dirty_mesh_chunks.clear()

        meshes = [
            self._mesh_cache[chunk]
            for chunk in sorted(self._mesh_cache)
            if len(self._mesh_cache[chunk])
        ]
        if not meshes:
            return np.empty((0, 7), dtype=np.float32)
        return np.concatenate(meshes)

    def _build_chunk_mesh(self, chunk: ChunkPosition) -> np.ndarray:
        chunk_start = tuple(coordinate * CHUNK_SIZE for coordinate in chunk)
        chunk_end = tuple(
            min(start + CHUNK_SIZE, self.size) for start in chunk_start
        )
        chunk_blocks = self.blocks[
            chunk_start[0]:chunk_end[0],
            chunk_start[1]:chunk_end[1],
            chunk_start[2]:chunk_end[2],
        ]
        vertices: list[
            tuple[
                float, float, float, float, float, float, float, float, float, float
            ]
        ] = []

        for local_x, local_y, local_z in np.argwhere(chunk_blocks):
            x = int(local_x) + chunk_start[0]
            y = int(local_y) + chunk_start[1]
            z = int(local_z) + chunk_start[2]
            block_type = int(self.blocks[x, y, z])
            for (dx, dy, dz), corners, brightness in _FACE_DATA:
                if self.is_solid(int(x + dx), int(y + dy), int(z + dz)):
                    continue

                texture_layer = self._texture_layer(block_type, (dx, dy, dz))
                points = [
                    (x + corner[0], y + corner[1], z + corner[2])
                    for corner in corners
                ]
                uvs = self._face_uvs((dx, dy, dz), corners)
                for index in (0, 1, 2, 0, 2, 3):
                    point = points[index]
                    vertices.append(
                        (
                            *point,
                            *uvs[index],
                            float(texture_layer),
                            brightness,
                            float(dx),
                            float(dy),
                            float(dz),
                        )
                    )

        return np.asarray(vertices, dtype=np.float32).reshape(-1, 10)

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
        raise ValueError(f"Неизвестный тип блока: {block_type}.")
