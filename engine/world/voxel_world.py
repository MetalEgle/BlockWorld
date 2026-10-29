import numpy as np

BlockPosition = tuple[int, int, int]
BlockType = int

BLOCK_DIRT = 1
BLOCK_GRASS = 2
BLOCK_STONE = 3
BLOCK_SAND = 4
BLOCK_COLORS: dict[BlockType, tuple[float, float, float]] = {
    BLOCK_DIRT: (0.50, 0.32, 0.18),
    BLOCK_GRASS: (0.35, 0.72, 0.29),
    BLOCK_STONE: (0.52, 0.55, 0.57),
    BLOCK_SAND: (0.78, 0.68, 0.39),
}
HOTBAR_BLOCKS: tuple[BlockType | None, ...] = (
    BLOCK_GRASS,
    BLOCK_DIRT,
    BLOCK_STONE,
    BLOCK_SAND,
    None,
    None,
    None,
    None,
    None,
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
    """A small, flat voxel world stored as a NumPy block array."""

    def __init__(self, size: int = 24) -> None:
        if size < 4:
            raise ValueError("Размер мира должен быть не меньше четырёх блоков.")

        self.size = size
        self.blocks = np.zeros((size, size, size), dtype=np.uint8)
        self.blocks[:, 0, :] = 1
        self.blocks[:, 1, :] = 2

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
        return True

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
        """Return visible block faces as interleaved position and color data."""
        vertices: list[tuple[float, float, float, float, float, float]] = []

        for x, y, z in np.argwhere(self.blocks):
            block_type = int(self.blocks[x, y, z])
            base_color = BLOCK_COLORS[block_type]
            for (dx, dy, dz), corners, brightness in _FACE_DATA:
                if self.is_solid(int(x + dx), int(y + dy), int(z + dz)):
                    continue

                color = tuple(channel * brightness for channel in base_color)
                points = [
                    (x + corner[0], y + corner[1], z + corner[2])
                    for corner in corners
                ]
                for index in (0, 1, 2, 0, 2, 3):
                    point = points[index]
                    vertices.append((*point, *color))

        return np.asarray(vertices, dtype=np.float32).reshape(-1, 6)
