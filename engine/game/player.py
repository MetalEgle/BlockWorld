import glfw
import numpy as np

from engine.world.voxel_world import BlockPosition, VoxelWorld


class Player:
    """First-person player with basic movement, jumping, and voxel collisions."""

    HEIGHT = 1.8
    RADIUS = 0.3
    MOVE_SPEED = 5.0
    JUMP_SPEED = 7.0
    GRAVITY = 20.0
    MOUSE_SENSITIVITY = 0.1

    def __init__(self, x: float, y: float, z: float) -> None:
        self.position = np.array((x, y, z), dtype=np.float64)
        self.yaw = -90.0
        self.pitch = 0.0
        self._vertical_velocity = 0.0
        self._grounded = False

    def look(self, delta_x: float, delta_y: float) -> None:
        self.yaw += delta_x * self.MOUSE_SENSITIVITY
        self.pitch = float(
            np.clip(
                self.pitch - delta_y * self.MOUSE_SENSITIVITY,
                -89.0,
                89.0,
            )
        )

    def update(self, window: glfw._GLFWwindow, world: VoxelWorld, delta_time: float) -> None:
        delta_time = min(max(delta_time, 0.0), 0.1)
        yaw_radians = np.deg2rad(self.yaw)
        forward = np.array((np.cos(yaw_radians), 0.0, np.sin(yaw_radians)))
        right = np.array((-forward[2], 0.0, forward[0]))

        movement = np.zeros(3, dtype=np.float64)
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            movement += forward
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            movement -= forward
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            movement += right
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            movement -= right

        if np.any(movement):
            movement /= np.linalg.norm(movement)
            self._move_axis(world, 0, movement[0] * self.MOVE_SPEED * delta_time)
            self._move_axis(world, 2, movement[2] * self.MOVE_SPEED * delta_time)

        if self._grounded and glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            self._vertical_velocity = self.JUMP_SPEED
            self._grounded = False

        self._vertical_velocity -= self.GRAVITY * delta_time
        vertical_delta = self._vertical_velocity * delta_time
        previous_y = self.position[1]
        self.position[1] += vertical_delta
        if self._collides(world):
            self.position[1] = previous_y
            if vertical_delta < 0:
                self._grounded = True
            self._vertical_velocity = 0.0
        else:
            self._grounded = False

    def view_matrix(self) -> np.ndarray:
        forward = self.forward_direction()
        right = np.cross(forward, np.array((0.0, 1.0, 0.0)))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        eye = self.eye_position()

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, eye)
        view[1, 3] = -np.dot(up, eye)
        view[2, 3] = np.dot(forward, eye)
        return view

    def eye_position(self) -> np.ndarray:
        return self.position + np.array((0.0, self.HEIGHT * 0.9, 0.0))

    def forward_direction(self) -> np.ndarray:
        yaw = np.deg2rad(self.yaw)
        pitch = np.deg2rad(self.pitch)
        forward = np.array(
            (
                np.cos(pitch) * np.cos(yaw),
                np.sin(pitch),
                np.cos(pitch) * np.sin(yaw),
            ),
            dtype=np.float64,
        )
        return forward / np.linalg.norm(forward)

    def intersects_block(self, block: BlockPosition) -> bool:
        x, y, z = block
        return bool(
            self.position[0] + self.RADIUS > x
            and self.position[0] - self.RADIUS < x + 1
            and self.position[1] + self.HEIGHT > y
            and self.position[1] < y + 1
            and self.position[2] + self.RADIUS > z
            and self.position[2] - self.RADIUS < z + 1
        )

    def _move_axis(self, world: VoxelWorld, axis: int, distance: float) -> None:
        if distance == 0.0:
            return

        previous_position = self.position[axis]
        self.position[axis] += distance
        if self._collides(world):
            self.position[axis] = previous_position

    def _collides(self, world: VoxelWorld) -> bool:
        epsilon = 1e-6
        if (
            self.position[0] - self.RADIUS < 0.0
            or self.position[0] + self.RADIUS > world.size
            or self.position[2] - self.RADIUS < 0.0
            or self.position[2] + self.RADIUS > world.size
        ):
            return True

        minimum = (
            int(np.floor(self.position[0] - self.RADIUS + epsilon)),
            int(np.floor(self.position[1] + epsilon)),
            int(np.floor(self.position[2] - self.RADIUS + epsilon)),
        )
        maximum = (
            int(np.floor(self.position[0] + self.RADIUS - epsilon)),
            int(np.floor(self.position[1] + self.HEIGHT - epsilon)),
            int(np.floor(self.position[2] + self.RADIUS - epsilon)),
        )
        for x in range(minimum[0], maximum[0] + 1):
            for y in range(minimum[1], maximum[1] + 1):
                for z in range(minimum[2], maximum[2] + 1):
                    if world.is_solid(x, y, z):
                        return True
        return False
