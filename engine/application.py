
import glfw
import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    glClear,
    glClearColor,
    glViewport,
)

from engine.graphics.hud import Hud
from engine.graphics.renderer import Renderer
from engine.game.player import Player
from engine.world.voxel_world import HOTBAR_BLOCKS, VoxelWorld


class Application:
    def __init__(self) -> None:
        self.window: glfw._GLFWwindow | None = None
        self._glfw_initialized = False
        self.renderer: Renderer | None = None
        self.world: VoxelWorld | None = None
        self.player: Player | None = None
        self._last_mouse_position: tuple[float, float] | None = None
        self._mouse_buttons_down: set[int] = set()
        self.selected_hotbar_slot = 0
        self.menu_open = False
        self.settings_open = False
        self.texture_quality = 1
        self.shadow_quality = 1

    def initialize(self) -> None:
        if not glfw.init():
            raise RuntimeError(
                "Не удалось инициализировать GLFW. "
                "Проверьте драйвер видеокарты и доступность графического окружения."
            )
        self._glfw_initialized = True

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(
            1280, 720, "BlockWorld Engine", None, None
        )

        if self.window is None:
            raise RuntimeError(
                "Не удалось создать окно OpenGL 3.3 Core Profile. "
                "Убедитесь, что видеодрайвер поддерживает OpenGL 3.3."
            )

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        glfw.set_framebuffer_size_callback(self.window, self._on_framebuffer_resize)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        width, height = glfw.get_framebuffer_size(self.window)
        self._on_framebuffer_resize(self.window, width, height)
        glClearColor(0.04, 0.07, 0.14, 1.0)
        self.world = VoxelWorld()
        self.player = Player(
            self.world.size / 2,
            2.0,
            self.world.size / 2,
        )
        self.renderer = Renderer(self.world.build_mesh())

    @staticmethod
    def _on_framebuffer_resize(
        _window: glfw._GLFWwindow, width: int, height: int
    ) -> None:
        del _window
        glViewport(0, 0, width, height)

    def run(self) -> None:
        try:
            self.initialize()
            window = self.window
            if window is None:
                raise RuntimeError("Окно приложения не было создано.")
            renderer = self.renderer
            if renderer is None:
                raise RuntimeError("Рендерер не был инициализирован.")
            world = self.world
            if world is None:
                raise RuntimeError("Мир не был инициализирован.")
            player = self.player
            if player is None:
                raise RuntimeError("Игрок не был инициализирован.")

            previous_time = glfw.get_time()
            while not glfw.window_should_close(window):
                glfw.poll_events()

                current_time = glfw.get_time()
                if not self.menu_open:
                    player.update(window, world, current_time - previous_time)
                previous_time = current_time
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                width, height = glfw.get_framebuffer_size(window)
                renderer.render(
                    player.view_matrix(),
                    self._projection_matrix(width, height),
                    width,
                    height,
                )
                if self.menu_open:
                    hovered_button = (
                        None
                        if self.settings_open
                        else self._menu_hovered_button(window, width, height)
                    )
                    renderer.draw_menu(
                        width,
                        height,
                        self.settings_open,
                        hovered_button,
                        self.texture_quality,
                        self.shadow_quality,
                    )
                else:
                    renderer.draw_hud(width, height, self.selected_hotbar_slot)
                glfw.swap_buffers(window)

        finally:
            self.close()

    def close(self) -> None:
        window, self.window = self.window, None
        try:
            self.player = None
            self.world = None
            self._mouse_buttons_down.clear()
            self._last_mouse_position = None
            renderer, self.renderer = self.renderer, None
            if renderer is not None:
                renderer.close()
            if window is not None:
                glfw.destroy_window(window)
        finally:
            if self._glfw_initialized:
                glfw.terminate()
                self._glfw_initialized = False

    def _on_mouse_move(
        self, _window: glfw._GLFWwindow, x_position: float, y_position: float
    ) -> None:
        del _window
        if self._last_mouse_position is None:
            self._last_mouse_position = (x_position, y_position)
            return

        previous_x, previous_y = self._last_mouse_position
        self._last_mouse_position = (x_position, y_position)
        if not self.menu_open and self.player is not None:
            self.player.look(x_position - previous_x, y_position - previous_y)

    def _on_mouse_button(
        self,
        window: glfw._GLFWwindow,
        button: int,
        action: int,
        _modifiers: int,
    ) -> None:
        del _modifiers
        if action == glfw.RELEASE:
            self._mouse_buttons_down.discard(button)
            return
        if action != glfw.PRESS or button in self._mouse_buttons_down:
            return

        self._mouse_buttons_down.add(button)
        if self.menu_open:
            if button == glfw.MOUSE_BUTTON_LEFT:
                width, height = glfw.get_framebuffer_size(window)
                window_width, window_height = glfw.get_window_size(window)
                if window_width <= 0 or window_height <= 0:
                    return
                cursor_x, cursor_y = glfw.get_cursor_pos(window)
                menu_button = Hud.menu_button_at(
                    width,
                    height,
                    cursor_x * width / window_width,
                    height - cursor_y * height / window_height,
                )
                if self.settings_open:
                    setting = Hud.settings_option_at(
                        width,
                        height,
                        cursor_x * width / window_width,
                        height - cursor_y * height / window_height,
                    )
                    if setting == (2, 0):
                        self.settings_open = False
                    elif setting is not None:
                        row, quality = setting
                        if row == 0:
                            self.texture_quality = quality
                            if self.renderer is not None:
                                self.renderer.set_texture_quality(quality)
                        elif row == 1:
                            self.shadow_quality = quality
                            if self.renderer is not None:
                                self.renderer.set_shadow_quality(quality)
                elif menu_button == 0:
                    self._close_menu()
                elif menu_button == 1:
                    self.settings_open = True
                elif menu_button == 2:
                    glfw.set_window_should_close(window, True)
            return

        if button not in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            return

        world = self.world
        player = self.player
        renderer = self.renderer
        if world is None or player is None or renderer is None:
            return

        hit = world.raycast(player.eye_position(), player.forward_direction(), 6.0)
        if hit is None:
            return

        hit_block, adjacent_block = hit
        if button == glfw.MOUSE_BUTTON_LEFT:
            changed = world.set_block(hit_block, 0)
        elif adjacent_block is not None and not player.intersects_block(adjacent_block):
            block_type = HOTBAR_BLOCKS[self.selected_hotbar_slot]
            changed = (
                world.set_block(adjacent_block, block_type)
                if block_type is not None
                else False
            )
        else:
            changed = False

        if changed:
            renderer.update_mesh(world.build_mesh())

    def _on_key(
        self,
        _window: glfw._GLFWwindow,
        key: int,
        _scancode: int,
        action: int,
        _modifiers: int,
    ) -> None:
        del _window, _scancode, _modifiers
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            self._handle_escape()
            return
        if self.menu_open:
            return

        if glfw.KEY_1 <= key <= glfw.KEY_9:
            self.selected_hotbar_slot = key - glfw.KEY_1

    def _on_scroll(
        self, _window: glfw._GLFWwindow, _x_offset: float, y_offset: float
    ) -> None:
        del _window, _x_offset
        if y_offset != 0 and not self.menu_open:
            self.selected_hotbar_slot = (
                self.selected_hotbar_slot - int(np.sign(y_offset))
            ) % len(HOTBAR_BLOCKS)

    def _handle_escape(self) -> None:
        if self.settings_open:
            self.settings_open = False
            return

        if self.menu_open:
            self._close_menu()
            return

        self.menu_open = True
        self._last_mouse_position = None
        self._mouse_buttons_down.clear()
        if self.window is not None:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
            self._position_cursor_on_continue(self.window)

    def _close_menu(self) -> None:
        self.settings_open = False
        self.menu_open = False
        self._last_mouse_position = None
        self._mouse_buttons_down.clear()
        if self.window is not None:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    @staticmethod
    def _menu_hovered_button(
        window: glfw._GLFWwindow, framebuffer_width: int, framebuffer_height: int
    ) -> int | None:
        window_width, window_height = glfw.get_window_size(window)
        if window_width <= 0 or window_height <= 0:
            return None
        cursor_x, cursor_y = glfw.get_cursor_pos(window)
        return Hud.menu_button_at(
            framebuffer_width,
            framebuffer_height,
            cursor_x * framebuffer_width / window_width,
            framebuffer_height - cursor_y * framebuffer_height / window_height,
        )

    @staticmethod
    def _position_cursor_on_continue(window: glfw._GLFWwindow) -> None:
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)
        window_width, window_height = glfw.get_window_size(window)
        if (
            framebuffer_width <= 0
            or framebuffer_height <= 0
            or window_width <= 0
            or window_height <= 0
        ):
            return

        button_x, button_y, button_width, button_height = Hud._menu_button_bounds(
            framebuffer_width, framebuffer_height
        )[0]
        cursor_x = (button_x + button_width / 2) * window_width / framebuffer_width
        cursor_y = (
            framebuffer_height - button_y - button_height / 2
        ) * window_height / framebuffer_height
        glfw.set_cursor_pos(window, cursor_x, cursor_y)

    @staticmethod
    def _projection_matrix(width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0:
            return np.eye(4, dtype=np.float32)

        field_of_view = np.deg2rad(70.0)
        focal_length = 1.0 / np.tan(field_of_view / 2.0)
        near_plane = 0.1
        far_plane = 100.0
        projection = np.zeros((4, 4), dtype=np.float32)
        projection[0, 0] = focal_length / (width / height)
        projection[1, 1] = focal_length
        projection[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
        projection[2, 3] = (2.0 * far_plane * near_plane) / (
            near_plane - far_plane
        )
        projection[3, 2] = -1.0
        return projection
