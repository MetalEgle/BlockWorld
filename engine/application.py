
import glfw
import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    glClear,
    glClearColor,
    glViewport,
)

from engine.diagnostics import PerformanceMonitor
from engine.game.player import Player
from engine.graphics.hud import Hud
from engine.graphics.renderer import Renderer
from engine.world.voxel_world import (
    BLOCK_DIRT,
    BLOCK_GRASS,
    BLOCK_TYPES,
    CHUNK_SIZE,
    HOTBAR_BLOCKS,
    VoxelWorld,
)


class Application:
    def __init__(self) -> None:
        self.window: glfw._GLFWwindow | None = None
        self._glfw_initialized = False
        self.renderer: Renderer | None = None
        self.world: VoxelWorld | None = None
        self.player: Player | None = None
        self._rendered_mesh_chunk: tuple[int, int] | None = None
        self._last_mouse_position: tuple[float, float] | None = None
        self._mouse_buttons_down: set[int] = set()
        self.selected_hotbar_slot = 0
        self.hotbar_blocks = list(HOTBAR_BLOCKS)
        self.menu_open = False
        self.inventory_open = False
        self.inventory_tab = 0
        self.inventory_slots: list[int | None] = [None] * 27
        self._inventory_drag_source: tuple[str, int] | None = None
        self._inventory_drag_block: int | None = None
        self._inventory_carried_block: int | None = None
        self.settings_open = False
        self.texture_quality = 1
        self.shadow_quality = 1
        self.block_lighting_enabled = True
        self.color_grade = 0
        self.effects_enabled = True
        self.performance_monitor = PerformanceMonitor()
        self.show_performance_overlay = False
        self.show_chunk_grid = False

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
        spawn_x = self.world.size // 2
        spawn_z = self.world.size // 2
        self.player = Player(
            spawn_x + 0.5,
            self.world.surface_height_at(spawn_x, spawn_z) + 1.0,
            spawn_z + 0.5,
        )
        self._rendered_mesh_chunk = (
            int(self.player.position[0]) // CHUNK_SIZE,
            int(self.player.position[2]) // CHUNK_SIZE,
        )
        self.renderer = Renderer(
            self.world.build_mesh(tuple(self.player.position)),
            self.world.surface_heights,
        )

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
                performance = self.performance_monitor.update(
                    current_time - previous_time
                )
                if not self.menu_open and not self.inventory_open:
                    player.update(window, world, current_time - previous_time)
                    current_chunk = (
                        int(player.position[0]) // CHUNK_SIZE,
                        int(player.position[2]) // CHUNK_SIZE,
                    )
                    if current_chunk != self._rendered_mesh_chunk:
                        renderer.update_mesh(
                            world.build_mesh(tuple(player.position))
                        )
                        self._rendered_mesh_chunk = current_chunk
                previous_time = current_time
                glClearColor(*self._clear_color_for_grade(self.color_grade), 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                width, height = glfw.get_framebuffer_size(window)
                targeted_block = None
                if not self.menu_open and not self.inventory_open:
                    hit = world.raycast(
                        player.eye_position(), player.forward_direction(), 6.0
                    )
                    if hit is not None:
                        targeted_block = hit[0]
                renderer.render(
                    player.view_matrix(),
                    self._projection_matrix(width, height),
                    width,
                    height,
                    targeted_block,
                    self.show_chunk_grid
                    and not self.menu_open
                    and not self.inventory_open,
                    current_time,
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
                        self.block_lighting_enabled,
                        self.color_grade,
                        self.effects_enabled,
                    )
                elif self.inventory_open:
                    renderer.draw_inventory(
                        width,
                        height,
                        self.inventory_tab,
                        self.hotbar_blocks,
                        self.selected_hotbar_slot,
                        self.inventory_slots,
                        self._inventory_carried_block,
                    )
                else:
                    renderer.draw_hud(
                        width,
                        height,
                        self.selected_hotbar_slot,
                        self.hotbar_blocks,
                    )
                if self.show_performance_overlay:
                    renderer.draw_debug_overlay(
                        width,
                        height,
                        performance,
                        world.block_count,
                    )
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
        if (
            not self.menu_open
            and not self.inventory_open
            and self.player is not None
        ):
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
            if button == glfw.MOUSE_BUTTON_LEFT and self.inventory_open:
                cursor_position = self._inventory_cursor_position(window)
                if cursor_position is not None:
                    width, height = glfw.get_framebuffer_size(window)
                    destination = Hud.inventory_drop_target(
                        width,
                        height,
                        self.inventory_tab,
                        *cursor_position,
                    )
                    self._finish_inventory_drag(destination)
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
                    if setting == (5, 0):
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
                        elif row == 2:
                            self.block_lighting_enabled = bool(quality)
                            if self.renderer is not None:
                                self.renderer.set_block_lighting_enabled(
                                    self.block_lighting_enabled
                                )
                        elif row == 3:
                            self.color_grade = quality
                            if self.renderer is not None:
                                self.renderer.set_color_grade(quality)
                        elif row == 4:
                            self.effects_enabled = bool(quality)
                            if self.renderer is not None:
                                self.renderer.set_effects_enabled(self.effects_enabled)
                elif menu_button == 0:
                    self._close_menu()
                elif menu_button == 1:
                    self.settings_open = True
                elif menu_button == 2:
                    glfw.set_window_should_close(window, True)
            return

        if self.inventory_open:
            if button == glfw.MOUSE_BUTTON_LEFT:
                framebuffer_position = self._inventory_cursor_position(window)
                if framebuffer_position is not None:
                    width, height = glfw.get_framebuffer_size(window)
                    source = Hud.inventory_item_at(
                        width,
                        height,
                        self.inventory_tab,
                        *framebuffer_position,
                    )
                    if source is not None:
                        self._begin_inventory_drag(source)
                    else:
                        tab = Hud.inventory_tab_at(
                            width, height, *framebuffer_position
                        )
                        if tab is not None:
                            self.inventory_tab = tab
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
            block_type = self.hotbar_blocks[self.selected_hotbar_slot]
            changed = (
                world.set_block(adjacent_block, block_type)
                if block_type is not None
                else False
            )
            if changed:
                below_block = (
                    adjacent_block[0],
                    adjacent_block[1] - 1,
                    adjacent_block[2],
                )
                if world.is_solid(*below_block) and (
                    world.blocks[below_block] == BLOCK_GRASS
                ):
                    world.set_block(below_block, BLOCK_DIRT)
        else:
            changed = False

        if changed:
            renderer.update_mesh(world.build_mesh(tuple(player.position)))

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

        if key == glfw.KEY_ESCAPE and self.inventory_open:
            self._close_inventory()
            return
        if key == glfw.KEY_ESCAPE:
            self._handle_escape()
            return
        if key == glfw.KEY_F3:
            self.show_performance_overlay = not self.show_performance_overlay
            return
        if key == glfw.KEY_F4:
            self.show_chunk_grid = not self.show_chunk_grid
            return
        if self.menu_open:
            return

        if key == glfw.KEY_I:
            if self.inventory_open:
                self._close_inventory()
            else:
                self._open_inventory()
            return
        if self.inventory_open:
            if key == glfw.KEY_1:
                self.inventory_tab = 0
            elif key == glfw.KEY_2:
                self.inventory_tab = 1
            return

        if glfw.KEY_1 <= key <= glfw.KEY_9:
            self.selected_hotbar_slot = key - glfw.KEY_1
        elif key == glfw.KEY_0:
            self.selected_hotbar_slot = min(9, len(self.hotbar_blocks) - 1)

    def _on_scroll(
        self, _window: glfw._GLFWwindow, _x_offset: float, y_offset: float
    ) -> None:
        del _window, _x_offset
        if y_offset != 0 and not self.menu_open and not self.inventory_open:
            self.selected_hotbar_slot = (
                self.selected_hotbar_slot - int(np.sign(y_offset))
            ) % len(HOTBAR_BLOCKS)

    def _open_inventory(self) -> None:
        self.inventory_open = True
        self._inventory_drag_source = None
        self._inventory_drag_block = None
        self._last_mouse_position = None
        self._mouse_buttons_down.clear()
        if self.window is not None:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def _close_inventory(self) -> None:
        if self._inventory_drag_block is not None:
            self._finish_inventory_drag(None)
        self.inventory_open = False
        self._inventory_drag_source = None
        self._inventory_drag_block = None
        self._last_mouse_position = None
        self._mouse_buttons_down.clear()
        if self.window is not None:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    @staticmethod
    def _inventory_cursor_position(
        window: glfw._GLFWwindow,
    ) -> tuple[float, float] | None:
        width, height = glfw.get_framebuffer_size(window)
        window_width, window_height = glfw.get_window_size(window)
        if width <= 0 or height <= 0 or window_width <= 0 or window_height <= 0:
            return None
        cursor_x, cursor_y = glfw.get_cursor_pos(window)
        return (
            cursor_x * width / window_width,
            height - cursor_y * height / window_height,
        )

    def _begin_inventory_drag(self, source: tuple[str, int]) -> None:
        kind, index = source
        if kind == "catalog":
            block_type = BLOCK_TYPES[index]
            self._inventory_drag_source = None
            self._inventory_drag_block = (
                self._inventory_carried_block
                if self._inventory_carried_block is not None
                else block_type
            )
            self._inventory_carried_block = (
                block_type
                if self._inventory_carried_block is not None
                else None
            )
            return

        if self._inventory_carried_block is not None:
            self._inventory_drag_source = None
            self._inventory_drag_block = self._inventory_carried_block
            return

        slots = self.hotbar_blocks if kind == "hotbar" else self.inventory_slots
        block_type = slots[index]
        if block_type is None:
            return
        self._inventory_drag_source = source
        self._inventory_drag_block = block_type

    def _finish_inventory_drag(
        self, destination: tuple[str, int] | None
    ) -> None:
        source = self._inventory_drag_source
        block_type = self._inventory_drag_block
        if block_type is None:
            self._inventory_drag_source = None
            return

        if destination is None:
            if source is not None:
                self._inventory_slots_for(source[0])[source[1]] = None
            self._inventory_carried_block = block_type
        else:
            destination_slots = self._inventory_slots_for(destination[0])
            if source is not None:
                source_slots = self._inventory_slots_for(source[0])
                if source != destination:
                    displaced_block = destination_slots[destination[1]]
                    destination_slots[destination[1]] = block_type
                    source_slots[source[1]] = displaced_block
                self._inventory_carried_block = None
            else:
                self._inventory_carried_block = destination_slots[destination[1]]
                destination_slots[destination[1]] = block_type
            if destination[0] == "hotbar":
                self.selected_hotbar_slot = destination[1]

        self._inventory_drag_source = None
        self._inventory_drag_block = None

    def _inventory_slots_for(self, kind: str) -> list[int | None]:
        if kind == "hotbar":
            return self.hotbar_blocks
        if kind == "inventory":
            return self.inventory_slots
        raise ValueError(f"Неизвестный тип слота инвентаря: {kind}.")

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
    def _clear_color_for_grade(grade: int) -> tuple[float, float, float]:
        clear_color = (0.04, 0.07, 0.14)
        if grade == 0:
            return clear_color
        if grade == 1:
            return (
                min(1.0, clear_color[0] * 1.10),
                min(1.0, clear_color[1] * 1.03),
                min(1.0, clear_color[2] * 0.88),
            )
        if grade == 2:
            return (
                min(1.0, clear_color[0] * 0.88),
                min(1.0, clear_color[1] * 1.01),
                min(1.0, clear_color[2] * 1.14),
            )
        if grade == 3:
            luminance = (
                clear_color[0] * 0.2126
                + clear_color[1] * 0.7152
                + clear_color[2] * 0.0722
            )
            color = tuple(
                luminance + (component - luminance) * 0.82
                for component in clear_color
            )
            return (
                min(1.0, max(0.0, (color[0] - 0.5) * 1.12 + 0.5) * 1.05),
                min(1.0, max(0.0, (color[1] - 0.5) * 1.12 + 0.5)),
                min(1.0, max(0.0, (color[2] - 0.5) * 1.12 + 0.5) * 0.91),
            )
        raise ValueError(f"Неизвестный вариант цветокоррекции: {grade}.")

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
