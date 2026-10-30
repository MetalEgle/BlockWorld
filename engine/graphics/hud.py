from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_SCISSOR_TEST,
    glClear,
    glClearColor,
    glDisable,
    glEnable,
    glScissor,
)

from engine.world.voxel_world import (
    BLOCK_COLORS,
    BLOCK_DIRT,
    BLOCK_GRASS,
    BLOCK_SAND,
    BLOCK_STONE,
    HOTBAR_BLOCKS,
)


_HEART = (
    ".##..##.",
    "########",
    "########",
    ".######.",
    "..####..",
    "...##...",
)

_HUNGER = (
    "...##...",
    "..####..",
    ".######.",
    "#######.",
    ".#####..",
    "..###...",
    "..##....",
)


class Hud:
    """Draw a compact, pixel-style first-person status display."""

    _MENU_GLYPHS = {
        "А": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        "В": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
        "Г": ("11111", "10000", "10000", "10000", "10000", "10000", "10000"),
        "Д": ("00110", "01010", "01010", "01010", "10001", "11111", "10001"),
        "Е": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
        "Ж": ("10101", "10101", "01110", "00100", "01110", "10101", "10101"),
        "З": ("01110", "10001", "00001", "00110", "00001", "10001", "01110"),
        "И": ("10001", "10011", "10101", "11001", "10001", "10001", "10001"),
        "Й": ("01010", "00100", "10001", "10011", "10101", "11001", "10001"),
        "К": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
        "Л": ("00111", "01001", "01001", "01001", "10001", "10001", "10001"),
        "М": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
        "Н": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
        "О": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
        "П": ("11111", "10001", "10001", "10001", "10001", "10001", "10001"),
        "Р": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
        "С": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
        "Т": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
        "У": ("10001", "10001", "10001", "01111", "00001", "10001", "01110"),
        "Х": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
        "Ы": ("10001", "10001", "11101", "10011", "10011", "10001", "11101"),
        "Ь": ("10000", "10000", "10000", "11110", "10001", "10001", "11110"),
        " ": ("00000",) * 7,
    }

    @staticmethod
    def draw(width: int, height: int, selected_slot: int = 0) -> None:
        if width <= 0 or height <= 0:
            return

        scale = max(1, height // 360)
        center_x = width // 2
        hotbar_y = 12 * scale
        slot_size = 20 * scale
        slot_gap = 2 * scale
        hotbar_width = 9 * slot_size + 8 * slot_gap
        hotbar_x = center_x - hotbar_width // 2

        Hud._draw_hotbar(
            hotbar_x, hotbar_y, slot_size, slot_gap, scale, selected_slot
        )
        Hud._draw_status_rows(center_x, hotbar_y, slot_size, scale)
        Hud._draw_experience_bar(center_x, hotbar_y, scale)
        Hud._draw_crosshair(width, height)

    @staticmethod
    def draw_menu(
        width: int,
        height: int,
        settings_open: bool = False,
        hovered_button: int | None = None,
        texture_quality: int = 1,
        shadow_quality: int = 1,
    ) -> None:
        if width <= 0 or height <= 0:
            return

        scale = max(1, height // 540)
        panel_width = min(360 * scale, width - 32 * scale)
        panel_height = min(330 * scale, height - 32 * scale)
        panel_x = (width - panel_width) // 2
        panel_y = (height - panel_height) // 2
        Hud._rectangle(panel_x, panel_y, panel_width, panel_height, (0.09, 0.12, 0.17))
        Hud._rectangle(
            panel_x,
            panel_y + panel_height - 3 * scale,
            panel_width,
            3 * scale,
            (0.38, 0.66, 0.84),
        )

        title = "НАСТРОЙКИ" if settings_open else "ПАУЗА"
        Hud._draw_menu_text(
            title,
            width // 2,
            panel_y + panel_height - 62 * scale,
            scale,
            (0.94, 0.96, 1.0),
        )

        if settings_open:
            Hud._draw_settings(width, height, texture_quality, shadow_quality)
            return

        labels = ("ПРОДОЛЖИТЬ", "НАСТРОЙКИ", "ВЫХОД ИЗ ИГРЫ")
        for index, (button_x, button_y, button_width, button_height) in enumerate(
            Hud._menu_button_bounds(width, height)
        ):
            color = (
                (0.31, 0.49, 0.61)
                if index == hovered_button
                else (0.17, 0.23, 0.31)
            )
            Hud._rectangle(button_x, button_y, button_width, button_height, color)
            Hud._rectangle(
                button_x,
                button_y + button_height - 2 * scale,
                button_width,
                2 * scale,
                (0.46, 0.61, 0.71),
            )
            Hud._draw_menu_text(
                labels[index],
                width // 2,
                button_y + (button_height - 7 * scale) // 2,
                scale,
                (0.96, 0.97, 0.99),
            )

    @staticmethod
    def menu_button_at(
        width: int, height: int, x: float, y: float
    ) -> int | None:
        for index, (button_x, button_y, button_width, button_height) in enumerate(
            Hud._menu_button_bounds(width, height)
        ):
            if (
                button_x <= x < button_x + button_width
                and button_y <= y < button_y + button_height
            ):
                return index
        return None

    @staticmethod
    def settings_option_at(
        width: int, height: int, x: float, y: float
    ) -> tuple[int, int] | None:
        for row in range(2):
            for option, bounds in enumerate(
                Hud._settings_choice_bounds(width, height, row)
            ):
                button_x, button_y, button_width, button_height = bounds
                if (
                    button_x <= x < button_x + button_width
                    and button_y <= y < button_y + button_height
                ):
                    return row, option

        back_x, back_y, back_width, back_height = Hud._settings_back_bounds(
            width, height
        )
        if (
            back_x <= x < back_x + back_width
            and back_y <= y < back_y + back_height
        ):
            return 2, 0
        return None

    @staticmethod
    def _draw_settings(
        width: int, height: int, texture_quality: int, shadow_quality: int
    ) -> None:
        scale = max(1, height // 540)
        row_labels = ("ТЕКСТУРЫ", "ТЕНИ")
        for row, (label, selected) in enumerate(
            zip(row_labels, (texture_quality, shadow_quality))
        ):
            if selected not in range(3):
                raise ValueError(f"Неизвестный уровень качества настройки: {selected}.")
            option_labels = (
                ("НИЗКОЕ", "СРЕДНЕЕ", "ВЫСОКОЕ")
                if row == 0
                else ("ВЫКЛ", "НИЗКОЕ", "ВЫСОКОЕ")
            )
            choice_bounds = Hud._settings_choice_bounds(width, height, row)
            label_y = choice_bounds[0][1] + choice_bounds[0][3] + 8 * scale
            Hud._draw_menu_text(
                label,
                width // 2,
                label_y,
                scale,
                (0.82, 0.88, 0.94),
            )
            for option, (button_x, button_y, button_width, button_height) in enumerate(
                choice_bounds
            ):
                color = (
                    (0.31, 0.49, 0.61)
                    if option == selected
                    else (0.17, 0.23, 0.31)
                )
                Hud._rectangle(button_x, button_y, button_width, button_height, color)
                Hud._draw_menu_text(
                    option_labels[option],
                    button_x + button_width // 2,
                    button_y + (button_height - 7 * scale) // 2,
                    scale,
                    (0.96, 0.97, 0.99),
                )

        back_x, back_y, back_width, back_height = Hud._settings_back_bounds(
            width, height
        )
        Hud._rectangle(back_x, back_y, back_width, back_height, (0.17, 0.23, 0.31))
        Hud._draw_menu_text(
            "НАЗАД",
            width // 2,
            back_y + (back_height - 7 * scale) // 2,
            scale,
            (0.96, 0.97, 0.99),
        )

    @staticmethod
    def _settings_choice_bounds(
        width: int, height: int, row: int
    ) -> tuple[tuple[int, int, int, int], ...]:
        scale = max(1, height // 540)
        panel_height = min(330 * scale, height - 32 * scale)
        panel_y = (height - panel_height) // 2
        button_width = min(280 * scale, width - 48 * scale)
        choice_width = button_width // 3
        button_height = 30 * scale
        button_y = panel_y + panel_height - (144 + row * 82) * scale
        return tuple(
            (
                (width - button_width) // 2 + option * choice_width,
                button_y,
                choice_width if option < 2 else button_width - 2 * choice_width,
                button_height,
            )
            for option in range(3)
        )

    @staticmethod
    def _settings_back_bounds(width: int, height: int) -> tuple[int, int, int, int]:
        scale = max(1, height // 540)
        panel_height = min(330 * scale, height - 32 * scale)
        panel_y = (height - panel_height) // 2
        button_width = min(280 * scale, width - 48 * scale)
        button_height = 30 * scale
        return (
            (width - button_width) // 2,
            panel_y + 12 * scale,
            button_width,
            button_height,
        )

    @staticmethod
    def _menu_button_bounds(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
        scale = max(1, height // 540)
        panel_height = min(330 * scale, height - 32 * scale)
        panel_y = (height - panel_height) // 2
        button_width = min(280 * scale, width - 48 * scale)
        button_height = 42 * scale
        button_gap = 12 * scale
        button_x = (width - button_width) // 2
        stack_height = 3 * button_height + 2 * button_gap
        first_y = panel_y + (panel_height + stack_height) // 2 - button_height
        return tuple(
            (
                button_x,
                first_y - index * (button_height + button_gap),
                button_width,
                button_height,
            )
            for index in range(3)
        )

    @staticmethod
    def _draw_menu_text(
        text: str,
        center_x: int,
        y: int,
        scale: int,
        color: tuple[float, float, float],
    ) -> None:
        character_width = 5 * scale
        character_gap = scale
        text_width = len(text) * (character_width + character_gap) - character_gap
        x = center_x - text_width // 2

        for character in text:
            glyph = Hud._MENU_GLYPHS[character]
            for row_index, row in enumerate(glyph):
                column = 0
                while column < len(row):
                    if row[column] != "1":
                        column += 1
                        continue
                    start = column
                    while column < len(row) and row[column] == "1":
                        column += 1
                    Hud._rectangle(
                        x + start * scale,
                        y + (len(glyph) - row_index - 1) * scale,
                        (column - start) * scale,
                        scale,
                        color,
                    )
            x += character_width + character_gap

    @staticmethod
    def _draw_hotbar(
        x: int, y: int, slot_size: int, gap: int, scale: int, selected_slot: int
    ) -> None:
        outer_size = slot_size + 2 * scale
        for index in range(9):
            slot_x = x + index * (slot_size + gap)
            border_color = (
                (0.95, 0.95, 0.95)
                if index == selected_slot
                else (0.12, 0.12, 0.12)
            )
            Hud._rectangle(slot_x - scale, y - scale, outer_size, outer_size, border_color)
            Hud._rectangle(
                slot_x,
                y,
                slot_size,
                slot_size,
                (0.34, 0.34, 0.34) if index == 0 else (0.23, 0.23, 0.23),
            )

            block_type = HOTBAR_BLOCKS[index]
            if block_type is not None:
                Hud._draw_block_icon(slot_x, y, slot_size, scale, block_type)

    @staticmethod
    def _draw_block_icon(
        x: int, y: int, slot_size: int, scale: int, block_type: int
    ) -> None:
        inset = (slot_size - 12 * scale) // 2
        icon_x = x + inset
        icon_y = y + inset
        icon_size = 12 * scale

        if block_type == BLOCK_GRASS:
            Hud._rectangle(icon_x, icon_y, icon_size, icon_size, (0.27, 0.70, 0.26))
            Hud._rectangle(
                icon_x,
                icon_y + 7 * scale,
                icon_size,
                5 * scale,
                (0.43, 0.29, 0.17),
            )
            Hud._rectangle(
                icon_x + 2 * scale,
                icon_y + 1 * scale,
                icon_size - 4 * scale,
                2 * scale,
                (0.58, 0.92, 0.38),
            )
            Hud._rectangle(
                icon_x + 4 * scale,
                icon_y + 5 * scale,
                2 * scale,
                2 * scale,
                (0.18, 0.56, 0.20),
            )
            Hud._rectangle(
                icon_x + 8 * scale,
                icon_y + 4 * scale,
                2 * scale,
                2 * scale,
                (0.19, 0.58, 0.22),
            )
            return

        if block_type == BLOCK_DIRT:
            Hud._rectangle(icon_x, icon_y, icon_size, icon_size, (0.49, 0.31, 0.17))
            Hud._rectangle(
                icon_x + 2 * scale,
                icon_y + 3 * scale,
                icon_size - 4 * scale,
                2 * scale,
                (0.60, 0.40, 0.22),
            )
            Hud._rectangle(
                icon_x + 1 * scale,
                icon_y + 8 * scale,
                3 * scale,
                2 * scale,
                (0.63, 0.42, 0.22),
            )
            Hud._rectangle(
                icon_x + 7 * scale,
                icon_y + 6 * scale,
                3 * scale,
                2 * scale,
                (0.55, 0.37, 0.18),
            )
            return

        if block_type == BLOCK_STONE:
            Hud._rectangle(icon_x, icon_y, icon_size, icon_size, (0.58, 0.60, 0.63))
            for offset_y in (2, 6, 9):
                for offset_x in (1, 5, 9):
                    Hud._rectangle(
                        icon_x + offset_x * scale,
                        icon_y + offset_y * scale,
                        2 * scale,
                        2 * scale,
                        (0.40, 0.42, 0.46),
                    )
            return

        if block_type == BLOCK_SAND:
            Hud._rectangle(icon_x, icon_y, icon_size, icon_size, (0.83, 0.74, 0.47))
            Hud._rectangle(
                icon_x + 1 * scale,
                icon_y + 4 * scale,
                icon_size - 2 * scale,
                2 * scale,
                (0.92, 0.86, 0.62),
            )
            Hud._rectangle(
                icon_x + 5 * scale,
                icon_y + 8 * scale,
                4 * scale,
                2 * scale,
                (0.88, 0.80, 0.57),
            )
            return

        color = BLOCK_COLORS[block_type]
        Hud._rectangle(icon_x, icon_y, icon_size, icon_size, tuple(c * 0.7 for c in color))
        Hud._rectangle(icon_x, icon_y + 7 * scale, icon_size, 5 * scale, color)
        Hud._rectangle(
            icon_x + scale,
            icon_y + scale,
            icon_size - 2 * scale,
            scale,
            tuple(min(c * 1.2, 1.0) for c in color),
        )

    @staticmethod
    def _draw_status_rows(center_x: int, hotbar_y: int, slot_size: int, scale: int) -> None:
        icon_width = 8 * scale
        gap = scale
        row_width = 10 * icon_width + 9 * gap
        row_y = hotbar_y + slot_size + 12 * scale
        health_x = center_x - 8 * scale - row_width
        hunger_x = center_x + 8 * scale

        for index in range(10):
            x = health_x + index * (icon_width + gap)
            Hud._draw_bitmap(
                _HEART,
                x + scale,
                row_y - scale,
                scale,
                (0.18, 0.08, 0.08),
            )
            Hud._draw_bitmap(_HEART, x, row_y, scale, (0.87, 0.12, 0.16))
            Hud._draw_bitmap(
                _HUNGER,
                hunger_x + index * (icon_width + gap),
                row_y,
                scale,
                (0.82, 0.48, 0.14),
            )

    @staticmethod
    def _draw_experience_bar(center_x: int, hotbar_y: int, scale: int) -> None:
        width = 182 * scale
        x = center_x - width // 2
        y = hotbar_y - 7 * scale
        Hud._rectangle(x, y, width, 5 * scale, (0.08, 0.08, 0.08))
        Hud._rectangle(x + scale, y + scale, width - 2 * scale, 3 * scale, (0.20, 0.20, 0.20))

    @staticmethod
    def _draw_bitmap(
        bitmap: tuple[str, ...],
        x: int,
        y: int,
        scale: int,
        color: tuple[float, float, float],
    ) -> None:
        for row_index, row in enumerate(bitmap):
            column = 0
            while column < len(row):
                if row[column] != "#":
                    column += 1
                    continue
                start = column
                while column < len(row) and row[column] == "#":
                    column += 1
                Hud._rectangle(
                    x + start * scale,
                    y + (len(bitmap) - 1 - row_index) * scale,
                    (column - start) * scale,
                    scale,
                    color,
                )

    @staticmethod
    def _draw_crosshair(width: int, height: int) -> None:
        center_x = width // 2
        center_y = height // 2
        Hud._rectangle(center_x - 1, center_y - 6, 3, 13, (0.02, 0.02, 0.02))
        Hud._rectangle(center_x - 6, center_y - 1, 13, 3, (0.02, 0.02, 0.02))
        Hud._rectangle(center_x, center_y - 4, 1, 9, (0.95, 0.95, 0.95))
        Hud._rectangle(center_x - 4, center_y, 9, 1, (0.95, 0.95, 0.95))

    @staticmethod
    def _rectangle(
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[float, float, float],
    ) -> None:
        if width <= 0 or height <= 0:
            return

        glEnable(GL_SCISSOR_TEST)
        glScissor(x, y, width, height)
        glClearColor(*color, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_SCISSOR_TEST)
