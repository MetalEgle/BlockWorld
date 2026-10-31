from functools import lru_cache
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_SCISSOR_TEST,
    glClear,
    glClearColor,
    glDisable,
    glEnable,
    glScissor,
)
from PIL import Image

from engine.diagnostics import PerformanceStats
from engine.world.voxel_world import (
    BLOCK_COLORS,
    BLOCK_DIRT,
    BLOCK_GRASS,
    BLOCK_GLOWSTONE,
    BLOCK_LANTERN,
    BLOCK_LEAF,
    BLOCK_SAND,
    BLOCK_STONE,
    BLOCK_STONE_BRICKS,
    BLOCK_STONE_COBBLE,
    BLOCK_TYPES,
    HOTBAR_BLOCKS,
    BLOCK_WOOD,
    BLOCK_WOOD_BOARD,
)

_BLOCK_ICON_TEXTURES = {
    BLOCK_GRASS: "Grass_Top.jfif",
    BLOCK_DIRT: "Dirt.jfif",
    BLOCK_STONE: "Stone.jfif",
    BLOCK_SAND: "Send.jfif",
    BLOCK_LEAF: "Leaf.jfif",
    BLOCK_STONE_BRICKS: "Stone_Briks.jfif",
    BLOCK_STONE_COBBLE: "Stone_Cobble.jfif",
    BLOCK_WOOD: "Wood_Side.jfif",
    BLOCK_WOOD_BOARD: "Wood_Board.jfif",
    BLOCK_GLOWSTONE: "Send.jfif",
    BLOCK_LANTERN: "Lighter.jfif",
}


@lru_cache(maxsize=None)
def _load_block_icon_pixels(
    block_type: int,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    try:
        filename = _BLOCK_ICON_TEXTURES[block_type]
    except KeyError as error:
        raise ValueError(f"Нет текстуры для блока хотбара: {block_type}.") from error

    texture_path = (
        Path(__file__).resolve().parents[2] / "assets" / "textures" / filename
    )
    if not texture_path.is_file():
        raise FileNotFoundError(f"Не найдена текстура блока: {texture_path}.")
    with Image.open(texture_path) as image:
        pixels = image.convert("RGB").resize((8, 8), Image.Resampling.NEAREST)
        return tuple(
            tuple(
                tuple(channel / 255.0 for channel in pixels.getpixel((x, y)))
                for x in range(8)
            )
            for y in range(8)
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

    _DEBUG_GLYPHS = {
        "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
        "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
        "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
        "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
        "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
        "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
        "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
        "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
        "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
        "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
        "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
        "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
        "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
        "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
        "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
        "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
        "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
        "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
        "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
        "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
        "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
        "N": ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
        "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
        "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
        "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
        "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
        "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
        "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
        "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
        "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
        "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
        " ": ("00000",) * 7,
        "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
        ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
        "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    }

    _MENU_GLYPHS = {
        "А": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        "Б": ("11111", "10000", "10000", "11110", "10001", "10001", "11110"),
        "В": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
        "Г": ("11111", "10000", "10000", "10000", "10000", "10000", "10000"),
        "Д": ("00110", "01010", "01010", "01010", "10001", "11111", "10001"),
        "Е": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
        "Э": ("11111", "00001", "00111", "00001", "00001", "10001", "01110"),
        "Ф": ("00100", "01110", "10101", "10101", "01110", "00100", "00100"),
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
        "Ц": ("10010", "10010", "10010", "10010", "10010", "11111", "00001"),
        "Ч": ("10001", "10001", "10001", "01111", "00001", "00001", "00001"),
        "Х": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
        "Ы": ("10001", "10001", "11101", "10011", "10011", "10001", "11101"),
        "Ь": ("10000", "10000", "10000", "11110", "10001", "10001", "11110"),
        "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
        "Ш": ("10101", "10101", "10101", "10101", "10101", "10101", "11111"),
        "Щ": ("10101", "10101", "10101", "10101", "10101", "10101", "11111"),
        "Я": ("01111", "10001", "10001", "01111", "00101", "01001", "10001"),
        " ": ("00000",) * 7,
    }
    _INVENTORY_BLOCK_NAMES = (
        "ТРАВА",
        "ЗЕМЛЯ",
        "КАМЕНЬ",
        "ПЕСОК",
        "ЛИСТВА",
        "КИРПИЧ",
        "БУЛЫЖНИК",
        "ДЕРЕВО",
        "ДОСКИ",
        "СВЕТЯЩИЙСЯ",
        "ФОНАРЬ",
    )

    @staticmethod
    def draw(
        width: int,
        height: int,
        selected_slot: int = 0,
        hotbar_blocks: tuple[int | None, ...] | list[int | None] = HOTBAR_BLOCKS,
    ) -> None:
        if width <= 0 or height <= 0:
            return

        scale = max(1, height // 360)
        center_x = width // 2
        hotbar_y = 12 * scale
        slot_size = 20 * scale
        slot_gap = 2 * scale
        hotbar_width = (
            len(hotbar_blocks) * slot_size + (len(hotbar_blocks) - 1) * slot_gap
        )
        hotbar_x = center_x - hotbar_width // 2

        Hud._draw_hotbar(
            hotbar_x,
            hotbar_y,
            slot_size,
            slot_gap,
            scale,
            selected_slot,
            hotbar_blocks,
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
        block_lighting_enabled: bool = True,
        color_grade: int = 0,
        effects_enabled: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            return

        scale = max(1, height // 540)
        panel_width = min(360 * scale, width - 32 * scale)
        panel_height = min((500 if settings_open else 330) * scale, height - 32 * scale)
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
            Hud._draw_settings(
                width,
                height,
                texture_quality,
                shadow_quality,
                block_lighting_enabled,
                color_grade,
                effects_enabled,
            )
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
    def build_debug_overlay_vertices(
        width: int,
        height: int,
        stats: PerformanceStats,
        block_count: int,
    ) -> np.ndarray:
        if width <= 0 or height <= 0:
            return np.empty((0, 5), dtype=np.float32)

        scale = max(1, height // 720)
        lines = (
            f"FPS {stats.fps:.1f}",
            f"FRAME {stats.frame_time_ms:.1f} MS",
            f"FRAMES {stats.frame_count}",
            f"CPU {stats.process_cpu_percent:.0f}% GAME",
            f"CPU {stats.system_cpu_percent:.0f}% SYS",
            f"RAM {stats.process_memory_mb:.0f} MB GAME",
            (
                f"RAM {stats.system_memory_used_gb:.1f}/"
                f"{stats.system_memory_total_gb:.1f} GB SYS"
            ),
            f"BLOCKS {block_count}",
            f"RES {width}X{height}",
        )
        padding = 6 * scale
        line_height = 9 * scale
        text_width = max(len(line) * 6 * scale - scale for line in lines)
        panel_width = text_width + 2 * padding
        panel_height = len(lines) * line_height + 2 * padding
        panel_x = 8 * scale
        panel_y = height - panel_height - 8 * scale

        vertices: list[float] = []
        Hud._append_debug_rectangle(
            vertices,
            panel_x,
            panel_y,
            panel_width,
            panel_height,
            (0.04, 0.05, 0.07),
        )
        for index, line in enumerate(lines):
            Hud._append_debug_text(
                vertices,
                line,
                panel_x + padding,
                panel_y + panel_height - padding - (index + 1) * line_height + scale,
                scale,
                (0.92, 0.95, 0.98),
            )
        return np.asarray(vertices, dtype=np.float32).reshape(-1, 5)

    @staticmethod
    def _append_debug_text(
        vertices: list[float],
        text: str,
        x: int,
        y: int,
        scale: int,
        color: tuple[float, float, float],
    ) -> None:
        for character in text:
            glyph = Hud._DEBUG_GLYPHS[character]
            for row_index, row in enumerate(glyph):
                column = 0
                while column < len(row):
                    if row[column] != "1":
                        column += 1
                        continue
                    start = column
                    while column < len(row) and row[column] == "1":
                        column += 1
                    Hud._append_debug_rectangle(
                        vertices,
                        x + start * scale,
                        y + (len(glyph) - row_index - 1) * scale,
                        (column - start) * scale,
                        scale,
                        color,
                    )
            x += 6 * scale

    @staticmethod
    def _append_debug_rectangle(
        vertices: list[float],
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[float, float, float],
    ) -> None:
        if width <= 0 or height <= 0:
            return

        x2 = x + width
        y2 = y + height
        for point_x, point_y in (
            (x, y),
            (x2, y),
            (x2, y2),
            (x, y),
            (x2, y2),
            (x, y2),
        ):
            vertices.extend((point_x, point_y, *color))

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
        for row in range(5):
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
            return 5, 0
        return None

    @staticmethod
    def _draw_settings(
        width: int,
        height: int,
        texture_quality: int,
        shadow_quality: int,
        block_lighting_enabled: bool,
        color_grade: int,
        effects_enabled: bool,
    ) -> None:
        scale = max(1, height // 540)
        row_labels = ("ТЕКСТУРЫ", "ТЕНИ", "БЛОЧНЫЙ СВЕТ", "ЦВЕТОКОР", "ЭФФЕКТЫ")
        for row, (label, selected) in enumerate(
            zip(
                row_labels,
                (
                    texture_quality,
                    shadow_quality,
                    int(block_lighting_enabled),
                    color_grade,
                    int(effects_enabled),
                ),
            )
        ):
            option_count = 4 if row == 3 else 2 if row in (2, 4) else 3
            if selected not in range(option_count):
                raise ValueError(f"Неизвестный уровень качества настройки: {selected}.")
            option_labels = {
                0: ("НИЗКОЕ", "СРЕДНЕЕ", "ВЫСОКОЕ"),
                1: ("ВЫКЛ", "НИЗКОЕ", "ВЫСОКОЕ"),
                2: ("ВЫКЛ", "ВКЛ"),
                3: ("ВЫКЛ", "ТЕПЛО", "ХОЛОД", "КИНО"),
                4: ("ВЫКЛ", "ВКЛ"),
            }[row]
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
        panel_height = min(500 * scale, height - 32 * scale)
        panel_y = (height - panel_height) // 2
        button_width = min(280 * scale, width - 48 * scale)
        option_count = 4 if row == 3 else 2 if row in (2, 4) else 3
        choice_width = button_width // option_count
        button_height = 30 * scale
        back_y = panel_y + 2 * scale
        first_y = panel_y + panel_height - 135 * scale
        row_spacing = min(
            70 * scale,
            (first_y - back_y - button_height - 8 * scale) / 4,
        )
        button_y = round(first_y - row * row_spacing)
        return tuple(
            (
                (width - button_width) // 2 + option * choice_width,
                button_y,
                (
                    choice_width
                    if option < option_count - 1
                    else button_width - (option_count - 1) * choice_width
                ),
                button_height,
            )
            for option in range(option_count)
        )

    @staticmethod
    def _settings_back_bounds(width: int, height: int) -> tuple[int, int, int, int]:
        scale = max(1, height // 540)
        panel_height = min(500 * scale, height - 32 * scale)
        panel_y = (height - panel_height) // 2
        button_width = min(280 * scale, width - 48 * scale)
        button_height = 30 * scale
        return (
            (width - button_width) // 2,
            panel_y + 2 * scale,
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
    def draw_inventory(
        width: int,
        height: int,
        selected_tab: int = 0,
        hotbar_blocks: tuple[int | None, ...] | list[int | None] = HOTBAR_BLOCKS,
        selected_slot: int = 0,
        inventory_slots: tuple[int | None, ...] | list[int | None] = (),
        carried_block: int | None = None,
    ) -> None:
        if width <= 64 or height <= 64:
            return

        scale = max(1, height // 540)
        panel_x, panel_y, panel_width, panel_height = Hud._inventory_panel_bounds(
            width, height
        )
        Hud._rectangle(0, 0, width, height, (0.015, 0.025, 0.04))
        Hud._rectangle(
            panel_x, panel_y, panel_width, panel_height, (0.09, 0.12, 0.17)
        )
        Hud._rectangle(
            panel_x,
            panel_y + panel_height - 3 * scale,
            panel_width,
            3 * scale,
            (0.38, 0.66, 0.84),
        )
        Hud._draw_menu_text(
            "ИНВЕНТАРЬ",
            width // 2,
            panel_y + panel_height - 48 * scale,
            scale,
            (0.94, 0.96, 1.0),
        )

        for tab, label in enumerate(("ИНВЕНТАРЬ", "БЛОКИ")):
            x, y, tab_width, tab_height = Hud._inventory_tab_bounds(
                width, height
            )[tab]
            color = (0.24, 0.39, 0.52) if tab == selected_tab else (0.15, 0.19, 0.25)
            Hud._rectangle(x, y, tab_width, tab_height, color)
            Hud._draw_menu_text(
                label,
                x + tab_width // 2,
                y + (tab_height - 7 * scale) // 2,
                scale,
                (0.94, 0.96, 1.0),
            )

        if selected_tab == 0:
            Hud._draw_inventory_items(
                panel_x,
                panel_y,
                panel_width,
                panel_height,
                scale,
                inventory_slots,
            )
        else:
            Hud._draw_block_catalog(width, height, scale)

        Hud._draw_inventory_hotbar(
            width,
            height,
            hotbar_blocks,
            selected_slot,
            scale,
        )
        if carried_block is not None:
            Hud._draw_menu_text(
                "В РУКЕ",
                panel_x + 50 * scale,
                panel_y + 14 * scale,
                scale,
                (0.78, 0.84, 0.91),
            )
            Hud._draw_block_texture_icon(
                panel_x + 76 * scale,
                panel_y + 7 * scale,
                24 * scale,
                scale,
                carried_block,
            )
        Hud._draw_menu_text(
            "I ЗАКРЫТЬ",
            width // 2,
            panel_y + 14 * scale,
            scale,
            (0.68, 0.74, 0.82),
        )

    @staticmethod
    def inventory_tab_at(
        width: int, height: int, cursor_x: float, cursor_y: float
    ) -> int | None:
        if width <= 0 or height <= 0:
            return None
        for tab, (x, y, tab_width, tab_height) in enumerate(
            Hud._inventory_tab_bounds(width, height)
        ):
            if x <= cursor_x < x + tab_width and y <= cursor_y < y + tab_height:
                return tab
        return None

    @staticmethod
    def inventory_item_at(
        width: int,
        height: int,
        selected_tab: int,
        cursor_x: float,
        cursor_y: float,
    ) -> tuple[str, int] | None:
        hotbar_slot = Hud.inventory_hotbar_slot_at(
            width, height, cursor_x, cursor_y
        )
        if hotbar_slot is not None:
            return "hotbar", hotbar_slot

        if selected_tab == 0:
            for index, (x, y, slot_size, _) in enumerate(
                Hud._inventory_storage_slot_bounds(width, height)
            ):
                if (
                    x <= cursor_x < x + slot_size
                    and y <= cursor_y < y + slot_size
                ):
                    return "inventory", index
        elif selected_tab == 1:
            for index, (x, y, slot_size, _) in enumerate(
                Hud._inventory_storage_slot_bounds(width, height)
            ):
                if (
                    x <= cursor_x < x + slot_size
                    and y <= cursor_y < y + slot_size
                    and index < len(BLOCK_TYPES)
                ):
                    return "catalog", index
        return None

    @staticmethod
    def inventory_drop_target(
        width: int,
        height: int,
        selected_tab: int,
        cursor_x: float,
        cursor_y: float,
    ) -> tuple[str, int] | None:
        hotbar_slot = Hud.inventory_hotbar_slot_at(
            width, height, cursor_x, cursor_y
        )
        if hotbar_slot is not None:
            return "hotbar", hotbar_slot
        if selected_tab == 0:
            for index, (x, y, slot_size, _) in enumerate(
                Hud._inventory_storage_slot_bounds(width, height)
            ):
                if (
                    x <= cursor_x < x + slot_size
                    and y <= cursor_y < y + slot_size
                ):
                    return "inventory", index
        return None

    @staticmethod
    def inventory_hotbar_slot_at(
        width: int, height: int, cursor_x: float, cursor_y: float
    ) -> int | None:
        for index, (x, y, slot_size, _) in enumerate(
            Hud._inventory_hotbar_slot_bounds(width, height)
        ):
            if x <= cursor_x < x + slot_size and y <= cursor_y < y + slot_size:
                return index
        return None

    @staticmethod
    def _inventory_panel_bounds(
        width: int, height: int
    ) -> tuple[int, int, int, int]:
        scale = max(1, height // 540)
        panel_width = min(820 * scale, width - 32 * scale)
        panel_height = min(540 * scale, height - 32 * scale)
        return (
            (width - panel_width) // 2,
            (height - panel_height) // 2,
            panel_width,
            panel_height,
        )

    @staticmethod
    def _inventory_tab_bounds(
        width: int, height: int
    ) -> tuple[tuple[int, int, int, int], ...]:
        panel_x, panel_y, panel_width, panel_height = Hud._inventory_panel_bounds(
            width, height
        )
        scale = max(1, height // 540)
        tab_width = min(190 * scale, (panel_width - 24 * scale) // 2)
        tab_height = 34 * scale
        gap = 8 * scale
        total_width = tab_width * 2 + gap
        x = panel_x + (panel_width - total_width) // 2
        y = panel_y + panel_height - 104 * scale
        return (
            (x, y, tab_width, tab_height),
            (x + tab_width + gap, y, tab_width, tab_height),
        )

    @staticmethod
    def _draw_inventory_items(
        panel_x: int,
        panel_y: int,
        panel_width: int,
        panel_height: int,
        scale: int,
        inventory_slots: tuple[int | None, ...] | list[int | None],
    ) -> None:
        cell_gap = 4 * scale
        cell_size = max(
            18 * scale,
            min(34 * scale, (panel_width * 54 // 100 - 32 * scale) // 9 - cell_gap),
        )
        grid_x = panel_x + 24 * scale
        grid_y = panel_y + 92 * scale
        Hud._draw_menu_text(
            "ПРЕДМЕТЫ",
            grid_x + (9 * cell_size + 8 * cell_gap) // 2,
            grid_y + 3 * (cell_size + cell_gap) + 10 * scale,
            scale,
            (0.78, 0.84, 0.91),
        )
        for row in range(3):
            for column in range(9):
                x = grid_x + column * (cell_size + cell_gap)
                y = grid_y + row * (cell_size + cell_gap)
                Hud._draw_empty_slot(x, y, cell_size, scale)
                index = row * 9 + column
                if index < len(inventory_slots) and inventory_slots[index] is not None:
                    Hud._draw_block_texture_icon(
                        x, y, cell_size, scale, inventory_slots[index]
                    )

        equipment_x = panel_x + panel_width * 56 // 100
        equipment_step = cell_size + 14 * scale
        tab_y = panel_y + panel_height - 104 * scale
        equipment_y = min(
            grid_y + 2 * equipment_step,
            tab_y - cell_size - 16 * scale,
        )
        equipment_title_y = equipment_y + cell_size + 24 * scale
        if equipment_title_y >= tab_y - 10 * scale:
            equipment_title_y = panel_y + 14 * scale
        Hud._draw_menu_text(
            "ЭКИПИРОВКА",
            equipment_x + (panel_width * 40 // 100) // 2,
            equipment_title_y,
            scale,
            (0.78, 0.84, 0.91),
        )
        equipment = ("ШЛЕМ", "НАГРУДНИК", "ПОНОЖИ", "ОРУЖИЕ", "ОРУЖИЕ")
        for index, label in enumerate(equipment):
            column = index % 2
            row = index // 2
            x = equipment_x + column * (cell_size + 86 * scale)
            y = equipment_y - row * equipment_step
            Hud._draw_menu_text(
                label,
                x + cell_size // 2,
                y + cell_size + 5 * scale,
                scale,
                (0.76, 0.81, 0.88),
            )
            Hud._draw_empty_slot(x, y, cell_size, scale)

    @staticmethod
    def _draw_block_catalog(
        width: int,
        height: int,
        scale: int,
    ) -> None:
        for index, (x, y, slot_size, _) in enumerate(
            Hud._inventory_storage_slot_bounds(width, height)
        ):
            Hud._draw_empty_slot(x, y, slot_size, scale)
            if index < len(BLOCK_TYPES):
                Hud._draw_block_texture_icon(
                    x, y, slot_size, scale, BLOCK_TYPES[index]
                )

    @staticmethod
    def _inventory_hotbar_slot_bounds(
        width: int, height: int
    ) -> tuple[tuple[int, int, int, int], ...]:
        if width <= 64 or height <= 64:
            return ()
        panel_x, panel_y, panel_width, _ = Hud._inventory_panel_bounds(width, height)
        scale = max(1, height // 540)
        slot_size = 34 * scale
        slot_gap = 4 * scale
        total_width = (
            len(HOTBAR_BLOCKS) * slot_size
            + (len(HOTBAR_BLOCKS) - 1) * slot_gap
        )
        first_x = panel_x + (panel_width - total_width) // 2
        slot_y = panel_y + 48 * scale
        return tuple(
            (
                first_x + index * (slot_size + slot_gap),
                slot_y,
                slot_size,
                slot_size,
            )
            for index in range(len(HOTBAR_BLOCKS))
        )

    @staticmethod
    def _inventory_storage_slot_bounds(
        width: int, height: int
    ) -> tuple[tuple[int, int, int, int], ...]:
        if width <= 64 or height <= 64:
            return ()
        panel_x, panel_y, panel_width, _ = Hud._inventory_panel_bounds(width, height)
        scale = max(1, height // 540)
        cell_gap = 4 * scale
        cell_size = max(
            18 * scale,
            min(34 * scale, (panel_width * 54 // 100 - 32 * scale) // 9 - cell_gap),
        )
        grid_x = panel_x + 24 * scale
        grid_y = panel_y + 92 * scale
        return tuple(
            (
                grid_x + column * (cell_size + cell_gap),
                grid_y + row * (cell_size + cell_gap),
                cell_size,
                cell_size,
            )
            for row in range(3)
            for column in range(9)
        )

    @staticmethod
    @staticmethod
    def _draw_inventory_hotbar(
        width: int,
        height: int,
        hotbar_blocks: tuple[int | None, ...] | list[int | None],
        selected_slot: int,
        scale: int,
    ) -> None:
        for index, (x, y, slot_size, _) in enumerate(
            Hud._inventory_hotbar_slot_bounds(width, height)
        ):
            Hud._draw_empty_slot(x, y, slot_size, scale)
            if index < len(hotbar_blocks) and hotbar_blocks[index] is not None:
                Hud._draw_block_texture_icon(
                    x, y, slot_size, scale, hotbar_blocks[index]
                )
            if index == selected_slot:
                border = 2 * scale
                Hud._rectangle(x, y, slot_size, border, (0.95, 0.95, 0.95))
                Hud._rectangle(
                    x, y + slot_size - border, slot_size, border, (0.95, 0.95, 0.95)
                )
                Hud._rectangle(x, y, border, slot_size, (0.95, 0.95, 0.95))
                Hud._rectangle(
                    x + slot_size - border,
                    y,
                    border,
                    slot_size,
                    (0.95, 0.95, 0.95),
                )

    @staticmethod
    def _draw_empty_slot(x: int, y: int, size: int, scale: int) -> None:
        Hud._rectangle(x, y, size, size, (0.42, 0.47, 0.53))
        Hud._rectangle(
            x + scale,
            y + scale,
            size - 2 * scale,
            size - 2 * scale,
            (0.16, 0.19, 0.24),
        )

    @staticmethod
    def _draw_hotbar(
        x: int,
        y: int,
        slot_size: int,
        gap: int,
        scale: int,
        selected_slot: int,
        hotbar_blocks: tuple[int | None, ...] | list[int | None] = HOTBAR_BLOCKS,
    ) -> None:
        outer_size = slot_size + 2 * scale
        for index, block_type in enumerate(hotbar_blocks):
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

            if block_type is not None:
                Hud._draw_block_texture_icon(
                    slot_x, y, slot_size, scale, block_type
                )

    @staticmethod
    def _draw_block_texture_icon(
        x: int, y: int, slot_size: int, scale: int, block_type: int
    ) -> None:
        pixels = _load_block_icon_pixels(block_type)
        pixel_size = max(
            1, min(3 * scale, (slot_size - 4 * scale) // len(pixels))
        )
        icon_size = pixel_size * len(pixels)
        icon_x = x + (slot_size - icon_size) // 2
        icon_y = y + (slot_size - icon_size) // 2
        for row, colors in enumerate(pixels):
            for column, color in enumerate(colors):
                Hud._rectangle(
                    icon_x + column * pixel_size,
                    icon_y + (len(pixels) - row - 1) * pixel_size,
                    pixel_size,
                    pixel_size,
                    color,
                )

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

        if block_type == BLOCK_GLOWSTONE:
            Hud._rectangle(icon_x, icon_y, icon_size, icon_size, (0.83, 0.49, 0.13))
            Hud._rectangle(
                icon_x + 2 * scale,
                icon_y + 2 * scale,
                icon_size - 4 * scale,
                icon_size - 4 * scale,
                (1.0, 0.84, 0.39),
            )
            Hud._rectangle(
                icon_x + 4 * scale,
                icon_y + 4 * scale,
                icon_size - 8 * scale,
                icon_size - 8 * scale,
                (1.0, 0.96, 0.72),
            )
            return

        if block_type == BLOCK_LANTERN:
            Hud._rectangle(icon_x, icon_y, icon_size, icon_size, (0.35, 0.23, 0.12))
            Hud._rectangle(
                icon_x + 3 * scale,
                icon_y + 2 * scale,
                icon_size - 6 * scale,
                icon_size - 4 * scale,
                (1.0, 0.68, 0.2),
            )
            Hud._rectangle(
                icon_x + 5 * scale,
                icon_y + 4 * scale,
                icon_size - 10 * scale,
                icon_size - 8 * scale,
                (1.0, 0.94, 0.62),
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
