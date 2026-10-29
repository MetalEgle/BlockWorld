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
