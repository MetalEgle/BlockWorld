from ctypes import c_void_p
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_DEPTH_TEST,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FALSE,
    GL_LINEAR_MIPMAP_NEAREST,
    GL_NEAREST,
    GL_REPEAT,
    GL_RGBA,
    GL_RGBA8,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D_ARRAY,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindBuffer,
    glBindTexture,
    glBindVertexArray,
    glBufferData,
    glDeleteBuffers,
    glDeleteProgram,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenerateMipmap,
    glGenBuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetUniformLocation,
    glPixelStorei,
    glTexImage3D,
    glTexParameteri,
    glTexSubImage3D,
    glUniformMatrix4fv,
    glUniform1i,
    glUseProgram,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image

from engine.graphics.hud import Hud
from engine.world.voxel_world import (
    TEXTURE_DIRT,
    TEXTURE_GRASS_SIDE,
    TEXTURE_GRASS_TOP,
    TEXTURE_SAND,
    TEXTURE_STONE,
)

_TEXTURE_FILES = (
    "Grass_Top.jfif",
    "Grass_Side.jfif",
    "Dirt.jfif",
    "Gemini_Generated_Image_cjzo89cjzo89cjzo (2).jfif",
    "Send.jfif",
)

_TEXTURE_LAYERS = (
    TEXTURE_GRASS_TOP,
    TEXTURE_GRASS_SIDE,
    TEXTURE_DIRT,
    TEXTURE_STONE,
    TEXTURE_SAND,
)


class Renderer:
    def __init__(self, vertices: np.ndarray) -> None:
        self._program = 0
        self._vertex_array = 0
        self._vertex_buffer = 0
        self._texture_array = 0
        self._vertex_count = len(vertices)
        self._mvp_location = -1
        self._texture_location = -1

        assets_dir = Path(__file__).resolve().parents[2] / "assets"
        shader_dir = assets_dir / "shaders"
        try:
            vertex_source = (shader_dir / "world.vert").read_text(encoding="utf-8")
            fragment_source = (shader_dir / "world.frag").read_text(encoding="utf-8")
            self._program = compileProgram(
                compileShader(vertex_source, GL_VERTEX_SHADER),
                compileShader(fragment_source, GL_FRAGMENT_SHADER),
            )
            self._mvp_location = glGetUniformLocation(self._program, "u_mvp")
            self._texture_location = glGetUniformLocation(
                self._program, "u_block_textures"
            )
            self._texture_array = self._load_textures(assets_dir / "textures")
            glUseProgram(self._program)
            glUniform1i(self._texture_location, 0)

            self._vertex_array = glGenVertexArrays(1)
            self._vertex_buffer = glGenBuffers(1)

            glBindVertexArray(self._vertex_array)
            glBindBuffer(GL_ARRAY_BUFFER, self._vertex_buffer)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

            stride = 7 * vertices.itemsize
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(
                1,
                2,
                GL_FLOAT,
                GL_FALSE,
                stride,
                c_void_p(3 * vertices.itemsize),
            )
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(
                2,
                1,
                GL_FLOAT,
                GL_FALSE,
                stride,
                c_void_p(5 * vertices.itemsize),
            )
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(
                3,
                1,
                GL_FLOAT,
                GL_FALSE,
                stride,
                c_void_p(6 * vertices.itemsize),
            )
            glEnableVertexAttribArray(3)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            glEnable(GL_DEPTH_TEST)
        except Exception:
            self.close()
            raise

    def render(
        self, view: np.ndarray, projection: np.ndarray, width: int, height: int
    ) -> None:
        if width <= 0 or height <= 0:
            return

        mvp = projection @ view

        glUseProgram(self._program)
        glUniformMatrix4fv(self._mvp_location, 1, GL_FALSE, mvp.flatten(order="F"))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._texture_array)
        glBindVertexArray(self._vertex_array)
        glDrawArrays(GL_TRIANGLES, 0, self._vertex_count)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    @staticmethod
    def _load_textures(texture_dir: Path) -> int:
        images: list[np.ndarray] = []
        expected_size: tuple[int, int] | None = None

        for filename in _TEXTURE_FILES:
            path = texture_dir / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"Не найдена текстура блока: {path}. "
                    "Проверьте имя файла в assets/textures."
                )
            with Image.open(path) as image:
                pixels = np.ascontiguousarray(
                    np.flipud(np.asarray(image.convert("RGBA"), dtype=np.uint8))
                )
            size = (pixels.shape[1], pixels.shape[0])
            if expected_size is None:
                expected_size = size
            elif size != expected_size:
                raise ValueError(
                    f"Размер текстуры {path.name} — {size}, "
                    f"ожидался размер {expected_size}."
                )
            images.append(pixels)

        if expected_size is None:
            raise RuntimeError("Не удалось загрузить текстуры блоков.")

        texture_array = glGenTextures(1)
        try:
            glBindTexture(GL_TEXTURE_2D_ARRAY, texture_array)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage3D(
                GL_TEXTURE_2D_ARRAY,
                0,
                GL_RGBA8,
                expected_size[0],
                expected_size[1],
                len(images),
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                None,
            )
            for image, layer in zip(images, _TEXTURE_LAYERS):
                glTexSubImage3D(
                    GL_TEXTURE_2D_ARRAY,
                    0,
                    0,
                    0,
                    layer,
                    expected_size[0],
                    expected_size[1],
                    1,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    image,
                )
            glTexParameteri(
                GL_TEXTURE_2D_ARRAY,
                GL_TEXTURE_MIN_FILTER,
                GL_LINEAR_MIPMAP_NEAREST,
            )
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glGenerateMipmap(GL_TEXTURE_2D_ARRAY)
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
            return int(texture_array)
        except Exception:
            glDeleteTextures([texture_array])
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
            raise

    def update_mesh(self, vertices: np.ndarray) -> None:
        self._vertex_count = len(vertices)
        glBindBuffer(GL_ARRAY_BUFFER, self._vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    @staticmethod
    def draw_hud(width: int, height: int, selected_slot: int) -> None:
        Hud.draw(width, height, selected_slot)

    def close(self) -> None:
        if self._vertex_buffer:
            glDeleteBuffers(1, [self._vertex_buffer])
            self._vertex_buffer = 0
        if self._vertex_array:
            glDeleteVertexArrays(1, [self._vertex_array])
            self._vertex_array = 0
        if self._program:
            glDeleteProgram(self._program)
            self._program = 0
        if self._texture_array:
            glDeleteTextures([self._texture_array])
            self._texture_array = 0
