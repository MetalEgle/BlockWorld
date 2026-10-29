from ctypes import c_void_p
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_DEPTH_TEST,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FALSE,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_VERTEX_SHADER,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glDeleteBuffers,
    glDeleteProgram,
    glDeleteVertexArrays,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetUniformLocation,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader

from engine.graphics.hud import Hud


class Renderer:
    def __init__(self, vertices: np.ndarray) -> None:
        self._program = 0
        self._vertex_array = 0
        self._vertex_buffer = 0
        self._vertex_count = len(vertices)
        self._mvp_location = -1

        shader_dir = Path(__file__).resolve().parents[2] / "assets" / "shaders"
        try:
            vertex_source = (shader_dir / "world.vert").read_text(encoding="utf-8")
            fragment_source = (shader_dir / "world.frag").read_text(encoding="utf-8")
            self._program = compileProgram(
                compileShader(vertex_source, GL_VERTEX_SHADER),
                compileShader(fragment_source, GL_FRAGMENT_SHADER),
            )
            self._mvp_location = glGetUniformLocation(self._program, "u_mvp")
            self._vertex_array = glGenVertexArrays(1)
            self._vertex_buffer = glGenBuffers(1)

            glBindVertexArray(self._vertex_array)
            glBindBuffer(GL_ARRAY_BUFFER, self._vertex_buffer)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

            stride = 6 * vertices.itemsize
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(
                1,
                3,
                GL_FLOAT,
                GL_FALSE,
                stride,
                c_void_p(3 * vertices.itemsize),
            )
            glEnableVertexAttribArray(1)
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
        glBindVertexArray(self._vertex_array)
        glDrawArrays(GL_TRIANGLES, 0, self._vertex_count)
        glBindVertexArray(0)

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
