from ctypes import c_void_p
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_CLAMP_TO_EDGE,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_COMPONENT,
    GL_DEPTH_COMPONENT24,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DYNAMIC_DRAW,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FALSE,
    GL_LINES,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_LINEAR_MIPMAP_NEAREST,
    GL_NONE,
    GL_NEAREST,
    GL_POLYGON_OFFSET_FILL,
    GL_REPEAT,
    GL_RGBA,
    GL_RGBA8,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
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
    glDeleteFramebuffers,
    glDeleteProgram,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDrawArrays,
    glDrawBuffer,
    glReadBuffer,
    glClear,
    glDisable,
    glEnable,
    glEnableVertexAttribArray,
    glGenerateMipmap,
    glGenBuffers,
    glGenFramebuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetUniformLocation,
    glLineWidth,
    glCheckFramebufferStatus,
    glCopyTexImage2D,
    glFramebufferTexture2D,
    glPixelStorei,
    glPolygonOffset,
    glTexImage3D,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage3D,
    glUniformMatrix4fv,
    glUniform1i,
    glUniform1f,
    glUniform2f,
    glUniform3f,
    glBindFramebuffer,
    glUseProgram,
    glViewport,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image

from engine.diagnostics import PerformanceStats
from engine.graphics.hud import Hud
from engine.world.voxel_world import (
    CHUNK_SIZE,
    TEXTURE_DIRT,
    TEXTURE_GRASS_SIDE,
    TEXTURE_GRASS_TOP,
    TEXTURE_LEAF,
    TEXTURE_SAND,
    TEXTURE_STONE,
    TEXTURE_STONE_BRICKS,
    TEXTURE_STONE_COBBLE,
    TEXTURE_WOOD_SIDE,
    TEXTURE_WOOD_TOP,
    TEXTURE_WOOD_BOARD,
)

_TEXTURE_FILES = (
    "Grass_Top.jfif",
    "Grass_Side.jfif",
    "Dirt.jfif",
    "Stone.jfif",
    "Send.jfif",
    "Leaf.jfif",
    "Stone_Briks.jfif",
    "Stone_Cobble.jfif",
    "Wood_Side.jfif",
    "Wood_Top.jfif",
    "Wood_Board.jfif",
)

_TEXTURE_LAYERS = (
    TEXTURE_GRASS_TOP,
    TEXTURE_GRASS_SIDE,
    TEXTURE_DIRT,
    TEXTURE_STONE,
    TEXTURE_SAND,
    TEXTURE_LEAF,
    TEXTURE_STONE_BRICKS,
    TEXTURE_STONE_COBBLE,
    TEXTURE_WOOD_SIDE,
    TEXTURE_WOOD_TOP,
    TEXTURE_WOOD_BOARD,
)

_SUN_DIRECTION = np.asarray((0.65, 1.5, 0.65), dtype=np.float32)
_SUN_DIRECTION /= np.linalg.norm(_SUN_DIRECTION)


class Renderer:
    _SHADOW_RESOLUTIONS = (512, 512, 1024)

    def __init__(self, vertices: np.ndarray, surface_heights: np.ndarray) -> None:
        self._program = 0
        self._vertex_array = 0
        self._vertex_buffer = 0
        self._outline_vertex_array = 0
        self._outline_vertex_buffer = 0
        self._chunk_grid_vertex_array = 0
        self._chunk_grid_vertex_buffer = 0
        self._chunk_grid_vertex_count = 0
        self._debug_program = 0
        self._debug_vertex_array = 0
        self._debug_vertex_buffer = 0
        self._sky_program = 0
        self._sky_vertex_array = 0
        self._texture_array = 0
        self._blur_program = 0
        self._blur_vertex_array = 0
        self._blur_vertex_buffer = 0
        self._blur_texture = 0
        self._shadow_program = 0
        self._shadow_texture = 0
        self._shadow_framebuffer = 0
        self._shadow_resolution = 0
        self._shadow_matrix_location = -1
        self._vertex_count = len(vertices)
        self._mvp_location = -1
        self._texture_location = -1
        self._blur_texel_size_location = -1
        self._light_mvp_location = -1
        self._shadow_enabled_location = -1
        self._sun_direction_location = -1
        self._outline_location = -1
        self._outline_offset_location = -1
        self._debug_resolution_location = -1
        self._sky_inverse_view_projection_location = -1
        self._sky_camera_position_location = -1
        self._sky_sun_direction_location = -1
        self._sky_time_location = -1
        self._sky_color_grade_location = -1
        self._sky_effects_enabled_location = -1
        self._color_grade_location = -1
        self._effects_enabled_location = -1
        self._camera_position_location = -1
        self._shadow_matrix = np.eye(4, dtype=np.float32)
        self._shadow_map_dirty = True
        self.texture_quality = 1
        self.shadow_quality = 1
        self.color_grade = 0
        self.effects_enabled = True

        assets_dir = Path(__file__).resolve().parents[2] / "assets"
        shader_dir = assets_dir / "shaders"
        try:
            vertex_source = (shader_dir / "world.vert").read_text(encoding="utf-8")
            fragment_source = (shader_dir / "world.frag").read_text(encoding="utf-8")
            self._program = compileProgram(
                compileShader(vertex_source, GL_VERTEX_SHADER),
                compileShader(fragment_source, GL_FRAGMENT_SHADER),
                # PyOpenGL validates before the two sampler uniforms get distinct units.
                validate=False,
            )
            self._mvp_location = glGetUniformLocation(self._program, "u_mvp")
            self._texture_location = glGetUniformLocation(
                self._program, "u_block_textures"
            )
            self._light_mvp_location = glGetUniformLocation(
                self._program, "u_light_mvp"
            )
            self._shadow_enabled_location = glGetUniformLocation(
                self._program, "u_shadows_enabled"
            )
            self._sun_direction_location = glGetUniformLocation(
                self._program, "u_sun_direction"
            )
            self._outline_location = glGetUniformLocation(
                self._program, "u_outline"
            )
            self._outline_offset_location = glGetUniformLocation(
                self._program, "u_outline_offset"
            )
            self._color_grade_location = glGetUniformLocation(
                self._program, "u_color_grade"
            )
            self._effects_enabled_location = glGetUniformLocation(
                self._program, "u_effects_enabled"
            )
            self._camera_position_location = glGetUniformLocation(
                self._program, "u_camera_position"
            )
            self._texture_array = self._load_textures(assets_dir / "textures")
            glUseProgram(self._program)
            glUniform1i(self._texture_location, 0)
            glUniform1i(glGetUniformLocation(self._program, "u_shadow_map"), 1)

            shadow_vertex_source = (shader_dir / "shadow.vert").read_text(
                encoding="utf-8"
            )
            shadow_fragment_source = (shader_dir / "shadow.frag").read_text(
                encoding="utf-8"
            )
            self._shadow_program = compileProgram(
                compileShader(shadow_vertex_source, GL_VERTEX_SHADER),
                compileShader(shadow_fragment_source, GL_FRAGMENT_SHADER),
            )
            self._shadow_matrix_location = glGetUniformLocation(
                self._shadow_program, "u_light_mvp"
            )
            self._shadow_matrix = self._make_shadow_matrix(vertices)
            self._create_shadow_map(self._SHADOW_RESOLUTIONS[self.shadow_quality])

            sky_vertex_source = (shader_dir / "sky.vert").read_text(
                encoding="utf-8"
            )
            sky_fragment_source = (shader_dir / "sky.frag").read_text(
                encoding="utf-8"
            )
            self._sky_program = compileProgram(
                compileShader(sky_vertex_source, GL_VERTEX_SHADER),
                compileShader(sky_fragment_source, GL_FRAGMENT_SHADER),
            )
            self._sky_inverse_view_projection_location = glGetUniformLocation(
                self._sky_program, "u_inverse_view_projection"
            )
            self._sky_camera_position_location = glGetUniformLocation(
                self._sky_program, "u_camera_position"
            )
            self._sky_sun_direction_location = glGetUniformLocation(
                self._sky_program, "u_sun_direction"
            )
            self._sky_time_location = glGetUniformLocation(
                self._sky_program, "u_time"
            )
            self._sky_color_grade_location = glGetUniformLocation(
                self._sky_program, "u_color_grade"
            )
            self._sky_effects_enabled_location = glGetUniformLocation(
                self._sky_program, "u_effects_enabled"
            )
            self._sky_vertex_array = glGenVertexArrays(1)

            blur_vertex_source = (shader_dir / "blur.vert").read_text(
                encoding="utf-8"
            )
            blur_fragment_source = (shader_dir / "blur.frag").read_text(
                encoding="utf-8"
            )
            self._blur_program = compileProgram(
                compileShader(blur_vertex_source, GL_VERTEX_SHADER),
                compileShader(blur_fragment_source, GL_FRAGMENT_SHADER),
            )
            self._blur_texel_size_location = glGetUniformLocation(
                self._blur_program, "u_texel_size"
            )
            blur_texture_location = glGetUniformLocation(
                self._blur_program, "u_scene"
            )
            glUseProgram(self._blur_program)
            glUniform1i(blur_texture_location, 0)
            self._blur_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self._blur_texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glBindTexture(GL_TEXTURE_2D, 0)

            blur_vertices = np.array(
                (
                    -1.0, -1.0, 0.0, 0.0,
                    1.0, -1.0, 1.0, 0.0,
                    1.0, 1.0, 1.0, 1.0,
                    -1.0, -1.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0,
                    -1.0, 1.0, 0.0, 1.0,
                ),
                dtype=np.float32,
            )
            self._blur_vertex_array = glGenVertexArrays(1)
            self._blur_vertex_buffer = glGenBuffers(1)
            glBindVertexArray(self._blur_vertex_array)
            glBindBuffer(GL_ARRAY_BUFFER, self._blur_vertex_buffer)
            glBufferData(
                GL_ARRAY_BUFFER,
                blur_vertices.nbytes,
                blur_vertices,
                GL_STATIC_DRAW,
            )
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, None)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(
                1,
                2,
                GL_FLOAT,
                GL_FALSE,
                4 * vertices.itemsize,
                c_void_p(2 * vertices.itemsize),
            )
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)

            self._vertex_array = glGenVertexArrays(1)
            self._vertex_buffer = glGenBuffers(1)

            glBindVertexArray(self._vertex_array)
            glBindBuffer(GL_ARRAY_BUFFER, self._vertex_buffer)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

            stride = 10 * vertices.itemsize
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
            glVertexAttribPointer(
                4,
                3,
                GL_FLOAT,
                GL_FALSE,
                stride,
                c_void_p(7 * vertices.itemsize),
            )
            glEnableVertexAttribArray(4)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            glEnable(GL_DEPTH_TEST)

            chunk_grid_vertices = self._build_chunk_grid_vertices(surface_heights)
            self._chunk_grid_vertex_count = len(chunk_grid_vertices)
            self._chunk_grid_vertex_array = glGenVertexArrays(1)
            self._chunk_grid_vertex_buffer = glGenBuffers(1)
            glBindVertexArray(self._chunk_grid_vertex_array)
            glBindBuffer(GL_ARRAY_BUFFER, self._chunk_grid_vertex_buffer)
            glBufferData(
                GL_ARRAY_BUFFER,
                chunk_grid_vertices.nbytes,
                chunk_grid_vertices,
                GL_STATIC_DRAW,
            )
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, None)
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)

            debug_vertex_source = (shader_dir / "debug_overlay.vert").read_text(
                encoding="utf-8"
            )
            debug_fragment_source = (shader_dir / "debug_overlay.frag").read_text(
                encoding="utf-8"
            )
            self._debug_program = compileProgram(
                compileShader(debug_vertex_source, GL_VERTEX_SHADER),
                compileShader(debug_fragment_source, GL_FRAGMENT_SHADER),
            )
            self._debug_resolution_location = glGetUniformLocation(
                self._debug_program, "u_resolution"
            )
            self._debug_vertex_array = glGenVertexArrays(1)
            self._debug_vertex_buffer = glGenBuffers(1)
            glBindVertexArray(self._debug_vertex_array)
            glBindBuffer(GL_ARRAY_BUFFER, self._debug_vertex_buffer)
            glVertexAttribPointer(
                0, 2, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, None
            )
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(
                1,
                3,
                GL_FLOAT,
                GL_FALSE,
                5 * vertices.itemsize,
                c_void_p(2 * vertices.itemsize),
            )
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)

            outline_vertices = np.array(
                (
                    0, 0, 0, 1, 0, 0,
                    1, 0, 0, 1, 1, 0,
                    1, 1, 0, 0, 1, 0,
                    0, 1, 0, 0, 0, 0,
                    0, 0, 1, 1, 0, 1,
                    1, 0, 1, 1, 1, 1,
                    1, 1, 1, 0, 1, 1,
                    0, 1, 1, 0, 0, 1,
                    0, 0, 0, 0, 0, 1,
                    1, 0, 0, 1, 0, 1,
                    1, 1, 0, 1, 1, 1,
                    0, 1, 0, 0, 1, 1,
                ),
                dtype=np.float32,
            )
            outline_vertices = outline_vertices.reshape(-1, 3)
            outline_vertices = np.where(
                outline_vertices == 0.0, -0.002, 1.002
            ).astype(np.float32)
            self._outline_vertex_array = glGenVertexArrays(1)
            self._outline_vertex_buffer = glGenBuffers(1)
            glBindVertexArray(self._outline_vertex_array)
            glBindBuffer(GL_ARRAY_BUFFER, self._outline_vertex_buffer)
            glBufferData(
                GL_ARRAY_BUFFER,
                outline_vertices.nbytes,
                outline_vertices,
                GL_STATIC_DRAW,
            )
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, None)
            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
        except Exception:
            self.close()
            raise

    def render(
        self,
        view: np.ndarray,
        projection: np.ndarray,
        width: int,
        height: int,
        outline_position: tuple[int, int, int] | None = None,
        show_chunk_grid: bool = False,
        time_seconds: float = 0.0,
    ) -> None:
        if width <= 0 or height <= 0:
            return

        if self.shadow_quality and self._shadow_map_dirty:
            self._render_shadow_map(width, height)

        mvp = projection @ view
        inverse_view_projection = np.linalg.inv(mvp)
        camera_position = np.linalg.inv(view)[:3, 3]
        self._draw_sky(
            inverse_view_projection,
            camera_position,
            time_seconds,
        )

        glUseProgram(self._program)
        glUniformMatrix4fv(self._mvp_location, 1, GL_FALSE, mvp.flatten(order="F"))
        glUniformMatrix4fv(
            self._light_mvp_location, 1, GL_FALSE, self._shadow_matrix.flatten(order="F")
        )
        glUniform1i(self._shadow_enabled_location, int(self.shadow_quality > 0))
        glUniform1i(self._outline_location, 0)
        glUniform1i(self._color_grade_location, self.color_grade)
        glUniform1i(self._effects_enabled_location, int(self.effects_enabled))
        glUniform3f(
            self._camera_position_location,
            float(camera_position[0]),
            float(camera_position[1]),
            float(camera_position[2]),
        )
        glUniform3f(
            self._sun_direction_location,
            float(_SUN_DIRECTION[0]),
            float(_SUN_DIRECTION[1]),
            float(_SUN_DIRECTION[2]),
        )
        glUniform3f(self._outline_offset_location, 0.0, 0.0, 0.0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._texture_array)
        glActiveTexture(GL_TEXTURE0 + 1)
        glBindTexture(GL_TEXTURE_2D, self._shadow_texture)
        glBindVertexArray(self._vertex_array)
        glDrawArrays(GL_TRIANGLES, 0, self._vertex_count)
        glBindVertexArray(0)
        if show_chunk_grid and self._chunk_grid_vertex_count:
            glUniform1i(self._outline_location, 1)
            glUniform3f(self._outline_offset_location, 0.0, 0.0, 0.0)
            glBindVertexArray(self._chunk_grid_vertex_array)
            glDrawArrays(GL_LINES, 0, self._chunk_grid_vertex_count)
            glBindVertexArray(0)
        if outline_position is not None:
            glUniform1i(self._outline_location, 1)
            glUniform3f(
                self._outline_offset_location,
                float(outline_position[0]),
                float(outline_position[1]),
                float(outline_position[2]),
            )
            glLineWidth(2.0)
            glBindVertexArray(self._outline_vertex_array)
            glDrawArrays(GL_LINES, 0, 24)
            glBindVertexArray(0)
            glLineWidth(1.0)
        glActiveTexture(GL_TEXTURE0 + 1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def _draw_sky(
        self,
        inverse_view_projection: np.ndarray,
        camera_position: np.ndarray,
        time_seconds: float,
    ) -> None:
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self._sky_program)
        glUniformMatrix4fv(
            self._sky_inverse_view_projection_location,
            1,
            GL_FALSE,
            inverse_view_projection.flatten(order="F"),
        )
        glUniform3f(
            self._sky_camera_position_location,
            float(camera_position[0]),
            float(camera_position[1]),
            float(camera_position[2]),
        )
        glUniform3f(
            self._sky_sun_direction_location,
            float(_SUN_DIRECTION[0]),
            float(_SUN_DIRECTION[1]),
            float(_SUN_DIRECTION[2]),
        )
        glUniform1f(self._sky_time_location, time_seconds)
        glUniform1i(self._sky_color_grade_location, self.color_grade)
        glUniform1i(
            self._sky_effects_enabled_location, int(self.effects_enabled)
        )
        glBindVertexArray(self._sky_vertex_array)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)

    @staticmethod
    def _build_chunk_grid_vertices(surface_heights: np.ndarray) -> np.ndarray:
        world_size = surface_heights.shape[0]
        coordinates = list(range(0, world_size, CHUNK_SIZE))
        coordinates.append(world_size)
        vertices: list[tuple[float, float, float]] = []

        for coordinate in coordinates:
            left_x = max(coordinate - 1, 0)
            right_x = min(coordinate, world_size - 1)
            for z in range(world_size):
                next_z = min(z + 1, world_size - 1)
                start_y = max(
                    int(surface_heights[left_x, z]),
                    int(surface_heights[right_x, z]),
                ) + 1.02
                end_y = max(
                    int(surface_heights[left_x, next_z]),
                    int(surface_heights[right_x, next_z]),
                ) + 1.02
                vertices.extend(
                    (
                        (float(coordinate), start_y, float(z)),
                        (float(coordinate), end_y, float(z + 1)),
                    )
                )

            near_z = max(coordinate - 1, 0)
            far_z = min(coordinate, world_size - 1)
            for x in range(world_size):
                next_x = min(x + 1, world_size - 1)
                start_y = max(
                    int(surface_heights[x, near_z]),
                    int(surface_heights[x, far_z]),
                ) + 1.02
                end_y = max(
                    int(surface_heights[next_x, near_z]),
                    int(surface_heights[next_x, far_z]),
                ) + 1.02
                vertices.extend(
                    (
                        (float(x), start_y, float(coordinate)),
                        (float(x + 1), end_y, float(coordinate)),
                    )
                )

        return np.asarray(vertices, dtype=np.float32).reshape(-1, 3)

    @staticmethod
    def _make_shadow_matrix(vertices: np.ndarray) -> np.ndarray:
        if len(vertices):
            minimum = vertices[:, :3].min(axis=0)
            maximum = vertices[:, :3].max(axis=0)
            center = (minimum + maximum) * 0.5
            radius = max(float(np.max(maximum - minimum)) * 0.75, 4.0)
        else:
            center = np.zeros(3, dtype=np.float32)
            radius = 16.0

        eye = center + radius * _SUN_DIRECTION
        forward = center - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array((0.0, 1.0, 0.0)))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, eye)
        view[1, 3] = -np.dot(up, eye)
        view[2, 3] = np.dot(forward, eye)

        near_plane = 0.1
        far_plane = radius * 4.0
        projection = np.zeros((4, 4), dtype=np.float32)
        projection[0, 0] = 1.0 / radius
        projection[1, 1] = 1.0 / radius
        projection[2, 2] = -2.0 / (far_plane - near_plane)
        projection[2, 3] = -(far_plane + near_plane) / (far_plane - near_plane)
        projection[3, 3] = 1.0
        return projection @ view

    def _create_shadow_map(self, resolution: int) -> None:
        texture = int(glGenTextures(1))
        framebuffer = int(glGenFramebuffers(1))
        try:
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_DEPTH_COMPONENT24,
                resolution,
                resolution,
                0,
                GL_DEPTH_COMPONENT,
                GL_FLOAT,
                None,
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT,
                GL_TEXTURE_2D,
                texture,
                0,
            )
            glDrawBuffer(GL_NONE)
            glReadBuffer(GL_NONE)
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("Не удалось создать буфер карты теней OpenGL.")
        except Exception:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glBindTexture(GL_TEXTURE_2D, 0)
            if framebuffer:
                glDeleteFramebuffers(1, [framebuffer])
            if texture:
                glDeleteTextures([texture])
            raise

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        old_texture, old_framebuffer = self._shadow_texture, self._shadow_framebuffer
        self._shadow_texture, self._shadow_framebuffer = texture, framebuffer
        self._shadow_resolution = resolution
        if old_framebuffer:
            glDeleteFramebuffers(1, [old_framebuffer])
        if old_texture:
            glDeleteTextures([old_texture])
        self._shadow_map_dirty = True

    def _render_shadow_map(self, width: int, height: int) -> None:
        resolution = self._shadow_resolution
        glBindFramebuffer(GL_FRAMEBUFFER, self._shadow_framebuffer)
        glViewport(0, 0, resolution, resolution)
        try:
            glClear(GL_DEPTH_BUFFER_BIT)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(1.1, 2.0)
            glUseProgram(self._shadow_program)
            glUniformMatrix4fv(
                self._shadow_matrix_location,
                1,
                GL_FALSE,
                self._shadow_matrix.flatten(order="F"),
            )
            glBindVertexArray(self._vertex_array)
            glDrawArrays(GL_TRIANGLES, 0, self._vertex_count)
            glBindVertexArray(0)
        finally:
            glDisable(GL_POLYGON_OFFSET_FILL)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, width, height)
        self._shadow_map_dirty = False

    def set_texture_quality(self, quality: int) -> None:
        if quality not in (0, 1, 2):
            raise ValueError(f"Неизвестный уровень качества текстур: {quality}.")
        if quality == self.texture_quality:
            return
        self.texture_quality = quality
        min_filter = (
            GL_NEAREST
            if quality == 0
            else GL_LINEAR_MIPMAP_NEAREST
            if quality == 1
            else GL_LINEAR_MIPMAP_LINEAR
        )
        mag_filter = GL_LINEAR if quality == 2 else GL_NEAREST
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._texture_array)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, min_filter)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, mag_filter)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def set_shadow_quality(self, quality: int) -> None:
        if quality not in (0, 1, 2):
            raise ValueError(f"Неизвестный уровень качества теней: {quality}.")
        if quality == self.shadow_quality:
            return
        resolution = self._SHADOW_RESOLUTIONS[quality]
        if quality and resolution != self._shadow_resolution:
            self._create_shadow_map(resolution)
        self.shadow_quality = quality
        self._shadow_map_dirty = True

    def set_color_grade(self, grade: int) -> None:
        if grade not in (0, 1, 2, 3):
            raise ValueError(f"Неизвестный вариант цветокоррекции: {grade}.")
        self.color_grade = grade

    def set_effects_enabled(self, enabled: bool) -> None:
        self.effects_enabled = enabled

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
        self._shadow_matrix = self._make_shadow_matrix(vertices)
        self._shadow_map_dirty = True
        glBindBuffer(GL_ARRAY_BUFFER, self._vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    @staticmethod
    def draw_hud(width: int, height: int, selected_slot: int) -> None:
        Hud.draw(width, height, selected_slot)

    def draw_debug_overlay(
        self,
        width: int, height: int, stats: PerformanceStats, block_count: int
    ) -> None:
        vertices = Hud.build_debug_overlay_vertices(
            width, height, stats, block_count
        )
        if not len(vertices):
            return

        glDisable(GL_DEPTH_TEST)
        glUseProgram(self._debug_program)
        glUniform2f(self._debug_resolution_location, float(width), float(height))
        glBindVertexArray(self._debug_vertex_array)
        glBindBuffer(GL_ARRAY_BUFFER, self._debug_vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, len(vertices))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)

    def draw_menu(
        self,
        width: int,
        height: int,
        settings_open: bool = False,
        hovered_button: int | None = None,
        texture_quality: int = 1,
        shadow_quality: int = 1,
        color_grade: int = 0,
        effects_enabled: bool = True,
    ) -> None:
        self._draw_blurred_background(width, height)
        Hud.draw_menu(
            width,
            height,
            settings_open,
            hovered_button,
            texture_quality,
            shadow_quality,
            color_grade,
            effects_enabled,
        )

    def _draw_blurred_background(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._blur_texture)
        glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 0, 0, width, height, 0)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self._blur_program)
        glUniform2f(
            self._blur_texel_size_location,
            2.5 / width,
            2.5 / height,
        )
        glBindVertexArray(self._blur_vertex_array)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_DEPTH_TEST)

    def close(self) -> None:
        if self._chunk_grid_vertex_buffer:
            glDeleteBuffers(1, [self._chunk_grid_vertex_buffer])
            self._chunk_grid_vertex_buffer = 0
        if self._chunk_grid_vertex_array:
            glDeleteVertexArrays(1, [self._chunk_grid_vertex_array])
            self._chunk_grid_vertex_array = 0
        if self._debug_vertex_buffer:
            glDeleteBuffers(1, [self._debug_vertex_buffer])
            self._debug_vertex_buffer = 0
        if self._debug_vertex_array:
            glDeleteVertexArrays(1, [self._debug_vertex_array])
            self._debug_vertex_array = 0
        if self._debug_program:
            glDeleteProgram(self._debug_program)
            self._debug_program = 0
        if self._sky_vertex_array:
            glDeleteVertexArrays(1, [self._sky_vertex_array])
            self._sky_vertex_array = 0
        if self._sky_program:
            glDeleteProgram(self._sky_program)
            self._sky_program = 0
        if self._outline_vertex_buffer:
            glDeleteBuffers(1, [self._outline_vertex_buffer])
            self._outline_vertex_buffer = 0
        if self._outline_vertex_array:
            glDeleteVertexArrays(1, [self._outline_vertex_array])
            self._outline_vertex_array = 0
        if self._vertex_buffer:
            glDeleteBuffers(1, [self._vertex_buffer])
            self._vertex_buffer = 0
        if self._vertex_array:
            glDeleteVertexArrays(1, [self._vertex_array])
            self._vertex_array = 0
        if self._program:
            glDeleteProgram(self._program)
            self._program = 0
        if self._blur_program:
            glDeleteProgram(self._blur_program)
            self._blur_program = 0
        if self._blur_vertex_buffer:
            glDeleteBuffers(1, [self._blur_vertex_buffer])
            self._blur_vertex_buffer = 0
        if self._blur_vertex_array:
            glDeleteVertexArrays(1, [self._blur_vertex_array])
            self._blur_vertex_array = 0
        if self._blur_texture:
            glDeleteTextures([self._blur_texture])
            self._blur_texture = 0
        if self._shadow_program:
            glDeleteProgram(self._shadow_program)
            self._shadow_program = 0
        if self._shadow_framebuffer:
            glDeleteFramebuffers(1, [self._shadow_framebuffer])
            self._shadow_framebuffer = 0
        if self._shadow_texture:
            glDeleteTextures([self._shadow_texture])
            self._shadow_texture = 0
        if self._texture_array:
            glDeleteTextures([self._texture_array])
            self._texture_array = 0
