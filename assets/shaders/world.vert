#version 330 core

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec2 a_uv;
layout (location = 2) in float a_texture_layer;
layout (location = 3) in float a_face_brightness;

uniform mat4 u_mvp;

out vec2 vertex_uv;
flat out int texture_layer;
out float face_brightness;

void main()
{
    gl_Position = u_mvp * vec4(a_position, 1.0);
    vertex_uv = a_uv;
    texture_layer = int(a_texture_layer);
    face_brightness = a_face_brightness;
}
