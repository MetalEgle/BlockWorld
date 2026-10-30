#version 330 core

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec2 a_uv;
layout (location = 2) in float a_texture_layer;
layout (location = 3) in float a_face_brightness;

uniform mat4 u_mvp;
uniform mat4 u_light_mvp;

out vec2 vertex_uv;
flat out int texture_layer;
out float face_brightness;
out vec4 light_position;

void main()
{
    vec4 position = vec4(a_position, 1.0);
    gl_Position = u_mvp * position;
    light_position = u_light_mvp * position;
    vertex_uv = a_uv;
    texture_layer = int(a_texture_layer);
    face_brightness = a_face_brightness;
}
