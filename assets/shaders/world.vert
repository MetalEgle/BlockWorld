#version 330 core

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec2 a_uv;
layout (location = 2) in float a_texture_layer;
layout (location = 3) in float a_face_brightness;
layout (location = 4) in vec3 a_normal;

uniform mat4 u_mvp;
uniform mat4 u_light_mvp;
uniform int u_outline;
uniform vec3 u_outline_offset;

out vec2 vertex_uv;
flat out int texture_layer;
out float face_brightness;
out vec3 world_normal;
out vec3 world_position;
out vec4 light_position;

void main()
{
    vec4 position = vec4(a_position, 1.0);
    if (u_outline != 0)
    {
        position.xyz += u_outline_offset;
    }
    gl_Position = u_mvp * position;
    light_position = u_light_mvp * position;
    vertex_uv = a_uv;
    texture_layer = int(a_texture_layer);
    face_brightness = a_face_brightness;
    world_normal = a_normal;
    world_position = a_position;
}
