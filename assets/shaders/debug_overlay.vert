#version 330 core

layout (location = 0) in vec2 a_position;
layout (location = 1) in vec3 a_color;

uniform vec2 u_resolution;

out vec3 vertex_color;

void main()
{
    vec2 clip_position = a_position / u_resolution * 2.0 - 1.0;
    gl_Position = vec4(clip_position, 0.0, 1.0);
    vertex_color = a_color;
}
