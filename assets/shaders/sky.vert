#version 330 core

out vec2 screen_position;

void main()
{
    vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );
    screen_position = positions[gl_VertexID];
    gl_Position = vec4(screen_position, 1.0, 1.0);
}
