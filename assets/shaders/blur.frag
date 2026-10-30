#version 330 core

in vec2 v_tex_coord;

out vec4 frag_color;

uniform sampler2D u_scene;
uniform vec2 u_texel_size;

void main()
{
    const float weights[4] = float[](0.227027, 0.1945946, 0.1216216, 0.054054);
    vec3 color = texture(u_scene, v_tex_coord).rgb * weights[0];

    for (int offset = 1; offset <= 3; ++offset)
    {
        vec2 delta = u_texel_size * float(offset);
        float weight = weights[offset] * 0.5;
        color += texture(u_scene, v_tex_coord + vec2(delta.x, 0.0)).rgb * weight;
        color += texture(u_scene, v_tex_coord - vec2(delta.x, 0.0)).rgb * weight;
        color += texture(u_scene, v_tex_coord + vec2(0.0, delta.y)).rgb * weight;
        color += texture(u_scene, v_tex_coord - vec2(0.0, delta.y)).rgb * weight;
    }

    frag_color = vec4(color * 0.55, 1.0);
}
