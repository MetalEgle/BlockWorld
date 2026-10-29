#version 330 core

in vec2 vertex_uv;
flat in int texture_layer;
in float face_brightness;

uniform sampler2DArray u_block_textures;

out vec4 fragment_color;

void main()
{
    vec4 texel = texture(u_block_textures, vec3(vertex_uv, float(texture_layer)));
    fragment_color = vec4(texel.rgb * face_brightness, texel.a);
}
