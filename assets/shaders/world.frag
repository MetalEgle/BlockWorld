#version 330 core

in vec2 vertex_uv;
flat in int texture_layer;
in float face_brightness;
in vec4 light_position;

uniform sampler2DArray u_block_textures;
uniform sampler2D u_shadow_map;
uniform int u_shadows_enabled;

out vec4 fragment_color;

float shadow_visibility()
{
    if (u_shadows_enabled == 0)
    {
        return 1.0;
    }

    vec3 projected = light_position.xyz / light_position.w;
    projected = projected * 0.5 + 0.5;
    if (projected.z > 1.0 || projected.x < 0.0 || projected.x > 1.0
        || projected.y < 0.0 || projected.y > 1.0)
    {
        return 1.0;
    }

    float bias = 0.0008 + 0.001 * (1.0 - face_brightness);
    vec2 texel_size = 1.0 / vec2(textureSize(u_shadow_map, 0));
    float visible_samples = 0.0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float closest_depth = texture(
                u_shadow_map, projected.xy + vec2(x, y) * texel_size
            ).r;
            visible_samples += projected.z - bias <= closest_depth ? 1.0 : 0.0;
        }
    }
    return mix(0.58, 1.0, visible_samples / 9.0);
}

void main()
{
    vec4 texel = texture(u_block_textures, vec3(vertex_uv, float(texture_layer)));
    float lighting = face_brightness * shadow_visibility();
    fragment_color = vec4(texel.rgb * lighting, texel.a);
}
