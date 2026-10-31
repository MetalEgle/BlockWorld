#version 330 core

in vec2 vertex_uv;
flat in int texture_layer;
in float face_brightness;
in vec3 world_normal;
in vec3 world_position;
in vec4 light_position;

uniform sampler2DArray u_block_textures;
uniform sampler2D u_shadow_map;
uniform int u_shadows_enabled;
uniform int u_outline;
uniform int u_color_grade;
uniform vec3 u_sun_direction;
uniform vec3 u_camera_position;
uniform int u_effects_enabled;

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
    for (int x = -2; x <= 2; ++x)
    {
        for (int y = -2; y <= 2; ++y)
        {
            float closest_depth = texture(
                u_shadow_map, projected.xy + vec2(x, y) * texel_size
            ).r;
            visible_samples += projected.z - bias <= closest_depth ? 1.0 : 0.0;
        }
    }
    return mix(0.22, 1.0, visible_samples / 25.0);
}

void main()
{
    if (u_outline != 0)
    {
        fragment_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec4 texel = texture(u_block_textures, vec3(vertex_uv, float(texture_layer)));
    float sunlight = max(dot(normalize(world_normal), u_sun_direction), 0.0);
    float lighting = face_brightness * (0.25 + 0.75 * sunlight)
        * shadow_visibility();
    vec3 color = texel.rgb * lighting;
    if (u_color_grade == 1)
    {
        color *= vec3(1.10, 1.03, 0.88);
    }
    else if (u_color_grade == 2)
    {
        color *= vec3(0.88, 1.01, 1.14);
    }
    else if (u_color_grade == 3)
    {
        float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = mix(vec3(luminance), color, 0.82);
        color = (color - 0.5) * 1.12 + 0.5;
        color *= vec3(1.05, 1.0, 0.91);
    }
    if (u_effects_enabled != 0)
    {
        float distance_to_camera = distance(world_position, u_camera_position);
        float fog = smoothstep(24.0, 72.0, distance_to_camera);
        color = mix(color, vec3(0.58, 0.72, 0.86), fog);
    }
    fragment_color = vec4(clamp(color, 0.0, 1.0), texel.a);
}
