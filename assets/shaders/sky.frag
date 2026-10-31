#version 330 core

in vec2 screen_position;

uniform mat4 u_inverse_view_projection;
uniform vec3 u_camera_position;
uniform vec3 u_sun_direction;
uniform float u_time;
uniform int u_color_grade;
uniform int u_effects_enabled;

out vec4 fragment_color;

float hash_value(vec3 point)
{
    point = fract(point * vec3(0.1031, 0.1030, 0.0973));
    point += dot(point, point.yxz + 33.33);
    return fract((point.x + point.y) * point.z);
}

float value_noise(vec3 point)
{
    vec3 cell = floor(point);
    vec3 blend = fract(point);
    blend = blend * blend * (3.0 - 2.0 * blend);
    float total = 0.0;
    for (int z = 0; z <= 1; ++z)
    {
        for (int y = 0; y <= 1; ++y)
        {
            for (int x = 0; x <= 1; ++x)
            {
                vec3 corner = vec3(float(x), float(y), float(z));
                vec3 weight = mix(1.0 - blend, blend, corner);
                total += hash_value(cell + corner)
                    * weight.x * weight.y * weight.z;
            }
        }
    }
    return total;
}

float cloud_noise(vec3 point)
{
    float total = 0.0;
    float amplitude = 0.55;
    for (int octave = 0; octave < 2; ++octave)
    {
        total += value_noise(point) * amplitude;
        point = point * 2.03 + vec3(13.7, 7.1, 5.3);
        amplitude *= 0.5;
    }
    return total;
}

void main()
{
    vec4 far_point = u_inverse_view_projection
        * vec4(screen_position, 1.0, 1.0);
    vec3 ray = normalize(far_point.xyz / far_point.w - u_camera_position);

    float sky_height = clamp(ray.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 horizon_color = vec3(0.72, 0.82, 0.96);
    vec3 zenith_color = vec3(0.12, 0.36, 0.72);
    vec3 sky_color = mix(horizon_color, zenith_color, pow(sky_height, 0.72));
    sky_color = mix(sky_color, vec3(0.98, 0.66, 0.43), 0.13 * (1.0 - sky_height));

    float sun_alignment = max(dot(ray, normalize(u_sun_direction)), 0.0);
    float sun_glow = pow(sun_alignment, 48.0) * 0.32;
    float sun_disk = smoothstep(0.99972, 0.99991, sun_alignment);
    sky_color += vec3(1.0, 0.62, 0.28) * sun_glow;
    sky_color = mix(sky_color, vec3(1.0, 0.91, 0.72), sun_disk);

    if (u_effects_enabled != 0)
    {
        vec3 cloud_position = ray * 15.0
            + vec3(u_time * 0.012, u_time * 0.004, -u_time * 0.008);
        float clouds = cloud_noise(cloud_position);
        float cloud_coverage = smoothstep(0.51, 0.72, clouds)
            * smoothstep(0.04, 0.25, ray.y);
        float cloud_light = 0.72 + 0.28 * max(dot(ray, u_sun_direction), 0.0);
        sky_color = mix(
            sky_color,
            vec3(cloud_light, cloud_light * 0.99, cloud_light * 0.96),
            cloud_coverage * 0.92
        );
    }

    if (u_color_grade == 1)
    {
        sky_color *= vec3(1.08, 1.02, 0.91);
    }
    else if (u_color_grade == 2)
    {
        sky_color *= vec3(0.90, 1.01, 1.13);
    }
    else if (u_color_grade == 3)
    {
        float luminance = dot(sky_color, vec3(0.2126, 0.7152, 0.0722));
        sky_color = mix(vec3(luminance), sky_color, 0.82);
        sky_color = (sky_color - 0.5) * 1.12 + 0.5;
        sky_color *= vec3(1.05, 1.0, 0.91);
    }

    fragment_color = vec4(clamp(sky_color, 0.0, 1.0), 1.0);
}
