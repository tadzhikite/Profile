#version 450
#pragma shader_stage(fragment)

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec2 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    if(texture(texSampler, fragTexCoord).r < .1)
        discard;
    outColor = texture(texSampler, fragTexCoord).r* vec4(1.0f, 1.f, 1.f, 1.f);
}
