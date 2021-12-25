#version 450
#pragma shader_stage(vertex)
layout(push_constant) uniform PER_OBJECT
{
    mat4 mvpIdx;
    mat4 imgIdx;
}pc2;
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = pc2.mvpIdx * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = (pc2.imgIdx * vec4(inTexCoord, 0.f, 1.f)).xy;
}
