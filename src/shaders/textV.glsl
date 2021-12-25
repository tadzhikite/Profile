#version 450
#pragma shader_stage(vertex)
layout(push_constant) uniform PER_OBJECT
{
    mat4 mvpIdx;
    mat4 imgIdx;
}pc;
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec2 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = pc.mvpIdx * vec4(inPosition, 0.f, 1.0);
    fragColor = inTexCoord;
    fragTexCoord = (pc.imgIdx * vec4(inTexCoord, 0.f, 1.f)).xy;
}
