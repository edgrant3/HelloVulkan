#version 450

// Uniform Buffer Implementation
layout(binding = 0) uniform UniformBufferObject {
    // TODO: move to single uniform mat4 MVP
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// Vertex Buffer Implementation

// `in` variables (vertex attributes from vertex buffer)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

// `out` variables to pass to fragment shader
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

float sF = 1.75;
mat4 scale = mat4( sF, 0.0, 0.0, 0.0,
                   0.0,  sF, 0.0, 0.0,
                   0.0, 0.0,  sF, 0.0,
                   0.0, 0.0, 0.0, 1.0);

void main() {
    mat4 model = ubo.model;
    model[0][0] *= 1.5;
    model[1][1] *= 1.5;
    model[2][2] *= 1.5;

    gl_Position = ubo.proj * ubo.view * ubo.model * scale * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}

/* // Hardcoded vertices for "Drawing a Triangle" tutorial
layout(location = 0) out vec3 fragColor;

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}
*/