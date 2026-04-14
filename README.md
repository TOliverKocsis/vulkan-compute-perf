# Vulkan Performance measurement

## Varying number of threads per workgroups
The original tutorial uses 
`commandBuffer.dispatch(PARTICLE_COUNT / 256, 1, 1);`
The compute shader declares:
[shader("compute")]
[numthreads(256,1,1)]

This means each work group runs 256 invocations (threads) of the shader.
Therefore if we want 1 invocation for every 1 particle, we need to launch enought work groups to cover all particles.
So to get the number of workgroups = Particle count / num_of_threads in the shader.

The benchmark here aims to visulize the effect of num_of_threads per workgroups from the API perspective.
On RDNA3 there  are 32 thread slots per wavefront (https://gpuopen.com/learn/occupancy-explained/), so ideally we want and exact multiplier for maximum occupancy. For example if we have 32+32+2 threads, than the third wavefront will be mostly empty, with 32-2=30 empty threads.
One SIMD can interleave multiple wavefronts. 
From the api perspective, what we can be sure of, is that the workgroup will run on a WGP that share a fast shared memory called LDS (Local Data Share - 128KiB), or 'groupshared'. If a workgroup uses a large amount of LDS, fewer workgroups can be resident on the same WGP simultaneously, reducing occupancy

So on the Vulkan API part we can vary the number of workgroups.
On the shader part we can vary the number of threads per workgroups.
On the GPU one workgroup will run on one WGP, where threads will be organized into wavefronts.

The below graph is my current understanding (claud supported), and have not been verified with AMD resource exactly.
GPU
└── Shader Engine (SE)
    └── Work Group Processor (WGP)
        └── CU × 2  (a WGP contains 2 CUs)
            └── SIMD × 4  (each CU has 4 SIMDs)
                └── 16 wavefront slots  (each SIMD can hold 16 waves)

## Notes on the code

# Fragile descriptor binding in order of declaration 
The integer passed to vk::DescriptorSetLayoutBinding(N, ...) on the CPU side must exactly match `[[vk::binding(N)]]` in the shader. There is no compiler enforcement of this correspondence in the used tutorial example. A mismatch produces silent corruption or a validation layer warning rather than a build error.

Slang will assign bindings implicitly by declaration order when `[[vk::binding(N)]]` is omitted, which is fragile — reordering declarations silently shifts all binding numbers.

Possible mitigations

Explicit annotations:  Quickest sollution is used here, declare `[[vk::binding(N)]]` on every resource in shader file (.slang).
SPIRV-Reflect: introspect the compiled SPIR-V binary at runtime to auto-discover binding layout, driving descriptor set creation from the shader itself rather than hardcoded constants.
Bindless / descriptor indexing: a modern Vulkan pattern that replaces per-binding wiring with a large descriptor array and runtime indices, sidestepping the problem at a design level.


## Building

Requires Vulkan SDK, GLFW, GLM, and CMake.
workgroup size can be customized, but must divide PARTICLE_COUNT (8192), for now its hardcoded, will be customizable in the next step also
```bash
cmake -B build -S .
cmake --build build
cd build
./VulkanComputePerf --workgroup-size 64
```

## Hardware Information

Hardware information (https://rocm.docs.amd.com/en/docs-6.3.3/reference/gpu-arch-specs.html):
Radeon RX 7900 XT: 
Wavefront Size: 32
VRAM: 20GB
Compute Units: 84
LDS (KiB): 128
Infinity Cache (MiB): 80
L2 Cache (MiB): 6
Graphics L1 Cache (KiB): 256
L0 Vector Cache (KiB): 32
L0 Scalar Cache (KiB): 16
L0 Instruction Cache (KiB): 32


## Credits

Based on [https://docs.vulkan.org/tutorial/](https://docs.vulkan.org/tutorial/latest/11_Compute_Shader.html) by Alexander Overvoorde, licensed under CC BY-SA 4.0.