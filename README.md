# Vulkan Compute Performance — RDNA3 Workgroup Sweep

<p align="center">
  <img src="assets/particle_demo.gif" alt="Particle system demo" width="500"/>
</p>

> **Central question:** how does workgroup size and particle count affect GPU compute throughput on RDNA3, and where does memory bandwidth become the bottleneck?

This project uses the Vulkan compute API to run a GPU particle simulation, then sweeps workgroup thread count (32-1k) and particle count (8k – 2M+) while recording precise GPU-side timing via Vulkan timestamp queries. The goal is to connect measured throughput numbers back to the architectural specs of the RX 7900 XT.

---

## What the application does

The simulation places particles in a circle, assigns each one a velocity, and bounces them off screen edges. Every frame, a compute shader updates all particle positions on the GPU. A separate graphics pipeline renders them as coloured point sprites.

The compute shader is deliberately minimal:

```
new_position = old_position + velocity * deltaTime
```

One thread computes one particle. There is no inter-thread communication, no shared memory use, and minimal arithmetic. The shader reads 32 bytes and writes 32 bytes per particle. **This makes it a near-pure memory bandwidth test**.

The initial particle buffer is allocated in device-local GPU memory. The CPU uploads initial positions once via a staging buffer, than all updates happen entirely on the GPU with compute shader.

---

## RDNA3 architecture and theoretical expectations

### Hardware
https://rocm.docs.amd.com/en/docs-6.3.3/reference/gpu-arch-specs.html

| Property | Value |
|---|---|
| GPU | AMD Radeon RX 7900 XT |
| Architecture | RDNA3 (gfx1100) |
| Compute Units | 84 |
| Wavefront size | 32 threads |
| LDS per WGP | 128 KiB |
| Infinity Cache | 80 MiB |
| L2 Cache | 6 MiB |
| Graphics L1 Cache | 256 KiB |
| L0 Vector Cache | 32 KiB (per CU) |
| GDDR6 bandwidth | up to 800 GB/s |
| Effective bandwidth (with Infinity Cache) | up to 2900 GB/s |
| VRAM | 20 GB |

### Memory traffic per dispatch

Each particle holds:

| Field | Type | Size |
|---|---|---|
| position | float2 | 8 bytes |
| velocity | float2 | 8 bytes |
| color | float4 | 16 bytes |
| **Total** | | **32 bytes** |

The shader reads from `particlesIn` and writes to `particlesOut`, so total memory traffic per particle per frame is **64 bytes** (32 read + 32 write).

| Particle count | Total traffic per frame |
|---|---|
| 65,536 | 4 MB |
| 262,144 | 16 MB |
| 524,288 | 32 MB |
| 1,048,576 | 64 MB |
| 2,097,152 | 128 MB |

### Infinity Cache and the expected bandwidth cliff

The Infinity Cache is 80 MB. It acts primarily as a read cache. This leads to a concrete prediction:

- At **≤ ~1.25M particles** (~40 MB of read data), the `particlesIn` buffer fits inside the Infinity Cache. Reads come from cache at effective bandwidth far above the 800 GB/s GDDR6 figure.
- Above **~2.5M particles** (~80 MB read data), the read working set spills out of cache and throughput should drop toward the GDDR6 ceiling of 800 GB/s.

**Prediction:** effective read bandwidth will look much higher than 800 GB/s at small particle counts, and will decline as particle count grows past the cache capacity. A visible throughput cliff is expected somewhere around 1–2.5M particles depending on cache eviction behaviour.

### Theoretical throughput ceiling

At (up to) 800 GB/s GDDR6 bandwidth and 64 bytes per particle:

```
Max throughput = 800 GB/s ÷ 64 bytes = 12,500M particles/s = 12.5B particles/s
```

At that rate, dispatching 1M particles should take:

```
1,000,000 particles ÷ 12,500,000,000 particles/s ≈ 0.08 ms = 80 µs
```
Therefore an expected worse case for 2m particles is 160us.

### Workgroup size 
The original tutorial uses 
`commandBuffer.dispatch(PARTICLE_COUNT / 256, 1, 1);`
The compute shader declares:
[shader("compute")]
[numthreads(256,1,1)]

This means each work group runs 256 invocations (threads) of the shader.
Therefore if we want 1 invocation for every 1 particle, we need to launch enought work groups to cover all particles.
So workgroups = Particle count / num_of_threads in the shader.

The benchmark here aims to visulize the effect of num_of_threads per workgroups from the API perspective.
On RDNA3 there  are 32 thread slots per wavefront (https://gpuopen.com/learn/occupancy-explained/), so ideally we want and exact multiplier for maximum occupancy. For example if we have 32+32+2 threads, than the third wavefront will be mostly empty, with 32-2=30 empty threads.
One SIMD can interleave multiple wavefronts. 
From the api perspective, what we can be sure of, is that the workgroup will run on a WGP that share a fast shared memory called LDS (Local Data Share - 128KiB), or 'groupshared'. If a workgroup uses a large amount of LDS, fewer workgroups can be resident on the same WGP simultaneously, reducing occupancy.

So on the Vulkan API part we can vary the number of workgroups.
On the shader part we can vary the number of threads per workgroups.
On the GPU one workgroup will run on one WGP, where threads will be organized into wavefronts.

The below graph is my current understanding of execution hierarchy:
GPU
└── Shader Engine (SE)
    └── Work Group Processor (WGP)
        └── CU × 2  (a WGP contains 2 CUs)
            └── SIMD × 4  (each CU has 4 SIMDs)
                └── 16 wavefront slots  (each SIMD can hold 16 waves)

## Measurement methodology

### Why not CPU timing?

Wrapping a `vkQueueSubmit` call in `std::chrono` on the CPU measures the wrong thing. The CPU call returns almost immediately — the GPU executes the work asynchronously. Even if you wait for the fence, you are measuring submission overhead, driver processing, fence signalling, and OS scheduling jitter on top of the actual GPU execution time.

### Vulkan timestamp queries

This project uses `VK_QUERY_TYPE_TIMESTAMP` to record GPU-side timing:

1. `vkCreateQueryPool` — creates a pool with 2 slots (start and end)
2. `vkCmdResetQueryPool` — resets slots inside the command buffer before use
3. `vkCmdWriteTimestamp` with `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` — written immediately before the dispatch
4. `vkCmdWriteTimestamp` again — written immediately after the dispatch
5. After the frame fence signals, `vkGetQueryPoolResults` retrieves the two tick values
6. Elapsed time in nanoseconds = `(ticks[1] - ticks[0]) × timestampPeriod`

`timestampPeriod` is retrieved from `VkPhysicalDeviceProperties` and converts GPU clock ticks to nanoseconds. The result is the actual GPU execution time of the dispatch, isolated from everything else.

## Results

![Latency vs Particle Count](assets/2026-04-18_15-46-52_latency_vs_particles.png)

![Latency vs Workgroup Size](assets/2026-04-18_15-46-52_latency_vs_workgroup.png)

![Frame Latency at WG256](assets/2026-04-18_15-46-52_frame_latency_wg256.png)

## Code notes

# Fragile descriptor binding in order of declaration 
The integer passed to vk::DescriptorSetLayoutBinding(N, ...) on the CPU side must exactly match `[[vk::binding(N)]]` in the shader. There is no compiler enforcement of this correspondence in the used tutorial example. A mismatch produces silent corruption or a validation layer warning rather than a build error.

Slang will assign bindings implicitly by declaration order when `[[vk::binding(N)]]` is omitted, which is fragile — reordering declarations silently shifts all binding numbers.

Possible mitigations

Explicit annotations:  Quickest sollution is used here, declare `[[vk::binding(N)]]` on every resource in shader file (.slang).
SPIRV-Reflect: introspect the compiled SPIR-V binary at runtime to auto-discover binding layout, driving descriptor set creation from the shader itself rather than hardcoded constants.
Bindless / descriptor indexing: a modern Vulkan pattern that replaces per-binding wiring with a large descriptor array and runtime indices, sidestepping the problem at a design level.


## Building

Requires Vulkan SDK, GLFW, GLM, and CMake.
workgroup size (default: 256) must divide PARTICLE_COUNT (default:8192) without remainder
Maximum workgroup size is 1024 by hadrware limitation
```bash
cmake -B build -S .
cmake --build build
cd build
./VulkanComputePerf --particle-count 32768 --workgroup-size 64 --duration 10
```

## Credits

Vulkan code largely based on [https://docs.vulkan.org/tutorial/](https://docs.vulkan.org/tutorial/latest/11_Compute_Shader.html) by Alexander Overvoorde, licensed under CC BY-SA 4.0.