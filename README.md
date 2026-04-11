# Vulkan 


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

```bash
cmake -B build -S .
cmake --build build
cd build
./VulkanComputePerf
```

## Credits

Based on [https://docs.vulkan.org/tutorial/](https://docs.vulkan.org/tutorial/latest/11_Compute_Shader.html) by Alexander Overvoorde, licensed under CC BY-SA 4.0.