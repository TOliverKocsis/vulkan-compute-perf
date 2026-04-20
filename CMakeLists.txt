cmake_minimum_required(VERSION 3.20)
project(VulkanComputePerf)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)
find_package(stb REQUIRED)

find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

set(SHADERS_DIR ${CMAKE_BINARY_DIR}/shaders)

add_custom_command(
    OUTPUT ${SHADERS_DIR}/slang.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    COMMAND ${SLANGC_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader_compute.slang
        -target spirv
        -profile spirv_1_4+spvRayQueryKHR
        -emit-spirv-directly
        -fvk-use-entrypoint-name
        -O3
        -entry vertMain -entry fragMain -entry compMain
        -o ${SHADERS_DIR}/slang.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader_compute.slang
    COMMENT "Compiling Slang shaders"
    VERBATIM
)
add_custom_target(slang_shader DEPENDS ${SHADERS_DIR}/slang.spv)

add_executable(VulkanComputePerf src/compute_shader.cpp)
set_target_properties(VulkanComputePerf PROPERTIES
    CXX_STANDARD 20
)
target_link_libraries(VulkanComputePerf PRIVATE Vulkan::Vulkan glfw glm::glm)
target_include_directories(VulkanComputePerf PRIVATE ${stb_INCLUDE_DIRS})
target_compile_definitions(VulkanComputePerf PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
)
add_dependencies(VulkanComputePerf slang_shader)
