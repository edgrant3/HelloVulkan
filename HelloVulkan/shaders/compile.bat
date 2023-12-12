:: REMARKS

:: following instructions at: 
:: 		https://vulkan-tutorial.com/en/Drawing_a_triangle/Graphics_pipeline_basics/Shader_modules#:~:text=the%20glslc%20program.-,Windows,-Create%20a%20compile

:: -o is output flag (other options include creating human-readible format to review compiler optimizations)

:: Compiling shaders on the commandline is one of the most straightforward options and it's the one that we'll 
:: use in this tutorial, but it's also possible to compile shaders directly from your own code. The Vulkan SDK 
:: includes libshaderc, which is a library to compile GLSL code to SPIR-V from within your program.

C:/VulkanSDK/1.3.261.1/Bin/glslc.exe shader.vert -o vert.spv
C:/VulkanSDK/1.3.261.1/Bin/glslc.exe shader.frag -o frag.spv
pause