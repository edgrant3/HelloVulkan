//#include <vulkan/vulkan.h> // include from LunarG SDK (funcs, structs, enumerations)
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>    // will automatically load vulkan header bc of prev define

#include <iostream>        // error reporting, logging to console
#include <fstream>         // in/out streams to operate on files (SPIR-V compiled shader files)
#include <stdexcept>       // error propagating
#include <cstdlib>         // provides EXIT_SUCCESS and EXIT_FAILUTE macros

#include <cstring>         // not necessary, but could be for other compilers when using strcmp
#include <optional>        // to use for determining queue family existence

//#include <cstdint>         // Necessary for uint32_t // ?
#include <limits>          // Necessary for std::numeric_limits
#include <algorithm>       // Necessary for std::clamp

#include <vector>
#include <set>


// screen space H, W
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2;

// Want to use validations layers (error checking) provided from the LunarG Vulkan SDK
const  std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation" // from SDK
};

// Need swap chain extension (swap chain owns queue of buffers we render to before presenting to screen)
// reasons why this is not included in Vulkan core:
// 1.) Not all GPUs are capable of presenting imgs to screen (e.g. servers with no display output)
// 2.) Img presentation inextricably tied to window system and surfaces which is separate from Vulkan
const std::vector<const char*> deviceExtensions = {
    // Will check is supported by device 
    // (although availability of presentation queue implies swap chain extension is supported already)
    VK_KHR_SWAPCHAIN_EXTENSION_NAME // macro to prevent misspelling
};

// Only enable validation layers when compiling in debug mode
// `NDEBUG` is part of C++ standard and means "not debug"
#ifdef NDEBUG 
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif


// because vkCreateDebugUtilsMessengerEXT() is an extension function, it is not auto-loaded
// and its address must be looked up using vkGetInstanceProcAddr. the following proxy func does this
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}
// proxy function to destroy debugMessenger
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}


struct QueueFamilyIndices {
    // optional wrapper contains no value until it is assigned
    // can be queried if it has value using has_value() method
    std::optional<uint32_t> graphicsFamily; // for drawing image
    std::optional<uint32_t> presentFamily;  // for presenting image to window surface

    /* NOTE: it is very likely that these above two will be the same family
    // but we'll treat them as if separate for a uniform approach.
    // however, could add logic to prefer physical device which supports
    // drawing and presentation in the same queue for better performance
    */

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};


// *****************************************************************************************************
// **************** MAIN APPLICATION *******************************************************************
// *****************************************************************************************************


class HelloTriangleApplication {
public:
    //HelloTriangleApplication() : physicalDevice(VK_NULL_HANDLE) {}

    void run() {
        std::cout << "\nRunning HelloVulkan Application...\n" << std::endl;
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    /// MEMBER VARIABLES | START
    GLFWwindow* window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;                             // Window surface on which we will draw

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // Implicitly destroyed when instance destroyed
    VkDevice device;                                  // Logical device handle

    VkQueue graphicsQueue;                            // Graphics Queue handle (implicitly destroyed w/ device)
    VkQueue presentQueue;                             // Presentation Queue handle

    VkSwapchainKHR swapChain;                         // Swap chain is a queue of images
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    std::vector<VkImageView>   swapChainImageViews;     // Needed to use VkImages (how and what part of img to access)
    std::vector<VkFramebuffer> swapChainFramebuffers;   // each framebuffer references all the VkImageView objects that represent the attachments

    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;                   // command pools manage memory that is used to store command buffers, and buffers are allocated from them
    std::vector<VkCommandBuffer> commandBuffers; // group commands (like drawing, mem transfers) to allow Vulkan to more efficiently process all commands together
    // command buffers are automatically freed when their associated command pool is destroyed, so no explicit cleanup required!

    // Sync Primitives
    // Semaphores -> swap chain operation synchronization (ordering GPU execution)
    std::vector<VkSemaphore> imageAvailableSemaphores; // to signal that img has been acquired from swap chain & is ready for rendering
    std::vector<VkSemaphore> renderFinishedSemaphores; // to signal that rendering is finished and presentation can happen
    // Fence -> block host (CPU) from drawing more than one frame at a time
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;
    bool framebufferResized = false; // Explicitly catch window resizing, not just depending on drivers/platforms to generate VK_ERROR_OUT_OF_DATE_KHR

    /// MEMBER VARIABLES  | END
    /// PRIMARY FUNCTIONS | START

    void initWindow() {
        glfwInit(); // initialize GLFW windowing library

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // disable OpenGL context creation
        //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);  // disable window resizing (extra effort)

        // params: (w, h, title, monitor, *something relevant only to OpenGL*)
        window = glfwCreateWindow(WIDTH, HEIGHT, "HelloVulkan", nullptr, nullptr);

        // Have static member callback for resizing... GLFW doesn't know how to call member function with the correct `this` ptr to our app
        // have to give it this pointer to use
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        vkDeviceWaitIdle(device);
    }

    void cleanup() {

        std::cout << "\nGLEAMING THE CUBE!!!\n" << std::endl;

        cleanupSwapChain();

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr); // instance, optional (de-)allocation callback
        
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void cleanupSwapChain() {
        for (auto fb : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, fb, nullptr);
        }

        for (auto iv : swapChainImageViews) {
            vkDestroyImageView(device, iv, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    /// PRIMARY FUNCTIONS  | END
    /// CREATION FUNCTIONS | START

    void createInstance() {

        // Check if requested validation layers are available
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        // *OPTIONAL* 
        // struct contains info for app to initialize
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;
        // appInfo.pnext = nullptr // by default so we point to no extension info

        // *MANDATORY* 
        // struct tells Vulkan driver which *global* extensions and validation layers to use.
        //     (global means they apply to entire program, not specific device)
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // Next 2 layers specify desired global extensions
        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // Last 2 layers specify the global validation layers to enable
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            // These next two lines ensure debug messaging covers vkCreateInstance and vkDestroyInstance calls too
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        //// Optionally retrieve a list of supported extensions before creating instance
        //// TODO: create function that checks if all glfwExtensions are included in these supported
        //uint32_t extensionCount = 0;
        //// params:
        ////    1) ptr which allows us to filter extensions by a validation layer (ignore for now)
        ////    2) ptr to var which stores number of extensions
        ////    3) ptr to array of VkExtensionProperties to store details
        //vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        //std::vector<VkExtensionProperties> extensions(extensionCount);
        //vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        //std::cout << "supported extensions: \n";
        //for (const auto& e : extensions) {
        //    std::cout << '\t' << e.extensionName << '\n';
        //}

        // Can finally issue creation of instance call:
        // params:
        //   1) ptr to struct with creation info
        //   2) ptr to custom allocator callbacks (always nullptr in this tutorial)
        //   3) ptr to variable that stores the handle to the new object
        // 
        // returns: VkResult that is either VK_SUCCESS or error code
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

        std::cout << "SUCCESS: Vulkan instance created!" << std::endl;
    }

    void createSurface() {
        // Handles creation of surface (Window System Integration)
        // GLFW handles this in a platform-agnostic way
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }

        std::cout << "SUCCESS: window surface created using GLFW!" << std::endl;
    }

    void pickPhysicalDevice() {
        // Can select any number of graphics cards and use them simultaneously,
        // but in this tutorial we'll stick to the first graphics card that suits our needs
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                // grab first suitable device
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
        std::cout << "SUCCESS: physical device selected: " << deviceProperties.deviceName << std::endl;

    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        // w/ set, duplicate entries are reduced to one unique entry
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            // queueCount is 1 becaue we don't really need more than one...
            // can create all command buffers on multiple threads and then
            // submit them all at once on the main thread with single, low-overhead call
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // Specify the set of device features we'll use... TODO
        VkPhysicalDeviceFeatures deviceFeatures{};

        // Now create the device
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());// 0;
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // Newer implementations of Vulkan have no distinction b/w instance and device - specific
        // validation layers, so next conditional is unecessary but good for compatibility
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        // Finally ready to instantiate the logical device!
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

        std::cout << "SUCCESS: logical device created!" << std::endl;

    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR   presentMode   = chooseSwapPresentMode(  swapChainSupport.presentModes);
        VkExtent2D         extent        = chooseSwapExtent(       swapChainSupport.capabilities);

        // how many images to have in swap chain? can get minimum required to function,
        // but request at least one more so we don't sometimes wait on driver to complete 
        // internal operations before we acquire another image to render to
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        // also ensure we don't exceed max (0 is special case where there is no max)
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        // Once again fill out large struct to create Vulkan object...
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1; // # of layers for each image (always 1 unless stereoscopic 3D application)
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // what operations to expect on these images (recall similar from OpenGL for depth buffer example)

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        // Need to handle how swap chain imgs will be used across multiple queue families
        // as will be the case if graphicsFamily is not the presentFamily
        //   -- AKA draw to images w/ graphics fam, present to screen w/ present fam
        if (indices.graphicsFamily != indices.presentFamily) {
            // CONCURRENT => imgs used across multiple queue fams without explicit ownership transfers
            // kind of a cop-out because better ways exist in our case, but will be in another tutorial
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            // EXCLUSIVE => img "owned" by 1 queue fam at a time and explictly transferred before use in another queue fam
            //            > gives better performance
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            //createInfo.queueFamilyIndexCount = 0;     // Optional
            //createInfo.pQueueFamilyIndices = nullptr; // Optional
        }

        // Specify transform to apply to imgs in swap chain e.g.) 90-degree rotation or horizontal flip
        // --- set to current transform if none desired
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

        // Specify if the alpha channel should be used for blending w/ other windows in the window system
        // --- almost always want to ignore alpha channel
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        createInfo.presentMode = presentMode;

        // Clipped: if true, we don't care about color of obscured pixels (e.g. another window in front)
        // unless absolutely needed, this can improve performance
        createInfo.clipped = VK_TRUE; 

        // swap chain can become invalid or unoptomized while app is running (e.g. window was resized)
        // In that case, swap chain needs to be recreated and a ref to the old one must be specified in this field
        // more on this in later chapter @: https://vulkan-tutorial.com/Drawing_a_triangle/Swap_chain_recreation
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain");
        }
        std::cout << "SUCCESS: swap chain created!" << std::endl;

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); ++i) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];

            // how to interpret image data
            // viewType can treat imgs as 1D, 2D, 3D textures and cube maps
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;

            // swizzling moves components of a vector around (like col.rgba or col.bbg would work in GLSL)
            // https://en.wikipedia.org/wiki/Swizzling_(computer_graphics)
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            // subresourceRange field gives img's purpose and which part to access
            // Our images are color targets without mipmapping levels or multiple layers (as would happen w/ stereographic 3D)
            createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel   = 0;
            createInfo.subresourceRange.levelCount     = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount     = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }

        std::cout << "SUCCESS: image views created!" << std::endl;
    }

    void createRenderPass() {
        // Before we can finish creating the graphics pipeline,
        // we need to specify the framebuffer attachments we will use
        // e.g.) how many color and depth buffers there will be, how many
        //  samples to use for each, and how their contents should be handled

        // In this simple case, just a single color buffer represented by one swap chain img
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // just 1 sample (not doing multisampling yet)

        // loadOp: how to handle existing data in attachment prior to rendering
        //         * _LOAD      : preserve existing contents
        //         * _CLEAR     : clear values to constant at start
        //         * _DONT_CARE : existing contents are undefined, don't care about them
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

        // storeOp: what to do with attachment data after rendering
        //         * _STORE     : rendered contents are stored and can be read later
        //         * _DONT_CARE : contents will be undefined after render operation
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // we want to SEE the triangle, so _STORE

        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;  // not using stencil so N/A
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // not using stencil so N/A

        // following two settings are discussed more in texturing chapter...
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // SUBPASSES
        // subpasses are subsequent rendering passes which depend upon framebuffer vals from prev passes

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0; // 0 because it is an index to the attachements in colorAttachment array (we only have one entry)
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        // vv have to be explicit bc Vulkan may soon (if not already in '23) support compute subpasses
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        // index of attachment in this array is referenced from the frag shader w/ the directive:
        //      layout(location = 0) out vec4 outColor

        // A way to handle synchronization via subpass dependencies
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL; // external refers to special case for implicit subpass before or after render pass
        dependency.dstSubpass = 0; // destination (0 is our single one here) must be higher than src
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;


        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }

        std::cout << "SUCCESS: render pass created!" << std::endl;
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        // for vert shader...
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main"; // defines entrypoint (main is standard)
        // this could allow user to combo multiple frag shaders, for example
        // next can specify values for shader constants (not used in tutorial, but worth looking into)
        vertShaderStageInfo.pSpecializationInfo = nullptr;

        // for frag shader...
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName  = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // FIXED FUNCTIONS ////////////////////////
        
        // VERTEX INPUT (format to expect from vertex data passed to vert shader)
        // - Bindings: spacing b/w data and whether data is per-vertex or per-instance
        // - Attribute Descriptions: type of attr passed to vert shader, which binding to load
        //       them from and which offset
        // *** for drawing a simple triangle, we set this up to expect no data
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount   = 0;
        vertexInputInfo.pVertexBindingDescriptions      = nullptr; // Optional
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions    = nullptr; // Optional

        // INPUT ASSEMBLY
        // - Topology: what kind of geometry will be drawn from the verts? 
        //        * VK_PRIMITIVE_TOPOLOGY_POINT_LIST:     points from vertices
        //        * VK_PRIMITIVE_TOPOLOGY_LINE_LIST:      line from every 2 vertices without reuse
        //        * VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:     the end vertex of every line is used as start vertex for the next line
        //        * VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:  triangle from every 3 vertices without reuse
        //        * VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: the second and third vertex of every triangle are used as first two vertices of the next triangle
        // 
        // should "primitive restart" be enabled? If yes, can do *element buffer* and break up strip topologies

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // VIEWPORTS and SCISSORS
        // - Viewport: region of framebuffer to which output will be rendered (almost always (0,0) to (w, h))
        // - Scissor rectangle: basically rectangle mask outside of which pixels are discarded by rasterizer
        // NOTE: recall that size of swap chain and its imgs may differ from WIDTH, HEIGHT of the window
        //       & swap chain imgs will be used as framebuffers later on

        // if we wanted to set scissor and viewport statically, we would do the following:
        /*
        VkViewport viewport{};
        viewport.x        = 0.0f;
        viewport.y        = 0.0f;
        viewport.width    = (float) swapChainExtent.width;
        viewport.height   = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        // We want to draw to entire framebuffer, so specify scissor accordingly
        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        */

        // However, we want to be able to set them dynamically, so we'll do the following:
        // Note: we only need to set their count at this stage
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount  = 1;

        // RASTERIZER
        // converts geometry shaped by verts in vertex shader into fragments to be colored by frag shader
        // Also performs depth testing, face culling, and scissor test
        // Can be configured to output fragments that fill polygons or just wireframe rendering
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE; // if true, clamp frags to near/far clip planes instead of discarding (good 4 shadowmapping), requires enabling a GPU feature
        rasterizer.rasterizerDiscardEnable = VK_FALSE; // if true, geometry never passes thru rasterizer, disables framebuffer output
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // can be LINE or POINT as well
        rasterizer.lineWidth = 1.0f; // any val above 1.0f requires enabling `wideLines` GPU feature
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // can disable, cull front, cull back , or cull both
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // vertex order that constitutes front face
        // Optionally, add bias and offset to depth value (sometimes used in shadowmapping)
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f; // optional
        rasterizer.depthBiasClamp = 0.0f;          // optional
        rasterizer.depthBiasSlopeFactor = 0.0f;    // optional

        // MULTISAMPLING
        // One way to handle anti-aliasing by combining frag shader results of multiple polygons that
        // rasterize to the same pixel. performant bc it avoids running frag shader multiple times if only one polygon maps to pixel
        // requires enabling a GPU feature though.. we'll leave it disabled for now
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable   = VK_FALSE;
        multisampling.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading      = 1.0f;     // optional
        multisampling.pSampleMask           = nullptr;  // optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // optional
        multisampling.alphaToOneEnable      = VK_FALSE; // optional

        // DEPTH and STENCIL testing
        // TODO...

        // COLOR BLENDING
        // combine frag shader output w/ existing color in framebuffer
        // mixing or combine using bitwise operation
        // NOTE: here we're disabling both modes so colors are written to framebuffer unmodified

        // Per-framebuffer color blending (we only have one rn)
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | 
                                              VK_COLOR_COMPONENT_G_BIT | 
                                              VK_COLOR_COMPONENT_B_BIT | 
                                              VK_COLOR_COMPONENT_A_BIT ;
        colorBlendAttachment.blendEnable         = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
        colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;      // optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  // optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
        colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;      // optional
        // NOTE: see tutorial for pseudocode for how to do this per-framebuffer blending
        // TODO: research this in the specification... (VkBlendFactor and VkBlendOp enum options)

        // Global color blending
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional
        
        // DYNAMIC STATES: part of pipeline that can be changed without
        // recreating the pipeline at draw time (like viewport size and line width, and blend constants)
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        // NOTE: doing this causes config of these values to be ignored and we must specify the data at draw time
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // PIPELINE LAYOUT
        // Used to specify uniform values to pass to shaders
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount         = 0;       // optional
        pipelineLayoutInfo.pSetLayouts            = nullptr; // optional
        pipelineLayoutInfo.pushConstantRangeCount = 0;       // optional
        pipelineLayoutInfo.pPushConstantRanges    = nullptr; // optional

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // FINALLY - can create the pipline (this is the Big Papa)
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        // reference all the structures describing the fixed-function stage
        pipelineInfo.pVertexInputState   = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState      = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState   = &multisampling;
        pipelineInfo.pDepthStencilState  = nullptr; // Optional
        pipelineInfo.pColorBlendState    = &colorBlending;
        pipelineInfo.pDynamicState       = &dynamicState;
        // pipeline layout is Vulkan handle, not struct ptr
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass; // more involved thing, member variable of class
        pipelineInfo.subpass = 0;             // index of sub pass where this pipeline will be used

        // vv can create pipeline by deriving from an existing one (two pipelines can share parent, less expensive)
        // these values are also only used if VK_PIPELINE_CREATE_DERIVATIVE_BIT flag is also specified in `flags` field of pipelineInfo
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
        pipelineInfo.basePipelineIndex  = -1;             // Optional

        // Now, create! (2nd param is an optional VkPipelineCache which can speed up pipeline creation for multiple)
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        std::cout << "SUCCESS: graphics pipeline created!" << std::endl;

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); ++i) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
        std::cout << "SUCCESS: " << swapChainImageViews.size() << " framebuffers created!" << std::endl;
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        // choose graphics family bc we're recording commands for drawing
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); 

        // for flag aspect:
        // ...COMMAND_BUFFER_BIT: allow command buffers to be rerecorded individually (w/o this all have to be reset together)
        // ...TRANSIENT_BIT: hint that bommand buffers are rerecorded very often (could change mem allocation behavior)
        
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }

        std::cout << "SUCCESS: created command pool!" << std::endl;
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

        // level params:
        // VK_COMMAND_BUFFER_LEVEL_PRIMARY  : can be submitted to a queue for execution, but not called from other command buffers
        // VK_COMMAND_BUFFER_LEVEL_SECONDARY: cannot be submitted directly, but can be called from primary command buffers

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        std::cout << "SUCCESS: allocated command buffers!" << std::endl;
    }

    void createSyncObjects() {
        // Core philosophy of Vulkan is synchronization of execution on GPU
        // for this, we need syncronization primitives (semaphores, fences)
        // NOTE: many Vulkan API calls which start executing work on the GPU are
        //       *asynchronous*, meaning they return before the operation finishes

        // Semaphore: adds order b/w queue operations. used both to order work in same queue and b/w queues
        // * NOTE: there are binary and timeline semaphores, but we'll only use binary in this tutorial
        // * 2 state: signaled or unsignaled
        //     * for processes A, B to complete in that order, use semaphore S as 'signal' for A and 'wait' for B
        // this waiting only happens on GPU, and because the queue submission functions return before their 
        // operations finish, we need another sync primitive to handle CPU waiting...

        // Fence: sync/order the execution on the CPU (host). use when host needs to know when GPU finishes something
        // * also signaled or unsignaled, but require manual reset to unsignaled
        // * generally prefer not to block host unless necessary, so prefer semaphores or other sync primitives

        // *** Fence -> blocking code execution on CPU until signaled by prev step completion
        // *** Semaphore -> submit commands in order to GPU quickly, but subsequent task waits until signaled by prev completion
        
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);


        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        // future versions of Vulkan API or extensions may add functionality for flags and pNext params

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // initializes state to signaled to first draw call does not wait indefinitely

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create sync objects (semaphores & fence)!");
            }
        }

        std::cout << "SUCCESS: created sync objects (semaphores & fence)!" << std::endl;
    }

    void recreateSwapChain() {
        // Disadvantage of this approach: stops all rendering before creating new swap chain...
        // Can create new SC while drawing commands on img from old SC are still in-flight,
        // just need to pass old SC to oldSwapChain field in VkSwapchainCreateInfoKHR and then destroy old one
        // as soon as we finish using it
        
        // Handle window minimization (frame buffer size is 0 in this case)
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        // NOTE: we don't recreate render pass here for simplicity...
        // in theory, swap chain img format could change (e.g. moving window to
        // std range to high dynamic range monitor) needing to recreate renderpass
        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    /// CREATION FUNCTIONS | END
    /// HELPER FUNCTIONS   | START

    void drawFrame() {
        // Vulkan frame rendering steps (simple & high-level):
        // 1) wait for prev frame to finish
        // 2) acquire img from swap chain
        // 3) record command buffer which draws the scene onto that img
        // 4) submit recorded command buffer
        // 5) present swap chain image

        // Wait until prev frame has finished, so command buffer and semaphores are available
        // Following takes array of fences & waits on host for ALL or ANY (VK_TRUE means wait on all) to be signaled before returning
        // * final param is timeout (max 64 int effectively disables timeout)
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // Acquire img from swap chain (RECALL the `vk*KHR` naming bc swap chain is an extension feature)
        uint32_t imageIndex; // index to the VkImage in our swapChainImages array
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        // Check if swap chain needs recreating for reasons:
        // * Out of date: SC incompatible with surface and can't be used for rendering (usually due to window resize)
        // * Suboptimal: SC can still be used to present to surface, but properties no longer match exactly
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            // could also recreate here, but we can still proceed because we've retrieved an img
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Reset Fence to unsignaled state
        // NOTE: moved this below out-of-date check so we reset only if we know we'll submit work
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // Record the command buffer
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        // Submit the command buffer
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        // specifying stage of pipeline that writes to color attachment should wait until img available
        // NOTE: elements of waitSempahores and waitStages correspond
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; 
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        // specifying which semaphores to signal once the command buffer(s) have finished execution
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        // Presentation
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr; // Optional if multiple swap chains to check successful presentation

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            // Now, recreate if suboptimal too, because we want best result
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    }
    
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;                  // Optional
        beginInfo.pInheritanceInfo = nullptr; // Optional, for secondary buffers

        // flags parameter (how we'll use the command buffer):
        // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT:      will be rerecorded right after executing it once
        // VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: is a secondary command buffer that will be entirely within a single render pass
        // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT:     can be resubmitted while it is also already pending execution

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            // NOTE: this implicity resets the buffer (can't append commands later)
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        //std::cout << "SUCCESS: began recording command buffer!" << std::endl;

        // START RENDER PASS
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}}; // black w/ 100% opacity
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        // Render pass can now begin...
        // all functions that record commands have `vkCmd` prefix (and all return void)
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        // since we specified viewport and scissor state as dymanic in the fixed function step,
        // we need to set them here before issuing our draw command
        
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width  = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // TIME TO DRAW!!!
        // vkCmdDraw params:
        // - commandBuffer:
        // - vertexCount: 3 for our tri despite not having vertex buffer
        // - instanceCount: 1
        // - firstVertex: offset into vert buffer (lowest val of gl_VertexIndex)
        // - firstInstance: offset for instanced rendering (lowest val of gl_InstanceIndex)
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        //std::cout << "SUCCESS: finished recording command buffer!" << std::endl;

    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        // takes buffer w/ bytecode and creates VkShaderModule
        // specify ptr to the buffer and its length

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        // See tutorial for more advanced device selection scheme
        VkPhysicalDeviceProperties deviceProperties;
        VkPhysicalDeviceFeatures   deviceFeatures;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        std::cout << "evaluating physical device: " << deviceProperties.deviceName << std::endl;

        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
            deviceFeatures.geometryShader && indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    std::vector<const char*> getRequiredExtensions() {
        // bc Vulkan is platform-agnostic API, need an extension to interface with windowing system
        // luckily, GLFW has a built-in function which returns the extension(s) needed to do so
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        // Range constructor of std::vector captures elements from char** array
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            // Following macro == "VK_EXT_debug_utils" literal, and it's used to avoid typos
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;
        
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());



        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) break;

            ++i;
        }

        return indices;
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }


    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {

        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData       = nullptr; // Optional
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);


        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    // Swap-Chain Settings selection | START

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        // surface format = color depth ?
        // each VkSurfaceFormatKHR entry contains a `format` and a `colorSpace` member
        // where `format`   specifies color channels and types
        // and `colorSpace` specifies if the SRGB color space is supported via the VK_COLOR_SPACE_SRGB_NONLINEAR_KHR flag

        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && 
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        // If the above fails, we could start ranking the available formats based on how "good" they are
        
        return availableFormats[0];
    }
    
    VkPresentModeKHR   chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        // presnetation mode arguably most important setting for the swap chain, bc it represents
        // the actual conditions for showing images to the screen

        // 4 possible modes available in Vulkan:
        // - VK_PRESENT_MODE_IMMEDIATE_KHR
        //       * images submitted by app are transferred to screen immediately - could result in tearing
        // - VK_PRESENT_MODE_FIFO_KHR
        //       * display image from front of queue when display is refreshed
        //         & insert rendered images to the back of the queue (if full, wait)
        //       * most similar to vertical sync (v-sync) ("vertical blank" = moment display refreshes)
        // - VK_PRESENT_MODE_FIFO_RELAXED_KHR
        //       * only differs from previous where program is late and queue is empty...
        //       * instead of waiting for next vertical blank, image is transferred immediately when it
        //         arrives, and this could result in visible tearing
        // - VK_PRESENT_MODE_MAILBOX_KHR
        //       * differs from FIFO by replacing images already queued with newer ones instead of
        //         blocking the application when queue is full
        //       * render frames as fast as possible while avoiding tearing (fewer latency issues than vsync)
        //       * commonly known as "triple buffering"

        // Only VK_PRESENT_MODE_FIFO_KHR mode is guranteed available
        // author of tutorial prefers mailbox but requires more energy than FIFO (mobile apps prefer lower power)

        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D         chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        // swap extent is resolution of the images (almost always same as draw window resolution in pixels)
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width,
                                            capabilities.minImageExtent.width,
                                            capabilities.maxImageExtent.width);

            actualExtent.height = std::clamp(actualExtent.height,
                                             capabilities.minImageExtent.height,
                                             capabilities.maxImageExtent.height);
            return actualExtent;
        }
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    // Swap-Chain Settings selection | END

    /// HELPER FUNCTIONS | END
    /// EXTRA FUNCTIONS  | START

    bool checkValidationLayerSupport() {
        // The Vulkan API is designed around the idea of minimal driver overhead and one of the 
        // manifestations of that goal is that there is very limited error checking in the API by default.

        // Checks can be added to the API via Vulkan's elegant system of *validation layers*
        // see for more: https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Validation_layers

        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }
        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        // read all bytes from file and return byte array
        // ios::ate    -> start reading at end of the file (helps to get size of file from read pos)
        // ios::binary -> read the file as a binary file (avoid text transformations)
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t) file.tellg();

        std::cout << "size of " << filename << " | " << fileSize << " bytes" << std::endl;

        std::vector<char> buffer(fileSize);

        file.seekg(0); // seek back to beginning
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
        // messageSeverity param enums begin with: VK_DEBUG_UTILS_MESSAGE_SEVERITY_
        // * VERBOSE_BIT_EXT: diagnostic msg
        // * INFO_BIT_EXT:    informational msg like creation of resource
        // * WARNING_BIT_EXT: msg about behavior that is likely a bug but not necessarily an error
        // * ERROR_BIT_EXT:   msg about behavior that is invalid and may cause crashes
        // NOTE: can use comparison operators to check severity level!

        // messageType param enums begin with: VK_DEBUG_UTILS_MESSAGE_TYPE_
        // * GENERAL_BIT_EXT:     Something has happened that is unrelated to the specification or performance
        // * VALIDATION_BIT_EXT:  Something violates the specification or indicates a possible mistake
        // * PERFORMANCE_BIT_EXT: Potential non-optimal use of Vulkan

        // pCallbackData param refers to VkDebugUtilsMessengerCallbackDataEXT struct:
        //  ->pMessage:    debug msg as a null-terminated string
        //  ->pObjects:    arr of Vulkan object handles related to the msg
        //  ->objectCount: num of objects in arr

        // pUserData param: ptr specified during setup of the callback to pass your own data to it

        // Returns a boolean that indicates if the Vulkan call that triggered the validation layer 
        // message should be aborted. If the callback returns true, then the call is aborted with 
        // the VK_ERROR_VALIDATION_FAILED_EXT error.This is normally only used to test the validation 
        // layers themselves, so you should always return VK_FALSE

        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    /// EXTRA FUNCTIONS | END
};


// **************************************************
// **************** ENTRY POINT *********************
// **************************************************

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}