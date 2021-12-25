#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <optional>
#include <set>
#include <unordered_map>
#include <map>

struct Model {
    Model() = default;

    Model(const std::vector<std::string>& texturePath, const std::vector<std::string>& modelPath, const VkDevice& device, const VkPhysicalDevice& physicalDevice, 
            const VkCommandPool& commandPool, const VkQueue& queue, const size_t swapchainImageCount, const VkDescriptorPool& descriptorPool, const VkExtent2D& swapChainExtent, const VkRenderPass& renderPass);

    void recreateSwapchain(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const size_t swapchainImageCount, const VkDescriptorPool& descriptorPool, const VkExtent2D& swapChainExtent, const VkRenderPass& renderPass);

    void cleanupSwapchain(const VkDevice& device);

    void Destroy(const VkDevice& device);

    void recordCommands(VkCommandBuffer& commandBuf, uint32_t imageIndex);

    void updateUniformBuffer(const VkDevice& device, uint32_t currentImage, const VkExtent2D& swapChainExtent);

    const unsigned int addModel(const size_t modelInd, const size_t textureInd, const glm::mat4& wt);
    void changeModel(const unsigned int id, const size_t modelInd, const size_t textureInd, const glm::mat4& wt);
    void deleteModel(const unsigned ind);
};
