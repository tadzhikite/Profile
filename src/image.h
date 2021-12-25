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

struct Image {
    VkImage faceImage;
    VkDeviceMemory faceImageMemory;
    VkImageView faceImageView;
    VkSampler faceSampler;
    static constexpr std::array<glm::vec2, 8> vertices{
        glm::vec2{1.f, 0.f}, {1.f, 1.f},
            {1.f, 1.f}, {1.f, 0.f},
            {0.f, 0.f}, {0.f, 1.f},
            {0.f, 1.f}, {0.f, 0.f}
    };
    static constexpr std::array<uint16_t, 6> indices{0,2,1,1,2,3};
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline textPipeline;
    Image() = default;
    Image(const std::string& texturePath, const VkDevice& device, const VkPhysicalDevice& physicalDevice, 
            const VkCommandPool& commandPool, const VkQueue& queue, const size_t swapchainImageCount, const VkDescriptorPool& descriptorPool, const VkExtent2D& swapChainExtent, const VkRenderPass& renderPass) {
        auto g = createTextureImage(texturePath, device, physicalDevice, commandPool, queue);
        faceImage = g.first;
        faceImageMemory = g.second;
        faceImageView = createFaceImageView(device, faceImage);
        faceSampler = createFaceSampler(device, physicalDevice);
        auto v = createVertexBuffer(device, physicalDevice, commandPool, queue);
        vertexBuffer = v.first;
        vertexBufferMemory = v.second;
        auto i = createIndexBuffer(device, physicalDevice, commandPool, queue);
        indexBuffer = i.first;
        indexBufferMemory = i.second;
        descriptorSetLayout = createTextDescriptorSetLayout(device);
        descriptorSet = createDescriptorSet(device, descriptorSetLayout, descriptorPool, faceSampler, faceImageView);
        auto p = createTextPipeline(device, swapChainExtent, descriptorSetLayout, renderPass);
        pipelineLayout = p.first;
        textPipeline = p.second;
    };
    void recreateSwapchain(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const size_t swapchainImageCount, const VkDescriptorPool& descriptorPool, const VkExtent2D& swapChainExtent, const VkRenderPass& renderPass){
        descriptorSet = createDescriptorSet(device, descriptorSetLayout, descriptorPool, faceSampler, faceImageView);
        auto p = createTextPipeline(device, swapChainExtent, descriptorSetLayout, renderPass);
        pipelineLayout = p.first;
        textPipeline = p.second;
    }
    void cleanupSwapchain(const VkDevice& device){
        vkDestroyPipeline(device, textPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    }
    void Destroy(const VkDevice& device) {
        vkDestroyImage(device, faceImage, nullptr);
        vkFreeMemory(device, faceImageMemory, nullptr);
        vkDestroyImageView(device, faceImageView, nullptr);
        vkDestroySampler(device, faceSampler, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
    }
    void recordCommands(VkCommandBuffer& commandBuf, uint32_t imageIndex){
        vkCmdBindDescriptorSets(commandBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdBindPipeline(commandBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipeline);
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuf, 0, 1, &vertexBuffer, offsets);
        vkCmdBindIndexBuffer(commandBuf, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(commandBuf, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    }
    private:
    std::vector<char> readFile(const std::string& filename);
    std::pair<VkImage, VkDeviceMemory> createTextureImage(const std::string& filePath, const VkDevice& device, 
            const VkPhysicalDevice& physicalDevice, const VkCommandPool& commandPool, const VkQueue& queue);
    VkImageView createFaceImageView(const VkDevice& device, const VkImage& image);
    VkSampler createFaceSampler(const VkDevice& device, const VkPhysicalDevice& physicalDevice);
    std::pair<VkBuffer, VkDeviceMemory> createVertexBuffer(const VkDevice& device, const VkPhysicalDevice& physicalDevice, 
            const VkCommandPool& commandPool, const VkQueue& graphicsQueue);
    std::pair<VkBuffer, VkDeviceMemory> createIndexBuffer(const VkDevice& device, const VkPhysicalDevice& physicalDevice, 
            const VkCommandPool& commandPool, const VkQueue& graphicsQueue);
    VkDescriptorSetLayout createTextDescriptorSetLayout(const VkDevice& device);
    VkDescriptorSet createDescriptorSet(const VkDevice& device, const VkDescriptorSetLayout& descriptorSetLayout, 
            const VkDescriptorPool& descriptorPool, const VkSampler& sampler, const VkImageView& imageView);
    std::pair<VkPipelineLayout, VkPipeline> createTextPipeline(const VkDevice& device, const VkExtent2D& swapChainExtent, 
            const VkDescriptorSetLayout& descriptorSetLayout, const VkRenderPass& renderPass);
};
