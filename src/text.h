#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <ft2build.h>
#include FT_FREETYPE_H


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
#include <functional>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <map>

struct Text {
    VkImage faceImage;
    VkDeviceMemory faceImageMemory;
    VkImageView faceImageView;
    VkSampler faceSampler;
    static constexpr int ASCII_COUNT = 128;
    struct GlyphDims {
        char c;
        glm::ivec2 bearing;
        unsigned int advance;
        unsigned int width;
        unsigned int height;
        unsigned int size;
        unsigned int offset;
    };
    std::array<GlyphDims, ASCII_COUNT> glyphDims;
    static constexpr std::array<uint16_t, 6> indices{0,2,1,1,2,3};
    std::array<glm::mat4, 128> textureCoordinateTransforms;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline textPipeline;

    FT_Library library;
    FT_Face face;
    std::map<const unsigned int, std::vector<std::array<glm::mat4, 2>>> text{};
    unsigned int id;

    Text() = default;
    Text(const std::string& fontFilePath, const VkDevice& device, const VkPhysicalDevice& physicalDevice, 
            const VkCommandPool& commandPool, const VkQueue& queue, const size_t swapchainImageCount, const VkDescriptorPool& descriptorPool, const VkExtent2D& swapChainExtent, const VkRenderPass& renderPass);
    void recreateSwapchain(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const size_t swapchainImageCount, const VkDescriptorPool& descriptorPool, const VkExtent2D& swapChainExtent, const VkRenderPass& renderPass);
    void cleanupSwapchain(const VkDevice& device);
    void Destroy(const VkDevice& device);
    const unsigned int addText(const std::string& msg, const glm::vec2& origin, const float extent);
    const std::function<bool(const glm::vec2&)> hover(const unsigned int msgID);
    void changeText(const unsigned int id, const std::string& msg, const glm::vec2& origin, const float extent);
    void removeText(const unsigned int id);
    void recordCommands(VkCommandBuffer& commandBuf);

    private:
    std::vector<std::array<glm::mat4, 2>> tiles(const std::string& msg, const glm::vec2& origin, const float extent);
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
