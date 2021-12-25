#include "model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <stb_image.h>
#include <limits>
#include <functional>
#include <algorithm>
#include <iterator>
#include <numeric>
VkImage textureImage;
VkDeviceMemory textureImageMemory;
VkImageView textureImageView;
VkSampler textureSampler;
VkBuffer vertexBuffer;
VkDeviceMemory vertexBufferMemory;
VkBuffer indexBuffer;
VkDeviceMemory indexBufferMemory;

std::vector<VkBuffer> uniformBuffers;
std::vector<VkDeviceMemory> uniformBuffersMemory;
std::vector<VkDescriptorSet> descriptorSets;
VkDescriptorSetLayout descriptorSetLayout;
VkPipelineLayout pipelineLayout;
VkPipeline graphicsPipeline;
struct ModelData{
    VkDeviceSize vertexOffset;
    VkDeviceSize indexOffset;
    uint32_t indexCount;
};
std::vector<ModelData> modelMap;
std::vector<glm::mat4> textureCoordinates;
std::map<const unsigned int, std::pair< std::pair<size_t, size_t>, glm::mat4 >> modelInstances;

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};
std::pair<VkImage, VkDeviceMemory> createTextureImage(const std::vector<std::string>& texturePath, const VkDevice& device, const VkPhysicalDevice& physicalDevice, const VkCommandPool& commandPool, const VkQueue& graphicsQueue);
std::pair<VkImage, VkDeviceMemory> createTexture(const std::vector<std::string>& paths, const VkDevice& dev, const VkPhysicalDevice& pDev, const VkCommandPool& cmdPool, const VkQueue& queue);
VkImageView createTextureImageView(const VkDevice& device, const VkImage& image);
VkSampler createTextureSampler(const VkDevice& device, const VkPhysicalDevice& physicalDevice);
std::pair<std::vector<VkBuffer>, std::vector<VkDeviceMemory>> createUniformBuffers(
        const VkDevice& device, const VkPhysicalDevice& physicalDevice, const size_t bufferCount);
std::vector<VkDescriptorSet> createDescriptorSets(const VkDevice& device, const VkDescriptorSetLayout& descriptorSetLayout, 
        const VkDescriptorPool& descriptorPool, const std::vector<VkBuffer> uniformBuffers, 
        const VkSampler& sampler, const VkImageView& imageView);
VkDescriptorSetLayout createDescriptorSetLayout(const VkDevice& device);
std::pair<VkPipelineLayout, VkPipeline> createGraphicsPipeline(const VkDevice& device, const VkExtent2D& swapChainExtent, const VkDescriptorSetLayout& descriptorSetLayout, const VkRenderPass& renderPass);



std::pair<std::vector<ModelData>, std::pair<std::pair<VkBuffer, VkDeviceMemory>, std::pair<VkBuffer, VkDeviceMemory>> > loadModel(const VkDevice& dev, const VkPhysicalDevice& pDev, const VkCommandPool& cmdPool, const VkQueue& queue, const std::vector<std::string>& modelPath);

Model::Model(const std::vector<std::string>& texturePath, const std::vector<std::string>& modelPaths, const VkDevice& device, const VkPhysicalDevice& physicalDevice, 
        const VkCommandPool& commandPool, const VkQueue& queue, const size_t swapchainImageCount, const VkDescriptorPool& descriptorPool, const VkExtent2D& swapChainExtent, const VkRenderPass& renderPass) {
    auto imagePair = createTextureImage(texturePath, device, physicalDevice, commandPool, queue);
    textureImage = imagePair.first;
    textureImageMemory = imagePair.second;
    textureImageView = createTextureImageView(device, textureImage);
    textureSampler = createTextureSampler(device, physicalDevice);

    auto b = createUniformBuffers(device, physicalDevice, swapchainImageCount);
    uniformBuffers = b.first;
    uniformBuffersMemory = b.second;

    descriptorSetLayout = createDescriptorSetLayout(device);
    descriptorSets = createDescriptorSets(device, descriptorSetLayout, descriptorPool, uniformBuffers, textureSampler, textureImageView);
    auto p = createGraphicsPipeline(device, swapChainExtent, descriptorSetLayout, renderPass);
    pipelineLayout = p.first;
    graphicsPipeline = p.second;

    auto m = loadModel(device, physicalDevice, commandPool, queue, modelPaths);
    modelMap = m.first;
    auto v = m.second.first;
    vertexBuffer = v.first;
    vertexBufferMemory = v.second;

    auto i = m.second.second;
    indexBuffer = i.first;
    indexBufferMemory = i.second;

}

void Model::recreateSwapchain(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const size_t swapchainImageCount, const VkDescriptorPool& descriptorPool, const VkExtent2D& swapChainExtent, const VkRenderPass& renderPass){
    auto b = createUniformBuffers(device, physicalDevice, swapchainImageCount);
    uniformBuffers = b.first;
    uniformBuffersMemory = b.second;
    descriptorSets = createDescriptorSets(device, descriptorSetLayout, descriptorPool, uniformBuffers, textureSampler, textureImageView);
    auto p = createGraphicsPipeline(device, swapChainExtent, descriptorSetLayout, renderPass);
    pipelineLayout = p.first;
    graphicsPipeline = p.second;
}

void Model::cleanupSwapchain(const VkDevice& device){
    for(int i = 0; i < uniformBuffers.size(); i++){
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

void Model::Destroy(const VkDevice& device) {
    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);
    vkDestroyImageView(device, textureImageView, nullptr);
    vkDestroySampler(device, textureSampler, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
}

UniformBufferObject ubo{};
void Model::recordCommands(VkCommandBuffer& commandBuf, uint32_t imageIndex) {
    vkCmdBindPipeline(commandBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuf, 0, 1, &vertexBuffer, offsets);
    vkCmdBindIndexBuffer(commandBuf, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(commandBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[imageIndex], 0, nullptr);
    for(const auto& m: modelInstances){
        auto model = modelMap[m.second.first.first];
        auto t = std::array<glm::mat4, 2>{m.second.second, textureCoordinates[m.second.first.second] };
        vkCmdPushConstants(commandBuf, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 2*sizeof(glm::mat4), &t);
        vkCmdDrawIndexed(commandBuf, model.indexCount, 1, model.indexOffset, model.vertexOffset, 0);
    }
}

void Model::updateUniformBuffer(const VkDevice& device, uint32_t currentImage, const VkExtent2D& swapChainExtent) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;
    void* data;
    vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
}
unsigned int id = 0;
const unsigned int Model::addModel(const size_t modelInd, const size_t textureInd, const glm::mat4& wt){
    modelInstances.insert({id, {{modelInd, textureInd}, wt}});
    id++;
    return id - 1; }
void Model::changeModel(const unsigned int id, const size_t modelInd, const size_t textureInd, const glm::mat4& wt){
    modelInstances[id] = {{modelInd, textureInd}, wt};
}
void Model::deleteModel(const unsigned ind){
   modelInstances.erase(ind);
}

std::vector<char> readFile(const std::string& filename);
std::pair<VkImage, VkDeviceMemory> createTextureImage(
        const std::vector<std::string>& texturePath, const VkDevice& device, const VkPhysicalDevice& physicalDevice,
        const VkCommandPool& commandPool, const VkQueue& graphicsQueue) {

    std::vector<char> pixelBuffer;
    std::vector<VkBufferImageCopy> bufferImageCopy{};
    std::transform(std::begin(texturePath), std::end(texturePath), std::back_inserter(bufferImageCopy), [&pixelBuffer, off = 0, imgOff = 0](const std::string& s) mutable {
            int texWidth, texHeight, texChannels;
            stbi_uc* pixels = stbi_load(s.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
            if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
            }
            auto bufferOffset = pixelBuffer.size();
            pixelBuffer.resize(pixelBuffer.size()+texWidth*texHeight*4);
            memcpy(&pixelBuffer[bufferOffset], pixels, 4*texWidth*texHeight);

            auto region = VkBufferImageCopy {
            .bufferOffset = off,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = VkImageSubresourceLayers{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageOffset = VkOffset3D{imgOff, 0, 0},
            .imageExtent = VkExtent3D{ static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1 } };
            off += 4*texWidth*texHeight;
            imgOff += texWidth;
            stbi_image_free(pixels);

            return region; });




    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    auto stagingBufferInfo = VkBufferCreateInfo {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = pixelBuffer.size(),
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };

    if (vkCreateBuffer(device, &stagingBufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    VkMemoryRequirements stagingReqs;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &stagingReqs);

    auto stagingBufferProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t stagingTypeIndex = 0;
    //check if indexed memory type for buffer is supported in physical device, then check if type includes required properties.
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((stagingReqs.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & stagingBufferProperties) == stagingBufferProperties) {
            stagingTypeIndex = i;
            break;
        }
    }
    auto stagingAllocationInfo = VkMemoryAllocateInfo {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = stagingReqs.size,
            .memoryTypeIndex = stagingTypeIndex };

    if (vkAllocateMemory(device, &stagingAllocationInfo, nullptr, &stagingBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, pixelBuffer.size(), 0, &data);
    memcpy(data, &pixelBuffer[0], pixelBuffer.size());
    vkUnmapMemory(device, stagingBufferMemory);

    auto imageExtent = std::accumulate(std::begin(bufferImageCopy), std::end(bufferImageCopy), VkExtent3D{}, [](const VkExtent3D& a, const VkBufferImageCopy& i){
                return VkExtent3D{a.width + i.imageExtent.width, std::max(a.height, i.imageExtent.height), 1};});
    float xt = 0.f;
    for(unsigned int i = 0; i < bufferImageCopy.size(); i++) {
        auto x = static_cast<float>(bufferImageCopy[i].imageExtent.width)/static_cast<float>(imageExtent.width);
        auto y = static_cast<float>(bufferImageCopy[i].imageExtent.height)/static_cast<float>(imageExtent.height);
        auto t = glm::mat4{
            x, 0.f, 0.f, 0.f,
                0.f, y, 0.f, 0.f,
                0.f, 0.f, 1.f, 0.f,
                xt, 0.f, 0.f, 1.f,
        };
        xt += x;
        textureCoordinates.push_back(t);
    }
    auto imageInfo = VkImageCreateInfo {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_SRGB,
            .extent = imageExtent,
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
    VkImage image;
    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    uint32_t typeIndex = 0;
    //check if indexed memory type for the image is supported in physical device, then check if type is required device local
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if((memRequirements.memoryTypeBits&(1<<i))&&(memProperties.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            typeIndex = i;
            break;
        }
    }

    auto textureMemoryAllocateInfo = VkMemoryAllocateInfo {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = typeIndex };
    VkDeviceMemory imageMemory;
    if (vkAllocateMemory(device, &textureMemoryAllocateInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);

    auto commandBufferAllocateInfo = VkCommandBufferAllocateInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1 };
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
    auto beginInfo = VkCommandBufferBeginInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };

    auto barrier1 = VkImageMemoryBarrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = VkImageSubresourceRange{ 
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
    auto barrier2 = VkImageMemoryBarrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = VkImageSubresourceRange{ 
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier1
            );
    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, bufferImageCopy.size(), &bufferImageCopy[0]);
    vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier2
            );
    vkEndCommandBuffer(commandBuffer);

    auto submitInfo = VkSubmitInfo {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer };
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
    return std::pair<VkImage, VkDeviceMemory> {image, imageMemory};
}
VkImageView createTextureImageView(const VkDevice& device, const VkImage& image) {
    auto subresourceRange = VkImageSubresourceRange {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1 };

    auto viewInfo = VkImageViewCreateInfo {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_SRGB,
            .subresourceRange = subresourceRange };
    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }
    return imageView;
}

VkSampler createTextureSampler(const VkDevice& device, const VkPhysicalDevice& physicalDevice) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);

    VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE};
    VkSampler sampler;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
    return sampler;
}
std::pair<VkBuffer, VkDeviceMemory> createVertexBuffer(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const VkCommandPool& commandPool, const VkQueue& graphicsQueue, const std::vector<Vertex>& vertices);
std::pair<VkBuffer, VkDeviceMemory> createIndexBuffer(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const VkCommandPool& commandPool, const VkQueue& graphicsQueue, const std::vector<uint32_t>& indices);
std::pair<std::vector<ModelData>, std::pair<std::pair<VkBuffer, VkDeviceMemory>, std::pair<VkBuffer, VkDeviceMemory>> > loadModel(const VkDevice& dev, const VkPhysicalDevice& pDev, const VkCommandPool& cmdPool, const VkQueue& queue, const std::vector<std::string>& modelPath) {
    std::vector<ModelData> modelData{};
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    for ( const auto& p : modelPath) {
        std::unordered_map<Vertex, uint32_t> uniqueVertices{};
        tinyobj::attrib_t attrib{};
        std::vector<tinyobj::shape_t> shapes{};
        std::vector<tinyobj::material_t> materials{};
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, p.c_str())) {
            throw std::runtime_error(warn + err);
        }

        VkDeviceSize vertexOffset = vertices.size(); 
        VkDeviceSize indexOffset = indices.size();
        uint32_t indexCount = 0;
        uint32_t vertexCount = 0;
        glm::vec3 bearing{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()};
        glm::vec3 minExtent{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()};
        glm::vec3 maxExtent{std::numeric_limits<float>::min(),std::numeric_limits<float>::min(),std::numeric_limits<float>::min()};
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                glm::vec3 v = glm::vec3{
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2] };
                minExtent = glm::vec3{std::min(v.x, minExtent.x),std::min(v.y, minExtent.y), std::min(v.z, minExtent.z)};
                maxExtent = glm::vec3{std::max(v.x, maxExtent.x),std::max(v.y, maxExtent.y), std::max(v.z, maxExtent.z)};
                bearing = glm::vec3{ std::min(v.x, bearing.x), std::min(v.y, bearing.y), std::min(v.z, bearing.z) };
            }
        }
        glm::vec3 extent = maxExtent - minExtent;
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{
                    .pos = glm::vec3{
                        (attrib.vertices[3 * index.vertex_index + 0] - bearing.x)/extent.x,
                        (attrib.vertices[3 * index.vertex_index + 1] - bearing.y)/extent.y,
                        (attrib.vertices[3 * index.vertex_index + 2] - bearing.z)/extent.z },
                        .color = {1.0f, 1.0f, 1.0f},
                        .texCoord = {
                            attrib.texcoords[2 * index.texcoord_index + 0],
                            1.0f - attrib.texcoords[2 * index.texcoord_index + 1] }
                };
                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = vertexCount++;
                    vertices.push_back(vertex);
                }
                indices.push_back(uniqueVertices[vertex]);
                indexCount++;
            }
        }
        modelData.push_back(ModelData{.vertexOffset = vertexOffset, .indexOffset = indexOffset, .indexCount = indexCount});
    }
    return {modelData, {createVertexBuffer(dev, pDev, cmdPool, queue, vertices), createIndexBuffer(dev, pDev, cmdPool, queue, indices)}};
}

std::pair<VkBuffer, VkDeviceMemory> createVertexBuffer(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const VkCommandPool& commandPool, const VkQueue& graphicsQueue, const std::vector<Vertex>& vertices) {
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(vertices[0]) * vertices.size(),
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    auto stagingBufferProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t stagingTypeIndex = 0;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & stagingBufferProperties) == stagingBufferProperties) {
            stagingTypeIndex = i;
            break;
        }
    }

    VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = stagingTypeIndex };

    if (vkAllocateMemory(device, &allocInfo, nullptr, &stagingBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t) bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    VkBufferCreateInfo vertexBufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(vertices[0]) * vertices.size(),
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };

    if (vkCreateBuffer(device, &vertexBufferInfo, nullptr, &vertexBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);
    uint32_t vertexTypeIndex = 0;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            vertexTypeIndex = i;
            break;
        }
    }
    VkMemoryAllocateInfo vertexAllocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = vertexTypeIndex };

    if (vkAllocateMemory(device, &vertexAllocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);

    auto commandBufferAllocateInfo = VkCommandBufferAllocateInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1 };

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{ .size = bufferSize };
    vkCmdCopyBuffer(commandBuffer, stagingBuffer, vertexBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    auto submitInfo = VkSubmitInfo {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer };

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
    return std::pair<VkBuffer, VkDeviceMemory> {vertexBuffer, vertexBufferMemory};
}
std::pair<VkBuffer, VkDeviceMemory> createIndexBuffer(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const VkCommandPool& commandPool, const VkQueue& graphicsQueue, const std::vector<uint32_t>& indices) {
    VkDeviceSize bufferSize = sizeof(uint32_t) * indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bufferSize,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    auto stagingBufferProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t stagingTypeIndex = 0;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & stagingBufferProperties) == stagingBufferProperties) {
            stagingTypeIndex = i;
            break;
        }
    }

    VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = stagingTypeIndex };

    if (vkAllocateMemory(device, &allocInfo, nullptr, &stagingBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t) bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    VkBufferCreateInfo indexBufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bufferSize,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE };

    if (vkCreateBuffer(device, &indexBufferInfo, nullptr, &indexBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    vkGetBufferMemoryRequirements(device, indexBuffer, &memRequirements);
    uint32_t indexTypeIndex = 0;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            indexTypeIndex = i;
            break;
        }
    }
    VkMemoryAllocateInfo indexAllocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = indexTypeIndex };

    if (vkAllocateMemory(device, &indexAllocInfo, nullptr, &indexBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, indexBuffer, indexBufferMemory, 0);

    auto commandBufferAllocateInfo = VkCommandBufferAllocateInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1 };

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{ .size = bufferSize };
    vkCmdCopyBuffer(commandBuffer, stagingBuffer, indexBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);
    auto submitInfo = VkSubmitInfo {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer };
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
    return std::pair<VkBuffer, VkDeviceMemory> {indexBuffer, indexBufferMemory};
}
std::pair<std::vector<VkBuffer>, std::vector<VkDeviceMemory>> createUniformBuffers(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const size_t bufferCount) {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    std::vector<VkBuffer> uniformBuffers{};
    std::vector<VkDeviceMemory> uniformBuffersMemory{};
    uniformBuffers.resize(bufferCount);
    uniformBuffersMemory.resize(bufferCount);

    for (size_t i = 0; i < bufferCount; i++) {
        VkBufferCreateInfo bufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = bufferSize,
                .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE };

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &uniformBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, uniformBuffers[i], &memRequirements);

        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        auto properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t typeIndex{};
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((memRequirements.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                typeIndex = i;
                break;
            }
        }

        VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memRequirements.size,
                .memoryTypeIndex = typeIndex };

        if (vkAllocateMemory(device, &allocInfo, nullptr, &uniformBuffersMemory[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }
        vkBindBufferMemory(device, uniformBuffers[i], uniformBuffersMemory[i], 0);
    }
    return std::pair<std::vector<VkBuffer>, std::vector<VkDeviceMemory>> {uniformBuffers, uniformBuffersMemory};
}
std::vector<VkDescriptorSet> createDescriptorSets(const VkDevice& device, const VkDescriptorSetLayout& descriptorSetLayout, const VkDescriptorPool& descriptorPool,
        const std::vector<VkBuffer> uniformBuffers, const VkSampler& sampler, const VkImageView& imageView) {
    std::vector<VkDescriptorSet> descriptorSets(uniformBuffers.size());
    std::vector<VkDescriptorSetLayout> layouts(uniformBuffers.size(), descriptorSetLayout);
    auto allocInfo = VkDescriptorSetAllocateInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(uniformBuffers.size()),
            .pSetLayouts = layouts.data()};

    descriptorSets.resize(uniformBuffers.size());
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < uniformBuffers.size(); i++) {
        VkDescriptorBufferInfo bufferInfo{
            .buffer = uniformBuffers[i],
            .offset = 0,
            .range = sizeof(UniformBufferObject)};

        VkDescriptorImageInfo imageInfo{
            .sampler = sampler,
                .imageView = imageView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        std::array<VkWriteDescriptorSet, 2> descriptorWrites{VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &bufferInfo}, VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &imageInfo}};

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
    return descriptorSets;
}

VkDescriptorSetLayout createDescriptorSetLayout(const VkDevice& device) {
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSetLayoutBinding uboLayoutBinding{
        .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .pImmutableSamplers = nullptr};

    VkDescriptorSetLayoutBinding samplerLayoutBinding{
        .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr };

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()};

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
    return descriptorSetLayout;
}
std::pair<VkPipelineLayout, VkPipeline> createGraphicsPipeline(const VkDevice& device, const VkExtent2D& swapChainExtent, const VkDescriptorSetLayout& descriptorSetLayout, const VkRenderPass& renderPass) {
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    auto vertShaderCode = readFile("build/modelV.spv");
    auto vertShaderModuleCreateInfo = VkShaderModuleCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = vertShaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(vertShaderCode.data()) };
    VkShaderModule vertShaderModule;
    if (vkCreateShaderModule(device, &vertShaderModuleCreateInfo, nullptr, &vertShaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    auto fragShaderCode = readFile("build/modelF.spv");
    auto fragShaderModuleCreateInfo = VkShaderModuleCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = fragShaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(fragShaderCode.data()) };
    VkShaderModule fragShaderModule;
    if (vkCreateShaderModule(device, &fragShaderModuleCreateInfo, nullptr, &fragShaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertShaderModule,
            .pName = "main"};
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragShaderModule,
            .pName = "main"};
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            //.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
            .primitiveRestartEnable = VK_FALSE};
    VkViewport viewport{
        .x = 0.0f,
            .y = 0.0f,
            .width = (float) swapChainExtent.width,
            .height = (float) swapChainExtent.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f};
    VkRect2D scissor{
        .offset = {0, 0},
            .extent = swapChainExtent};
    VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor};
    VkPipelineRasterizationStateCreateInfo rasterizer{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            //.cullMode = VK_CULL_MODE_BACK_BIT,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo multisampling{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE};
    VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE};
    VkPipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT};
    VkPipelineColorBlendStateCreateInfo colorBlending{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
            .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}};

    auto pushConstantRange = VkPushConstantRange { .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = 2*sizeof(glm::mat4)};

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange
    };
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }
    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .layout = pipelineLayout,
            .renderPass = renderPass,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE};
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    return std::pair<VkPipelineLayout, VkPipeline> {pipelineLayout, graphicsPipeline};
}
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}
