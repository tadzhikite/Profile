#include "text.h"
#include <functional>
#include <algorithm>
#include <iterator>
#include <numeric>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

const std::function<bool(const glm::vec2&)> Text::hover(const unsigned int msgID){
    return [this, msgID](const glm::vec2& cursor){ 
        auto bounds = std::accumulate(std::begin(text[msgID]), std::end(text[msgID]), std::array<float, 4>{1.f, 1.f, -1.f, -1.f}, [](const std::array<float, 4>& m, const std::array<glm::mat4, 2>& a){ 
            return std::array<float, 4>{ std::min(m[0], a[0][3].x), std::min(m[1], a[0][3].y), std::max(m[2], a[0][3].x + a[0][0].x), std::max(m[3], a[0][3].y - a[0][1].y) }; });  
        return (cursor[0] > bounds[0]) && (cursor[0] < bounds[2]) && (cursor[1] > bounds[1]) && (cursor[1] < bounds[3]);
    };
}
Text::Text(const std::string& filePath, const VkDevice& device, const VkPhysicalDevice& physicalDevice, 
        const VkCommandPool& commandPool, const VkQueue& queue, const size_t swapchainImageCount, const VkDescriptorPool& descriptorPool, const VkExtent2D& swapChainExtent, const VkRenderPass& renderPass) {
    auto g = createTextureImage(filePath, device, physicalDevice, commandPool, queue);
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
}
void Text::recreateSwapchain(const VkDevice& device, const VkPhysicalDevice& physicalDevice, const size_t swapchainImageCount, const VkDescriptorPool& descriptorPool, const VkExtent2D& swapChainExtent, const VkRenderPass& renderPass){
    descriptorSet = createDescriptorSet(device, descriptorSetLayout, descriptorPool, faceSampler, faceImageView);
    auto p = createTextPipeline(device, swapChainExtent, descriptorSetLayout, renderPass);
    pipelineLayout = p.first;
    textPipeline = p.second;
}
void Text::cleanupSwapchain(const VkDevice& device){
    vkDestroyPipeline(device, textPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}
void Text::Destroy(const VkDevice& device) {
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
const unsigned int Text::addText(const std::string& msg, const glm::vec2& origin, const float extent) {
    text.insert({id, tiles(msg, origin, extent)});
    id++;
    return id - 1;
}
void Text::changeText(const unsigned int id, const std::string& msg, const glm::vec2& origin, const float extent) {
    text.erase(id);
    text.insert({id, tiles(msg, origin, extent)});
}
void Text::removeText(const unsigned int id) {
    text.erase(id);
}
void Text::recordCommands(VkCommandBuffer& commandBuf) {
    vkCmdBindDescriptorSets(commandBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdBindPipeline(commandBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipeline);
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuf, 0, 1, &vertexBuffer, offsets);
    vkCmdBindIndexBuffer(commandBuf, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
    for(const auto& t : text){
        for(const auto& tile : t.second){
            vkCmdPushConstants(commandBuf, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 2*sizeof(glm::mat4), &tile);
            vkCmdDrawIndexed(commandBuf, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        }
    }
}
std::vector<std::array<glm::mat4, 2>> Text::tiles(const std::string& msg, const glm::vec2& origin, const float extent){
    float m = extent/std::accumulate(std::begin(msg), std::end(msg), 0.f, [this](const float& f, const char& c){ return f + static_cast<float>(glyphDims[c].advance/64); });
    float adv = 0.f;
    std::vector<std::array<glm::mat4, 2>> t{};
    for(int i = 0; i < msg.size(); i++) {
        float xb = m*static_cast<float>(glyphDims[msg[i]].bearing[0]);
        float yb = -m*static_cast<float>(glyphDims[msg[i]].bearing[1]);
        t.push_back(std::array<glm::mat4, 2> {
                glm::mat4{ 
                m*static_cast<float>(glyphDims[msg[i]].width), 0.f, 0.f, 0.f,
                0.f, m*static_cast<float>(glyphDims[msg[i]].height), 0.f, 0.f,
                0.f, 0.f, 1.f, 0.f,
                origin[0] + xb + adv, -origin[1] + yb, 0.f, 1.f},
                textureCoordinateTransforms[msg[i]] });
        adv += m*static_cast<float>(glyphDims[msg[i]].advance/64);
    }
    return t;
}
std::vector<char> Text::readFile(const std::string& filename) {
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
std::pair<VkPipelineLayout, VkPipeline> Text::createTextPipeline(const VkDevice& device, const VkExtent2D& swapChainExtent, const VkDescriptorSetLayout& descriptorSetLayout, const VkRenderPass& renderPass) {
    std::cout << "CREATING TEXT PIPELINE!\n";
    VkPipelineLayout pipelineLayout;
    VkPipeline textPipeline;
    auto vertShaderCode = readFile("build/textV.spv");
    auto vertShaderModuleCreateInfo = VkShaderModuleCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = vertShaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(vertShaderCode.data()) };
    VkShaderModule vertShaderModule;
    if (vkCreateShaderModule(device, &vertShaderModuleCreateInfo, nullptr, &vertShaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    auto fragShaderCode = readFile("build/textF.spv");
    auto fragShaderModuleCreateInfo = VkShaderModuleCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = fragShaderCode.size(),
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
    auto bindingDescription = VkVertexInputBindingDescription {
        .binding = 0, .stride = 2*sizeof(glm::vec2), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};

    auto attributeDescriptions = std::vector<VkVertexInputAttributeDescription>{VkVertexInputAttributeDescription{
        .location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = 0}, {
            .location = 1, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = sizeof(glm::vec2)} };
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
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
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
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
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &textPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    return std::pair<VkPipelineLayout, VkPipeline> {pipelineLayout, textPipeline};
}
std::pair<VkImage, VkDeviceMemory> Text::createTextureImage(const std::string& filePath, const VkDevice& device, 
        const VkPhysicalDevice& physicalDevice, const VkCommandPool& commandPool, const VkQueue& queue){

    FT_Init_FreeType(&library);
    FT_New_Face(library, filePath.c_str(), 0, &face);
    FT_Set_Pixel_Sizes(face, 0, 500);

    std::array<VkBufferImageCopy, ASCII_COUNT> bufferImageCopy{};
    std::vector<char> faceBuffer{};
    uint32_t offset = 0;
    int32_t imgOff = 0;
    for(unsigned int i = 0; i < ASCII_COUNT; i++) {
        if(FT_Load_Char(face, i, FT_LOAD_RENDER)){ std::cout << "error::FREETYPE: Failed to load Glyph\n";}

        uint32_t size = face->glyph->bitmap.width*face->glyph->bitmap.rows;
        glyphDims[i] = GlyphDims{
            .c = static_cast<char>(i),
                .bearing = glm::ivec2{face->glyph->bitmap_left, face->glyph->bitmap_top},
                .advance = static_cast<unsigned int>(face->glyph->advance.x),
                .width = face->glyph->bitmap.width,
                .height = face->glyph->bitmap.rows,
                .size = size,
                .offset = offset
        };
        bufferImageCopy[i] = VkBufferImageCopy {
            .bufferOffset = offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = VkImageSubresourceLayers { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                .imageOffset = VkOffset3D{imgOff, 0, 0},
                .imageExtent = VkExtent3D{face->glyph->bitmap.width, face->glyph->bitmap.rows, 1}
        };
        offset += size;
        imgOff += face->glyph->bitmap.width;

    }
    faceBuffer.resize(std::accumulate(std::begin(glyphDims), std::end(glyphDims), 0, [](const size_t& s, const GlyphDims& d){ return s+d.size; }));
    for(unsigned int i = 0; i < ASCII_COUNT; i++) {
        if(FT_Load_Char(face, i, FT_LOAD_RENDER)){ std::cout << "error::FREETYPE: Failed to load Glyph\n";}
        memcpy(&faceBuffer[glyphDims[i].offset], face->glyph->bitmap.buffer, glyphDims[i].size);
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    auto stagingBufferInfo = VkBufferCreateInfo {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = faceBuffer.size(),
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
    vkMapMemory(device, stagingBufferMemory, 0, faceBuffer.size(), 0, &data);
    memcpy(data, &faceBuffer[0], faceBuffer.size());
    vkUnmapMemory(device, stagingBufferMemory);
    auto imageExtent = std::accumulate(std::begin(glyphDims), std::end(glyphDims), VkExtent3D{}, [](const VkExtent3D& a, const GlyphDims& d){
            return VkExtent3D{a.width + d.width, std::max(a.height, d.height), 1};});
    float xt = 0.f;
    for(unsigned int i = 0; i < ASCII_COUNT; i++) {
        if(FT_Load_Char(face, i, FT_LOAD_RENDER)){ std::cout << "error::FREETYPE: Failed to load Glyph\n";}
        auto x = static_cast<float>(face->glyph->bitmap.width)/static_cast<float>(imageExtent.width);
        auto y = static_cast<float>(face->glyph->bitmap.rows)/static_cast<float>(imageExtent.height);
        auto t = glm::mat4{
            x, 0.f, 0.f, 0.f,
                0.f, y, 0.f, 0.f,
                0.f, 0.f, 1.f, 0.f,
                xt, 0.f, 0.f, 1.f,
        };
        xt += x;
        textureCoordinateTransforms[i] = t;
    }

    auto imageInfo = VkImageCreateInfo {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8_SRGB,
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
    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ASCII_COUNT, &bufferImageCopy[0]);
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
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
    return std::pair<VkImage, VkDeviceMemory> {image, imageMemory};
}
VkDescriptorSet Text::createDescriptorSet(const VkDevice& device, const VkDescriptorSetLayout& descriptorSetLayout, 
        const VkDescriptorPool& descriptorPool, const VkSampler& sampler, const VkImageView& imageView) {
    VkDescriptorSet descriptorSet;
    auto allocInfo = VkDescriptorSetAllocateInfo {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptorSetLayout };

    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    VkDescriptorImageInfo imageInfo{
        .sampler = sampler,
            .imageView = imageView,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    auto descriptorWrite = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &imageInfo};

    vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    return descriptorSet;
}
VkDescriptorSetLayout Text::createTextDescriptorSetLayout(const VkDevice& device) {
    VkDescriptorSetLayout descriptorSetLayout;

    VkDescriptorSetLayoutBinding samplerLayoutBinding{
        .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr };

    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &samplerLayoutBinding };

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
    return descriptorSetLayout;
}
std::pair<VkBuffer, VkDeviceMemory> Text::createIndexBuffer(const VkDevice& device, const VkPhysicalDevice& physicalDevice, 
        const VkCommandPool& commandPool, const VkQueue& graphicsQueue) {
    VkDeviceSize bufferSize = sizeof(uint16_t) * 6;

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
std::pair<VkBuffer, VkDeviceMemory> Text::createVertexBuffer(const VkDevice& device, const VkPhysicalDevice& physicalDevice, 
        const VkCommandPool& commandPool, const VkQueue& graphicsQueue) {
    VkDeviceSize bufferSize = 8*sizeof(glm::vec2);
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
    std::array<glm::vec2, 8> vertices{
        glm::vec2{1.f, 0.f}, {1.f, 0.f},
            {1.f, 1.f}, {1.f, 1.f},
            {0.f, 0.f}, {0.f, 0.f},
            {0.f, 1.f}, {0.f, 1.f}
    };

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t) bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    VkBufferCreateInfo vertexBufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bufferSize,
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
VkSampler Text::createFaceSampler(const VkDevice& device, const VkPhysicalDevice& physicalDevice) {
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
VkImageView Text::createFaceImageView(const VkDevice& device, const VkImage& image) {
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
            .format = VK_FORMAT_R8_SRGB,
            .subresourceRange = subresourceRange };
    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }
    return imageView;
}
