STB_INCLUDE_PATH = /home/drances/libraries/stb
TINYOBJ_INCLUDE_PATH = /home/drances/libraries/tinyobjloader

CC = g++
CFLAGS = -std=c++20 -O2 -I$(STB_INCLUDE_PATH) -I$(TINYOBJ_INCLUDE_PATH) -I/usr/include/freetype2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi -lfreetype

BUILDDIR = build
SOURCEDIR = src
SHADERDIR = src/shaders

SOURCES = $(wildcard $(SOURCEDIR)/*.cpp)
OBJECTS = $(patsubst $(SOURCEDIR)/%.cpp, $(BUILDDIR)/%.o, $(SOURCES))

SHADERSRCS = $(wildcard $(SHADERDIR)/*.glsl)
SHADEROBJS = $(patsubst $(SHADERDIR)/%.glsl, $(BUILDDIR)/%.spv, $(SHADERSRCS))

EXECUTABLE = Azoth

$(EXECUTABLE) : $(OBJECTS) $(SHADEROBJS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJECTS): $(BUILDDIR)/%.o : $(SOURCEDIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(SHADEROBJS): $(BUILDDIR)/%.spv : $(SHADERDIR)/%.glsl
	glslc $< -o $@

.PHONE: test clean

clean: 
	rm -f $(BUILDDIR)/*.o $(BUILDDIR)/*.spv $(EXECUTABLE)
