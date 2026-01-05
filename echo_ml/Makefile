# Echo ML - Makefile for standalone C/C++ compilation
#
# This builds the library for use outside of Node.js (e.g., embedded systems,
# direct C/C++ integration, or as a shared library).

CC = gcc
CXX = g++
AR = ar

# Optimization flags
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -std=c11
CXXFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -std=c++17

# SIMD flags (adjust for target architecture)
SIMD_FLAGS = -mavx2 -mfma

# Include paths
INCLUDES = -Iinclude

# Source files
C_SOURCES = src/echo_tensor.c src/echo_reservoir.c src/echo_layers.c
OBJECTS = $(C_SOURCES:.c=.o)

# Output
LIB_STATIC = libecho_ml.a
LIB_SHARED = libecho_ml.so
TEST_BIN = test_echo_ml

# Default target
all: $(LIB_STATIC) $(LIB_SHARED)

# Static library
$(LIB_STATIC): $(OBJECTS)
	$(AR) rcs $@ $^
	@echo "Built static library: $@"
	@ls -lh $@

# Shared library
$(LIB_SHARED): $(OBJECTS)
	$(CC) -shared -o $@ $^ -lm
	@echo "Built shared library: $@"
	@ls -lh $@

# Object files
%.o: %.c include/echo_ml.h
	$(CC) $(CFLAGS) $(SIMD_FLAGS) $(INCLUDES) -fPIC -c $< -o $@

# Test binary
test: $(LIB_STATIC) tests/test_standalone.c
	$(CC) $(CFLAGS) $(SIMD_FLAGS) $(INCLUDES) tests/test_standalone.c -L. -lecho_ml -lm -o $(TEST_BIN)
	./$(TEST_BIN)

# Clean
clean:
	rm -f $(OBJECTS) $(LIB_STATIC) $(LIB_SHARED) $(TEST_BIN)

# Install (to /usr/local by default)
PREFIX ?= /usr/local
install: $(LIB_STATIC) $(LIB_SHARED)
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install -m 644 $(LIB_STATIC) $(PREFIX)/lib/
	install -m 755 $(LIB_SHARED) $(PREFIX)/lib/
	install -m 644 include/echo_ml.h $(PREFIX)/include/

# Size analysis
size: $(LIB_STATIC) $(LIB_SHARED)
	@echo "\n=== Library Size Analysis ==="
	@echo "Static library:"
	@ls -lh $(LIB_STATIC)
	@size $(LIB_STATIC) 2>/dev/null || true
	@echo "\nShared library:"
	@ls -lh $(LIB_SHARED)
	@echo "\nObject files:"
	@ls -lh $(OBJECTS)
	@echo "\nTotal source lines:"
	@wc -l src/*.c include/*.h | tail -1

# Debug build
debug: CFLAGS = -O0 -g -Wall -Wextra -std=c11 -DDEBUG
debug: clean all

.PHONY: all test clean install size debug
