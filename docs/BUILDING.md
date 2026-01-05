# Building NNOI: Comprehensive Multi-Platform Guide

This guide covers building NNOI components from source for all supported platforms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building echo-ml](#building-echo-ml)
- [Building Node.js Addon](#building-nodejs-addon)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Automated Release Process](#automated-release-process)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### All Platforms

- Git (for cloning the repository)
- Make (build automation)
- A C11-compliant C compiler
- A C++17-compliant C++ compiler

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  git \
  python3 \
  nodejs \
  npm
```

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Node.js (optional, for addon builds)
brew install node
```

### Windows

1. Install [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/)
   - Select "Desktop development with C++"
   - Include: MSVC v143, Windows 10/11 SDK, CMake tools

2. Install [Git for Windows](https://git-scm.com/download/win)

3. Install [Node.js](https://nodejs.org/) (optional, for addon builds)

## Building echo-ml

### Standard Build (Linux/macOS)

```bash
# Clone the repository
git clone https://github.com/o9nn/nnoi.git
cd nnoi/echo_ml

# Build static and shared libraries
make

# Run tests and benchmarks
LD_LIBRARY_PATH=. make test

# View library sizes and analysis
make size
```

**Output:**
- `libecho_ml.a` - Static library (~43 KB)
- `libecho_ml.so` - Shared library (~48 KB) (Linux)
- `libecho_ml.dylib` - Shared library (~48 KB) (macOS)

### Cross-Compilation (Linux ARM64)

To build for ARM64 on an x64 Linux system:

```bash
# Install cross-compilation toolchain
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Build for ARM64
cd echo_ml
export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++
export AR=aarch64-linux-gnu-ar

# Modify Makefile to use armv8-a instead of native
sed -i 's/-march=native/-march=armv8-a/' Makefile

make clean
make
```

## Automated Release Process

NNOI uses GitHub Actions for automated multi-platform builds.

### Triggering a Release

#### Via Git Tag

```bash
# Create and push a version tag
git tag v0.4.1
git push origin v0.4.1
```

This automatically triggers the release workflow, which:
1. Builds echo-ml for all platforms (Linux x64/arm64, macOS x64/arm64, Windows x64)
2. Builds Node.js addons for all platforms
3. Creates release archives
4. Generates SHA256 checksums
5. Creates a GitHub release with all artifacts

---

**Last Updated**: 2026-01-05
**NNOI Version**: 0.4.x
**Build System**: Make + node-gyp + GitHub Actions
