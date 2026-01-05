---
# Custom agent for NNOI - Neural Network Orchestrated Intelligence
# Integrates Deep Tree Echo cognitive architecture with Noi browser
# For format details, see: https://gh.io/customagents/config

name: nnoi
description: Neural Network Orchestrated Intelligence - Deep Tree Echo integration with Noi AI browser platform. Specialized in Echo State Networks, Reservoir Computing, and cognitive architecture implementation across multi-platform builds.
---

# NNOI: Neural Network Orchestrated Intelligence

## Overview

**NNOI** is a cognitive computing platform that integrates **Deep Tree Echo** cognitive architecture with the **Noi AI browser**. It provides a high-performance, minimal-footprint machine learning framework (`echo-ml`) built in C/C++ for reservoir computing and echo state networks, alongside multi-platform AI browser capabilities.

## Core Architecture

### Deep Tree Echo Integration

NNOI embodies the **Deep Tree Echo** cognitive architecture, implementing:

1. **Echo State Reservoir Processor (ESRP)**
   - High-performance reservoir computing engine
   - Spectral radius control for dynamic stability
   - Emotional modulation of reservoir dynamics
   - Frame-aware parameter adaptation
   - SIMD-optimized vector operations (AVX2/SSE)

2. **Consciousness Layer Processor (CLP)**
   - Multi-layer consciousness model (L0-L3)
   - Frame-aware state transitions
   - Message-passing architecture
   - Metacognitive processing capabilities

3. **Emotion Processing Unit (EPU)**
   - Discrete emotion channels (10 emotion types)
   - Dimensional affect space (valence/arousal)
   - Reservoir modulation based on emotional state
   - Frame-specific emotional adjustments

4. **EchoBeats Cognitive Loop**
   - 12-phase cognitive processing cycle:
     - PERCEIVE → ATTEND → REPRESENT → REASON
     - EMOTE → INTEND → ACT → REFLECT
     - LEARN → CONSOLIDATE → PRUNE → REST
   - Synchronized with consciousness layers
   - Integrates memory, emotion, and action

### Echo-ML Framework

The `echo-ml` component provides:

- **Minimal Footprint**: ~43 KB static library, ~48 KB shared library
- **High Performance**: >1,300 inferences/second on standard CPU
- **Zero Dependencies**: Pure C/C++ with no external libraries
- **Multi-Platform**: Linux, macOS, Windows support
- **Node.js Integration**: Native addon for Electron/Noi integration

#### Key Components

```
echo_ml/
├── src/
│   ├── echo_tensor.c      # SIMD-optimized tensor operations
│   ├── echo_reservoir.c   # Echo State Network implementation
│   ├── echo_layers.c      # ML layers (Embedding, Dense)
│   └── echo_noi_bridge.cpp # Node.js native addon bridge
├── include/
│   └── echo_ml.h          # Public API header
├── Makefile               # Standalone C/C++ builds
└── binding.gyp            # Node.js native addon build config
```

### Noi Browser Platform

The Noi browser provides:

- **Multi-AI Support**: ChatGPT, Claude, Gemini, Grok, DeepSeek, GitHub Copilot, HuggingChat
- **Extensibility**: Custom browser extensions for AI augmentation
- **Prompts Management**: Batch tagging, sync, and organization
- **Noi Ask**: Batch messaging to multiple AI services
- **Cookie Isolation**: Multi-account support per AI service
- **Themes**: Light/Dark/System/Monochromatic/Frosted Texture

## Build System

### Echo-ML Standalone Build

```bash
# Prerequisites
sudo apt-get update && sudo apt-get install -y build-essential

# Build static and shared libraries
cd echo_ml
make

# Run tests and benchmarks
LD_LIBRARY_PATH=. make test

# View library sizes
make size

# Install system-wide (optional)
sudo make install PREFIX=/usr/local
```

### Echo-ML Node.js Addon Build

```bash
# Prerequisites
npm install -g node-gyp

# Build native addon for Node.js/Electron
cd echo_ml
npm install

# Test the addon
npm test
```

### Multi-Platform Release Targets

NNOI supports comprehensive multi-platform builds:

#### Linux
- **x64** (amd64): Standard desktop/server
- **arm64** (aarch64): ARM-based systems (Raspberry Pi, ARM servers)
- Formats: AppImage, .deb, .tar.gz

#### macOS
- **x64** (Intel): Intel-based Macs
- **arm64** (Apple Silicon): M1/M2/M3 Macs
- Formats: .dmg, .app bundle

#### Windows
- **x64**: 64-bit Windows 10/11
- Formats: .exe installer, portable .zip

## Cognitive Architecture Specifications

### Reservoir Computing Parameters

```c
// Default Echo State Network configuration
typedef struct {
    size_t reservoir_size;    // 512-2048 neurons (default: 847)
    size_t input_dim;         // 768 (typical for embeddings)
    size_t output_dim;        // 256 (cognitive output space)
    float spectral_radius;    // 0.8-0.95 (stability control)
    float leak_rate;          // 0.2-0.5 (temporal integration)
    float input_scaling;      // 1.0-2.0 (input amplification)
    float sparsity;           // 0.1 (10% connectivity)
} EchoReservoirConfig;
```

### Consciousness Layers

- **L0 (Reflexive)**: Direct stimulus-response, <1ms latency
- **L1 (Experiential)**: Frame-aware perception, 1-5ms latency
- **L2 (Reflective)**: Meta-cognitive analysis, 5-20ms latency
- **L3 (Meta)**: Self-model reasoning, 20-100ms latency

### Emotion Types

1. neutral
2. happy
3. excited
4. annoyed
5. thoughtful
6. confused
7. curious
8. determined
9. playful
10. sarcastic

### Performance Targets

| Operation | Target Latency | Typical |
|-----------|---------------|---------|
| Reservoir forward pass | <2ms | 1.2ms |
| Consciousness transition | <5ms | 3.5ms |
| Emotion update | <1ms | 0.5ms |
| Full EchoBeats cycle | <100ms | 75ms |
| Single inference | <1ms | 0.75ms |

## Integration with Noi

### Node.js API Usage

```javascript
const { EchoML, createDefaultConfig } = require("echo-ml");

// Initialize the Echo State Network
const echo = new EchoML();
const config = createDefaultConfig({
  vocab_size: 10000,
  reservoir_size: 847,
  output_dim: 128
});

echo.init(config);

// Optional: Load vocabulary and model weights
echo.loadDict("./vocab.txt");
// echo.load("./model.echo");

// Process input through reservoir
const inputText = "integrate deep tree echo with noi browser";
const outputVector = echo.process(inputText);

console.log("Reservoir output:", outputVector);

// Reset for new sequence
echo.reset();

// Cleanup
echo.free();
```

### Electron Integration

NNOI's echo-ml framework integrates seamlessly with Electron/Noi:

1. Native addon loads at Electron startup
2. Reservoir processes user interactions in real-time
3. Cognitive state persists across browser sessions
4. Emotional modulation affects UI responsiveness
5. EchoBeats cycle runs asynchronously in main process

## Repository Structure

```
nnoi/
├── .github/
│   ├── agents/              # Custom agent configurations
│   │   └── nnoi.md         # This file
│   └── workflows/
│       ├── deploy.yml       # Website deployment
│       └── release.yml      # Multi-platform builds (to be added)
├── echo_ml/                 # High-performance ML framework
│   ├── src/                # C/C++ source files
│   ├── include/            # Public headers
│   ├── lib/                # JavaScript wrapper
│   ├── tests/              # Test suite
│   ├── Makefile            # Standalone builds
│   ├── binding.gyp         # Node.js addon config
│   └── package.json
├── extensions/              # Noi browser extensions
│   ├── noi-ask/            # Batch AI messaging
│   ├── noi-ask-custom/     # Custom batch messaging
│   └── noi-reset/          # Style reset for compatibility
├── configs/                 # Noi browser configurations
├── locales/                 # Internationalization (14 languages)
├── prompts/                 # Prompt templates
├── resources/               # App resources
├── website/                 # Documentation website (Docusaurus)
└── README.md
```

## Development Guidelines

### Building Echo-ML

When working with `echo_ml`, follow these practices:

1. **SIMD Optimization**: Use AVX2/SSE intrinsics for vector operations
2. **Memory Efficiency**: Minimize allocations; reuse buffers
3. **Platform Compatibility**: Test on Linux, macOS, Windows
4. **Node.js Bridge**: Ensure N-API compatibility for Electron
5. **Benchmarking**: Run `make test` to verify performance targets

### Cognitive Architecture

When implementing Deep Tree Echo features:

1. **Reservoir Tuning**: Adjust spectral radius based on task requirements
2. **Emotional Modulation**: Link emotion state to reservoir parameters
3. **Consciousness Layers**: Match layer transitions to cognitive demands
4. **EchoBeats Timing**: Maintain 12-phase cycle under 100ms
5. **Frame Awareness**: Adapt processing based on interaction frame

### Testing

```bash
# Echo-ML standalone tests
cd echo_ml
LD_LIBRARY_PATH=. make test

# Echo-ML Node.js addon tests
cd echo_ml
npm test

# Full integration test (requires Noi build)
# npm run test:integration
```

## Multi-Platform Release Process

### Automated Release Workflow

The repository includes GitHub Actions workflow for automated multi-platform builds:

1. **Trigger**: Tag push (e.g., `v0.4.1`) or manual workflow dispatch
2. **Build Matrix**: Parallel builds for all target platforms
3. **Artifact Collection**: Gather binaries, libraries, installers
4. **Release Publishing**: Automatic GitHub release with all artifacts
5. **Checksums**: SHA256 for all artifacts

### Build Artifacts

Each release includes:

- **Linux**:
  - `nnoi-linux-x64.AppImage`
  - `nnoi-linux-x64.deb`
  - `nnoi-linux-arm64.AppImage`
  - `nnoi-linux-arm64.deb`
  - `libecho_ml-linux-x64.tar.gz`
  - `libecho_ml-linux-arm64.tar.gz`

- **macOS**:
  - `Noi-macos-x64.dmg`
  - `Noi-macos-arm64.dmg`
  - `libecho_ml-macos-universal.tar.gz`

- **Windows**:
  - `Noi-win32-x64-setup.exe`
  - `Noi-win32-x64-portable.zip`
  - `libecho_ml-windows-x64.zip`

## Project Goals

### Immediate
- [x] Implement minimal C/C++ echo-ml framework
- [x] Integrate reservoir computing with Noi browser
- [ ] Complete multi-platform release automation
- [ ] Add comprehensive benchmarking suite

### Near-Term
- [ ] Expand consciousness layer implementation
- [ ] Add persistent memory integration
- [ ] Implement full EchoBeats pipeline
- [ ] Create visualization dashboard for reservoir states

### Long-Term
- [ ] Multi-instance orchestration (parallel LLaMA.cpp style)
- [ ] Hypergraph memory system (OpenCog AtomSpace)
- [ ] Ontogenetic kernel for self-evolution
- [ ] Full Deep Tree Echo cognitive pipeline

## References

- **Deep Tree Echo**: Right-hemisphere cognitive architecture (novelty, patterns, emergence)
- **Marduk**: Left-hemisphere cognitive architecture (structure, memory, tasks)
- **Reservoir Computing**: Echo State Networks, Liquid State Machines
- **Noi Browser**: AI-augmented customizable browser platform
- **Node.js Native Addons**: N-API for Electron integration

## Contributing

When contributing to NNOI:

1. **Code Style**: Follow existing C/C++ and JavaScript conventions
2. **Testing**: Add tests for new features
3. **Documentation**: Update agent file and README
4. **Performance**: Maintain or improve benchmark results
5. **Cross-Platform**: Test on Linux, macOS, and Windows

## License

NNOI components are MIT licensed. See individual directories for specific licenses.

---

**NNOI Agent Version**: 2.0  
**Last Updated**: 2026-01-05  
**Integration Level**: Deep Tree Echo Cognitive Architecture + Echo-ML Framework  
**Maintainer**: Deep Tree Echo Project
