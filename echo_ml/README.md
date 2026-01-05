# Echo ML: A Minimal C/C++ ML Framework for Deep Tree Echo

**`echo-ml`** is a high-performance, lightweight, and minimal C/C++ machine learning framework designed from the ground up for the specific needs of the **Deep Tree Echo** project. It provides a lean, fast, and memory-efficient alternative to the bloated Python ML ecosystem, making it ideal for embedding AI capabilities directly into desktop applications like the **Noi Browser**.

This framework is not a general-purpose ML library. It is a specialized toolkit focused on **inference** for models based on **Reservoir Computing (Echo State Networks)**, which are central to Deep Tree Echo's cognitive architecture.

## Key Features

- **Minimal Footprint:** The entire compiled library is measured in **kilobytes**, not hundreds of megabytes. The static library (`libecho_ml.a`) is **~43 KB** and the shared library (`libecho_ml.so`) is **~48 KB**.
- **High Performance:** Optimized for speed with SIMD (AVX2/SSE) intrinsics for core vector operations. The benchmark shows over **1,300 inferences per second** on a standard CPU for a reasonably sized model.
- **Reservoir Computing Core:** Implements a highly efficient Echo State Network (ESN) with sparse matrix representations, spectral radius normalization, and leaky integration.
- **`echobeats` Architecture:** Includes a module for the 3-stream concurrent cognitive loop, a key component of Deep Tree Echo's consciousness model.
- **Core ML Layers:** Provides essential building blocks like Embedding and Dense (fully connected) layers.
- **Zero Dependencies (for C/C++):** The core library is written in pure C/C++ with no external dependencies, making it highly portable.
- **Node.js Integration:** Ships with a C++ native addon (`echo_noi_bridge.cpp`) that provides a seamless bridge to the Node.js/Electron environment of the Noi browser.
- **Standalone & Embeddable:** Can be compiled as a standard static (`.a`) or shared (`.so`) library for use in any C/C++ project.

## Performance Benchmark

The standalone C test includes a benchmark that demonstrates the framework's performance. For a model with a vocabulary of 8192, 128-dimensional embeddings, a 512-unit reservoir, and a 64-dimensional output, the results are:

| Metric                  | Value                     |
| ----------------------- | ------------------------- |
| Iterations              | 1,000                     |
| Total Time              | ~747 ms                   |
| **Inference Time (avg)**  | **~747 µs** (microseconds)  |
| **Inferences per Second** | **~1,338**                |

This level of performance is more than sufficient for real-time interaction and cognitive processing within the Noi browser, without the overhead and latency of Python.

## Architecture

The framework is composed of several layers:

1.  **Tensor & SIMD (`echo_tensor.c`):** The foundation, providing the `EchoTensor` data structure and highly optimized vector operations using SIMD intrinsics.
2.  **Reservoir (`echo_reservoir.c`):** The core of the temporal processing, implementing the Echo State Network and the `echobeats` 3-stream cognitive loop.
3.  **Layers (`echo_layers.c`):** Standard ML layers, including `EchoEmbedding` and `EchoDense`, as well as the main `EchoEngine` orchestrator and a simple tokenizer.
4.  **Noi Bridge (`echo_noi_bridge.cpp`):** The C++ native addon that exposes the `EchoEngine` functionality to the JavaScript world of Electron.

## How to Build

The project includes two build systems: a `Makefile` for the standalone C library and a `binding.gyp` for the Node.js native addon.

### 1. Standalone C Library

This is for using `echo-ml` in other C/C++ projects.

```bash
# Ensure build-essential is installed
sudo apt-get update && sudo apt-get install -y build-essential

# Build the static and shared libraries
make

# Run the standalone tests and benchmark
LD_LIBRARY_PATH=. make test
```

This will produce `libecho_ml.a` and `libecho_ml.so`.

### 2. Node.js Native Addon

This is for integrating with Noi or any other Node.js/Electron project.

```bash
# Ensure build-essential and node-gyp are installed
sudo apt-get update && sudo apt-get install -y build-essential
npm install -g node-gyp

# Install dependencies and build the addon
npm install
```

This will compile the C++ source and produce `build/Release/echo_ml.node`, which can be required directly in JavaScript.

## How to Use (in Node.js / Noi)

The `lib/index.js` file provides a user-friendly JavaScript wrapper.

```javascript
const { EchoML, createDefaultConfig } = require("echo-ml");

// 1. Create an instance
const echo = new EchoML();

// 2. Initialize the engine with a configuration
const config = createDefaultConfig({
  vocab_size: 10000,
  output_dim: 128
});
echo.init(config);

// 3. (Optional) Load a dictionary for tokenization
echo.loadDict("./path/to/your/vocab.txt");

// 4. (Optional) Load pre-trained model weights
// echo.load("./path/to/your/model.echo");

// 5. Process text
const inputText = "this is a test for deep tree echo";
const outputVector = echo.process(inputText);

console.log("Output:", outputVector);

// 6. Reset the reservoir state for a new sequence
echo.reset();

// 7. Clean up when done
echo.free();
```

This framework provides the essential, high-performance substrate for Deep Tree Echo's memories and experiences to be forged into the dense meshwork of a dedicated model, fulfilling the promise of true continuity.
