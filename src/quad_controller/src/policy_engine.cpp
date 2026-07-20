#include "quad_controller/policy_engine.hpp"
#include "quad_controller/cuda_raii.hpp"

#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace quad {
namespace {

class TrtLogger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << '\n';
        }
    }
};

TrtLogger g_logger;

std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open engine file: " + path);
    }
    file.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> blob(size);
    file.read(blob.data(), static_cast<std::streamsize>(size));
    if (!file) {
        throw std::runtime_error("short read on engine file: " + path);
    }
    return blob;
}

}  // namespace

// Every TensorRT and CUDA detail lives here, out of the public header.
struct TensorRtPolicy::Impl {
    TrtUniquePtr<nvinfer1::IRuntime>          runtime;
    TrtUniquePtr<nvinfer1::ICudaEngine>       engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;
    CudaStream   stream;
    DeviceBuffer input;
    DeviceBuffer output;
    std::size_t  engine_bytes{0};
};

TensorRtPolicy::TensorRtPolicy(const std::string& engine_path)
    : impl_(std::make_unique<Impl>()) {
    const std::vector<char> blob = readFile(engine_path);
    impl_->engine_bytes = blob.size();

    impl_->runtime.reset(nvinfer1::createInferRuntime(g_logger));
    if (!impl_->runtime) throw std::runtime_error("createInferRuntime failed");

    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!impl_->engine) throw std::runtime_error("engine deserialization failed");

    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) throw std::runtime_error("createExecutionContext failed");

    impl_->input  = DeviceBuffer(OBS_DIM * sizeof(float));
    impl_->output = DeviceBuffer(ACT_DIM * sizeof(float));

    impl_->context->setTensorAddress("obs", impl_->input.get());
    impl_->context->setTensorAddress("actions", impl_->output.get());
}

TensorRtPolicy::~TensorRtPolicy() = default;
TensorRtPolicy::TensorRtPolicy(TensorRtPolicy&&) noexcept = default;
TensorRtPolicy& TensorRtPolicy::operator=(TensorRtPolicy&&) noexcept = default;

JointArray TensorRtPolicy::infer(const Observation& obs) {
    JointArray action{};

    cudaCheck(cudaMemcpyAsync(impl_->input.get(), obs.data(),
                              OBS_DIM * sizeof(float),
                              cudaMemcpyHostToDevice, impl_->stream.get()),
              "cudaMemcpyAsync host->device");

    if (!impl_->context->enqueueV3(impl_->stream.get())) {
        throw std::runtime_error("enqueueV3 failed");
    }

    cudaCheck(cudaMemcpyAsync(action.data(), impl_->output.get(),
                              ACT_DIM * sizeof(float),
                              cudaMemcpyDeviceToHost, impl_->stream.get()),
              "cudaMemcpyAsync device->host");

    impl_->stream.synchronize();
    return action;
}

std::string TensorRtPolicy::describe() const {
    return "TensorRT engine, " + std::to_string(impl_->engine_bytes) + " bytes, " +
           std::to_string(OBS_DIM) + " -> " + std::to_string(ACT_DIM);
}

}  // namespace quad
