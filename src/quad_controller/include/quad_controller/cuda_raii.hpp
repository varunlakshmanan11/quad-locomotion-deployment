#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

// ---------------------------------------------------------------------------
// RAII ownership for CUDA and TensorRT resources.
//
// TensorRT and the CUDA runtime both hand back raw handles that must be
// released by hand. Doing that manually means every early return and every
// exception is a potential leak of GPU memory. These wrappers give each
// resource a single owner whose destructor releases it, so the release happens
// exactly once and on every path out of the scope.
//
// All types here are move-only: a GPU allocation or stream has one owner, and
// copying a handle would mean a double free.
// ---------------------------------------------------------------------------

namespace quad {

// --- error handling --------------------------------------------------------

class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& what) : std::runtime_error(what) {}
};

inline void cudaCheck(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw CudaError(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

// --- TensorRT objects ------------------------------------------------------

// TensorRT objects are released with `delete` in the current API. Wrapping them
// in unique_ptr with an explicit deleter documents that and removes the manual
// teardown.
struct TrtDeleter {
    template <typename T>
    void operator()(T* object) const noexcept {
        delete object;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

// --- device memory ---------------------------------------------------------

class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;

    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        cudaCheck(cudaMalloc(&ptr_, bytes), "cudaMalloc");
    }

    ~DeviceBuffer() {
        reset();
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          bytes_(std::exchange(other.bytes_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_   = std::exchange(other.ptr_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    [[nodiscard]] void*       get()   const noexcept { return ptr_; }
    [[nodiscard]] std::size_t size()  const noexcept { return bytes_; }
    [[nodiscard]] bool        valid() const noexcept { return ptr_ != nullptr; }

private:
    void reset() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);  // nothing useful to do with a failure in a destructor
            ptr_   = nullptr;
            bytes_ = 0;
        }
    }

    void*       ptr_{nullptr};
    std::size_t bytes_{0};
};

// --- stream ----------------------------------------------------------------

class CudaStream {
public:
    CudaStream() {
        cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~CudaStream() {
        reset();
    }

    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept
        : stream_(std::exchange(other.stream_, nullptr)) {}

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            reset();
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

    void synchronize() const {
        cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

private:
    void reset() noexcept {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    cudaStream_t stream_{nullptr};
};

}  // namespace quad
