#pragma once

#include "quad_controller/policy_types.hpp"

#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// Inference behind an interface.
//
// Callers depend on IPolicyEngine, not on TensorRT. That keeps the backend
// swappable (a different runtime, or a stub in tests) and, via the pimpl in
// TensorRtPolicy, keeps TensorRT and CUDA headers out of consumers entirely.
// ---------------------------------------------------------------------------

namespace quad {

class IPolicyEngine {
public:
    virtual ~IPolicyEngine() = default;

    IPolicyEngine(const IPolicyEngine&)            = delete;
    IPolicyEngine& operator=(const IPolicyEngine&) = delete;

    // Run one forward pass. Throws on inference failure.
    [[nodiscard]] virtual JointArray infer(const Observation& obs) = 0;

    // Human-readable description of the loaded backend, for logging.
    [[nodiscard]] virtual std::string describe() const = 0;

protected:
    // Declaring the copy operations above suppresses implicit generation of the
    // move operations, so derived classes cannot be moved unless the base
    // re-enables them here.
    IPolicyEngine()                                = default;
    IPolicyEngine(IPolicyEngine&&) noexcept        = default;
    IPolicyEngine& operator=(IPolicyEngine&&) noexcept = default;
};

class TensorRtPolicy final : public IPolicyEngine {
public:
    // Loads and deserializes a serialized engine. Throws std::runtime_error if
    // the file is missing or unusable, and quad::CudaError on allocation
    // failure.
    explicit TensorRtPolicy(const std::string& engine_path);
    ~TensorRtPolicy() override;

    TensorRtPolicy(TensorRtPolicy&&) noexcept;
    TensorRtPolicy& operator=(TensorRtPolicy&&) noexcept;

    [[nodiscard]] JointArray infer(const Observation& obs) override;
    [[nodiscard]] std::string describe() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace quad
