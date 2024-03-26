#ifndef MLLM_CPUBACKEND_H
#define MLLM_CPUBACKEND_H

#include "Backend.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include "quantize/Quantize.hpp"

namespace mllm {
class CPUBackend final : public Backend {
public:
    explicit CPUBackend(shared_ptr<MemoryManager> &mm);
    ~CPUBackend() override = default;

    class Creator {
    public:
        virtual Op *create(OpParam op_param, Backend *bn, string, int threadCount) const = 0;
    };
    bool addCreator(OpType t, Creator *c) {
        if (map_creator_.find(t) != map_creator_.end()) {
            printf("Error: %d type has be added\n", t);
            return false;
        }
        map_creator_.insert(std::make_pair(t, c));
        return true;
    }
    Op *opCreate(const OpParam &op_param, string name, int threadCount) override;
    TensorFunction *funcCreate(const TensorFuncType type) override;

    void registerOps() override;
    void registerFuncs() override;

    static int cpu_threads;

private:
    std::map<OpType, CPUBackend::Creator *> map_creator_;
    std::map<TensorFuncType, TensorFunction *> map_function_;
};

} // namespace mllm

#endif // MLLM_CPUBACKEND_H