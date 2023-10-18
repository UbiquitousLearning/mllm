#ifndef MLLM_CPUBACKEND_H
#define MLLM_CPUBACKEND_H

#include "Backend.hpp"
#include "Op.hpp"
#include "Types.hpp"
namespace mllm {
class CPUBackend final : public Backend {
public:
    CPUBackend(shared_ptr<MemoryManager> &mm);
    ~CPUBackend() = default;

    class Creator {
    public:
        // virtual Op* Create(const vector<shared_ptr<Tensor>>& inputs, const vector<shared_ptr<Tensor>>& outputs,
        //                             OpParam op_param, Backend* backend) const = 0;
        virtual Op *create(OpParam op_param, Backend *bn, string name) const = 0;
    };
    void initCreatorMap() {
        map_creator_ = new std::map<OpType, CPUBackend::Creator *>;
    }
    bool addCreator(OpType t, Creator *c) {
        auto *map = map_creator_;
        if (map->find(t) != map->end()) {
            printf("Error: %d type has be added\n", t);
            return false;
        }
        map->insert(std::make_pair(t, c));
        return true;
    }

    // virtual Op* OpCreate(const vector<shared_ptr<Tensor>>& inputs, const vector<shared_ptr<Tensor>>& outputs,
    //                             OpParam op_param) override;

    virtual Op *opCreate(const OpParam &op_param, string name) override;

    virtual void registerOps() override;

private:
    std::map<OpType, CPUBackend::Creator *> *map_creator_;
};

} // namespace mllm

#endif // MLLM_CPUBACKEND_H