#ifndef MLLM_QNNBACKEND_H
#define MLLM_QNNBACKEND_H

#include "Backend.hpp"
#include "Op.hpp"
#include "Types.hpp"

namespace mllm {
class QNNBackend : public Backend {
public:
    explicit QNNBackend(shared_ptr<MemoryManager> &mm);
    ~QNNBackend() override = default;

    class Creator {
    public:
        virtual Op *create(OpParam op_param, Backend *bn, string name) const = 0;
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

    void onSetUpStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) override{
        printf("QNNBackend onSetUpStart\n");
    };
    void onSetUpEnd(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) override{
        printf("QNNBackend onSetUpEnd\n");
    };
    void onExecuteStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName = "") override{
        printf("QNNBackend onExecuteStart\n");
    };
    void onExecuteEnd() override{
        printf("QNNBackend onExecuteEnd\n");
    };

    void registerOps() override;

    TensorFunction *funcCreate(const TensorFuncType type) override{
        return nullptr;
    }
    void registerFuncs() override{
        ;
    }

private:
    std::map<OpType, QNNBackend::Creator *> map_creator_;
};

} // namespace mllm

#endif // MLLM_QNNBACKEND_H