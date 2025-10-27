#ifndef MLLM_CPUBACKEND_H
#define MLLM_CPUBACKEND_H

#include "Backend.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include <map>
// #include "backends/cpu/third_party/ggml/Quantize.hpp"

namespace mllm {
class Module;
class Layer;
class CPUBackend final : public Backend {
public:
    explicit CPUBackend(shared_ptr<MemoryManager> &mm);
    ~CPUBackend() {
        for (auto &creator_pair : map_creator_) {
            delete creator_pair.second; // 手动删除用 new 创建的 Creator 对象
        }
        map_creator_.clear();
    }

    class Creator {
    public:
        virtual ~Creator() = default;
        virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const = 0;
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
    TensorFunction *funcCreate(TensorFuncType type) override;

    void registerOps() override;
    void registerFuncs() override;

    std::vector<Tensor> runLayer(Layer *layer, std::vector<Tensor> inputs, int N) override;

    std::vector<Tensor> runOp(Op *op, std::vector<Tensor> input, std::vector<std::string> out_names, bool in_place) override;
    std::vector<Tensor> runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) override;

    static int cpu_threads;

    // #ifdef USE_QNN
    void setCurSequenceLength(int sequence_length) {
        cur_sequence_length_ = sequence_length;
    }
    int getCurSequenceLength() {
        return cur_sequence_length_;
    }
    void setTotalSequenceLength(int sequence_length) {
        total_sequence_length_ = sequence_length;
    }
    int getTotalSequenceLength() {
        return total_sequence_length_;
    }
    void toggleSwitching() {
        isSwitchingStage = !isSwitchingStage;
    }
    void setChunkSize(int chunk_size) {
        chunk_size_ = chunk_size;
    }
    int getChunkSize() {
        return chunk_size_;
    }
    bool isStageSwitching() {
        return isSwitchingStage;
    }
    void setExecutionType(ExecutionType type) {
        execution_type = type;
    }
    ExecutionType getExecutionType() {
        return execution_type;
    }
    // #endif

    // #ifdef USE_SD
    void setLastDraftLength(unsigned int draft_length) {
        last_draft_length = draft_length;
    }
    void setLastVerifiedPositionIds(const std::vector<unsigned int> &verified_position_ids) {
        last_verified_position_ids = verified_position_ids;
    }
    void setUsingDraft(bool _usingDraft) {
        this->usingDraft = _usingDraft;
    }
    unsigned int getLastDraftLength() {
        return last_draft_length;
    }
    std::vector<unsigned int> getLastVerifiedPositionIds() {
        return last_verified_position_ids;
    }
    bool isUsingDraft() {
        return usingDraft;
    }

    void convert_fp_data(Tensor *src, Tensor *dest) override;
    // #endif
private:
    std::map<OpType, CPUBackend::Creator *> map_creator_;
    std::map<TensorFuncType, TensorFunction *> map_function_;
    // #ifdef USE_QNN
    // auto regression seq state
    int cur_sequence_length_ = 0;
    // total real seq length used for chunk&padding input
    int total_sequence_length_ = 0;
    // chunk size used in HeadLinear
    int chunk_size_ = 0;
    bool isSwitchingStage = false;
    ExecutionType execution_type = PROMPT;
    // #endif

    // #ifdef USE_SD
    bool usingDraft = false;
    std::vector<unsigned int> last_verified_position_ids;
    unsigned int last_draft_length = 0;
    // #endif

    void _create_output_tensors(
        std::vector<std::shared_ptr<Tensor>> &out_tensors,
        const std::vector<std::shared_ptr<Tensor>> &input_tensors,
        const std::vector<std::string> &out_names,
        Module *module,
        map<std::string, std::shared_ptr<Tensor>> &activation_tensors,
        Backend *backend);
    map<string, double> op_inference_time_;
    void _print_op_inference_time(bool sort = false);
};

} // namespace mllm

#endif // MLLM_CPUBACKEND_H