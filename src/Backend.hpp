#ifndef MLLM_BACKEND_H
#define MLLM_BACKEND_H

#include "MemoryManager.hpp"
#include "OpDefined.hpp"
#include "Types.hpp"
#include <memory>
#include <unordered_map>
using std::shared_ptr;
using std::unordered_map;
namespace mllm {
class Op;

class Tensor;
class Backend;

// KVCache map for QNN-CPU KVCache sharing
#ifdef USE_QNN
static std::unordered_map<string, Op *> kv_cache_map;
#endif

class TensorFunction {
public:
    virtual void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) = 0;
    virtual void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) = 0;
};
class Backend {
public:
    Backend(shared_ptr<MemoryManager> &mm) :
        mem_manager_(mm) {
    }
    virtual ~Backend() = default;

    /**
     * \brief Allocates memory of the given size and alignment.
     * \param ptr A pointer to the pointer where the start address of the allocated memory will be stored.
     * \param size The size of the memory to be allocated.
     * \param alignment The alignment of the memory to be allocated.
     */
    void alloc(void **ptr, size_t size, size_t alignment) {
        mem_manager_->alloc(ptr, size, alignment);
    }

    /**
     * \brief Frees the memory pointed to by ptr.
     * \param ptr A pointer to the memory to be freed.
     */
    void free(void *ptr) {
        mem_manager_->free(ptr);
    }

    /**
     * \brief Creates an operation(Op) with the given parameters.
     * \param op_param The parameters for the operation to be created.
     * \param name The name of the operation. Default is an empty string.
     * \param threadCount The number of threads to be used for the operation. Default is 4.
     * \return A pointer to the created operation.
     */
    virtual Op *opCreate(const OpParam &op_param, string name = "", int threadCount = 4) = 0;
    virtual TensorFunction *funcCreate(TensorFuncType type) = 0;

    virtual void onSetUpStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName = "") {};
    virtual void onSetUpEnd(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName = "") {};
    virtual void onExecuteStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName = "") {};
    virtual void onExecuteEnd(std::vector<std::shared_ptr<Tensor>> &outputs, const string &graph_name = "") {};

    /**
     * \brief Registers all the operations supported by the backend.
     * This function is expected to be overridden by each specific backend implementation.
     */
    virtual void registerOps() = 0;
    virtual void registerFuncs() = 0;

    BackendType type() const {
        return type_;
    }
    static map<BackendType, Backend *> global_backends;

protected:
    BackendType type_;
    shared_ptr<MemoryManager> mem_manager_;
};

/**
 * \brief abstract Runtime register
 */
class BackendCreator {
public:
    virtual Backend *create(BackendConfig config) = 0;
};

/**
 * \brief get registered backend creator for given backend type.
 * \param type  given backend type.
 * \return backend creator pointer if registered, nullptr otherwise.
 */
const std::shared_ptr<BackendCreator> GetBackendCreator(BackendType type);

/**
 * \brief register backend creator for given backend type.
 * \param type given backend type.
 * \param creator registering backend creator.
 * \return true if backend creator for given backend type was not registered before, false otherwise.
 */
bool InsertBackendCreatorMap(BackendType type, shared_ptr<BackendCreator> creator);

} // namespace mllm

#endif // MLLM_BACKEND_H