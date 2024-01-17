#ifndef MLLM_BACKEND_H
#define MLLM_BACKEND_H

#include "MemoryManager.hpp"
#include "Types.hpp"
#include <memory>
using std::shared_ptr;

namespace mllm {
class Op;

class Tensor;
class Backend;
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

    /**
     * \brief Registers all the operations supported by the backend.
     * This function is expected to be overridden by each specific backend implementation.
     */
    virtual void registerOps() = 0;

private:
    //
    shared_ptr<MemoryManager> mem_manager_;
};

} // namespace mllm

#endif // MLLM_BACKEND_H