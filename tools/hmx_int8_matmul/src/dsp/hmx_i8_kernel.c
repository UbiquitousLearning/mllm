#include "dsp/hmx_i8_kernel.h"

#include "dsp/hmx_raw.h"
#include "dsp/dma_utils.h"
#include "dsp/hvx_worker_pool.h"

#include <float.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <qurt.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef HMX_HVX_EPILOGUE_LANES
#define HMX_HVX_EPILOGUE_LANES 1
#endif
#ifndef HMX_HVX_EPILOGUE_BATCH_TILES
#define HMX_HVX_EPILOGUE_BATCH_TILES 8
#endif

#if HMX_HVX_EPILOGUE_LANES > 1
#define HMX_I8_ACC_PIPELINE_SLOTS (2 * HMX_HVX_EPILOGUE_BATCH_TILES)
#else
#define HMX_I8_ACC_PIPELINE_SLOTS 2
#endif

enum {
    ACC_CONFIG_BYTES = 16384,
    ACC_BIAS_CONFIG_BYTES = 4096,
    ACC_PLANE_BYTES = 2048,
    ACC_READOUT_SLOT_BYTES = 4 * ACC_PLANE_BYTES,
    ACC_PIPELINE_SLOTS = HMX_I8_ACC_PIPELINE_SLOTS,
    HMX_I8_MAX_PADDED_K = 65536,
    HMX_I8_DMA_N_TILES = 8,
};

static int hmx_i8_cpu_pack_progress(
    volatile int32_t *progress, int query_tiles, int key_tiles,
    int wait) {
    if (progress == NULL) return 1;
    const uint64_t begin = wait ? HAP_perf_get_qtimer_count() : 0;
    for (;;) {
        qurt_mem_cache_clean(
            (qurt_addr_t)progress, sizeof(*progress),
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        __asm__ __volatile__("barrier" : : : "memory");
        const int32_t value = __atomic_load_n(progress, __ATOMIC_ACQUIRE);
        if (value == 1) return 1;
        if (value < 0) return -1;
        if ((value & HMX_I8_CPU_PACK_STREAM_FLAG) != 0) {
            const int ready_query =
                (value & HMX_I8_CPU_PACK_QUERY_MASK)
                >> HMX_I8_CPU_PACK_QUERY_SHIFT;
            const int ready_key = value & HMX_I8_CPU_PACK_KEY_MASK;
            if (ready_query >= query_tiles && ready_key >= key_tiles) {
                return 1;
            }
        }
        if (!wait) return 0;
        if (HAP_perf_get_qtimer_count() - begin
                > UINT64_C(5) * UINT64_C(19200000)) {
            return -1;
        }
    }
}

static uintptr_t align_up_uintptr(uintptr_t value, uintptr_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static uintptr_t align_down_uintptr(uintptr_t value, uintptr_t alignment) {
    return value & ~(alignment - 1u);
}

/* Row-major 64x32 activation tile. This is the layout consumed by the
 * activation.ub instruction used by the recovered HexKL micro primitive. */
static void pack_activation_tile(
    uint8_t *destination,
    const uint8_t *source,
    int leading_dimension,
    int row_start,
    int column_start,
    int source_signed) {
    for (int row = 0; row < HMX_I8_M_TILE; ++row) {
        uint8_t *row_destination =
            destination + (size_t)row * HMX_I8_K_TILE;
        const uint8_t *row_source = source
            + (size_t)(row_start + row) * leading_dimension + column_start;
        if (!source_signed) {
            memcpy(row_destination, row_source, HMX_I8_K_TILE);
        } else {
            const uint32_t *source_words = (const uint32_t *)row_source;
            uint32_t *destination_words = (uint32_t *)row_destination;
            for (int word = 0; word < HMX_I8_K_TILE / 4; ++word) {
                destination_words[word] =
                    source_words[word] ^ UINT32_C(0x80808080);
            }
        }
    }
}

/* HMX WH layout recovered from the local transformer and QAIRT layout docs:
 * [K/32, N/32, 8:K, 32:N, 4:K]. For one 32x32 tile this interleaves four
 * consecutive K rows for every output column. */
static void pack_weight_tile(
    int8_t *destination,
    const int8_t *source,
    int leading_dimension,
    int row_start,
    int column_start,
    int source_transposed) {
    if (source_transposed) {
        for (int row_group = 0; row_group < HMX_I8_K_TILE; row_group += 4) {
            uint32_t *packed = (uint32_t *)(
                destination + (size_t)(row_group / 4) * 128u);
            for (int column = 0; column < HMX_I8_N_TILE; ++column) {
                const uint32_t *value = (const uint32_t *)(
                    source
                    + (size_t)(column_start + column) * leading_dimension
                    + row_start + row_group);
                packed[column] = *value;
            }
        }
    } else {
        for (int row = 0; row < HMX_I8_K_TILE; ++row) {
            for (int column = 0; column < HMX_I8_N_TILE; ++column) {
                size_t packed_index = (size_t)(row / 4) * 128u
                    + (size_t)column * 4u + (size_t)(row & 3);
                destination[packed_index] = source[
                    (size_t)(row_start + row) * leading_dimension
                    + column_start + column];
            }
        }
    }
}

static void setup_accumulator_read_config(uint8_t *config) {
    memset(config, 0, 4096);
    int32_t *plane0_config = (int32_t *)(config + 0);
    int32_t *plane1_config = (int32_t *)(config + 1024);
    int32_t *plane2_config = (int32_t *)(config + 2048);
    int32_t *plane3_config = (int32_t *)(config + 3072);
    for (int i = 0; i < 64; ++i) {
        plane0_config[i] = 0x6000;
        plane1_config[i] = 0x4000;
        plane2_config[i] = 0x2000;
        plane3_config[i] = 0x0000;
    }
}

static void read_accumulator_i32(uint8_t *config, int32_t *result) {
    uint8_t *plane0 = config + 4096;
    uint8_t *plane1 = config + 6144;
    uint8_t *plane2 = config + 8192;
    uint8_t *plane3 = config + 10240;

    hmx_raw_read_acc_byte_plane(plane0, config + 0);
    hmx_raw_read_acc_byte_plane(plane1, config + 1024);
    hmx_raw_read_acc_byte_plane(plane2, config + 2048);
    hmx_raw_read_acc_byte_plane(plane3, config + 3072);

    /* The four retained reads are byte planes. Rebuild little-endian int32
     * values exactly as hexkl_micro_shuffle_4xa8_to_a32 does. */
    uint8_t *result_bytes = (uint8_t *)result;
    for (int i = 0; i < HMX_I8_M_TILE * HMX_I8_N_TILE; ++i) {
        result_bytes[(size_t)i * 4u + 0u] = plane0[i];
        result_bytes[(size_t)i * 4u + 1u] = plane1[i];
        result_bytes[(size_t)i * 4u + 2u] = plane2[i];
        result_bytes[(size_t)i * 4u + 3u] = plane3[i];
    }
}

static void read_accumulator_planes(
    const uint8_t *config,
    uint8_t *readout_slot) {
    hmx_raw_read_acc_byte_plane(
        readout_slot + 0 * ACC_PLANE_BYTES, config + 0);
    hmx_raw_read_acc_byte_plane(
        readout_slot + 1 * ACC_PLANE_BYTES, config + 1024);
    hmx_raw_read_acc_byte_plane(
        readout_slot + 2 * ACC_PLANE_BYTES, config + 2048);
    hmx_raw_read_acc_byte_plane(
        readout_slot + 3 * ACC_PLANE_BYTES, config + 3072);
}

static int8_t requantize_i8(int32_t accumulator, float output_scale) {
    float scaled = (float)accumulator * output_scale;
    if (scaled >= 127.0f) {
        return INT8_MAX;
    }
    if (scaled <= -128.0f) {
        return INT8_MIN;
    }

    /* The cast truncates toward zero, so adding +/-0.5 implements
     * round-to-nearest with ties away from zero without a libm dependency. */
    int32_t rounded = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    return (int8_t)rounded;
}

static inline __attribute__((always_inline)) void store_vector_bytes(
    void *address,
    uint32_t count,
    HVX_Vector value) {
    value = Q6_V_vlalign_VVR(value, value, (size_t)address);
    const uint32_t right_offset =
        (uint32_t)((size_t)address & 127u) + count;
    HVX_VectorPred left_mask = Q6_Q_vsetq_R((size_t)address);
    HVX_VectorPred right_mask = Q6_Q_vsetq2_R(right_offset);
    if (right_offset > 128u) {
        Q6_vmem_QRIV(right_mask, (HVX_Vector *)address + 1, value);
        right_mask = Q6_Q_vcmp_eq_VbVb(value, value);
    }
    left_mask = Q6_Q_or_QQn(left_mask, right_mask);
    Q6_vmem_QnRIV(left_mask, (HVX_Vector *)address, value);
}

#define SHUFFLE_4XA8_VECTORS(v0, v1, v2, v3, out0, out1, out2, out3) do { \
    HVX_VectorPair pair_a_ = Q6_W_vshuff_VVR((v1), (v0), -1); \
    HVX_VectorPair pair_b_ = Q6_W_vshuff_VVR((v3), (v2), -1); \
    HVX_VectorPair pair_c_ = Q6_W_vshuff_VVR( \
        Q6_V_lo_W(pair_b_), Q6_V_lo_W(pair_a_), -2); \
    HVX_VectorPair pair_d_ = Q6_W_vshuff_VVR( \
        Q6_V_hi_W(pair_b_), Q6_V_hi_W(pair_a_), -2); \
    (out0) = Q6_V_lo_W(pair_c_); \
    (out1) = Q6_V_hi_W(pair_c_); \
    (out2) = Q6_V_lo_W(pair_d_); \
    (out3) = Q6_V_hi_W(pair_d_); \
} while (0)

static inline __attribute__((always_inline)) HVX_Vector
quantize_i32_vector_to_i8_low32(
    HVX_Vector accumulator,
    HVX_Vector correction,
    HVX_Vector scale,
    HVX_Vector positive_half,
    HVX_Vector negative_half) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector corrected =
        Q6_Vw_vsub_VwVw(accumulator, correction);
    const HVX_VectorPred negative =
        Q6_Q_vcmp_gt_VwVw(zero, corrected);
    const HVX_Vector rounding =
        Q6_V_vmux_QVV(negative, negative_half, positive_half);
    const HVX_Vector scaled = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(corrected), scale));
    const HVX_Vector rounded = Q6_Vw_equals_Vsf(
        Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(scaled, rounding)));
    const HVX_Vector int8_bias = Q6_V_vsplat_R(128);
    const HVX_Vector biased = Q6_Vw_vadd_VwVw(rounded, int8_bias);
    const HVX_Vector half = Q6_Vh_vpack_VwVw_sat(zero, biased);
    const HVX_Vector packed = Q6_Vub_vpack_VhVh_sat(zero, half);
    return Q6_V_vxor_VV(packed, Q6_V_vsplat_R((int32_t)0x80808080u));
}

static void quantize_readout_slot_i8_range(
    int8_t *output,
    int output_leading_dimension,
    int row_start,
    int column_start,
    const uint8_t *readout_slot,
    const int32_t *rhs_sums,
    float output_scale,
    int lane,
    int lanes) {
    const HVX_Vector *plane0 = (const HVX_Vector *)(
        readout_slot + 0 * ACC_PLANE_BYTES);
    const HVX_Vector *plane1 = (const HVX_Vector *)(
        readout_slot + 1 * ACC_PLANE_BYTES);
    const HVX_Vector *plane2 = (const HVX_Vector *)(
        readout_slot + 2 * ACC_PLANE_BYTES);
    const HVX_Vector *plane3 = (const HVX_Vector *)(
        readout_slot + 3 * ACC_PLANE_BYTES);
    const HVX_Vector rhs_sum = *(const HVX_Vector *)(rhs_sums + column_start);
    const HVX_Vector correction = Q6_Vw_vasl_VwR(rhs_sum, 7);
    union {
        float value;
        uint32_t bits;
    } scale_bits = {.value = output_scale};
    const HVX_Vector scale = Q6_V_vsplat_R((int32_t)scale_bits.bits);
    const HVX_Vector positive_half = Q6_V_vsplat_R(0x3f000000);
    const HVX_Vector negative_half = Q6_V_vsplat_R((int32_t)0xbf000000u);

    for (int stripe = lane; stripe < 16; stripe += lanes) {
        HVX_Vector row0;
        HVX_Vector row1;
        HVX_Vector row2;
        HVX_Vector row3;
        SHUFFLE_4XA8_VECTORS(
            plane0[stripe], plane1[stripe],
            plane2[stripe], plane3[stripe],
            row0, row1, row2, row3);
        row0 = quantize_i32_vector_to_i8_low32(
            row0, correction, scale, positive_half, negative_half);
        row1 = quantize_i32_vector_to_i8_low32(
            row1, correction, scale, positive_half, negative_half);
        row2 = quantize_i32_vector_to_i8_low32(
            row2, correction, scale, positive_half, negative_half);
        row3 = quantize_i32_vector_to_i8_low32(
            row3, correction, scale, positive_half, negative_half);

        int8_t *base = output
            + (size_t)(row_start + stripe * 4) * output_leading_dimension
            + column_start;
        store_vector_bytes(base + 0 * output_leading_dimension, 32, row0);
        store_vector_bytes(base + 1 * output_leading_dimension, 32, row1);
        store_vector_bytes(base + 2 * output_leading_dimension, 32, row2);
        store_vector_bytes(base + 3 * output_leading_dimension, 32, row3);
    }
}

static void quantize_readout_slot_i8(
    int8_t *output,
    int output_leading_dimension,
    int row_start,
    int column_start,
    const uint8_t *readout_slot,
    const int32_t *rhs_sums,
    float output_scale) {
    quantize_readout_slot_i8_range(
        output, output_leading_dimension, row_start, column_start,
        readout_slot, rhs_sums, output_scale, 0, 1);
}

static void store_readout_slot_i32_range(
    int32_t *output,
    int output_leading_dimension,
    int row_start,
    int column_start,
    const uint8_t *readout_slot,
    const int32_t *rhs_sums,
    int lane,
    int lanes) {
    const HVX_Vector *plane0 = (const HVX_Vector *)(
        readout_slot + 0 * ACC_PLANE_BYTES);
    const HVX_Vector *plane1 = (const HVX_Vector *)(
        readout_slot + 1 * ACC_PLANE_BYTES);
    const HVX_Vector *plane2 = (const HVX_Vector *)(
        readout_slot + 2 * ACC_PLANE_BYTES);
    const HVX_Vector *plane3 = (const HVX_Vector *)(
        readout_slot + 3 * ACC_PLANE_BYTES);
    const HVX_Vector rhs_sum = *(const HVX_Vector *)(rhs_sums + column_start);
    const HVX_Vector correction = Q6_Vw_vasl_VwR(rhs_sum, 7);

    for (int stripe = lane; stripe < 16; stripe += lanes) {
        HVX_Vector row0;
        HVX_Vector row1;
        HVX_Vector row2;
        HVX_Vector row3;
        SHUFFLE_4XA8_VECTORS(
            plane0[stripe], plane1[stripe],
            plane2[stripe], plane3[stripe],
            row0, row1, row2, row3);
        row0 = Q6_Vw_vsub_VwVw(row0, correction);
        row1 = Q6_Vw_vsub_VwVw(row1, correction);
        row2 = Q6_Vw_vsub_VwVw(row2, correction);
        row3 = Q6_Vw_vsub_VwVw(row3, correction);

        int32_t *base = output
            + (size_t)(row_start + stripe * 4) * output_leading_dimension
            + column_start;
        *(HVX_Vector *)(base + 0 * output_leading_dimension) = row0;
        *(HVX_Vector *)(base + 1 * output_leading_dimension) = row1;
        *(HVX_Vector *)(base + 2 * output_leading_dimension) = row2;
        *(HVX_Vector *)(base + 3 * output_leading_dimension) = row3;
    }
}

static void store_readout_slot_i32(
    int32_t *output,
    int output_leading_dimension,
    int row_start,
    int column_start,
    const uint8_t *readout_slot,
    const int32_t *rhs_sums) {
    store_readout_slot_i32_range(
        output, output_leading_dimension, row_start, column_start,
        readout_slot, rhs_sums, 0, 1);
}

#if HMX_HVX_EPILOGUE_LANES > 1
typedef struct hmx_epilogue_executor {
    qurt_sem_t start[HMX_HVX_MAX_WORKERS];
    qurt_sem_t done;
    void *output;
    int output_leading_dimension;
    int row_start[HMX_HVX_EPILOGUE_BATCH_TILES];
    int column_start[HMX_HVX_EPILOGUE_BATCH_TILES];
    const uint8_t *readout_slots;
    const int32_t *rhs_sums;
    float output_scale;
    int output_i32;
    int tile_count;
    int handler_participates;
    int lanes;
    int active;
    volatile int stop;
} hmx_epilogue_executor;

static void hmx_run_epilogue_tile(
    const hmx_epilogue_executor *executor, int tile) {
    const uint8_t *readout_slot = executor->readout_slots
        + (size_t)tile * ACC_READOUT_SLOT_BYTES;
    if (executor->output_i32) {
        store_readout_slot_i32(
            (int32_t *)executor->output,
            executor->output_leading_dimension,
            executor->row_start[tile],
            executor->column_start[tile],
            readout_slot,
            executor->rhs_sums);
    } else {
        quantize_readout_slot_i8(
            (int8_t *)executor->output,
            executor->output_leading_dimension,
            executor->row_start[tile],
            executor->column_start[tile],
            readout_slot,
            executor->rhs_sums,
            executor->output_scale);
    }
}

static void hmx_run_epilogue_lane(
    const hmx_epilogue_executor *executor, int lane, int lanes) {
    int tile = lane;
    int stride = lanes;
    if (!executor->handler_participates) {
        if (lane == 0) return;
        tile = lane - 1;
        stride = lanes - 1;
    }
    for (; tile < executor->tile_count; tile += stride) {
        hmx_run_epilogue_tile(executor, tile);
    }
}

static void hmx_epilogue_worker(void *opaque, int lane, int lanes) {
    hmx_epilogue_executor *executor = (hmx_epilogue_executor *)opaque;
    const int worker = lane - 1;
    if (worker < 0 || worker >= HMX_HVX_MAX_WORKERS) return;
    for (;;) {
        (void)qurt_sem_down(&executor->start[worker]);
        __asm__ __volatile__("barrier" : : : "memory");
        if (executor->stop) break;
        hmx_run_epilogue_lane(executor, lane, lanes);
        __asm__ __volatile__("barrier" : : : "memory");
        (void)qurt_sem_up(&executor->done);
    }
}

static int hmx_epilogue_executor_begin(hmx_epilogue_executor *executor) {
    memset(executor, 0, sizeof(*executor));
    if (hmx_hvx_worker_pool_init() != 0) return -1;
    executor->lanes = hmx_hvx_worker_pool_lanes();
    if (executor->lanes < 2) return -1;
    if (executor->lanes > HMX_HVX_EPILOGUE_LANES) {
        executor->lanes = HMX_HVX_EPILOGUE_LANES;
    }
    for (int worker = 0; worker < executor->lanes - 1; ++worker) {
        qurt_sem_init_val(&executor->start[worker], 0);
    }
    qurt_sem_init_val(&executor->done, 0);
    if (hmx_hvx_workers_start(hmx_epilogue_worker, executor) != 0) {
        for (int worker = 0; worker < executor->lanes - 1; ++worker) {
            qurt_sem_destroy(&executor->start[worker]);
        }
        qurt_sem_destroy(&executor->done);
        return -1;
    }
    return 0;
}

static void hmx_epilogue_executor_dispatch(
    hmx_epilogue_executor *executor,
    void *output,
    int output_leading_dimension,
    const int *row_start,
    const int *column_start,
    const uint8_t *readout_slots,
    int tile_count,
    const int32_t *rhs_sums,
    float output_scale,
    int output_i32,
    int handler_participates) {
    executor->output = output;
    executor->output_leading_dimension = output_leading_dimension;
    memcpy(executor->row_start, row_start,
           (size_t)tile_count * sizeof(row_start[0]));
    memcpy(executor->column_start, column_start,
           (size_t)tile_count * sizeof(column_start[0]));
    executor->readout_slots = readout_slots;
    executor->rhs_sums = rhs_sums;
    executor->output_scale = output_scale;
    executor->output_i32 = output_i32;
    executor->tile_count = tile_count;
    executor->handler_participates = handler_participates;
    executor->active = 1;
    __asm__ __volatile__("barrier" : : : "memory");
    for (int worker = 0; worker < executor->lanes - 1; ++worker) {
        (void)qurt_sem_up(&executor->start[worker]);
    }
    if (handler_participates) {
        hmx_run_epilogue_lane(executor, 0, executor->lanes);
    }
}

static void hmx_epilogue_executor_wait(hmx_epilogue_executor *executor) {
    if (!executor->active) return;
    for (int worker = 0; worker < executor->lanes - 1; ++worker) {
        (void)qurt_sem_down(&executor->done);
    }
    executor->active = 0;
}

static void hmx_epilogue_executor_end(hmx_epilogue_executor *executor) {
    hmx_epilogue_executor_wait(executor);
    executor->stop = 1;
    __asm__ __volatile__("barrier" : : : "memory");
    for (int worker = 0; worker < executor->lanes - 1; ++worker) {
        (void)qurt_sem_up(&executor->start[worker]);
    }
    (void)hmx_hvx_workers_wait();
    for (int worker = 0; worker < executor->lanes - 1; ++worker) {
        qurt_sem_destroy(&executor->start[worker]);
    }
    qurt_sem_destroy(&executor->done);
}

typedef struct hmx_external_epilogue_job hmx_external_epilogue_job;

typedef struct hmx_external_epilogue_task {
    hmx_external_epilogue_job *job;
    int lane;
} hmx_external_epilogue_task;

struct hmx_external_epilogue_job {
    /* Reuse the exact same batch description and lane partitioning as the
     * standalone worker-pool executor above. Its start/done semaphores are
     * deliberately unused in this externally scheduled mode. */
    hmx_epilogue_executor batch;
    qurt_sem_t done;
    hmx_external_epilogue_task tasks[HMX_HVX_MAX_WORKERS];
    volatile int remaining;
    int active;
};

static void hmx_external_epilogue_task_run(void *opaque) {
    hmx_external_epilogue_task *task =
        (hmx_external_epilogue_task *)opaque;
    hmx_external_epilogue_job *job = task->job;
    hmx_run_epilogue_lane(&job->batch, task->lane, job->batch.lanes);
    __asm__ __volatile__("barrier" : : : "memory");
    if (__sync_sub_and_fetch(&job->remaining, 1) == 0) {
        (void)qurt_sem_up(&job->done);
    }
}

static void hmx_external_epilogue_begin(hmx_external_epilogue_job *job) {
    memset(job, 0, sizeof(*job));
    qurt_sem_init_val(&job->done, 0);
}

static void hmx_external_epilogue_wait(hmx_external_epilogue_job *job) {
    if (!job->active) return;
    (void)qurt_sem_down(&job->done);
    job->active = 0;
}

static void hmx_external_epilogue_dispatch(
    hmx_external_epilogue_job *job,
    const hmx_i8_task_scheduler *scheduler,
    void *output,
    int output_leading_dimension,
    const int *row_start,
    const int *column_start,
    const uint8_t *readout_slots,
    int tile_count,
    const int32_t *rhs_sums,
    float output_scale,
    int output_i32,
    int handler_participates) {
    hmx_epilogue_executor *batch = &job->batch;
    batch->output = output;
    batch->output_leading_dimension = output_leading_dimension;
    memcpy(batch->row_start, row_start,
           (size_t)tile_count * sizeof(row_start[0]));
    memcpy(batch->column_start, column_start,
           (size_t)tile_count * sizeof(column_start[0]));
    batch->readout_slots = readout_slots;
    batch->rhs_sums = rhs_sums;
    batch->output_scale = output_scale;
    batch->output_i32 = output_i32;
    batch->tile_count = tile_count;
    batch->handler_participates = handler_participates;
    batch->lanes = scheduler->lanes;
    if (batch->lanes > HMX_HVX_EPILOGUE_LANES) {
        batch->lanes = HMX_HVX_EPILOGUE_LANES;
    }
    if (batch->lanes > HMX_HVX_MAX_LANES) {
        batch->lanes = HMX_HVX_MAX_LANES;
    }
    if (batch->lanes < 2) batch->lanes = 2;

    job->remaining = batch->lanes - 1;
    job->active = 1;
    __asm__ __volatile__("barrier" : : : "memory");
    for (int lane = 1; lane < batch->lanes; ++lane) {
        hmx_external_epilogue_task *task = &job->tasks[lane - 1];
        task->job = job;
        task->lane = lane;
        /* A full queue is not a correctness failure: run that lane on the
         * handler and retain the same completion accounting. */
        if (scheduler->submit_high(
                scheduler->context, hmx_external_epilogue_task_run,
                task) != 0) {
            hmx_external_epilogue_task_run(task);
        }
    }
    if (handler_participates) {
        hmx_run_epilogue_lane(batch, 0, batch->lanes);
    }
}

static void hmx_external_epilogue_end(hmx_external_epilogue_job *job) {
    hmx_external_epilogue_wait(job);
    qurt_sem_destroy(&job->done);
}
#endif

static inline __attribute__((always_inline)) int hmx_i8_kernel_impl(
    void *output,
    const uint8_t *lhs,
    const int8_t *rhs,
    int m,
    int k,
    int n,
    int output_i8,
    int rhs_transposed,
    int lhs_signed,
    float output_scale,
    uint8_t *vtcm,
    uint32_t vtcm_size) {
    if (!output || !lhs || !rhs || !vtcm || m <= 0 || k <= 0 || n <= 0 ||
        k > HMX_I8_MAX_PADDED_K ||
        (m % HMX_I8_M_TILE) != 0 ||
        (k % HMX_I8_K_TILE) != 0 ||
        (n % HMX_I8_N_TILE) != 0 ||
        (output_i8 && (!(output_scale >= 0.0f) || output_scale > FLT_MAX))) {
        return -1;
    }

    uintptr_t begin = align_up_uintptr((uintptr_t)vtcm, HMX_I8_ACT_TILE_BYTES);
    uintptr_t end = (uintptr_t)vtcm + vtcm_size;
    uintptr_t result_address = align_down_uintptr(end - HMX_I8_OUTPUT_TILE_BYTES, 2048);
    uintptr_t config_address = align_down_uintptr(result_address - ACC_CONFIG_BYTES, 2048);
    uintptr_t weight_address = align_down_uintptr(config_address - HMX_I8_WEIGHT_TILE_BYTES, 128);
    int32_t *rhs_sums_all = NULL;
    if (output_i8) {
        rhs_sums_all = (int32_t *)begin;
        begin = align_up_uintptr(
            begin + (uintptr_t)((size_t)n * sizeof(int32_t)),
            HMX_I8_ACT_TILE_BYTES);
    }
    if (weight_address <= begin) {
        return -1;
    }

    uint8_t *activation_slots = (uint8_t *)begin;
    int8_t *weight_tile = (int8_t *)weight_address;
    uint8_t *config = (uint8_t *)config_address;
    int32_t *result_tile = (int32_t *)result_address;
    int k_tiles = k / HMX_I8_K_TILE;
    int slot_count = (int)((weight_address - begin) / HMX_I8_ACT_TILE_BYTES);
    if (slot_count < 1) {
        return -1;
    }

    int cached_tiles = k_tiles;
    uint8_t *scratch_tile = NULL;
    if (cached_tiles > slot_count) {
        cached_tiles = slot_count - 1;
        scratch_tile = activation_slots + (size_t)cached_tiles * HMX_I8_ACT_TILE_BYTES;
    }

    setup_accumulator_read_config(config);

    if (output_i8) {
        for (int column = 0; column < n; ++column) {
            int32_t sum = 0;
            for (int inner = 0; inner < k; ++inner) {
                sum += rhs_transposed
                    ? rhs[(size_t)column * k + inner]
                    : rhs[(size_t)inner * n + column];
            }
            rhs_sums_all[column] = sum;
        }
    }

    for (int row_start = 0; row_start < m; row_start += HMX_I8_M_TILE) {
        for (int kt = 0; kt < cached_tiles; ++kt) {
            pack_activation_tile(
                activation_slots + (size_t)kt * HMX_I8_ACT_TILE_BYTES,
                lhs,
                k,
                row_start,
                kt * HMX_I8_K_TILE,
                lhs_signed);
        }

        for (int column_start = 0; column_start < n; column_start += HMX_I8_N_TILE) {
            hmx_raw_clear_i32();
            for (int kt = 0; kt < k_tiles; ++kt) {
                uint8_t *activation_tile;
                if (kt < cached_tiles) {
                    activation_tile =
                        activation_slots + (size_t)kt * HMX_I8_ACT_TILE_BYTES;
                } else {
                    activation_tile = scratch_tile;
                    pack_activation_tile(
                        activation_tile,
                        lhs,
                        k,
                        row_start,
                        kt * HMX_I8_K_TILE,
                        lhs_signed);
                }

                pack_weight_tile(
                    weight_tile,
                    rhs,
                    rhs_transposed ? k : n,
                    kt * HMX_I8_K_TILE,
                    column_start,
                    rhs_transposed);
                hmx_raw_mac_u8i8(activation_tile, weight_tile);
            }

            read_accumulator_i32(config, result_tile);
            if (output_i8) {
                int8_t *quantized_output = (int8_t *)output;
                for (int row = 0; row < HMX_I8_M_TILE; ++row) {
                    for (int column = 0; column < HMX_I8_N_TILE; ++column) {
                        int32_t corrected =
                            result_tile[(size_t)row * HMX_I8_N_TILE + column] -
                            128 * rhs_sums_all[column_start + column];
                        quantized_output[
                            (size_t)(row_start + row) * n + column_start + column] =
                            requantize_i8(corrected, output_scale);
                    }
                }
            } else {
                int32_t *output_i32 = (int32_t *)output;
                for (int row = 0; row < HMX_I8_M_TILE; ++row) {
                    memcpy(
                        output_i32 + (size_t)(row_start + row) * n + column_start,
                        result_tile + (size_t)row * HMX_I8_N_TILE,
                        HMX_I8_N_TILE * sizeof(int32_t));
                }
            }
        }
    }
    return 0;
}

int hmx_i8_kernel(
    int32_t *output,
    const uint8_t *lhs,
    const int8_t *rhs,
    int m,
    int k,
    int n,
    uint8_t *vtcm,
    uint32_t vtcm_size) {
    return hmx_i8_kernel_impl(
        output, lhs, rhs, m, k, n, 0, 0, 0, 1.0f, vtcm, vtcm_size);
}

int hmx_i8_kernel_fixed(
    int32_t *output,
    const uint8_t *lhs,
    const int8_t *rhs,
    uint8_t *vtcm,
    uint32_t vtcm_size) {
    return hmx_i8_kernel_impl(
        output,
        lhs,
        rhs,
        HMX_FIXED_M_PADDED,
        HMX_FIXED_K_PADDED,
        HMX_FIXED_N_PADDED,
        0,
        0,
        0,
        1.0f,
        vtcm,
        vtcm_size);
}

int hmx_i8_kernel_i8_fixed(
    int8_t *output,
    const uint8_t *lhs,
    const int8_t *rhs,
    float output_scale,
    uint8_t *vtcm,
    uint32_t vtcm_size) {
    return hmx_i8_kernel_impl(
        output,
        lhs,
        rhs,
        HMX_FIXED_M_PADDED,
        HMX_FIXED_K_PADDED,
        HMX_FIXED_N_PADDED,
        1,
        0,
        0,
        output_scale,
        vtcm,
        vtcm_size);
}

int hmx_i8_kernel_qk_i8_fixed(
    int8_t *output,
    const uint8_t *query,
    const int8_t *key,
    float output_scale,
    uint8_t *vtcm,
    uint32_t vtcm_size) {
    return hmx_i8_kernel_impl(
        output,
        query,
        key,
        HMX_FIXED_M_PADDED,
        HMX_FIXED_K_PADDED,
        HMX_FIXED_N_PADDED,
        1,
        1,
        0,
        output_scale,
        vtcm,
        vtcm_size);
}

int hmx_i8_kernel_qk_i8_direct(
    int8_t *output,
    const int8_t *query,
    const int8_t *key,
    int m,
    int k,
    int n,
    float output_scale,
    uint8_t *vtcm,
    uint32_t vtcm_size) {
    if (!output || !query || !key || !vtcm || m <= 0 || k <= 0 || n <= 0 ||
        (m % HMX_I8_M_TILE) != 0 || (k % HMX_I8_K_TILE) != 0 ||
        (n % HMX_I8_N_TILE) != 0 || !(output_scale >= 0.0f) ||
        output_scale > FLT_MAX) {
        return -1;
    }
    const uint32_t required = hmx_i8_qk_direct_vtcm_bytes(m, k, n);
    if (required == 0 || required > vtcm_size) {
        return -1;
    }

    uintptr_t cursor = align_up_uintptr((uintptr_t)vtcm, 2048u);
    int32_t *rhs_sums = (int32_t *)cursor;
    cursor = align_up_uintptr(
        cursor + (uintptr_t)((size_t)n * sizeof(int32_t)), 2048u);
    uint8_t *packed_query = (uint8_t *)cursor;
    cursor += (uintptr_t)((size_t)m * (size_t)k);
    cursor = align_up_uintptr(cursor, 2048u);
    int8_t *packed_key = (int8_t *)cursor;
    cursor += (uintptr_t)((size_t)n * (size_t)k);
    cursor = align_up_uintptr(cursor, 2048u);
    uint8_t *config = (uint8_t *)cursor;
    cursor += ACC_BIAS_CONFIG_BYTES;
    cursor = align_up_uintptr(cursor, 2048u);
    uint8_t *readout_slots = (uint8_t *)cursor;

    const int m_tiles = m / HMX_I8_M_TILE;
    const int k_tiles = k / HMX_I8_K_TILE;
    const int n_tiles = n / HMX_I8_N_TILE;
    for (int row_tile = 0; row_tile < m_tiles; ++row_tile) {
        for (int inner_tile = 0; inner_tile < k_tiles; ++inner_tile) {
            pack_activation_tile(
                packed_query
                    + ((size_t)row_tile * k_tiles + inner_tile)
                        * HMX_I8_ACT_TILE_BYTES,
                (const uint8_t *)query,
                k,
                row_tile * HMX_I8_M_TILE,
                inner_tile * HMX_I8_K_TILE,
                1);
        }
    }
    for (int column_tile = 0; column_tile < n_tiles; ++column_tile) {
        for (int inner_tile = 0; inner_tile < k_tiles; ++inner_tile) {
            pack_weight_tile(
                packed_key
                    + ((size_t)column_tile * k_tiles + inner_tile)
                        * HMX_I8_WEIGHT_TILE_BYTES,
                key,
                k,
                inner_tile * HMX_I8_K_TILE,
                column_tile * HMX_I8_N_TILE,
                1);
        }
    }
    for (int column = 0; column < n; ++column) {
        int32_t sum = 0;
        for (int inner = 0; inner < k; ++inner) {
            sum += key[(size_t)column * k + inner];
        }
        rhs_sums[column] = sum;
    }
    setup_accumulator_read_config(config);

    int have_previous = 0;
    int previous_slot = 0;
    int previous_row_start = 0;
    int previous_column_start = 0;
    int current_slot = 0;
    for (int row_tile = 0; row_tile < m_tiles; ++row_tile) {
        const uint8_t *query_tiles = packed_query
            + (size_t)row_tile * k_tiles * HMX_I8_ACT_TILE_BYTES;
        for (int column_tile = 0; column_tile < n_tiles; ++column_tile) {
            const int8_t *key_tiles = packed_key
                + (size_t)column_tile * k_tiles * HMX_I8_WEIGHT_TILE_BYTES;
            hmx_raw_clear_i32();
            hmx_raw_mac_u8i8_deep(query_tiles, key_tiles, k_tiles);

            /* HMX execution is decoupled from HVX execution.  Consume
             * the previous tile only after issuing the current tile, then
             * read the current accumulator into the other ring slot.  This
             * is the two-slot issue -> epilogue -> readout pipeline used by
             * the local HTPOPLIB direct E-plane implementation. */
            if (have_previous) {
                quantize_readout_slot_i8(
                    output,
                    n,
                    previous_row_start,
                    previous_column_start,
                    readout_slots
                        + (size_t)previous_slot * ACC_READOUT_SLOT_BYTES,
                    rhs_sums,
                    output_scale);
            }
            const int row_start = row_tile * HMX_I8_M_TILE;
            const int column_start = column_tile * HMX_I8_N_TILE;
            read_accumulator_planes(
                config,
                readout_slots + (size_t)current_slot * ACC_READOUT_SLOT_BYTES);
            previous_slot = current_slot;
            previous_row_start = row_start;
            previous_column_start = column_start;
            have_previous = 1;
            current_slot ^= 1;
        }
    }
    if (have_previous) {
        quantize_readout_slot_i8(
            output,
            n,
            previous_row_start,
            previous_column_start,
            readout_slots + (size_t)previous_slot * ACC_READOUT_SLOT_BYTES,
            rhs_sums,
            output_scale);
    }
    return 0;
}

uint32_t hmx_i8_qk_direct_vtcm_bytes(int m, int k, int n) {
    if (m <= 0 || k <= 0 || n <= 0 ||
        (m % HMX_I8_M_TILE) != 0 || (k % HMX_I8_K_TILE) != 0 ||
        (n % HMX_I8_N_TILE) != 0) {
        return 0;
    }
    size_t bytes = 2048u;
    bytes += (size_t)n * sizeof(int32_t) + 2048u;
    bytes += (size_t)m * (size_t)k + 2048u;
    bytes += (size_t)n * (size_t)k + 2048u;
    bytes += ACC_BIAS_CONFIG_BYTES + 2048u
        + ACC_PIPELINE_SLOTS * ACC_READOUT_SLOT_BYTES;
    bytes = (bytes + 4095u) & ~(size_t)4095u;
    return bytes <= UINT32_MAX ? (uint32_t)bytes : 0;
}

uint32_t hmx_i8_qk_packed_vtcm_bytes(int m, int k, int n) {
    if (m <= 0 || k <= 0 || n <= 0 ||
        (m % HMX_I8_M_TILE) != 0 || (k % HMX_I8_K_TILE) != 0 ||
        (n % HMX_I8_N_TILE) != 0) {
        return 0;
    }
    const int chunk_n = n < HMX_I8_DMA_N_TILES * HMX_I8_N_TILE
        ? n : HMX_I8_DMA_N_TILES * HMX_I8_N_TILE;
    size_t bytes = 2048u + ACC_BIAS_CONFIG_BYTES + 2048u;
    bytes += ACC_PIPELINE_SLOTS * ACC_READOUT_SLOT_BYTES + 2048u;
    /* Double-buffer query tiles so the final key chunk for row r can overlap
     * the VTCM DMA of query row r + 1. */
    bytes += 2u * (size_t)HMX_I8_M_TILE * (size_t)k + 4096u;
    bytes += 2u * (size_t)chunk_n * (size_t)k + 4096u;
    bytes = (bytes + 4095u) & ~(size_t)4095u;
    return bytes <= UINT32_MAX ? (uint32_t)bytes : 0;
}

static int hmx_i8_kernel_qk_packed_impl(
    void *output,
    const uint8_t *packed_query,
    const int8_t *packed_key,
    const int32_t *key_sums,
    int m,
    int k,
    int n,
    float output_scale,
    int output_i32,
    uint8_t *vtcm,
    uint32_t vtcm_size,
    const hmx_i8_task_scheduler *scheduler,
    volatile int32_t *producer_progress) {
#if HMX_HVX_EPILOGUE_LANES == 1
    (void)scheduler;
#endif
    if (!output || !packed_query || !packed_key || !key_sums || !vtcm ||
        m <= 0 || k <= 0 || n <= 0 ||
        (m % HMX_I8_M_TILE) != 0 || (k % HMX_I8_K_TILE) != 0 ||
        (n % HMX_I8_N_TILE) != 0 ||
        (!output_i32 && (!(output_scale >= 0.0f) ||
                         output_scale > FLT_MAX))) {
        return -1;
    }
    const uint32_t required = hmx_i8_qk_packed_vtcm_bytes(m, k, n);
    if (required == 0 || required > vtcm_size) return -1;

    uintptr_t cursor = align_up_uintptr((uintptr_t)vtcm, 2048u);
    uint8_t *config = (uint8_t *)cursor;
    cursor = align_up_uintptr(cursor + ACC_BIAS_CONFIG_BYTES, 2048u);
    uint8_t *readout_slots = (uint8_t *)cursor;
    cursor = align_up_uintptr(
        cursor + ACC_PIPELINE_SLOTS * ACC_READOUT_SLOT_BYTES, 2048u);
    uint8_t *query_buffers[2];
    query_buffers[0] = (uint8_t *)cursor;
    cursor = align_up_uintptr(
        cursor + (uintptr_t)((size_t)HMX_I8_M_TILE * (size_t)k), 2048u);
    query_buffers[1] = (uint8_t *)cursor;
    cursor = align_up_uintptr(
        cursor + (uintptr_t)((size_t)HMX_I8_M_TILE * (size_t)k), 2048u);

    const int n_tiles = n / HMX_I8_N_TILE;
    const int chunk_tiles = n_tiles < HMX_I8_DMA_N_TILES
        ? n_tiles : HMX_I8_DMA_N_TILES;
    const int chunk_n = chunk_tiles * HMX_I8_N_TILE;
    const size_t maximum_key_chunk_bytes = (size_t)chunk_n * (size_t)k;
    int8_t *key_buffers[2];
    key_buffers[0] = (int8_t *)cursor;
    cursor = align_up_uintptr(cursor + maximum_key_chunk_bytes, 2048u);
    key_buffers[1] = (int8_t *)cursor;
    cursor = align_up_uintptr(cursor + maximum_key_chunk_bytes, 2048u);
    if (cursor > (uintptr_t)vtcm + vtcm_size) return -1;

    hmx_dma_desc_1d query_dma __attribute__((aligned(64)));
    hmx_dma_desc_1d key_dma[2] __attribute__((aligned(64)));
    const int m_tiles = m / HMX_I8_M_TILE;
    const int k_tiles = k / HMX_I8_K_TILE;
    setup_accumulator_read_config(config);

#if HMX_HVX_EPILOGUE_LANES > 1
    hmx_epilogue_executor epilogue_executor;
    hmx_external_epilogue_job external_epilogue_job;
    const int external_epilogue = scheduler != NULL
        && scheduler->submit_high != NULL && scheduler->lanes >= 2;
    if (external_epilogue) {
        hmx_external_epilogue_begin(&external_epilogue_job);
    }
    const int internal_epilogue = !external_epilogue
        && hmx_epilogue_executor_begin(&epilogue_executor) == 0;
    const int parallel_epilogue = external_epilogue || internal_epilogue;
#endif

    int have_previous = 0;
    int previous_slot = 0;
    int previous_row_start = 0;
    int previous_column_start = 0;
    int current_slot = 0;
    int current_query_buffer = 0;
    int prefetched_query_row = -1;
#if HMX_HVX_EPILOGUE_LANES > 1
    int epilogue_batch_rows[HMX_HVX_EPILOGUE_BATCH_TILES];
    int epilogue_batch_columns[HMX_HVX_EPILOGUE_BATCH_TILES];
    int epilogue_batch_count = 0;
    int epilogue_batch_buffer = 0;
    int produced_tiles = 0;
    const int total_output_tiles = m_tiles * n_tiles;
#endif
    for (int row_tile = 0; row_tile < m_tiles; ++row_tile) {
        const uint8_t *query_source = packed_query
            + (size_t)row_tile * k_tiles * HMX_I8_ACT_TILE_BYTES;
        if (prefetched_query_row == row_tile) {
            if (!hmx_dma_wait_idle()) return -1;
            current_query_buffer ^= 1;
            prefetched_query_row = -1;
        } else {
            if (producer_progress != NULL) {
                if (hmx_i8_cpu_pack_progress(
                        producer_progress, row_tile + 1, 0, 1) <= 0) {
                    return -1;
                }
                qurt_mem_cache_clean(
                    (qurt_addr_t)query_source,
                    (size_t)HMX_I8_M_TILE * (size_t)k,
                    QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
            }
            hmx_dma_prepare_copy(
                &query_dma, query_source,
                query_buffers[current_query_buffer],
                (uint32_t)((size_t)HMX_I8_M_TILE * (size_t)k));
            if (!hmx_dma_wait_idle()) return -1;
            hmx_dma_start(&query_dma);
            if (!hmx_dma_wait_idle()) return -1;
        }

        int chunk_begin_tile = 0;
        int current_buffer = 0;
        int current_tiles = n_tiles < chunk_tiles ? n_tiles : chunk_tiles;
        if (producer_progress != NULL && row_tile == 0) {
            if (hmx_i8_cpu_pack_progress(
                    producer_progress, row_tile + 1,
                    current_tiles, 1) <= 0) {
                return -1;
            }
            qurt_mem_cache_clean(
                (qurt_addr_t)packed_key,
                (size_t)current_tiles * HMX_I8_N_TILE * (size_t)k,
                QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
            qurt_mem_cache_clean(
                (qurt_addr_t)key_sums,
                (size_t)current_tiles * HMX_I8_N_TILE * sizeof(int32_t),
                QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        }
        hmx_dma_prepare_copy(
            &key_dma[current_buffer], packed_key,
            key_buffers[current_buffer],
            (uint32_t)((size_t)current_tiles * HMX_I8_N_TILE * (size_t)k));
        hmx_dma_start(&key_dma[current_buffer]);
        if (!hmx_dma_wait_idle()) return -1;

        while (chunk_begin_tile < n_tiles) {
            const int next_begin_tile = chunk_begin_tile + current_tiles;
            const int next_buffer = current_buffer ^ 1;
            int next_tiles = 0;
            int next_dma_started = 0;
            if (next_begin_tile < n_tiles) {
                next_tiles = n_tiles - next_begin_tile;
                if (next_tiles > chunk_tiles) next_tiles = chunk_tiles;
                const int8_t *next_source = packed_key
                    + (size_t)next_begin_tile * k_tiles
                        * HMX_I8_WEIGHT_TILE_BYTES;
                const int next_ready = producer_progress == NULL
                        || row_tile != 0 ? 1
                    : hmx_i8_cpu_pack_progress(
                        producer_progress, row_tile + 1,
                        next_begin_tile + next_tiles, 0);
                if (next_ready < 0) return -1;
                if (next_ready > 0) {
                    if (producer_progress != NULL && row_tile == 0) {
                        qurt_mem_cache_clean(
                            (qurt_addr_t)next_source,
                            (size_t)next_tiles * HMX_I8_N_TILE * (size_t)k,
                            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
                        qurt_mem_cache_clean(
                            (qurt_addr_t)(key_sums
                                + (size_t)next_begin_tile * HMX_I8_N_TILE),
                            (size_t)next_tiles * HMX_I8_N_TILE
                                * sizeof(int32_t),
                            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
                    }
                    hmx_dma_prepare_copy(
                        &key_dma[next_buffer], next_source,
                        key_buffers[next_buffer],
                        (uint32_t)((size_t)next_tiles * HMX_I8_N_TILE
                                   * (size_t)k));
                    if (!hmx_dma_wait_idle()) return -1;
                    hmx_dma_start(&key_dma[next_buffer]);
                    next_dma_started = 1;
                }
            }

            /* The DMA engine has no next key chunk to fetch on the final
             * chunk. Use that otherwise-idle interval to stage the next query
             * row while HMX consumes the current query buffer. Never wait for
             * a streaming CPU producer here: if the row is not ready yet, the
             * next loop iteration falls back to the original synchronous
             * transfer. */
            if (next_begin_tile >= n_tiles && row_tile + 1 < m_tiles
                && prefetched_query_row < 0) {
                const int next_query_ready = producer_progress == NULL ? 1
                    : hmx_i8_cpu_pack_progress(
                        producer_progress, row_tile + 2, 0, 0);
                if (next_query_ready < 0) return -1;
                if (next_query_ready > 0) {
                    const uint8_t *next_query_source = packed_query
                        + (size_t)(row_tile + 1) * k_tiles
                            * HMX_I8_ACT_TILE_BYTES;
                    if (producer_progress != NULL) {
                        qurt_mem_cache_clean(
                            (qurt_addr_t)next_query_source,
                            (size_t)HMX_I8_M_TILE * (size_t)k,
                            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
                    }
                    hmx_dma_prepare_copy(
                        &query_dma, next_query_source,
                        query_buffers[current_query_buffer ^ 1],
                        (uint32_t)((size_t)HMX_I8_M_TILE * (size_t)k));
                    if (!hmx_dma_wait_idle()) return -1;
                    hmx_dma_start(&query_dma);
                    prefetched_query_row = row_tile + 1;
                }
            }

            for (int local_tile = 0; local_tile < current_tiles;
                 ++local_tile) {
                const int column_tile = chunk_begin_tile + local_tile;
                const int8_t *key_tiles = key_buffers[current_buffer]
                    + (size_t)local_tile * k_tiles
                        * HMX_I8_WEIGHT_TILE_BYTES;
                hmx_raw_clear_i32();
                hmx_raw_mac_u8i8_deep(
                    query_buffers[current_query_buffer], key_tiles, k_tiles);
#if HMX_HVX_EPILOGUE_LANES > 1
                if (parallel_epilogue) {
                    const int readout_slot =
                        epilogue_batch_buffer
                            * HMX_HVX_EPILOGUE_BATCH_TILES
                        + epilogue_batch_count;
                    read_accumulator_planes(
                        config,
                        readout_slots + (size_t)readout_slot
                            * ACC_READOUT_SLOT_BYTES);
                    epilogue_batch_rows[epilogue_batch_count] =
                        row_tile * HMX_I8_M_TILE;
                    epilogue_batch_columns[epilogue_batch_count] =
                        column_tile * HMX_I8_N_TILE;
                    ++epilogue_batch_count;
                    ++produced_tiles;
                    if (epilogue_batch_count
                            == HMX_HVX_EPILOGUE_BATCH_TILES
                        || produced_tiles == total_output_tiles) {
                        const int final_batch =
                            produced_tiles == total_output_tiles;
                        /* The workers consumed the other VTCM half while the
                         * handler produced this batch. Reclaim them only at
                         * the coarse batch boundary, then publish this half. */
                        const uint8_t *batch_readout = readout_slots
                            + (size_t)epilogue_batch_buffer
                                * HMX_HVX_EPILOGUE_BATCH_TILES
                                * ACC_READOUT_SLOT_BYTES;
                        if (external_epilogue) {
                            hmx_external_epilogue_wait(
                                &external_epilogue_job);
                            hmx_external_epilogue_dispatch(
                                &external_epilogue_job, scheduler, output, n,
                                epilogue_batch_rows,
                                epilogue_batch_columns, batch_readout,
                                epilogue_batch_count, key_sums, output_scale,
                                output_i32, final_batch);
                            if (final_batch) {
                                hmx_external_epilogue_wait(
                                    &external_epilogue_job);
                            }
                        } else {
                            hmx_epilogue_executor_wait(&epilogue_executor);
                            hmx_epilogue_executor_dispatch(
                                &epilogue_executor, output, n,
                                epilogue_batch_rows,
                                epilogue_batch_columns, batch_readout,
                                epilogue_batch_count, key_sums, output_scale,
                                output_i32, final_batch);
                            if (final_batch) {
                                hmx_epilogue_executor_wait(
                                    &epilogue_executor);
                            }
                        }
                        epilogue_batch_count = 0;
                        epilogue_batch_buffer ^= 1;
                    }
                } else
#endif
                {
                    if (have_previous) {
                        if (output_i32) {
                            store_readout_slot_i32(
                                (int32_t *)output, n, previous_row_start,
                                previous_column_start,
                                readout_slots + (size_t)previous_slot
                                    * ACC_READOUT_SLOT_BYTES,
                                key_sums);
                        } else {
                            quantize_readout_slot_i8(
                                (int8_t *)output, n, previous_row_start,
                                previous_column_start,
                                readout_slots + (size_t)previous_slot
                                    * ACC_READOUT_SLOT_BYTES,
                                key_sums, output_scale);
                        }
                    }
                    read_accumulator_planes(
                        config,
                        readout_slots
                            + (size_t)current_slot
                                * ACC_READOUT_SLOT_BYTES);
                    previous_slot = current_slot;
                    previous_row_start = row_tile * HMX_I8_M_TILE;
                    previous_column_start = column_tile * HMX_I8_N_TILE;
                    have_previous = 1;
                    current_slot ^= 1;
                }
            }

            if (next_tiles != 0 && !next_dma_started) {
                const int8_t *next_source = packed_key
                    + (size_t)next_begin_tile * k_tiles
                        * HMX_I8_WEIGHT_TILE_BYTES;
                if (hmx_i8_cpu_pack_progress(
                        producer_progress, row_tile + 1,
                        next_begin_tile + next_tiles, 1) <= 0) {
                    return -1;
                }
                if (row_tile == 0) {
                    qurt_mem_cache_clean(
                        (qurt_addr_t)next_source,
                        (size_t)next_tiles * HMX_I8_N_TILE * (size_t)k,
                        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
                    qurt_mem_cache_clean(
                        (qurt_addr_t)(key_sums
                            + (size_t)next_begin_tile * HMX_I8_N_TILE),
                        (size_t)next_tiles * HMX_I8_N_TILE * sizeof(int32_t),
                        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
                }
                hmx_dma_prepare_copy(
                    &key_dma[next_buffer], next_source,
                    key_buffers[next_buffer],
                    (uint32_t)((size_t)next_tiles * HMX_I8_N_TILE
                               * (size_t)k));
                if (!hmx_dma_wait_idle()) return -1;
                hmx_dma_start(&key_dma[next_buffer]);
                next_dma_started = 1;
            }
            if (next_dma_started && !hmx_dma_wait_idle()) return -1;
            chunk_begin_tile = next_begin_tile;
            current_buffer = next_buffer;
            current_tiles = next_tiles;
        }
    }
#if HMX_HVX_EPILOGUE_LANES > 1
    if (parallel_epilogue) {
        if (external_epilogue) {
            hmx_external_epilogue_end(&external_epilogue_job);
        } else {
            hmx_epilogue_executor_wait(&epilogue_executor);
            hmx_epilogue_executor_end(&epilogue_executor);
        }
    } else
#endif
    if (have_previous) {
        if (output_i32) {
            store_readout_slot_i32(
                (int32_t *)output, n, previous_row_start,
                previous_column_start,
                readout_slots + (size_t)previous_slot
                    * ACC_READOUT_SLOT_BYTES,
                key_sums);
        } else {
            quantize_readout_slot_i8(
                (int8_t *)output, n, previous_row_start,
                previous_column_start,
                readout_slots + (size_t)previous_slot
                    * ACC_READOUT_SLOT_BYTES,
            key_sums, output_scale);
        }
    }
    return 0;
}

int hmx_i8_kernel_qk_i8_packed(
    int8_t *output,
    const uint8_t *packed_query,
    const int8_t *packed_key,
    const int32_t *key_sums,
    int m,
    int k,
    int n,
    float output_scale,
    uint8_t *vtcm,
    uint32_t vtcm_size) {
    return hmx_i8_kernel_qk_packed_impl(
        output, packed_query, packed_key, key_sums, m, k, n,
        output_scale, 0, vtcm, vtcm_size, NULL, NULL);
}

int hmx_i8_kernel_qk_i8_packed_streamed(
    int8_t *output,
    const uint8_t *packed_query,
    const int8_t *packed_key,
    const int32_t *key_sums,
    int m,
    int k,
    int n,
    float output_scale,
    volatile int32_t *producer_progress,
    uint8_t *vtcm,
    uint32_t vtcm_size) {
    return hmx_i8_kernel_qk_packed_impl(
        output, packed_query, packed_key, key_sums, m, k, n,
        output_scale, 0, vtcm, vtcm_size, NULL, producer_progress);
}

int hmx_i8_kernel_qk_i8_packed_scheduled(
    int8_t *output,
    const uint8_t *packed_query,
    const int8_t *packed_key,
    const int32_t *key_sums,
    int m,
    int k,
    int n,
    float output_scale,
    uint8_t *vtcm,
    uint32_t vtcm_size,
    const hmx_i8_task_scheduler *scheduler) {
    return hmx_i8_kernel_qk_packed_impl(
        output, packed_query, packed_key, key_sums, m, k, n,
        output_scale, 0, vtcm, vtcm_size, scheduler, NULL);
}

int hmx_i8_kernel_qk_i32_packed(
    int32_t *output,
    const uint8_t *packed_query,
    const int8_t *packed_key,
    const int32_t *key_sums,
    int m,
    int k,
    int n,
    uint8_t *vtcm,
    uint32_t vtcm_size) {
    return hmx_i8_kernel_qk_packed_impl(
        output, packed_query, packed_key, key_sums, m, k, n,
        0.0f, 1, vtcm, vtcm_size, NULL, NULL);
}

static inline HVX_Vector hmx_i8_quantize_f32x32(
    HVX_Vector input, HVX_Vector inverse_scale) {
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector scaled = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_VsfVsf(input, inverse_scale));
    const HVX_Vector positive_half = Q6_V_vsplat_R(0x3f000000);
    const HVX_Vector negative_half = Q6_V_vsplat_R((int32_t)0xbf000000u);
    const HVX_VectorPred negative =
        Q6_Q_vcmp_gt_VsfVsf(zero, scaled);
    const HVX_Vector rounding = Q6_V_vmux_QVV(
        negative, negative_half, positive_half);
    scaled = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_VsfVsf(scaled, rounding));
    HVX_Vector words = Q6_Vw_equals_Vsf(scaled);
    words = Q6_Vw_vmax_VwVw(words, Q6_V_vsplat_R(-127));
    words = Q6_Vw_vmin_VwVw(words, Q6_V_vsplat_R(127));
    return words;
}

static inline HVX_Vector hmx_i8_pack_four_i32_vectors(
    HVX_Vector row0, HVX_Vector row1,
    HVX_Vector row2, HVX_Vector row3) {
    const HVX_Vector low = Q6_Vh_vpack_VwVw_sat(row1, row0);
    const HVX_Vector high = Q6_Vh_vpack_VwVw_sat(row3, row2);
    return Q6_Vb_vpack_VhVh_sat(high, low);
}

int hmx_i8_hvx_pack_query_f32_ah(
    uint8_t *packed_query,
    const float *query,
    int rows,
    int m,
    int k,
    float inverse_scale) {
    if (!packed_query || !query || rows <= 0 || rows > m || m <= 0 ||
        k <= 0 || (m % HMX_I8_M_TILE) != 0 ||
        (k % HMX_I8_K_TILE) != 0 || !(inverse_scale > 0.0f) ||
        inverse_scale > FLT_MAX) {
        return -1;
    }
    uint32_t scale_bits = 0;
    memcpy(&scale_bits, &inverse_scale, sizeof(scale_bits));
    const HVX_Vector scale = Q6_V_vsplat_R((int32_t)scale_bits);
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector sign_bias = Q6_V_vsplat_R((int32_t)0x80808080u);
    const int k_tiles = k / HMX_I8_K_TILE;
    for (int row_tile = 0; row_tile < m; row_tile += HMX_I8_M_TILE) {
        for (int inner_tile = 0; inner_tile < k_tiles; ++inner_tile) {
            uint8_t *tile = packed_query
                + ((size_t)(row_tile / HMX_I8_M_TILE) * (size_t)k_tiles
                   + (size_t)inner_tile) * HMX_I8_ACT_TILE_BYTES;
            for (int row = 0; row < HMX_I8_M_TILE; row += 4) {
                HVX_Vector words[4];
                for (int lane_row = 0; lane_row < 4; ++lane_row) {
                    const int source_row = row_tile + row + lane_row;
                    if (source_row < rows) {
                        const float *source = query
                            + (size_t)source_row * (size_t)k
                            + (size_t)inner_tile * HMX_I8_K_TILE;
                        words[lane_row] = hmx_i8_quantize_f32x32(
                            *(const HVX_UVector *)source, scale);
                    } else {
                        words[lane_row] = zero;
                    }
                }
                const HVX_Vector packed = Q6_V_vxor_VV(
                    hmx_i8_pack_four_i32_vectors(
                        words[0], words[1], words[2], words[3]),
                    sign_bias);
                *(HVX_UVector *)(tile + (size_t)row * HMX_I8_K_TILE)
                    = packed;
            }
        }
    }
    asm volatile("barrier" : : : "memory");
    return 0;
}

int hmx_i8_hvx_pack_key_f16_wh(
    int8_t *packed_key,
    int32_t *key_sums,
    const uint16_t *key,
    int n,
    int k,
    int key_begin,
    float inverse_scale) {
    if (!packed_key || !key_sums || !key || n <= 0 || k <= 0 ||
        (n % HMX_I8_N_TILE) != 0 || (k % HMX_I8_K_TILE) != 0 ||
        key_begin < 0 || key_begin > n || !(inverse_scale > 0.0f) ||
        inverse_scale > FLT_MAX) {
        return -1;
    }
    uint32_t scale_bits = 0;
    memcpy(&scale_bits, &inverse_scale, sizeof(scale_bits));
    const HVX_Vector scale = Q6_V_vsplat_R((int32_t)scale_bits);
    const HVX_Vector one_hf = Q6_Vh_vsplat_R(0x3c00);
    const HVX_Vector zero = Q6_V_vzero();
    const int first_token = (key_begin / HMX_I8_N_TILE) * HMX_I8_N_TILE;
    const int k_tiles = k / HMX_I8_K_TILE;
    _Alignas(128) int8_t quantized_rows[2][128];
    for (int token = first_token; token < n; token += 2) {
        int32_t sums[2] = {0, 0};
        for (int inner = 0; inner < k; inner += 64) {
            HVX_Vector packed_rows[2];
            for (int row = 0; row < 2; ++row) {
                const uint16_t *source = key
                    + (size_t)(token + row) * (size_t)k + (size_t)inner;
                const HVX_Vector halves = *(const HVX_UVector *)source;
                const HVX_VectorPair fp32 =
                    Q6_Wqf32_vmpy_VhfVhf(halves, one_hf);
                HVX_Vector lo = hmx_i8_quantize_f32x32(
                    Q6_Vsf_equals_Vqf32(Q6_V_lo_W(fp32)), scale);
                HVX_Vector hi = hmx_i8_quantize_f32x32(
                    Q6_Vsf_equals_Vqf32(Q6_V_hi_W(fp32)), scale);
                /* HF->QF32 expands even and odd half lanes into the low/high
                 * vectors. Quantize those vectors independently, then
                 * interleave them back into ordinary row-major byte order. */
                const HVX_Vector even_half =
                    Q6_Vh_vpack_VwVw_sat(zero, lo);
                const HVX_Vector odd_half =
                    Q6_Vh_vpack_VwVw_sat(zero, hi);
                const HVX_Vector even_bytes =
                    Q6_Vb_vpack_VhVh_sat(zero, even_half);
                const HVX_Vector odd_bytes =
                    Q6_Vb_vpack_VhVh_sat(zero, odd_half);
                packed_rows[row] = Q6_V_lo_W(
                    Q6_W_vshuff_VVR(odd_bytes, even_bytes, -1));
                *(HVX_Vector *)quantized_rows[row] = packed_rows[row];
                const int valid = k - inner < 64 ? k - inner : 64;
                for (int lane = 0; lane < valid; ++lane) {
                    sums[row] += quantized_rows[row][lane];
                }
            }
            for (int half_tile = 0; half_tile < 2; ++half_tile) {
                const int inner_tile = inner / HMX_I8_K_TILE + half_tile;
                if (inner_tile >= k_tiles) break;
                int8_t *tile = packed_key
                    + ((size_t)(token / HMX_I8_N_TILE) * (size_t)k_tiles
                       + (size_t)inner_tile) * HMX_I8_WEIGHT_TILE_BYTES;
                for (int row = 0; row < 2; ++row) {
                    const uint32_t *source_words = (const uint32_t *)(
                        quantized_rows[row] + half_tile * HMX_I8_K_TILE);
                    uint32_t *destination_words = (uint32_t *)(
                        tile
                        + (size_t)((token + row) % HMX_I8_N_TILE) * 4u);
                    for (int group = 0; group < 8; ++group) {
                        destination_words[group * 32] = source_words[group];
                    }
                }
            }
        }
        key_sums[token] = sums[0];
        key_sums[token + 1] = sums[1];
    }
    asm volatile("barrier" : : : "memory");
    return 0;
}

int hmx_i8_hvx_scale_f32(
    const float *input, int rows, int stride, int columns, float *scale) {
    if (!input || !scale || rows <= 0 || stride < columns || columns <= 0) {
        return -1;
    }
    const HVX_Vector abs_mask = Q6_V_vsplat_R(0x7fffffff);
    _Alignas(128) uint32_t lanes[32];
    HVX_Vector vector_maximum = Q6_V_vzero();
    uint32_t tail_maximum = 0;
    for (int row = 0; row < rows; ++row) {
        const float *source = input + (size_t)row * (size_t)stride;
        int column = 0;
        for (; column + 32 <= columns; column += 32) {
            const HVX_Vector values = *(const HVX_UVector *)(source + column);
            vector_maximum = Q6_Vw_vmax_VwVw(
                vector_maximum, Q6_V_vand_VV(values, abs_mask));
        }
        for (; column < columns; ++column) {
            uint32_t bits;
            memcpy(&bits, source + column, sizeof(bits));
            bits &= UINT32_C(0x7fffffff);
            if (bits >= UINT32_C(0x7f800000)) return -1;
            if (bits > tail_maximum) tail_maximum = bits;
        }
    }
    *(HVX_Vector *)lanes = vector_maximum;
    uint32_t maximum_bits = tail_maximum;
    for (int lane = 0; lane < 32; ++lane) {
        if (lanes[lane] >= UINT32_C(0x7f800000)) return -1;
        if (lanes[lane] > maximum_bits) maximum_bits = lanes[lane];
    }
    float maximum;
    memcpy(&maximum, &maximum_bits, sizeof(maximum));
    *scale = maximum > 0.0f ? maximum * (1.0f / 127.0f) : FLT_MIN;
    return 0;
}

static float hmx_i8_positive_half_to_float(uint16_t half) {
    const uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t mantissa = half & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) return 0.0f;
        int shift = 0;
        while ((mantissa & 0x0400u) == 0) {
            mantissa <<= 1;
            ++shift;
        }
        mantissa &= 0x03ffu;
        bits = (uint32_t)(113 - shift) << 23;
        bits |= mantissa << 13;
    } else {
        bits = (exponent + 112u) << 23;
        bits |= mantissa << 13;
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

int hmx_i8_hvx_scale_f16(
    const uint16_t *input, int rows, int stride, int columns, float *scale) {
    if (!input || !scale || rows <= 0 || stride < columns || columns <= 0) {
        return -1;
    }
    const HVX_Vector abs_mask = Q6_V_vsplat_R(0x7fff7fff);
    _Alignas(128) uint16_t lanes[64];
    HVX_Vector vector_maximum = Q6_V_vzero();
    uint16_t tail_maximum = 0;
    for (int row = 0; row < rows; ++row) {
        const uint16_t *source = input + (size_t)row * (size_t)stride;
        int column = 0;
        for (; column + 64 <= columns; column += 64) {
            const HVX_Vector values = *(const HVX_UVector *)(source + column);
            vector_maximum = Q6_Vuh_vmax_VuhVuh(
                vector_maximum, Q6_V_vand_VV(values, abs_mask));
        }
        for (; column < columns; ++column) {
            const uint16_t bits = source[column] & 0x7fffu;
            if (bits >= 0x7c00u) return -1;
            if (bits > tail_maximum) tail_maximum = bits;
        }
    }
    *(HVX_Vector *)lanes = vector_maximum;
    uint16_t maximum_bits = tail_maximum;
    for (int lane = 0; lane < 64; ++lane) {
        if (lanes[lane] >= 0x7c00u) return -1;
        if (lanes[lane] > maximum_bits) maximum_bits = lanes[lane];
    }
    const float maximum = hmx_i8_positive_half_to_float(maximum_bits);
    *scale = maximum > 0.0f ? maximum * (1.0f / 127.0f) : FLT_MIN;
    return 0;
}
