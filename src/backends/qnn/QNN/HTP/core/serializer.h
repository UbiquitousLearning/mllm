//==============================================================================
//
// Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SERIALIZER_H
#define SERIALIZER_H 1

#include <cstdio>
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <array>
#include <vector>
#include <map>
#include <typeinfo>
#include <typeindex>
#include <string_view>
#include "limits.h"
#include "dtype.h"
#include "log.h"
#include "allocator.h"
#include "op_extra_info.h"

#include "serialize_defs.h"
#include "forward_classes.h"
#include "minihash.h"
#include "serialize_oplist.h"
#include "serialize_tensors.h"
#include "macros_attribute.h"
#include "heap_stats.h"

namespace fa {
struct FancyAllocator;
} // namespace fa

namespace hnnx {
class BlockTableReducerIfc;

//
// An instance of this is in the Serializer, as public data member 'size_info';
// it used to pass 'size' information, generated during the serialization process,
// to the 'htp_header_insert_picklesize' operation.
// It also contains flags indicating how long the 'SIZE' header will need to be,
// which are used in serialize_header (when that is called, we don't know any of the values,
// but we need to know which will be non-zero).
//
// Note that Serializer has bool crate_accounting_enabled() method; but that may be gated on or off
// during the serialization process, whereas 'has_crate_accounting' below is constant for a given
// run.
//
struct size_info_for_header_t {
    bool has_const_extent; // Are we are making a separate const-extent
    bool has_crate_accounting;
    // fields below are not usable by serialize_header
    size_t pickle_size; // basic picklesize, bytes, not including 'const_extent'
    uint64_t const_extent_size; // size of 'const_extent' section, 0 if absent
    size_t crate_size; // crate size prediction, 0 if none.

    // values specifically for the 'MEMORY' recorded
    heap_stats::heapstats_array heap_info;
};

/**
 * @brief \ref Serializer and \ref Deserializer modules that provides
 * a mechanism to flatten (serialize) and reconstruct (deserialize)
 * primitive and user-defined data types. The initial objective
 * was to create an in-memory representation of the optimized
 * \ref Graph on x86 which can then be reconstructed and executed on
 * a qdsp target, essentially, a means to Graph caching.
 *
 */

/**
 * @class Serializer
 *
 * @brief Serializer class that converts data of type T to
 * a flat buffer representation.  The Serializer object
 * needs to know before-hand the total buffer size required.
 * Incremental calls to the serializer object continually
 * appends data to the allocated buffer.
 */

class Serializer : public DeSerError, public SerOpsInterface {
  private:
    using type_index_map = minihash_noerase<std::type_info const *, unsigned>;
    type_index_map optype_idx_map;
    type_index_map tensortype_idx_map;

    // the state of the 'tensor connectivity' serialize engine.
    SerTensorConn tensorconn;

  protected:
    const Graph &graph_ref;
    const char *bufstart; // start of current buffer
    const char *bufend; // first byte we can't write
    char *bufp; // next to write
    size_t bytes_flushed; // bytes previously flushed
    bool overflowed;
    bool measuring; // true if we are only measuring;
    // this is used to serialize shape, interface object without
    // duplicating shared objects. The key is the object address; the value is
    // is an index used to encode; these are assigned 1,2,3 ...
    minihash_noerase<void const *, unsigned> objmap;

    bool aligned_const_format_flag = false;

    // This is used to assist in generating 'multi-section' files.
    // In 'single section' mode, it is always 0. See comment above get_section_id()
    // for more info.
    int multi_file_section_id = 0;

    // 'format version'. Currently only ones used are 0 = classic, 1 = f0723
    // Only access through methods like .classic_format(), set_format_classic();
    int format_version = 1;

    std::vector<uint32_t> blocktable_buffer;
    std::vector<std::pair<unsigned, unsigned>> blocktable_ptr_buffer;

    // results from the 'prescan' phase
    std::vector<std::pair<std::string_view, unsigned>> op_class_index; // for generating "Co" Aux Data record.
    std::vector<std::pair<std::string_view, unsigned>> tensor_class_index; // for generating "Ct" Aux Data
    unsigned prescan_tensor_count = 0;

  public:
    fa::FancyAllocator *allocator;
    // this is a simple state machine to help serialize 'aux-data' properly:
    // it is managed within Graph::serialize, FancyAllocator::serialize, and auxdata_serialize.
    enum {
        auxdata_not_ok, // can't insert aux data here
        auxdata_ok, // can insert, no action needed
        auxdata_after_alloc, // can insert but need a marker word first and -> after_alloc_ok
        auxdata_after_alloc_ok, // can insert, no action; but end marker is pending.
    } aux_data_state = auxdata_not_ok;
    // info for 'serialize_header' and 'htp_header_insert_picklesize' to use.

    size_info_for_header_t size_info = {};
    /**
	 * @brief Construct a new Serializer object
	 *
	 * @param[in] allocator_in \ref fa::FancyAllocator to get access
	 * to memory used by tensors
	 * @param[in] p allocated buffer
	 * @param[in] n size of buffer
	 */
    Serializer(const Graph &graph_in, Allocator *allocator_in, char *p, size_t n);
    /**
	 * @brief Destroy the Serializer object
	 *
	 */
    API_EXPORT virtual ~Serializer(); // please keep this as first virtual method declared.

    const Graph &graph() const { return graph_ref; };

    // This used to inform serialize_header() (and similar operations) what part of
    // a multi-section file they are generating.
    // 0 - no multi-section operation
    // -1  -> main header of multi-section file
    // 1...0xFFFF -> section id of current section in multi-section file
    inline int get_section_id() const { return multi_file_section_id; }
    inline void set_section_id(int section) { multi_file_section_id = section; }

    constexpr bool is_aligned_const_format() const { return aligned_const_format_flag; }
    void set_aligned_const_format(const bool v = true) { aligned_const_format_flag = v; }

    inline constexpr bool classic_format() const { return format_version == 0; }
    inline void set_format_classic() { format_version = 0; }

  protected:
    /**
	 * @brief Tracks data writes to buffer for over flow cases and sets
	 * buffer pointer to start while recording \ref bytes_flushed
	 *
	 * Note: all serializing is done via serialize_fwrite, or by simple_serialize (which
	 * calls flush_buffer when needed).
	 */
    API_EXPORT virtual void flush_buffer(); // call for overflow on short operation

    /**
	 * @brief Call for an arbitary length data that needs to be written
	 * to the serializer buffer
	 *
	 * @param[in] p data that needs to be copied into serializer buffer
	 * @param[in] len length of the data to write
	 * @param[in] align True to align data write to a boundary of 4
	 *
	 * Note: all serializing is done via serialize_fwrite or by simple_serialize
	 */
    API_EXPORT virtual void serialize_fwrite(const void *p, size_t len, bool align);

    /**
	 * @brief Call for an arbitary length data that needs to be written
	 * to the serializer buffer. This is used only when we simply want
	 * to count the length of the buffer. see \ref NullSerializer
	 *
	 * @param[in] p data that needs to be copied into serializer buffer
	 * @param[in] len length of the data to write
	 * @param[in] align True to align data write to a boundary of 4
	 *
	 */
    API_EXPORT void serialize_dummy_fwrite(const void *p, size_t len, bool align);
    /**
	 * @brief Tracks data writes to buffer for over flow cases and sets
	 * buffer pointer to start while recording num bytes flushed.
	 * This is used only when we simply want
	 * to count the length of the buffer. see \ref NullSerializer
	 *
	 */
    API_EXPORT void dummy_flush_buffer();

    /**
	 * @brief Get current position of buffer at which next data will be written
	 *
	 * @return size_t offset from buffer start
	 */
    size_t buffer_offset() const { return bufp - bufstart; }

    /**
	 * @brief Available buffer size remaining for serialization
	 *
	 * @return size_t remaining bytes size
	 */
    size_t buffer_remain() const { return bufend - bufp; }

    /**
	 * @brief serialize data of type T
	 *
	 * @tparam[in] value to serialize of type T
	 */
    // note: flush_buffer must always flush in multiples of 4, and copy any extra
    // to the front of the buffer.
    template <typename T> void simple_serialize(T val)
    {
        constexpr size_t W = (sizeof(T) < 4) ? 4 : sizeof(T);
        char *buf_next = bufp + W;
        if (buf_next > bufend) { // no room
            flush_buffer();
            buf_next = bufp + W;
        }
        if (sizeof(T) < 4) {
            // shorter int, ensure it fills the whole 32 bits
            *reinterpret_cast<uint32_t *>(bufp) = val;
        } else {
            *reinterpret_cast<T *>(bufp) = val;
        }
        bufp = buf_next;
    }
    API_EXPORT bool serialize_shared_obj_func(void const *p);

    API_EXPORT unsigned serialize_X_type(type_index_map &tidxmap, std::type_info const &tid, char const *name = nullptr,
                                         uint32_t flags = 0);

    // gateway for all types of 'generic' functions.
    API_EXPORT virtual void op_serialize_func(Op const *op, unsigned n_in, Tensor const *const *in_tens, unsigned n_out,
                                              uptr_Tensor const *out_tens, unsigned mode) override;
    // Used for ConstWrapperOp, ShapeWrapperOp, DummyN
    API_EXPORT virtual void op_for_tensor_func(Op const *op, unsigned n_out, uptr_Tensor const *out_tens) override;

    // used for all tensor cases
    API_EXPORT virtual void tensor_serialize(Tensor const *tens) override;

    // 'top-level' sequencing calls from SerOpsInterface
    // Before serializing the allocator, present all Ops to prescan_ops in the same order as they will be serialized.
    // This may be done in more than one call, with 'last = true' on the last one.
    API_EXPORT virtual void prescan_ops_func(Op *const *seq_of_ops, unsigned n_ops, bool last = false) override;

    API_EXPORT void prescan_tensor(Tensor const *);
    API_EXPORT void make_class_index_aux_record(std::vector<std::pair<std::string_view, unsigned>> const &table,
                                                bool is_tensor);

  public:
    API_EXPORT virtual void graph_io_tensors(unsigned n_in, uptr_Tensor const *in_tensors, unsigned n_out,
                                             uptr_Tensor const *out_tensors, bool is_prescan = false) override;
    API_EXPORT virtual void checkpoints_table(hnnx::Checkpoints const &) override;
    API_EXPORT virtual void before_runlists(unsigned nops_norun, unsigned nops_main, unsigned nops_vector,
                                            unsigned nops_mtx) override; // call before serializing 'non-runlist'
    API_EXPORT virtual void
    after_non_runlist() override; // call after serializing 'non_runlist', before 'combined runlist'
    API_EXPORT virtual void after_runlist() override; // call after runlist complete.

    // This is called directly before serializing each op; op_seqno is the 0-based index
    // (i.e. the number of ops previously serialized).
    API_EXPORT void add_op_marker(unsigned op_seqno) override;

    // used to handle 'framework' ops
    API_EXPORT virtual OpSerHandle op_special(Op const *op) override;

    // Given a pointer to a ShapeFlags which is really a Shape<RANK>, serialize its content.
    // Does not include 'shared object' protocol (only called when we have a new one)
    API_EXPORT virtual void shape_serialize(ShapeFlags const *basep, unsigned rank) override;

    // 'crate size accounting'
    void enable_create_accounting(bool newval = true) { crate_acct_enable = newval; }
    size_t crate_accounting_total_size() const { return crate_accounting_info.accum_size; }

  protected:
    bool crate_accounting_enabled() const { return crate_acct_enable; }
    API_EXPORT void crate_accounting(unsigned obj_size, unsigned obj_align);
    inline void crate_accounting(std::pair<unsigned, unsigned> szal) { crate_accounting(szal.first, szal.second); }
    // ops made with op_special are trickier for crate accounting: we don't know the size until spcl_done is called,
    // and the op may have a connected cratevec (or other object). So, these are accumulated here, and when spcl_done
    // is called, it issues one or two crate_accounting (the 'extra' being second, if present).
    struct spcl_op_acct {
        unsigned op_len; // current len of the op
        unsigned op_align; // current align of the op...
        unsigned extra_len; // len of 'extra' object (or 0 if none)
        unsigned extra_align; // align of 'extra' object
    } spcl_op_crate_accounting = {};
    bool crate_acct_enable = false;
    unsigned crate_preload_spacing = 32768; // 'target' spacing for preload ops

    void account_more_to_spcl_op(unsigned size, unsigned align); // adds incrementally to spcl_op_account
    API_EXPORT void do_insert_preload_op(); // actually inserts a preload (just does the accounting, in fact)
    // this keeps track of the 'crate accounting' when enabled:
    // - number of 4-byte-aligned requests; total size
    // - number of 8-byte-aligned requests; total size
    // - cumulative total (if objects appended in order respecting alignment of each)
    struct crate_acct_stats {
        unsigned num_a4;
        unsigned num_a8;
        size_t total_a4;
        size_t total_a8;
        size_t accum_size;
        size_t prev_preload_posn; // accum_size at the end of the most recent ChunkPreloadOp (0 if none)
        unsigned num_preload; // number inserted.
    } crate_accounting_info = {};

    // the 'blocktable reducer' object - or NULL if we are not doing that.
    std::unique_ptr<BlockTableReducerIfc> blocktable_reducer;
    // methods called by methods of the 'OpSerHandle'
    // See OpSerHandle to see what they do.
    API_EXPORT virtual void spcl_done(OpSerHandle &) override; // called by ~OpSerHandle
    API_EXPORT virtual void spcl_add_u32(OpSerHandle &, uint32_t const *p, unsigned n) override;
    API_EXPORT virtual void spcl_add_sized_vec(OpSerHandle &, uint32_t const *data, bool extra) override;
    API_EXPORT virtual void spcl_fill_nullptr(OpSerHandle &, unsigned n) override;

    // Get the 'op_format_code' for a given op. It is expected that this will be called at most
    // once per distinct Op type, when serializing.
    // The return value is {op_format_code, valid_flag}
    // valid_flag = false indicates that an error happened.
    API_EXPORT static std::pair<uint32_t, bool> obtain_op_format_code(Op const *op);

    // Generate block tables for tensors
    // these include generating the 'TENSOR_MAGIC' marker in classic format.
    // One for tensors which are not indirect:
    API_EXPORT void serialize_single_tensor_pointer(void const *ptr);
    // .. and one which serializes the block table on indirect tensors.
    // The return value is the size of created block table, for crate accounting.
    // (with shared block tables, it could be zero, or could be larger than num_blocks)
    API_EXPORT unsigned serialize_blocktable(void *const *blktable, unsigned num_blocks, bool is_tcm);

    API_EXPORT void write_blocktable_buffer();

  public:
    /**
	 * @brief a 'serialize' operation can call is_measuring(); if it returns true,
	 * the serialization op is only being done to measure size; in this case,
	 * you can instead call measure_bytes() and supply the data length,rather
	 * than serializing.
	 *
	 * NOTE: the parameter to measure_bytes will always be rounded up to a multiple
	 * of 4.
	 *
	 */
    bool is_measuring() const { return measuring; }
    /**
	 * @brief Called only when just 'measuring' and is used
	 * to account for new size required for the serialization data
	 * using 'len' bytes (rounded up to a multiple of 4). The call
	 * also flushes the buffer. Note that the current position
	 * must also be a multiple of 4.
	 *
	 * @param[in] len length that needs to be accounted for
	 */
    API_EXPORT virtual void measure_bytes(size_t len);

    /**
	 * @brief Gets the total bytes required for the serialization data
	 *
	 * @return size_t Length of bytes required for serialization
	 */
    size_t total_bytes() const { return bytes_flushed + buffer_offset(); }

    /**
	 * @brief Get status on the overflowed bit
	 *
	 * @return true if target serialization buffer has overflowed
	 * @return false if target serialization buffer has unfilled bytes
	 */
    bool is_overflowed() const { return overflowed; }

    /**
	 * @brief Final flush of the buffer which is called by client
	 * at the end of serialization process
	 *
	 * @return size_t total bytes serialized
	 */
    virtual size_t finalize() { return total_bytes(); }

    /**
	 * @brief serialize data of type which calls simple_serialize
	 *
	 * @param val data to serialize
	 *
	 * Note: the below are the only types supported for serialize_type<T>
	 */
    void serialize_uint64(uint64_t val); // inline below
    inline void serialize_float(float val) { simple_serialize<float>(val); }
    inline void serialize_uint32(uint32_t val) { simple_serialize<uint32_t>(val); }
    inline void serialize_int32(NN_INT32_T val) { simple_serialize<NN_INT32_T>(val); }
    inline void serialize_uint16(uint16_t val) { simple_serialize<uint16_t>(val); }
    inline void serialize_int16(int16_t val) { simple_serialize<int16_t>(val); }
    inline void serialize_uint8(uint8_t val) { simple_serialize<uint8_t>(val); }
    inline void serialize_int8(int8_t val) { simple_serialize<int8_t>(val); }

    // serialize string
    void serialize_str(const std::string_view &val) { serialize_buf_withlen(val.data(), val.size()); }
    // serialize 'n' bytes, padding with zeros to a multiple of 4
    void serialize_bytes(void const *p, unsigned len) { serialize_fwrite(p, len, true); }

    // namesig as alias for uint64, in case we want to change namesig_t
    inline void serialize_namesig(const uint64_t val) { serialize_uint64(val); }

    // serialize a pointer as 64 bits
    inline void serialize_ptr(void const *p) { serialize_uint64(size_t(p)); }

    // For each op serialized:
    //  - serialize all the input refs  with serialize_tensor_ref and/or serialize_tensor_refs
    //  - register addresses of all out tensors with serialize_tensor_def.
    //  Note that Tensorsubclass->serialize() will call serialize_def,
    //  so it's only needed to call it separately on special tensor subclasses
    //  which don't get serialized themselves (typically these are embedded in special Op classes).
    // NOTE: currently, each tensor must be registered with serialize_tensor_def before
    // it can be passed to serialize_tensor_ref.
    //
    inline void serialize_tensor_def(Tensor const *p) { tensorconn.tensor_def(*this, p); }
    inline void serialize_tensor_ref(Tensor const *p) { tensorconn.tensor_ref(*this, p); }
    inline void serialize_tensor_refs(Tensor const *const *p, unsigned n) { tensorconn.tensor_refs(*this, p, n); }
    // can call serialize_tensor_refs on pointers to subclass of Tensor.
    template <typename T> inline void serialize_tensor_refs(T const *const *p, unsigned n)
    {
        static_assert(std::is_base_of<Tensor, T>::value);
        tensorconn.tensor_refs(*this, (Tensor const *const *)p, n);
    }

    void serialize_finish_tensors() { tensorconn.store_pending(*this); }

    void serialize_tensor_type(std::type_info const &ty, char const *name)
    {
        serialize_X_type(tensortype_idx_map, ty, name);
    }
    void serialize_op_type(std::type_info const &ty, char const *name, uint32_t flags = 0)
    {
        serialize_X_type(optype_idx_map, ty, name, flags);
    }

    // write 32-bit header and then data
    API_EXPORT void serialize_buf_withlen(const void *buf, size_t bufsize);

    template <typename T> void serialize_type(T val);

    // serialize a shared object (shape or interface:
    // - serializes the id
    // - if not seen before, returns true, and caller must actually
    //   serialize the object;
    // - otherwise returns false, caller does nothing else.
    //
    template <typename T> bool serialize_shared_obj(T const *p) { return serialize_shared_obj_func((void const *)p); }

    /**
	 * @brief convernience wrappers for serialize fuctions that
	 * take in different number of arguments of uint32_t type
	 *
	 * @param[in] x0 first uint32_t data to serialize
	 * @param[in] x1 second uint32_t data to serialize
	 */
    // convenience wrappers (to reduce inlined code size w/o much loss of speed)
    API_EXPORT void serialize_uint32(uint32_t x0, uint32_t x1);
    API_EXPORT void serialize_uint32(uint32_t x0, uint32_t x1, uint32_t x2);
    API_EXPORT void serialize_uint32(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3);

    // to reduce code size in the templates, we can serialize arrays of
    // N sizet to uint32
    API_EXPORT void serialize_uint32_arr_sizet(size_t const *p, size_t N);
    API_EXPORT void serialize_uint32_arr(uint32_t const *p, size_t N);
    inline void serialize_unsigned_arr(unsigned const *p, size_t N)
    {
        static_assert(sizeof(uint32_t) == sizeof(unsigned));
        serialize_uint32_arr((uint32_t const *)p, N);
    }
    /**
	 * @brief serialize array of type std::array
	 *
	 * @tparam N num items in array
	 * @param arr actual array to serialize
	 */
    template <size_t N> inline void serialize_uint32_arr(std::array<size_t, N> const &arr)
    {
        serialize_uint32_arr_sizet(&arr[0], N);
    }
    /**
	 * @brief serialize vector
	 *
	 * @param[in] arr vector containing data of type uint32_t to serialize
	 */
    void serialize_uint32_arr(std::vector<size_t> const &arr) { serialize_uint32_arr_sizet(&arr[0], arr.size()); }
};

// unaligned store of 64-bits (two 32-bit aligned store)
template <> inline void Serializer::simple_serialize<uint64_t>(uint64_t val)
{
    constexpr size_t W = sizeof(uint64_t);
    char *buf_next = bufp + W;
    if (buf_next > bufend) { // no room
        flush_buffer();
        buf_next = bufp + W;
    }
    uint32_t *const p = reinterpret_cast<uint32_t *>(bufp);
    p[0] = (uint32_t)val;
    p[1] = (uint32_t)(val >> 32);
    bufp += W;
}

inline void Serializer::serialize_uint64(uint64_t val)
{
    simple_serialize<uint64_t>(val);
}

template <> inline void Serializer::serialize_type<uint64_t>(uint64_t val)
{
    serialize_uint64(val);
}
template <> inline void Serializer::serialize_type<float>(float val)
{
    serialize_float(val);
}
// sometimes uint32_t is unsigned long, sometimes it's unsigned
// sometimes unsigned long is uint64. Hopefully this should cover it all.
#if ULONG_MAX == UINT_MAX
template <> inline void Serializer::serialize_type<unsigned long>(unsigned long val)
{
    serialize_uint32(val);
}
template <> inline void Serializer::serialize_type<long>(long val)
{
    serialize_int32(val);
}
#endif
template <> inline void Serializer::serialize_type<int>(int val)
{
    serialize_int32(val);
}
template <> inline void Serializer::serialize_type<unsigned>(unsigned val)
{
    serialize_uint32(val);
}
template <> inline void Serializer::serialize_type<uint16_t>(uint16_t val)
{
    serialize_uint16(val);
}
template <> inline void Serializer::serialize_type<int16_t>(int16_t val)
{
    serialize_int16(val);
}
template <> inline void Serializer::serialize_type<uint8_t>(uint8_t val)
{
    serialize_uint8(val);
}
template <> inline void Serializer::serialize_type<int8_t>(int8_t val)
{
    serialize_int8(val);
}

/**
 * @brief NullSerializer class is a 'fake' serializer, to measure the len of
 *  an operation. It has a 64-byte dummy buffer built in.
 *
 */
class NullSerializer : public Serializer {
    union {
        uint32_t for_align;
        char tbuf[64];
    } uu;

  public:
    /**
	 * @brief Construct a new Null Serializer object
	 *
	 * @param[in] allocator_in \ref fa::FancyAllocator to get access
	 * to memory used by tensors
	 */
    NullSerializer(const Graph &graph_in, Allocator *allocator_in)
        : Serializer(graph_in, allocator_in, uu.tbuf, sizeof(uu.tbuf))
    {
        measuring = true;
    }
    API_EXPORT virtual ~NullSerializer();

  private:
    /**
	 * @brief Tracks data writes to buffer for over flow cases and sets
	 * buffer pointer to start while recording num bytes flushed
	 *
	 * Note: all serializing is done via serialize_fwrite, or by simple_serialize (which
	 * calls flush_buffer when needed).
	 */
    virtual void flush_buffer() override
    {
        // call for overflow on short operation
        dummy_flush_buffer();
    }
    /**
	 * @brief Call for an arbitary length data that needs to be written
	 * to the serializer buffer. Dummys out the actual write but sets
	 * up the internal len members to later provide the actual length
	 * of bytes required for serialization
	 *
	 * @param[in] p data that needs to be copied into serializer buffer
	 * @param[in] len length of the data to write
	 * @param[in] align True to align data write to a boundary of 4
	 *
	 */
    virtual void serialize_fwrite(const void *p, size_t len, bool align) override
    {
        serialize_dummy_fwrite(p, len, align);
    }
};

#if !defined(PREPARE_DISABLED) && !defined(__hexagon__)
/**
 * @brief FileSerializer class is constructed with a 'fileno' (open
 * output file) and buffersize, and can serialize to the file in one pass.
 * It has support for re-reading the first part of the file when done, modifying the
 * header, and rewriting it (this requires the file to be open in RD|WR more; it
 * should work even if bytes were written to the file prior to using FileSerializer).
 * If a buffer write operation fails, an error is recorded, the 'overflowed' flag
 * will be set, and further write operations will be suppressed. In this state the
 * total_bytes() result may be meaningless.
 */
class FileSerializer : public Serializer {

    std::unique_ptr<uint64_t[]> buffer_owner; // holds the allocated buffer
    int filedesc; // the output file
    // if, on first flush, there appears to be a pickle header with size <= buffersize,
    // that len is stored here (for use by prepare_to_modify_header).
    unsigned apparent_header_len;
    // this is used for 'header_rewrite' mechanism.
    size_t header_rewrite_len; // length of write() to be done by done_header_modify()
    int64_t saved_offset; // so 'header_rewrite' can restore the file position.

    static constexpr size_t CHUNKSIZE = 16384;
    static constexpr size_t MINBUFFER = 4 * CHUNKSIZE;

  public:
    /**
	 * @brief Construct a new FileSerializer object
	 *
	 * @param[in] allocator_in \ref fa::FancyAllocator to get access
	 * to memory used by tensors
	 */
    // The supplied buffer size may be rounded up (e.g. to a multiple of 16K)
    API_EXPORT FileSerializer(const Graph &graph_in, Allocator *allocator_in, int file_no, size_t buf_len = MINBUFFER);
    API_EXPORT virtual ~FileSerializer();

    // note, finalize can be done *before* rewriting header
    API_EXPORT virtual size_t finalize() override;
    // This must be called after serialization is complete to flush any partial
    // buffer data. It does not change the value returned by total_bytes().
    // Always check any_error() after calling this!
    API_EXPORT void flush_file();

    // Special methods to allow pickle_header to be updated, after all serialization
    // has been done:
    // prepare_to_modify_header will flush any pending writes, and then re-read the first buffer-full
    // of data (returning a pointer to it), so that the pickle-header data can be
    // updated to reflect the entire length. This *must* be followed
    // done_header_modify(), after which the file is complete, and
    // the FileSerializer can be deleted (no need for flush_file(), but it won't hurt).
    // After prepare_to_modify_header(), no serialization calls are allowed, or
    // things will get corrupted.
    // header_avail_len() can be called (after prepare_to_modify_header) to find out
    // how many bytes are available at the returned pointer.
    // The value returned by total_bytes() is not affected by these calls.
    // Important: done_header_modify must be called after prepare_to_modify_header, even
    // if you didn't change anything.
    // done_header_modify leaves the file positioned at 'end', where it was before
    // 'prepare_to_modify_header' was called.
    //
    // If a write() operation failed, prepare_to_modify_header will do nothing and
    // return nullptr, and done_header_modify will do nothing.

    API_EXPORT char *prepare_to_modify_header();
    API_EXPORT void done_header_modify();
    size_t header_avail_len() const { return header_rewrite_len; }

    // if you want to serialize some data, and then append a bunch of data with direct
    // file writes, and then use prepare_to_modify_header, you can
    //    (a) flush_buffer, after all serialize activity;
    //    (b) write some data using direct writes to the file;
    //    (c) call update_total_bytes(n) to tell it how many bytes you wrote;
    //    (d)  (repeat (b) and (c) as needed)
    //    (e) total_bytes() will reflect the extra bytes; you can now call prepare_to_modify_header
    //       and done_header_modify as usual.
    //
    //
    void update_total_bytes(size_t n) { bytes_flushed += n; }

  private:
    API_EXPORT virtual void flush_buffer() override final;
    API_EXPORT virtual void serialize_fwrite(const void *p, size_t len, bool align) override final;
};
#endif // !defined(PREPARE_DISABLED) && !defined(__hexagon__)

} // namespace hnnx

#endif // SERIALIZER_H
