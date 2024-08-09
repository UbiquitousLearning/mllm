//==============================================================================
//
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef CONST_EXTENT_DESCRIPTOR_H
#define CONST_EXTENT_DESCRIPTOR_H 1

#include <cstdio>
#include <vector>
#include <cassert>
#include <string>
#include "forward_classes.h"
#include "serialize_defs.h"
#include "pickle_header_tags.h"

namespace hnnx {

// definitions pertaining to the 'const extent descriptor'.

constexpr unsigned CONST_EXTENT_DESC_MAGIC = 0x71c43c9b;
// if a const extent descriptor has a 'cbname' in it, the last 32-bit slot
// is this value. The 0x3e, 0x00 is the ">\0" at the end of the cbname
constexpr unsigned CONST_EXTENT_CBNAME_TAG = 0xebbe003e;

// This must be a power of 2, and >= 64.
// This is effectively a 'quiet' minimum on options.serialize_const_alignment, which sets
// the actual alignment.
// It is not necessary for the decoder to know what value of alignment was used in the encoder.
constexpr unsigned CONST_EXTENT_MIN_ALIGN = 256;
//
// this is a (non-quiet) maximum on options.serialize_const_alignment
constexpr unsigned CONST_EXTENT_MAX_ALIGN = 1024 * 1024;

// This function is used by deserializer to help it extract the extent-desc table (as a vector<uint32_t>) from some
// arbitrary point down the pickle. Parameter is a pointer to the first 4 words; the return value is
//  0 if the first two words do not look like CEDesc header;
//  n otherwise (where 'n' is the number of 32-bit words to extract).
//
inline unsigned const_extent_hdr_check(uint32_t const *const hdrp)
{
    if (hdrp[0] != CONST_EXTENT_DESC_MAGIC) return 0;
    const unsigned word0 = hdrp[1];
    const unsigned hdr_len16 = word0 >> 24u; // units of 16 bytes
    const unsigned desc_len64 = word0 & 0xFFFFFFu; // units of 64 bytes
    const unsigned n_extent = hdrp[2] & 0xFFFFFFu;
    const unsigned n_mempool = hdrp[3] & 0xFFFFFFu;
    // no. of words actually needed
    const unsigned desc_words = 4 * (hdr_len16 + n_extent + n_mempool);

    // note, n_extent == n_mempool == 0 is allowed.
    if (hdr_len16 == 0 || desc_len64 == 0 || n_extent > n_mempool || desc_words > desc_len64 * 16) {
        return -1;
    }
    return desc_words;
}

// This class is used, on both encoder and decoder, to contain a 'const extent descriptor' in its raw form, (just an array of uint32)
// and provide higher-level access to the contents.

class ConstExtentDesc {
  protected:
    using table_t = std::vector<uint32_t>;
    // The 'table' may or may not contain the 'padding' section at the end; this is not accessed,
    // and the serialize method will always generate the required padding.
    table_t table;
    // some values broken out from the header...
    unsigned extab_n = 0, extab_idx = 0; // number of extents, and word index where they start
    unsigned mptab_n = 0, mptab_idx = 0; // number of memory pools, and word index where they start.
    unsigned desc_len = 0; // length of the entire descriptor in bytes (0 if invalid descriptor)

    bool scan_table(); // sanity check, and unpacks the above; returns true if OK.

  public:
    // Return from 'extent_info'.
    struct extab_entry {
        uint32_t extent_flags;
        uint32_t align; // a power of 2, >= 64
        uint64_t offset; // offset, in bytes, from the start of the descriptor, to where the data is.
        uint64_t length; // length of the data in bytes.
    };
    // Return from 'mempool_info'.
    // Note: if 'adjust_offset' is true, the 'offset' field from the containing extent will be added to offset,
    // so that the offset is from the start of the descriptor, instead of the start of the containing extent.
    struct mempool_entry {
        uint32_t mempool_id; // a mempool id >=2 indicating a const mempool
        uint32_t extent_id; // an extent_id, >=1
        uint64_t offset; // offset in bytes of the data from the start of the extent (see note above)
        uint64_t length; // length in bytes of the data
    };
    // optional name of the const_extent this descriptor corresponds to. Used for matching in weight_sharing.
    std::string name = std::string{};

    ConstExtentDesc() {}
    ConstExtentDesc(table_t &&table_in);
    void serialize(Serializer &) const;
    inline bool load_table(table_t &&table_in)
    {
        table = std::move(table_in);
        return scan_table();
    }

    constexpr bool is_valid() const { return desc_len != 0; }

    constexpr unsigned descriptor_length() const { return desc_len; }

    constexpr unsigned num_extents() const { return extab_n; }
    constexpr unsigned num_mempools() const { return mptab_n; }

    // unpack a row of the extent table
    // NOTE: extent_id is 1-based, must be 1 .. num_extents()
    extab_entry extent_info(unsigned extent_id) const;

    // unpack a row of the mempool table.
    // note: idx is not a mempool idx, it is a 1-based row in range 1...num_mempools();
    // if adjust_offset, the offset of the containing extent is added to the offset
    // of the mempool in the returned value.
    mempool_entry mempool_info(unsigned idx, bool adjust_offset = false) const;

    // The ordering of the data and the descriptors is such that:
    //
    // (1)  extent_info(1).offset >= descriptor_length()
    //      mempool_info(1,true).offset >= descriptor_length()
    // (2) for i >=2,
    //      extent_info(i).offset >= extent_info(i+1).offset + extent_info(i+1).length
    //      mempool_info(i,true).offset >= mempool_info(1-1,true).offset + mempool_info(1-1).length
    //
};
#ifndef PREPARE_DISABLED
// Called at the end of serializing a graph, if 'const extent' mode is enabled.
// See comment in const_extent_descriptor.cc for full details.
size_t write_aligned_const_info(Graph const &gr, Serializer &sctx);
#else
inline constexpr size_t write_aligned_const_info(Graph const &gr, Serializer const &sctx)
{
    return 0;
}
#endif

} // namespace hnnx

#endif // CONST_EXTENT_DESCRIPTOR_H
