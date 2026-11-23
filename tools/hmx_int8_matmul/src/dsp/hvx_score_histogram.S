.text
.p2align 4
.globl hmx_i8_hvx_histogram_tiles
.hidden hmx_i8_hvx_histogram_tiles
.type hmx_i8_hvx_histogram_tiles,@function
// r0: aligned 4x32-byte tiles, r1: tile count, r2: aligned lane histogram.
hmx_i8_hvx_histogram_tiles:
    { v0 = vxor(v0,v0) }
    .irp pair,1:0,3:2,5:4,7:6,9:8,11:10,13:12,15:14,17:16,19:18,21:20,23:22,25:24,27:26,29:28,31:30
    { v\pair = vcombine(v0,v0) }
    .endr
    { loop0(.Lscore_tiles,r1) }
    .p2align 4
.Lscore_tiles:
    { v0.tmp = vmem(r0++#1)
      vhist
    }:endloop0
    .irp vector,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    { vmem(r2++#1) = v\vector }
    .endr
    { jumpr r31 }
.size hmx_i8_hvx_histogram_tiles,.-hmx_i8_hvx_histogram_tiles
