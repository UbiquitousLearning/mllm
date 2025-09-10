// Auto generated: 2025-09-09 15:27:37
// do not modify this file
#pragma once
namespace mllm::ir {

// traits
#ifdef RTTI_NODE_IMPL
template<typename T>
struct NodeRTTIClassOfImpl {
  static inline bool classof(Node* v) { return false; }
};
#endif  //! RTTI_Node_IMPL

#define RTTI_RK_OP_IMPL(v) return (v)->getKind() >= RK_Op && (v)->getKind() <= RK_Op_Last

#define RTTI_RK_OP_LINALGIROP_IMPL(v) return (v)->getKind() >= RK_Op_LinalgIROp && (v)->getKind() <= RK_Op_LinalgIROp_Last

#define RTTI_RK_OP_LINALGIROP_REGISTEROP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_RegisterOp && (v)->getKind() <= RK_Op_LinalgIROp_RegisterOp

#define RTTI_RK_OP_LINALGIROP_CUSTOMKERNELOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_CustomKernelOp && (v)->getKind() <= RK_Op_LinalgIROp_CustomKernelOp

#define RTTI_RK_OP_LINALGIROP_FILLOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_FillOp && (v)->getKind() <= RK_Op_LinalgIROp_FillOp

#define RTTI_RK_OP_LINALGIROP_ADDOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_AddOp && (v)->getKind() <= RK_Op_LinalgIROp_AddOp

#define RTTI_RK_OP_LINALGIROP_SUBOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_SubOp && (v)->getKind() <= RK_Op_LinalgIROp_SubOp

#define RTTI_RK_OP_LINALGIROP_STFTOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_STFTOp && (v)->getKind() <= RK_Op_LinalgIROp_STFTOp

#define RTTI_RK_OP_LINALGIROP_ISTFTOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ISTFTOp && (v)->getKind() <= RK_Op_LinalgIROp_ISTFTOp

#define RTTI_RK_OP_LINALGIROP_MULOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_MulOp && (v)->getKind() <= RK_Op_LinalgIROp_MulOp

#define RTTI_RK_OP_LINALGIROP_DIVOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_DivOp && (v)->getKind() <= RK_Op_LinalgIROp_DivOp

#define RTTI_RK_OP_LINALGIROP_ABSOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_AbsOp && (v)->getKind() <= RK_Op_LinalgIROp_AbsOp

#define RTTI_RK_OP_LINALGIROP_LOGOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_LogOp && (v)->getKind() <= RK_Op_LinalgIROp_LogOp

#define RTTI_RK_OP_LINALGIROP_MATMULOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_MatMulOp && (v)->getKind() <= RK_Op_LinalgIROp_MatMulOp

#define RTTI_RK_OP_LINALGIROP_EMBEDDINGOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_EmbeddingOp && (v)->getKind() <= RK_Op_LinalgIROp_EmbeddingOp

#define RTTI_RK_OP_LINALGIROP_LINEAROP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_LinearOp && (v)->getKind() <= RK_Op_LinalgIROp_LinearOp

#define RTTI_RK_OP_LINALGIROP_ROPEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_RoPEOp && (v)->getKind() <= RK_Op_LinalgIROp_RoPEOp

#define RTTI_RK_OP_LINALGIROP_SOFTMAXOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_SoftmaxOp && (v)->getKind() <= RK_Op_LinalgIROp_SoftmaxOp

#define RTTI_RK_OP_LINALGIROP_TRANSPOSEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_TransposeOp && (v)->getKind() <= RK_Op_LinalgIROp_TransposeOp

#define RTTI_RK_OP_LINALGIROP_RMSNORMOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_RMSNormOp && (v)->getKind() <= RK_Op_LinalgIROp_RMSNormOp

#define RTTI_RK_OP_LINALGIROP_SILUOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_SiLUOp && (v)->getKind() <= RK_Op_LinalgIROp_SiLUOp

#define RTTI_RK_OP_LINALGIROP_KVCACHEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_KVCacheOp && (v)->getKind() <= RK_Op_LinalgIROp_KVCacheOp

#define RTTI_RK_OP_LINALGIROP_CAUSALMASKOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_CausalMaskOp && (v)->getKind() <= RK_Op_LinalgIROp_CausalMaskOp

#define RTTI_RK_OP_LINALGIROP_CASTTYPEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_CastTypeOp && (v)->getKind() <= RK_Op_LinalgIROp_CastTypeOp

#define RTTI_RK_OP_LINALGIROP_X2XOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_X2XOp && (v)->getKind() <= RK_Op_LinalgIROp_X2XOp

#define RTTI_RK_OP_LINALGIROP_VIEWOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ViewOp && (v)->getKind() <= RK_Op_LinalgIROp_ViewOp

#define RTTI_RK_OP_LINALGIROP_SPLITOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_SplitOp && (v)->getKind() <= RK_Op_LinalgIROp_SplitOp

#define RTTI_RK_OP_LINALGIROP_FLASHATTENTION2OP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_FlashAttention2Op && (v)->getKind() <= RK_Op_LinalgIROp_FlashAttention2Op

#define RTTI_RK_OP_LINALGIROP_REPEATOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_RepeatOp && (v)->getKind() <= RK_Op_LinalgIROp_RepeatOp

#define RTTI_RK_OP_LINALGIROP_PERMUTEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_PermuteOp && (v)->getKind() <= RK_Op_LinalgIROp_PermuteOp

#define RTTI_RK_OP_LINALGIROP_CONV1DOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_Conv1DOp && (v)->getKind() <= RK_Op_LinalgIROp_Conv1DOp

#define RTTI_RK_OP_LINALGIROP_CONV2DOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_Conv2DOp && (v)->getKind() <= RK_Op_LinalgIROp_Conv2DOp

#define RTTI_RK_OP_LINALGIROP_CONV3DOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_Conv3DOp && (v)->getKind() <= RK_Op_LinalgIROp_Conv3DOp

#define RTTI_RK_OP_LINALGIROP_GELUOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_GELUOp && (v)->getKind() <= RK_Op_LinalgIROp_GELUOp

#define RTTI_RK_OP_LINALGIROP_LAYERNORMOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_LayerNormOp && (v)->getKind() <= RK_Op_LinalgIROp_LayerNormOp

#define RTTI_RK_OP_LINALGIROP_MULTIMODALROPEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_MultimodalRoPEOp && (v)->getKind() <= RK_Op_LinalgIROp_MultimodalRoPEOp

#define RTTI_RK_OP_LINALGIROP_VISIONROPEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_VisionRoPEOp && (v)->getKind() <= RK_Op_LinalgIROp_VisionRoPEOp

#define RTTI_RK_OP_LINALGIROP_QUICKGELUOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_QuickGELUOp && (v)->getKind() <= RK_Op_LinalgIROp_QuickGELUOp

#define RTTI_RK_OP_LINALGIROP_COPYOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_CopyOp && (v)->getKind() <= RK_Op_LinalgIROp_CopyOp

#define RTTI_RK_OP_LINALGIROP_CLONEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_CloneOp && (v)->getKind() <= RK_Op_LinalgIROp_CloneOp

#define RTTI_RK_OP_LINALGIROP_NEGOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_NegOp && (v)->getKind() <= RK_Op_LinalgIROp_NegOp

#define RTTI_RK_OP_LINALGIROP_CONCATOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ConcatOp && (v)->getKind() <= RK_Op_LinalgIROp_ConcatOp

#define RTTI_RK_OP_LINALGIROP_REDUCEMINOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ReduceMinOp && (v)->getKind() <= RK_Op_LinalgIROp_ReduceMinOp

#define RTTI_RK_OP_LINALGIROP_REDUCEMAXOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ReduceMaxOp && (v)->getKind() <= RK_Op_LinalgIROp_ReduceMaxOp

#define RTTI_RK_OP_LINALGIROP_REDUCESUMOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ReduceSumOp && (v)->getKind() <= RK_Op_LinalgIROp_ReduceSumOp

#define RTTI_RK_OP_LINALGIROP_RELUOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ReLUOp && (v)->getKind() <= RK_Op_LinalgIROp_ReLUOp

#define RTTI_RK_OP_LINALGIROP_CONTIGUOUSOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ContiguousOp && (v)->getKind() <= RK_Op_LinalgIROp_ContiguousOp

#define RTTI_RK_OP_LINALGIROP_RESHAPEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ReshapeOp && (v)->getKind() <= RK_Op_LinalgIROp_ReshapeOp

#define RTTI_RK_OP_LINALGIROP_SLICEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_SliceOp && (v)->getKind() <= RK_Op_LinalgIROp_SliceOp

#define RTTI_RK_OP_LINALGIROP_PARAMOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ParamOp && (v)->getKind() <= RK_Op_LinalgIROp_ParamOp

#define RTTI_RK_OP_LINALGIROP_INDEXOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_IndexOp && (v)->getKind() <= RK_Op_LinalgIROp_IndexOp

#define RTTI_RK_OP_LINALGIROP_TOPKOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_TopKOp && (v)->getKind() <= RK_Op_LinalgIROp_TopKOp

#define RTTI_RK_OP_LINALGIROP_MEANOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_MeanOp && (v)->getKind() <= RK_Op_LinalgIROp_MeanOp

#define RTTI_RK_OP_LINALGIROP_CLIPOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ClipOp && (v)->getKind() <= RK_Op_LinalgIROp_ClipOp

#define RTTI_RK_OP_LINALGIROP_EXPOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_ExpOp && (v)->getKind() <= RK_Op_LinalgIROp_ExpOp

#define RTTI_RK_OP_LINALGIROP_SINOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_SinOp && (v)->getKind() <= RK_Op_LinalgIROp_SinOp

#define RTTI_RK_OP_LINALGIROP_COSOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_CosOp && (v)->getKind() <= RK_Op_LinalgIROp_CosOp

#define RTTI_RK_OP_LINALGIROP_PAGEDATTNOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_LinalgIROp_PagedAttnOp && (v)->getKind() <= RK_Op_LinalgIROp_PagedAttnOp

#define RTTI_RK_OP_GRAPHIROP_IMPL(v) return (v)->getKind() >= RK_Op_GraphIROp && (v)->getKind() <= RK_Op_GraphIROp_Last

#define RTTI_RK_OP_GRAPHIROP_SUBGRAPHOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_GraphIROp_SubGraphOp && (v)->getKind() <= RK_Op_GraphIROp_SubGraphOp

#define RTTI_RK_OP_GRAPHIROP_CALLGRAPHOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_GraphIROp_CallGraphOp && (v)->getKind() <= RK_Op_GraphIROp_CallGraphOp

#define RTTI_RK_OP_TENSORIROP_IMPL(v) return (v)->getKind() >= RK_Op_TensorIROp && (v)->getKind() <= RK_Op_TensorIROp_Last

#define RTTI_RK_OP_TENSORIROP_ALLOCOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_TensorIROp_AllocOp && (v)->getKind() <= RK_Op_TensorIROp_AllocOp

#define RTTI_RK_OP_TENSORIROP_REGISTEROP_IMPL(v) \
  return (v)->getKind() >= RK_Op_TensorIROp_RegisterOp && (v)->getKind() <= RK_Op_TensorIROp_RegisterOp

#define RTTI_RK_OP_TENSORIROP_FREEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_TensorIROp_FreeOp && (v)->getKind() <= RK_Op_TensorIROp_FreeOp

#define RTTI_RK_OP_BUILTINIROP_IMPL(v) return (v)->getKind() >= RK_Op_BuiltinIROp && (v)->getKind() <= RK_Op_BuiltinIROp_Last

#define RTTI_RK_OP_BUILTINIROP_MODULEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_BuiltinIROp_ModuleOp && (v)->getKind() <= RK_Op_BuiltinIROp_ModuleOp

#define RTTI_RK_OP_CONTROLFLOWIROP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ControlFlowIROp && (v)->getKind() <= RK_Op_ControlFlowIROp_Last

#define RTTI_RK_OP_CONTROLFLOWIROP_RETURNOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ControlFlowIROp_ReturnOp && (v)->getKind() <= RK_Op_ControlFlowIROp_ReturnOp

#define RTTI_RK_OP_PROGRAMIROP_IMPL(v) return (v)->getKind() >= RK_Op_ProgramIROp && (v)->getKind() <= RK_Op_ProgramIROp_Last

#define RTTI_RK_OP_PROGRAMIROP_FRAGMENTOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_FragmentOp && (v)->getKind() <= RK_Op_ProgramIROp_FragmentOp

#define RTTI_RK_OP_PROGRAMIROP_KERNELLAUNCHOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_KernelLaunchOp && (v)->getKind() <= RK_Op_ProgramIROp_KernelLaunchOp

#define RTTI_RK_OP_PROGRAMIROP_KERNELSYMBOLOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_KernelSymbolOp && (v)->getKind() <= RK_Op_ProgramIROp_KernelSymbolOp

#define RTTI_RK_OP_PROGRAMIROP_VALUESYMBOLOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_ValueSymbolOp && (v)->getKind() <= RK_Op_ProgramIROp_ValueSymbolOp

#define RTTI_RK_OP_PROGRAMIROP_JUMPOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_JumpOp && (v)->getKind() <= RK_Op_ProgramIROp_JumpOp

#define RTTI_RK_OP_PROGRAMIROP_LABELOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_LabelOp && (v)->getKind() <= RK_Op_ProgramIROp_LabelOp

#define RTTI_RK_OP_PROGRAMIROP_EXITOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_ExitOp && (v)->getKind() <= RK_Op_ProgramIROp_ExitOp

#define RTTI_RK_OP_PROGRAMIROP_RETOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_RetOp && (v)->getKind() <= RK_Op_ProgramIROp_RetOp

#define RTTI_RK_OP_PROGRAMIROP_ENTRYPOINTOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_EntryPointOp && (v)->getKind() <= RK_Op_ProgramIROp_EntryPointOp

#define RTTI_RK_OP_PROGRAMIROP_ALLOCOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_AllocOp && (v)->getKind() <= RK_Op_ProgramIROp_AllocOp

#define RTTI_RK_OP_PROGRAMIROP_FREEOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_FreeOp && (v)->getKind() <= RK_Op_ProgramIROp_FreeOp

#define RTTI_RK_OP_PROGRAMIROP_MODECONFIGOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_ModeConfigOp && (v)->getKind() <= RK_Op_ProgramIROp_ModeConfigOp

#define RTTI_RK_OP_PROGRAMIROP_BINDOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_ProgramIROp_BindOp && (v)->getKind() <= RK_Op_ProgramIROp_BindOp

#define RTTI_RK_OP_DBGIROP_IMPL(v) return (v)->getKind() >= RK_Op_DbgIROp && (v)->getKind() <= RK_Op_DbgIROp_Last

#define RTTI_RK_OP_DBGIROP_COMMENTOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_DbgIROp_CommentOp && (v)->getKind() <= RK_Op_DbgIROp_CommentOp

#define RTTI_RK_OP_DBGIROP_HINTSOP_IMPL(v) \
  return (v)->getKind() >= RK_Op_DbgIROp_HintsOp && (v)->getKind() <= RK_Op_DbgIROp_HintsOp

#define RTTI_RK_VAL_IMPL(v) return (v)->getKind() >= RK_Val && (v)->getKind() <= RK_Val_Last

#define RTTI_RK_VAL_LINALGIRVAL_IMPL(v) return (v)->getKind() >= RK_Val_LinalgIRVal && (v)->getKind() <= RK_Val_LinalgIRVal

#define RTTI_RK_VAL_GRAPHIRVAL_IMPL(v) return (v)->getKind() >= RK_Val_GraphIRVal && (v)->getKind() <= RK_Val_GraphIRVal

#define RTTI_RK_VAL_TENSORIRVAL_IMPL(v) return (v)->getKind() >= RK_Val_TensorIRVal && (v)->getKind() <= RK_Val_TensorIRVal_Last

#define RTTI_RK_VAL_TENSORIRVAL_TENSORVAL_IMPL(v) \
  return (v)->getKind() >= RK_Val_TensorIRVal_TensorVal && (v)->getKind() <= RK_Val_TensorIRVal_TensorVal

#define RTTI_RK_VAL_BUILTINIRVAL_IMPL(v) return (v)->getKind() >= RK_Val_BuiltinIRVal && (v)->getKind() <= RK_Val_BuiltinIRVal

#define RTTI_RK_VAL_CONTROLFLOWIRVAL_IMPL(v) \
  return (v)->getKind() >= RK_Val_ControlFlowIRVal && (v)->getKind() <= RK_Val_ControlFlowIRVal

#define RTTI_RK_VAL_PROGRAMIRVAL_IMPL(v) return (v)->getKind() >= RK_Val_ProgramIRVal && (v)->getKind() <= RK_Val_ProgramIRVal

#define RTTI_RK_VAL_DBGIRVAL_IMPL(v) return (v)->getKind() >= RK_Val_DbgIRVal && (v)->getKind() <= RK_Val_DbgIRVal

#define RTTI_RK_ATTR_IMPL(v) return (v)->getKind() >= RK_Attr && (v)->getKind() <= RK_Attr_Last

#define RTTI_RK_ATTR_LINALGIRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_LinalgIRAttr && (v)->getKind() <= RK_Attr_LinalgIRAttr

#define RTTI_RK_ATTR_GRAPHIRATTR_IMPL(v) return (v)->getKind() >= RK_Attr_GraphIRAttr && (v)->getKind() <= RK_Attr_GraphIRAttr

#define RTTI_RK_ATTR_TENSORIRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_TensorIRAttr && (v)->getKind() <= RK_Attr_TensorIRAttr

#define RTTI_RK_ATTR_BUILTINIRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr && (v)->getKind() <= RK_Attr_BuiltinIRAttr_Last

#define RTTI_RK_ATTR_BUILTINIRATTR_INTATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr_IntAttr && (v)->getKind() <= RK_Attr_BuiltinIRAttr_IntAttr

#define RTTI_RK_ATTR_BUILTINIRATTR_FPATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr_FPAttr && (v)->getKind() <= RK_Attr_BuiltinIRAttr_FPAttr

#define RTTI_RK_ATTR_BUILTINIRATTR_STRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr_StrAttr && (v)->getKind() <= RK_Attr_BuiltinIRAttr_StrAttr

#define RTTI_RK_ATTR_BUILTINIRATTR_SYMBOLATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr_SymbolAttr && (v)->getKind() <= RK_Attr_BuiltinIRAttr_SymbolAttr

#define RTTI_RK_ATTR_BUILTINIRATTR_BOOLATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr_BoolAttr && (v)->getKind() <= RK_Attr_BuiltinIRAttr_BoolAttr

#define RTTI_RK_ATTR_BUILTINIRATTR_VECTORFP32ATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_BuiltinIRAttr_VectorFP32Attr && (v)->getKind() <= RK_Attr_BuiltinIRAttr_VectorFP32Attr

#define RTTI_RK_ATTR_CONTROLFLOWIRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_ControlFlowIRAttr && (v)->getKind() <= RK_Attr_ControlFlowIRAttr

#define RTTI_RK_ATTR_PROGRAMIRATTR_IMPL(v) \
  return (v)->getKind() >= RK_Attr_ProgramIRAttr && (v)->getKind() <= RK_Attr_ProgramIRAttr

#define RTTI_RK_ATTR_DBGIRATTR_IMPL(v) return (v)->getKind() >= RK_Attr_DbgIRAttr && (v)->getKind() <= RK_Attr_DbgIRAttr

}  // namespace mllm::ir
