(*===-- llvm_scalar_opts.mli - LLVM Ocaml Interface ------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Scalar Transforms.

    This interface provides an ocaml API for LLVM scalar transforms, the
    classes in the [LLVMScalarOpts] library. *)

(** See the [llvm::createConstantPropogationPass] function. *)
val add_constant_propagation : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createSCCPPass] function. *)
val add_sccp : 'a Llvm_safe.PassManager.t -> unit

(** See [llvm::createDeadStoreEliminationPass] function. *)
val add_dead_store_elimination : 'a Llvm_safe.PassManager.t -> unit

(** See The [llvm::createAggressiveDCEPass] function. *)
val add_aggressive_dce : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createScalarReplAggregatesPass] function. *)
val add_scalar_repl_aggregation : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createScalarReplAggregatesPassSSA] function. *)
val add_scalar_repl_aggregation_ssa : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createScalarReplAggregatesWithThreshold] function. *)
val add_scalar_repl_aggregation_with_threshold
  : int -> 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createIndVarSimplifyPass] function. *)
val add_ind_var_simplification
  : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createInstructionCombiningPass] function. *)
val add_instruction_combination : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createLICMPass] function. *)
val add_licm : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createLoopUnswitchPass] function. *)
val add_loop_unswitch : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createLoopUnrollPass] function. *)
val add_loop_unroll : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createLoopRotatePass] function. *)
val add_loop_rotation : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createPromoteMemoryToRegisterPass] function. *)
val add_memory_to_register_promotion : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createDemoteMemoryToRegisterPass] function. *)
val add_memory_to_register_demotion : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createReassociatePass] function. *)
val add_reassociation : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createJumpThreadingPass] function. *)
val add_jump_threading : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createCFGSimplificationPass] function. *)
val add_cfg_simplification : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createTailCallEliminationPass] function. *)
val add_tail_call_elimination : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createGVNPass] function. *)
val add_gvn : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createMemCpyOptPass] function. *)
val add_memcpy_opt : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createLoopDeletionPass] function. *)
val add_loop_deletion : 'a Llvm_safe.PassManager.t -> unit

val add_loop_idiom : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createSimplifyLibCallsPass] function. *)
val add_lib_call_simplification : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createVerifierPass] function. *)
val add_verifier : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createCorrelatedValuePropagationPass] function. *)
val add_correlated_value_propagation : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createEarlyCSE] function. *)
val add_early_cse : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createLowerExpectIntrinsicPass] function. *)
val add_lower_expect_intrinsic : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createTypeBasedAliasAnalysisPass] function. *)
val add_type_based_alias_analysis : 'a Llvm_safe.PassManager.t -> unit

(** See the [llvm::createBasicAliasAnalysisPass] function. *)
val add_basic_alias_analysis : 'a Llvm_safe.PassManager.t -> unit
