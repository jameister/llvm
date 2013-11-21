(*===-- llvm_scalar_opts.ml - LLVM Ocaml Interface -------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

external add_constant_propagation
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_constant_propagation"

external add_sccp
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_sccp"

external add_dead_store_elimination
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_dead_store_elimination"

external add_aggressive_dce
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_aggressive_dce"

external add_scalar_repl_aggregation
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_scalar_repl_aggregation"

external add_scalar_repl_aggregation_ssa
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_scalar_repl_aggregation_ssa"

external add_scalar_repl_aggregation_with_threshold
  : int -> 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_scalar_repl_aggregation_with_threshold"

external add_ind_var_simplification
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_ind_var_simplification"

external add_instruction_combination
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_instruction_combination"

external add_licm
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_licm"

external add_loop_unswitch
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_loop_unswitch"

external add_loop_unroll
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_loop_unroll"

external add_loop_rotation
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_loop_rotation"

external add_memory_to_register_promotion
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_memory_to_register_promotion"

external add_memory_to_register_demotion
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_memory_to_register_demotion"

external add_reassociation
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_reassociation"

external add_jump_threading
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_jump_threading"

external add_cfg_simplification
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_cfg_simplification"

external add_tail_call_elimination
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_tail_call_elimination" 

external add_gvn
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_gvn"

external add_memcpy_opt
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_memcpy_opt"

external add_loop_deletion
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_loop_deletion"

external add_loop_idiom
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_loop_idiom"

external add_lib_call_simplification
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_lib_call_simplification"

external add_verifier
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_verifier"

external add_correlated_value_propagation
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_correlated_value_propagation"

external add_early_cse
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_early_cse"

external add_lower_expect_intrinsic
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_lower_expect_intrinsic"

external add_type_based_alias_analysis
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_type_based_alias_analysis"

external add_basic_alias_analysis
  : 'a Llvm_safe.PassManager.t -> unit
  = "llvm_add_basic_alias_analysis"
