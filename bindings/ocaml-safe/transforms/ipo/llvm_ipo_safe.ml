(*===-- llvm_ipo.mli - LLVM Ocaml Interface ------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** IPO Transforms.

    This interface provides an ocaml API for LLVM interprocedural optimizations, the
    classes in the [LLVMIPO] library. *)

type module_pass = Llvm_safe.PassManager.module_pass Llvm_safe.PassManager.t

external add_argument_promotion
  : module_pass -> unit
  = "llvm_add_argument_promotion"

external add_constant_merge
  : module_pass -> unit
  = "llvm_add_constant_merge"

external add_dead_arg_elimination
  : module_pass -> unit
  = "llvm_add_dead_arg_elimination"

external add_function_attrs
  : module_pass -> unit
  = "llvm_add_function_attrs"

external add_function_inlining
  : module_pass -> unit
  = "llvm_add_function_inlining"

external add_global_dce
  : module_pass -> unit
  = "llvm_add_global_dce"

external add_global_optimizer
  : module_pass -> unit
  = "llvm_add_global_optimizer"

external add_ipc_propagation
  : module_pass -> unit
  = "llvm_add_ipc_propagation"

external add_prune_eh
  : module_pass -> unit
  = "llvm_add_prune_eh"

external add_ipsccp
  : module_pass -> unit
  = "llvm_add_ipsccp"

external add_internalize
  : module_pass -> bool -> unit
  = "llvm_add_internalize"

external add_strip_dead_prototypes
  : module_pass -> unit
  = "llvm_add_strip_dead_prototypes"

external add_strip_symbols
  : module_pass -> unit
  = "llvm_add_strip_symbols"
