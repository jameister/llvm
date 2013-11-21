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

(** See llvm::createAddArgumentPromotionPass *)
val add_argument_promotion : module_pass -> unit

(** See llvm::createConstantMergePass function. *)
val add_constant_merge : module_pass -> unit

(** See llvm::createDeadArgEliminationPass function. *)
val add_dead_arg_elimination : module_pass -> unit

(** See llvm::createFunctionAttrsPass function. *)
val add_function_attrs : module_pass -> unit

(** See llvm::createFunctionInliningPass function. *)
val add_function_inlining : module_pass -> unit

(** See llvm::createGlobalDCEPass function. *)
val add_global_dce : module_pass -> unit

(** See llvm::createGlobalOptimizerPass function. *)
val add_global_optimizer : module_pass -> unit

(** See llvm::createIPConstantPropagationPass function. *)
val add_ipc_propagation : module_pass -> unit

(** See llvm::createPruneEHPass function. *)
val add_prune_eh : module_pass -> unit

(** See llvm::createIPSCCPPass function. *)
val add_ipsccp : module_pass -> unit

(** See llvm::createInternalizePass function. *)
val add_internalize : module_pass -> bool -> unit

(** See llvm::createStripDeadPrototypesPass function. *)
val add_strip_dead_prototypes : module_pass -> unit

(** See llvm::createStripSymbolsPass function. *)
val add_strip_symbols : module_pass -> unit
