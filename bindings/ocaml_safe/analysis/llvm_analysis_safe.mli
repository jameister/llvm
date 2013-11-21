(*===-- llvm_analysis.mli - LLVM Ocaml Interface ----------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Intermediate representation analysis.

    This interface provides an ocaml API for LLVM IR analyses, the classes in
    the Analysis library. *)

(** [verify_module m] returns [None] if the module [m] is valid, and
    [Some reason] if it is invalid. [reason] is a string containing a
    human-readable validation report. See [llvm::verifyModule]. *)
val verify_module : Llvm_safe.Module.t -> string option

(** [verify_function f] returns [None] if the function [f] is valid, and
    [Some reason] if it is invalid. [reason] is a string containing a
    human-readable validation report. See [llvm::verifyFunction]. *)
val verify_function : Llvm_safe.Function.c -> bool

(** [verify_module m] returns if the module [m] is valid, but prints a
    validation report to [stderr] and aborts the program if it is invalid. See
    [llvm::verifyModule]. *)
val assert_valid_module : Llvm_safe.Module.t -> unit

(** [verify_function f] returns if the function [f] is valid, but prints a
    validation report to [stderr] and aborts the program if it is invalid. See
    [llvm::verifyFunction]. *)
val assert_valid_function : Llvm_safe.Function.c -> unit

(** [view_function_cfg f] opens up a ghostscript window displaying the CFG of
    the current function with the code for each basic block inside.
    See [llvm::Function::viewCFG]. *)
val view_function_cfg : Llvm_safe.Function.c -> unit

(** [view_function_cfg_only f] works just like [view_function_cfg], but does
    not include the contents of basic blocks into the nodes. See
    [llvm::Function::viewCFGOnly]. *)
val view_function_cfg_only : Llvm_safe.Function.c -> unit
