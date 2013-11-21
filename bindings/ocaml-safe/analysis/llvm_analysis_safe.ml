(*===-- llvm_analysis.ml - LLVM Ocaml Interface -----------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)


external verify_module
  : Llvm_safe.Module.t -> string option
  = "llvm_verify_module"
external llvm_verify_function
  : Llvm_safe.Value.t -> bool
  = "llvm_verify_function"
external assert_valid_module
  : Llvm_safe.Module.t -> unit
  = "llvm_assert_valid_module"
external llvm_assert_valid_function
  : Llvm_safe.Value.t -> unit
  = "llvm_assert_valid_function"
external llvm_view_function_cfg
  : Llvm_safe.Value.t -> unit
  = "llvm_view_function_cfg"
external llvm_view_function_cfg_only
  : Llvm_safe.Value.t -> unit
  = "llvm_view_function_cfg_only"

let verify_function f = llvm_verify_function f#ptr
let assert_valid_function f = llvm_assert_valid_function f#ptr
let view_function_cfg f = llvm_view_function_cfg f#ptr
let view_function_cfg_only f = llvm_view_function_cfg_only f#ptr
