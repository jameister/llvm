(*===-- llvm_target.ml - LLVM Ocaml Interface ------------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

module Endian = struct
  type t = Big | Little
end

module DataLayout = struct
  type t

  external create
    : string -> t
    = "llvm_targetdata_create"
  external add
    : t -> 'a Llvm_safe.PassManager.t -> unit
    = "llvm_targetdata_add"
  external as_string
    : t -> string
    = "llvm_targetdata_as_string"
  external dispose
    : t -> unit
    = "llvm_targetdata_dispose"
end

external byte_order
  : DataLayout.t -> Endian.t
  = "llvm_byte_order"
external pointer_size
  : DataLayout.t -> int
  = "llvm_pointer_size"
external intptr_type
  : DataLayout.t -> Llvm_safe.IntegerType.t
  = "LLVMIntPtrType"
external size_in_bits
  : DataLayout.t -> Llvm_safe.Type.t -> Int64.t
  = "llvm_size_in_bits"
external store_size
  : DataLayout.t -> Llvm_safe.Type.t -> Int64.t
  = "llvm_store_size"
external abi_size
  : DataLayout.t -> Llvm_safe.Type.t -> Int64.t
  = "llvm_abi_size"
external abi_align
  : DataLayout.t -> Llvm_safe.Type.t -> int
  = "llvm_abi_align"
external stack_align
  : DataLayout.t -> Llvm_safe.Type.t -> int
  = "llvm_stack_align"
external preferred_align
  : DataLayout.t -> Llvm_safe.Type.t -> int
  = "llvm_preferred_align"
external preferred_align_of_global
  : DataLayout.t -> Llvm_safe.Value.t -> int
  = "llvm_preferred_align_of_global"
external element_at_offset
  : DataLayout.t -> Llvm_safe.Type.t -> Int64.t -> int
  = "llvm_element_at_offset"
external offset_of_element
  : DataLayout.t -> Llvm_safe.Type.t -> int -> Int64.t
  = "llvm_offset_of_element"

let intptr_type td = new Llvm_safe.IntegerType.c (intptr_type td)
let size_in_bits td ty = size_in_bits td ty#ptr
let store_size td ty = store_size td ty#ptr
let abi_size td ty = abi_size td ty#ptr
let abi_align td ty = abi_align td ty#ptr
let stack_align td ty = stack_align td ty#ptr
let preferred_align td ty = preferred_align td ty#ptr
let preferred_align_of_global td gv = preferred_align_of_global td gv#ptr
let element_at_offset td ty ofs = element_at_offset td ty#ptr ofs
let offset_of_element td ty elt = offset_of_element td ty#ptr elt
