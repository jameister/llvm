(*===-- llvm_target.mli - LLVM Ocaml Interface -----------------*- OCaml -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Target Information.

    This interface provides an ocaml API for LLVM target information,
    the classes in the Target library. *)

module Endian : sig
  type t = Big | Little
end

module DataLayout : sig
  type t

  (** [DataLayout.create rep] parses the target data string representation [rep].
      See the constructor llvm::DataLayout::DataLayout. *)
  val create : string -> t

  (** [add_target_data td pm] adds the target data [td] to the pass manager [pm].
      Does not take ownership of the target data.
      See the method llvm::PassManagerBase::add. *)
  val add : t -> 'a Llvm_safe.PassManager.t -> unit

  (** [as_string td] is the string representation of the target data [td].
      See the constructor llvm::DataLayout::DataLayout. *)
  val as_string : t -> string

  (** Deallocates a DataLayout.
      See the destructor llvm::DataLayout::~DataLayout. *)
  val dispose : t -> unit
end

(** Returns the byte order of a target, either LLVMBigEndian or
    LLVMLittleEndian.
    See the method llvm::DataLayout::isLittleEndian. *)
val byte_order : DataLayout.t -> Endian.t

(** Returns the pointer size in bytes for a target.
    See the method llvm::DataLayout::getPointerSize. *)
val pointer_size : DataLayout.t -> int

(** Returns the integer type that is the same size as a pointer on a target.
    See the method llvm::DataLayout::getIntPtrType. *)
val intptr_type : DataLayout.t -> Llvm_safe.IntegerType.c

(** Computes the size of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeSizeInBits. *)
val size_in_bits : DataLayout.t -> Llvm_safe.Type.c -> Int64.t

(** Computes the storage size of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeStoreSize. *)
val store_size : DataLayout.t -> Llvm_safe.Type.c -> Int64.t

(** Computes the ABI size of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeAllocSize. *)
val abi_size : DataLayout.t -> Llvm_safe.Type.c -> Int64.t

(** Computes the ABI alignment of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeABISize. *)
val abi_align : DataLayout.t -> Llvm_safe.Type.c -> int

(** Computes the call frame alignment of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeABISize. *)
val stack_align : DataLayout.t -> Llvm_safe.Type.c -> int

(** Computes the preferred alignment of a type in bytes for a target.
    See the method llvm::DataLayout::getTypeABISize. *)
val preferred_align : DataLayout.t -> Llvm_safe.Type.c -> int

(** Computes the preferred alignment of a global variable in bytes for a target.
    See the method llvm::DataLayout::getPreferredAlignment. *)
val preferred_align_of_global : DataLayout.t -> Llvm_safe.GlobalVariable.c -> int

(** Computes the structure element that contains the byte offset for a target.
    See the method llvm::StructLayout::getElementContainingOffset. *)
val element_at_offset : DataLayout.t -> Llvm_safe.Type.c -> Int64.t -> int

(** Computes the byte offset of the indexed struct element for a target.
    See the method llvm::StructLayout::getElementContainingOffset. *)
val offset_of_element : DataLayout.t -> Llvm_safe.Type.c -> int -> Int64.t
