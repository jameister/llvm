(*===-- llvm_bitwriter.mli - LLVM Ocaml Interface ---------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Bitcode writer.

    This interface provides an ocaml API for the LLVM bitcode writer, the
    classes in the Bitwriter library. *)

(** [write_bitcode_file m path] writes the bitcode for module [m] to the file at
    [path]. Returns [true] if successful, [false] otherwise. *)
val write_bitcode_file : Llvm_safe.Module.t -> string -> bool

(** [write_bitcode_to_fd ~unbuffered fd m] writes the bitcode for module
    [m] to the channel [c]. If [unbuffered] is [true], after every write the fd
    will be flushed. Returns [true] if successful, [false] otherwise. *)
val write_bitcode_to_fd
  : ?unbuffered:bool -> Llvm_safe.Module.t -> Unix.file_descr -> bool

(** [output_bitcode ~unbuffered c m] writes the bitcode for module [m]
    to the channel [c]. If [unbuffered] is [true], after every write the fd
    will be flushed. Returns [true] if successful, [false] otherwise. *)
val output_bitcode
  : ?unbuffered:bool -> out_channel -> Llvm_safe.Module.t -> bool
