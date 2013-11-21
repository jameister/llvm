(*===-- llvm_executionengine.ml - LLVM Ocaml Interface ----------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)


exception Error of string

external register_exns : exn -> unit = "llvm_register_ee_exns"
let _ = register_exns (Error "")

let maybe f = function
  | None -> None
  | Some x -> Some (f x)

module GenericValue = struct
  type t
  
  external of_float
    : Llvm_safe.Type.t -> float -> t
    = "llvm_genericvalue_of_float"
  external of_pointer
    : 'a -> t
    = "llvm_genericvalue_of_pointer"
  external of_int32
    : Llvm_safe.Type.t -> int32 -> t
    = "llvm_genericvalue_of_int32"
  external of_int
    : Llvm_safe.Type.t -> int -> t
    = "llvm_genericvalue_of_int"
  external of_nativeint
    : Llvm_safe.Type.t -> nativeint -> t
    = "llvm_genericvalue_of_nativeint"
  external of_int64
    : Llvm_safe.Type.t -> int64 -> t
    = "llvm_genericvalue_of_int64"
  let of_float fty n = of_float fty#ptr n
  let of_int32 ity n = of_int32 ity#ptr n
  let of_int ity n = of_int ity#ptr n
  let of_nativeint ity n = of_nativeint ity#ptr n
  let of_int64 ity n = of_int64 ity#ptr n
  
  external as_float
    : Llvm_safe.Type.t -> t -> float
    = "llvm_genericvalue_as_float"
  external as_pointer
    : t -> 'a
    = "llvm_genericvalue_as_pointer"
  external as_int32
    : t -> int32
    = "llvm_genericvalue_as_int32"
  external as_int
    : t -> int
    = "llvm_genericvalue_as_int"
  external as_nativeint
    : t -> nativeint
    = "llvm_genericvalue_as_nativeint"
  external as_int64
    : t -> int64
    = "llvm_genericvalue_as_int64"
  let as_float fty gv = as_float fty#ptr gv
end

module ExecutionEngine = struct
  type t
  
  external create
    : Llvm_safe.Module.t -> t
    = "llvm_ee_create"
  external create_interpreter
    : Llvm_safe.Module.t -> t
    = "llvm_ee_create_interpreter"
  external create_jit
    : Llvm_safe.Module.t -> int -> t
    = "llvm_ee_create_jit"
  external dispose
    : t -> unit
    = "llvm_ee_dispose"
  external add_module
    : Llvm_safe.Module.t -> t -> unit
    = "llvm_ee_add_module"
  external remove_module
    : Llvm_safe.Module.t -> t -> Llvm_safe.Module.t
    = "llvm_ee_remove_module"
  external find_function
    : string -> t -> Llvm_safe.Function.t option
    = "llvm_ee_find_function"
  external run_function
    : Llvm_safe.Value.t -> GenericValue.t array -> t -> GenericValue.t
    = "llvm_ee_run_function"
  external run_static_ctors
    : t -> unit
    = "llvm_ee_run_static_ctors"
  external run_static_dtors
    : t -> unit
    = "llvm_ee_run_static_dtors"
  external run_function_as_main
    : Llvm_safe.Value.t -> string array -> (string * string) array -> t -> int
    = "llvm_ee_run_function_as_main"
  external free_machine_code: Llvm_safe.Value.t -> t -> unit
    = "llvm_ee_free_machine_code"
  let find_function name ee =
    maybe (new Llvm_safe.Function.c) (find_function name ee)
  let run_function f = run_function f#ptr
  let run_function_as_main f = run_function_as_main f#ptr
  let free_machine_code f = free_machine_code f#ptr

  external target_data
    : t -> Llvm_target_safe.DataLayout.t
    = "LLVMGetExecutionEngineTargetData"
  
  (* The following are not bound. Patches are welcome.
  
  get_target_data: t -> lltargetdata
  add_global_mapping: llvalue -> llgenericvalue -> t -> unit
  clear_all_global_mappings: t -> unit
  update_global_mapping: llvalue -> llgenericvalue -> t -> unit
  get_pointer_to_global_if_available: llvalue -> t -> llgenericvalue
  get_pointer_to_global: llvalue -> t -> llgenericvalue
  get_pointer_to_function: llvalue -> t -> llgenericvalue
  get_pointer_to_function_or_stub: llvalue -> t -> llgenericvalue
  get_global_value_at_address: llgenericvalue -> t -> llvalue option
  store_value_to_memory: llgenericvalue -> llgenericvalue -> lltype -> unit
  initialize_memory: llvalue -> llgenericvalue -> t -> unit
  recompile_and_relink_function: llvalue -> t -> llgenericvalue
  get_or_emit_global_variable: llvalue -> t -> llgenericvalue
  disable_lazy_compilation: t -> unit
  lazy_compilation_enabled: t -> bool
  install_lazy_function_creator: (string -> llgenericvalue) -> t -> unit
  
   *)
end

external initialize_native_target
  : unit -> bool
  = "llvm_initialize_native_target"
