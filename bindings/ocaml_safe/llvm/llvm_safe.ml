(*===-- llvm/llvm.ml - LLVM Ocaml Interface --------------------------------===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

type ('a, 'b) pos = ('a, 'b) Llvm.llpos =
  | At_end of 'a
  | Before of 'b
let wrapl f = function
  | At_end coll -> At_end (f coll)
  | Before _ as pos -> pos
let wrapr f = function
  | At_end _ as pos -> pos
  | Before item -> Before (f item)

type ('a, 'b) rev_pos = ('a, 'b) Llvm.llrev_pos =
  | At_start of 'a
  | After of 'b
let rev_wrapl f = function
  | At_start coll -> At_start (f coll)
  | After _ as pos -> pos
let rev_wrapr f = function
  | At_start _ as pos -> pos
  | After item -> After (f item)

module type POSITION = sig
  type collection
  type item
  val first : collection -> (collection, item) pos
  val last : collection -> (collection, item) rev_pos
  val succ : item -> (collection, item) pos
  val pred : item -> (collection, item) rev_pos
end

module Iterable(P : POSITION) = struct
  type collection = P.collection
  type item = P.item
  let iter coll f = 
    let rec step = function
      | Llvm.At_end _ -> ()
      | Llvm.Before itm -> f itm; step (P.succ itm)
    in step (P.first coll)
  let fold_left init coll f =
    let rec step acc = function
      | Llvm.At_end _ -> acc
      | Llvm.Before itm -> step (f acc itm) (P.succ itm)
    in step init (P.first coll)
  let rev_iter coll f =
    let rec step = function
      | Llvm.At_start _ -> ()
      | Llvm.After itm -> f itm; step (P.pred itm)
    in step (P.last coll)
  let fold_right coll init f = 
    let rec step elt acc = match elt with
      | Llvm.At_start _ -> acc
      | Llvm.After itm -> step (P.pred itm) (f itm acc)
    in step (P.last coll) init
end

type llcontext = Llvm.llcontext
type lltype = Llvm.lltype
type llmodule = Llvm.llmodule
type llvalue = Llvm.llvalue
type lluse = Llvm.lluse
type llbasicblock = Llvm.llbasicblock
type llbuilder = Llvm.llbuilder
type llmemorybuffer = Llvm.llmemorybuffer

exception IoError of string
external register_exns : exn -> unit = "llvm_register_core_exns"
let _ = register_exns (IoError "")

exception Cast_failure of string

let maybe f = function
  | None -> None
  | Some x -> Some (f x)

module Opcode = struct
  type t = Llvm.Opcode.t =
    | Invalid
    | Ret | Br | Switch | IndirectBr | Invoke | Invalid2 | Unreachable
    | Add | FAdd | Sub | FSub | Mul | FMul
    | UDiv | SDiv | FDiv | URem | SRem | FRem
    | Shl | LShr | AShr | And | Or | Xor
    | Alloca | Load | Store | GetElementPtr
    | Trunc | ZExt | SExt | FPToUI | FPToSI | UIToFP | SIToFP
    | FPTrunc | FPExt | PtrToInt | IntToPtr | BitCast
    | ICmp | FCmp | PHI | Call | Select | UserOp1 | UserOp2 | VAArg
    | ExtractElement | InsertElement | ShuffleVector
    | ExtractValue | InsertValue | Fence | AtomicCmpXchg | AtomicRMW
    | Resume | LandingPad | Unwind
end

module Operator = struct
  type unary =
    | Neg | NSWNeg | NUWNeg | FNeg | Not
  type binary =
    | Add | NSWAdd | NUWAdd | FAdd | Sub | NSWSub | NUWSub | FSub
    | Mul | NSWMul | NUWMul | FMul | UDiv | SDiv | ExactSDiv | FDiv
    | URem | SRem | FRem | Shl | LShr | AShr | And | Or | Xor
  type cast =
    | Trunc | ZExt | SExt | FPToUI | FPToSI | UIToFP | SIToFP
    | FPTrunc | FPExt | PtrToInt | IntToPtr | BitCast
    | ZExtOrBitCast | SExtOrBitCast | TruncOrBitCast
    | PointerCast | IntCast | FPCast
end

module Predicate = struct
  type icmp = Llvm.Icmp.t =
    | Eq | Ne | Ugt | Uge | Ult | Ule | Sgt | Sge | Slt | Sle
  type fcmp = Llvm.Fcmp.t =
    | False | Oeq | Ogt | Oge | Olt | Ole | One | Ord
    | Uno | Ueq | Ugt | Uge | Ult | Ule | Une | True
end

module Attribute = struct
  type t = Llvm.Attribute.t =
    | Zext | Sext | Noreturn | Inreg | Structret | Nounwind | Noalias | Byval
    | Nest | Readnone | Readonly | Noinline | Alwaysinline | Optsize
    | Ssp | Sspreq
    | Alignment of int
    | Nocapture | Noredzone | Noimplicitfloat | Naked | Inlinehint
    | Stackalignment of int
    | ReturnsTwice | UWTable | NonLazyBind
end

module Context = struct
  type t = llcontext
  let create = Llvm.create_context
  let dispose = Llvm.dispose_context
  let global = Llvm.global_context
  let mdkind_id = Llvm.mdkind_id
end

module Type = struct
  type kind = Llvm.TypeKind.t =
    | Void | Half | Float | Double | X86fp80 | Fp128 | Ppc_fp128 | Label
    | Integer | Function | Struct | Array | Pointer | Vector | Metadata
  type t = lltype
  class c p = object
    method ptr = p
    method classify = Llvm.classify_type p
    method is_sized = Llvm.type_is_sized p
    method context = Llvm.type_context p
    method to_string = Llvm.string_of_lltype p
  end
  let void ctx = new c (Llvm.void_type ctx)
  let float ctx = new c (Llvm.float_type ctx)
  let double ctx = new c (Llvm.double_type ctx)
  let x86fp80 ctx = new c (Llvm.x86fp80_type ctx)
  let fp128 ctx = new c (Llvm.fp128_type ctx)
  let ppc_fp128 ctx = new c (Llvm.ppc_fp128_type ctx)
  let label ctx = new c (Llvm.label_type ctx)
end

module IntegerType = struct
  type t = lltype
  class c p = object
    inherit Type.c p
    method bitwidth = Llvm.integer_bitwidth p
  end
  let i1 ctx = new c (Llvm.i1_type ctx)
  let i8 ctx = new c (Llvm.i8_type ctx)
  let i16 ctx = new c (Llvm.i16_type ctx)
  let i32 ctx = new c (Llvm.i32_type ctx)
  let i64 ctx = new c (Llvm.i64_type ctx)
  let make ctx ~bits = new c (Llvm.integer_type ctx bits)
  let test ty = ty#classify = Llvm.TypeKind.Integer
  let from ty =
    if test ty then new c ty#ptr
    else raise (Cast_failure "Llvm_safe.IntegerType")
end

module FunctionType = struct
  type t = lltype
  class c p = object
    inherit Type.c p
    method return_type = new Type.c (Llvm.return_type p)
    method param_types = Array.map (new Type.c) (Llvm.param_types p)
    method is_var_arg = Llvm.is_var_arg p
  end
  let make ?(vararg = false) ~ret ~params =
    let paramps = Array.map (fun x -> x#ptr) params in
      if not vararg then new c (Llvm.function_type ret#ptr paramps)
      else new c (Llvm.var_arg_function_type ret#ptr paramps)
  let test ty = ty#classify = Llvm.TypeKind.Function
  let from ty =
    if test ty then new c ty#ptr
    else raise (Cast_failure "Llvm_safe.FunctionType")
end

module StructType = struct
  type t = lltype
  class c p = object
    inherit Type.c p
    method name = Llvm.struct_name p
    method set_body
        : ?packed:bool -> elts:Type.c array -> unit
        = fun ?(packed = false) ~elts ->
      let eltps = Array.map (fun x -> x#ptr) elts in
      Llvm.struct_set_body p eltps packed
    method element_types = Array.map (new Type.c) (Llvm.struct_element_types p)
    method is_packed = Llvm.is_packed p
    method is_opaque = Llvm.is_opaque p
  end
  let make ?(packed = false) ctx ~elts =
    let eltps = Array.map (fun x -> x#ptr) elts in
      if not packed then new c (Llvm.struct_type ctx eltps)
      else new c (Llvm.packed_struct_type ctx eltps)
  let named ctx name = new c (Llvm.named_struct_type ctx name)
  let test ty = ty#classify = Llvm.TypeKind.Struct
  let from ty =
    if test ty then new c ty#ptr
    else raise (Cast_failure "Llvm_safe.StructType")
end

module SequentialType = struct
  type t = lltype
  class c p = object
    inherit Type.c p
    method element_type = Llvm.element_type p
  end
end

module ArrayType = struct
  type t = lltype
  class c p = object
    inherit SequentialType.c p
    method length = Llvm.array_length p
  end
  let make ty ~len = new c (Llvm.array_type ty#ptr len)
  let test ty = ty#classify = Llvm.TypeKind.Array
  let from ty =
    if test ty then new c ty#ptr
    else raise (Cast_failure "Llvm_safe.ArrayType")
end

module PointerType = struct
  type t = lltype
  class c p = object
    inherit SequentialType.c p
    method address_space = Llvm.address_space p
  end
  let make ?(addrspace = 0) ty =
    if addrspace = 0 then new c (Llvm.pointer_type ty#ptr)
    else new c (Llvm.qualified_pointer_type ty#ptr addrspace)
  let test ty = ty#classify = Llvm.TypeKind.Pointer
  let from ty =
    if test ty then new c ty#ptr
    else raise (Cast_failure "Llvm_safe.PointerType")
end

module VectorType = struct
  type t = lltype
  class c p = object
    inherit SequentialType.c p
    method size = Llvm.vector_size p
  end
  let make ty ~size = new c (Llvm.vector_type ty#ptr size)
  let test ty = ty#classify = Llvm.TypeKind.Vector
  let from ty =
    if test ty then new c ty#ptr
    else raise (Cast_failure "Llvm_safe.VectorType")
end

module Value = struct
  type t = llvalue
  type kind = Llvm.ValueKind.t =
    | NullValue | Argument | BasicBlock | InlineAsm
    | MDNode | MDString | BlockAddress
    | ConstantAggregateZero | ConstantArray | ConstantExpr
    | ConstantFP | ConstantInt | ConstantPointerNull
    | ConstantStruct | ConstantVector
    | Function | GlobalAlias | GlobalVariable
    | UndefValue
    | Instruction of Opcode.t
  class c p = object
    method ptr = p
    method type_of = new Type.c (Llvm.type_of p)
    method classify = Llvm.classify_value p
    method name = Llvm.value_name p
    method set_name s = Llvm.set_value_name s p
    method dump = Llvm.dump_value p
    method replace_all_uses_with (v : c) = Llvm.replace_all_uses_with p v#ptr
  end
end

module User = struct
  type t = llvalue
  class c p = object
    inherit Value.c p
    method operand i = new Value.c (Llvm.operand p i)
    method set_operand i (v : Value.c) = Llvm.set_operand p i v#ptr
    method num_operands = Llvm.num_operands p
  end
end

module Use = struct
  type t = lluse
  let user u = new User.c (Llvm.user u)
  let used u = new Value.c (Llvm.used_value u)
  let first v = Llvm.use_begin v#ptr
  let succ u = Llvm.use_succ u
  let iter f v =
    let rec aux = function
      | None -> ()
      | Some u ->
          f u;
          aux (succ u)
    in
    aux (first v)
  let fold_left f init v =
    let rec aux init u =
      match u with
      | None -> init
      | Some u -> aux (f init u) (succ u)
    in
    aux init (first v)
  let fold_right f v init =
    let rec aux u init =
      match u with
      | None -> init
      | Some u -> f u (aux (succ u) init)
    in
    aux (first v) init
end

module Constant = struct
  type t = llvalue
  class c p = object
    inherit User.c p
    method is_null = Llvm.is_null p
    method is_undef = Llvm.is_undef p
  end
  let null ty = new c (Llvm.const_null ty#ptr)
  let undef ty = new c (Llvm.undef ty#ptr)
  let test v = Llvm.is_constant v#ptr
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.Constant")
end

module ConstantInt = struct
  type t = llvalue
  class c p = object
    inherit Constant.c p
    method int64_value = Llvm.int64_of_const p
  end
  let all_ones ty = new c (Llvm.const_all_ones ty#ptr)
  let of_int ty i = new c (Llvm.const_int ty#ptr i)
  let of_int64 ?(signext = false) ty i64 =
    new c (Llvm.const_of_int64 ty#ptr i64 signext)
  let of_string ty s ~radix = new c (Llvm.const_int_of_string ty#ptr s radix)
  let test (v : Value.c) = v#classify = Llvm.ValueKind.ConstantInt
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.ConstantInt")
end

module ConstantFP = struct
  type t = llvalue
  class c p = object inherit Constant.c p end
  let of_float ty f = new c (Llvm.const_float ty#ptr f)
  let of_string ty s = new c (Llvm.const_float_of_string ty#ptr s)
  let test (v : Value.c) = v#classify = Llvm.ValueKind.ConstantFP
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.ConstantFP")
end

module ConstantStruct = struct
  type t = llvalue
  class c p = object inherit Constant.c p end
  let make ?(packed = false) ctx vals =
    let valps = Array.map (fun x -> x#ptr) vals in
      if not packed then new c (Llvm.const_struct ctx valps)
      else new c (Llvm.const_packed_struct ctx valps)
  let named namedty elts =
    let eltps = Array.map (fun x -> x#ptr) elts in
    new c (Llvm.const_named_struct namedty#ptr eltps)
  let test v = v#classify = Llvm.ValueKind.ConstantStruct
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.ConstantStruct")
end

module ConstantArray = struct
  type t = llvalue
  class c p = object inherit Constant.c p end
  let make ty vals =
    let valps = Array.map (fun x -> x#ptr) vals in
      new c (Llvm.const_array ty#ptr valps)
  let of_string ?(nullterm = false) ctx s =
    if not nullterm then new c (Llvm.const_string ctx s)
    else new c (Llvm.const_stringz ctx s)
  let test v = v#classify = Llvm.ValueKind.ConstantArray
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.ConstantArray")
end

module ConstantPointerNull = struct
  type t = llvalue
  class c p = object inherit Constant.c p end
  let make ty = new c (Llvm.const_pointer_null ty#ptr)
  let test v = v#classify = Llvm.ValueKind.ConstantPointerNull
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.ConstantPointerNull")
end

module ConstantVector = struct
  type t = llvalue
  class c p = object inherit Constant.c p end
  let all_ones ty = new c (Llvm.const_all_ones ty#ptr)
  let make vals = new c (Llvm.const_vector (Array.map (fun x -> x#ptr) vals))
  let test v = v#classify = Llvm.ValueKind.ConstantVector
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.ConstantVector")
end

module ConstantExpr = struct
  type t = llvalue
  class c p = object
    inherit Constant.c p
    method opcode = Llvm.constexpr_opcode p
  end
  let size_of ty = new c (Llvm.size_of ty#ptr)
  let align_of ty = new c (Llvm.align_of ty#ptr)
  let unary op arg = new c (let open Operator in match op with
    | Neg -> Llvm.const_neg arg#ptr
    | NSWNeg -> Llvm.const_nsw_neg arg#ptr
    | NUWNeg -> Llvm.const_nuw_neg arg#ptr
    | FNeg -> Llvm.const_fneg arg#ptr
    | Not -> Llvm.const_not arg#ptr)
  let binary op ~lhs ~rhs = new c (let open Operator in match op with
    | Add -> Llvm.const_add lhs#ptr rhs#ptr
    | NSWAdd -> Llvm.const_nsw_add lhs#ptr rhs#ptr
    | NUWAdd -> Llvm.const_nuw_add lhs#ptr rhs#ptr
    | FAdd -> Llvm.const_fadd lhs#ptr rhs#ptr
    | Sub -> Llvm.const_sub lhs#ptr rhs#ptr
    | NSWSub -> Llvm.const_nsw_sub lhs#ptr rhs#ptr
    | NUWSub -> Llvm.const_nuw_sub lhs#ptr rhs#ptr
    | FSub -> Llvm.const_fsub lhs#ptr rhs#ptr
    | Mul -> Llvm.const_mul lhs#ptr rhs#ptr
    | NSWMul -> Llvm.const_nsw_mul lhs#ptr rhs#ptr
    | NUWMul -> Llvm.const_nuw_mul lhs#ptr rhs#ptr
    | FMul -> Llvm.const_fmul lhs#ptr rhs#ptr
    | UDiv -> Llvm.const_udiv lhs#ptr rhs#ptr
    | SDiv -> Llvm.const_sdiv lhs#ptr rhs#ptr
    | FDiv -> Llvm.const_fdiv lhs#ptr rhs#ptr
    | ExactSDiv -> Llvm.const_exact_sdiv lhs#ptr rhs#ptr
    | URem -> Llvm.const_urem lhs#ptr rhs#ptr
    | SRem -> Llvm.const_srem lhs#ptr rhs#ptr
    | FRem -> Llvm.const_frem lhs#ptr rhs#ptr
    | Shl -> Llvm.const_shl lhs#ptr rhs#ptr
    | LShr -> Llvm.const_lshr lhs#ptr rhs#ptr
    | AShr -> Llvm.const_ashr lhs#ptr rhs#ptr
    | And -> Llvm.const_and lhs#ptr rhs#ptr
    | Or -> Llvm.const_or lhs#ptr rhs#ptr
    | Xor -> Llvm.const_xor lhs#ptr rhs#ptr)
  let icmp pred ~lhs ~rhs = new c (Llvm.const_icmp pred lhs#ptr rhs#ptr)
  let fcmp pred ~lhs ~rhs = new c (Llvm.const_fcmp pred lhs#ptr rhs#ptr)
  let gep ?(inbounds = false) arg idxs =
    let idxps = Array.map (fun x -> x#ptr) idxs in
      if not inbounds then new c (Llvm.const_gep arg#ptr idxps)
      else new c (Llvm.const_in_bounds_gep arg#ptr idxps)
  let cast op arg ty = new c (let open Operator in match op with
    | Trunc -> Llvm.const_trunc arg#ptr ty#ptr
    | ZExt -> Llvm.const_zext arg#ptr ty#ptr
    | SExt -> Llvm.const_sext arg#ptr ty#ptr
    | FPToUI -> Llvm.const_fptoui arg#ptr ty#ptr
    | FPToSI -> Llvm.const_fptosi arg#ptr ty#ptr
    | UIToFP -> Llvm.const_uitofp arg#ptr ty#ptr
    | SIToFP -> Llvm.const_sitofp arg#ptr ty#ptr
    | FPTrunc -> Llvm.const_fptrunc arg#ptr ty#ptr
    | FPExt -> Llvm.const_fpext arg#ptr ty#ptr
    | PtrToInt -> Llvm.const_ptrtoint arg#ptr ty#ptr
    | IntToPtr -> Llvm.const_inttoptr arg#ptr ty#ptr
    | BitCast -> Llvm.const_bitcast arg#ptr ty#ptr
    | ZExtOrBitCast -> Llvm.const_zext_or_bitcast arg#ptr ty#ptr
    | SExtOrBitCast -> Llvm.const_sext_or_bitcast arg#ptr ty#ptr
    | TruncOrBitCast -> Llvm.const_trunc_or_bitcast arg#ptr ty#ptr
    | PointerCast -> Llvm.const_pointercast arg#ptr ty#ptr
    | IntCast -> Llvm.const_intcast arg#ptr ty#ptr
    | FPCast -> Llvm.const_fpcast arg#ptr ty#ptr)
  let select ~cond ~t ~f = new c (Llvm.const_select cond#ptr t#ptr f#ptr)
  let extract_element ~vec ~idx =
    new c (Llvm.const_extractelement vec#ptr idx#ptr)
  let insert_element ~vec ~elt ~idx =
    new c (Llvm.const_insertelement vec#ptr elt#ptr idx#ptr)
  let shuffle_vector ~v1 ~v2 ~mask =
    new c (Llvm.const_shufflevector v1#ptr v2#ptr mask#ptr)
  let extract_value ~agg ~idxs =
    new c (Llvm.const_extractvalue agg#ptr idxs)
  let insert_value ~agg ~elt ~idxs =
    new c (Llvm.const_insertvalue agg#ptr elt#ptr idxs)
  let test v = v#classify = Llvm.ValueKind.ConstantExpr
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.ConstantExpr")
end

module InlineAsm = struct
  type t = llvalue
  class c p = object inherit Value.c p end
  let make ty ~asm ~constraints ~effects ~align_stack =
    new c (Llvm.const_inline_asm ty#ptr asm constraints effects align_stack)
  let test v = v#classify = Llvm.ValueKind.InlineAsm
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.InlineAsm")
end

module MDNode = struct
  type t = llvalue
  class c p = object inherit Value.c p end
  let make ctx vals = new c (Llvm.mdnode ctx (Array.map (fun x -> x#ptr) vals))
  let named_metadata m name = Array.map (new c) (Llvm.get_named_metadata m name)
end

module MDString = struct
  type t = llvalue
  class c p = object
    inherit Value.c p
    method get = Llvm.get_mdstring p
  end
  let make ctx s = new c (Llvm.mdstring ctx s)
end

module Module = struct
  type t = llmodule
  let create = Llvm.create_module
  let dispose = Llvm.dispose_module
  let target_triple = Llvm.target_triple
  let set_target_triple m s = Llvm.set_target_triple s m
  let data_layout = Llvm.data_layout
  let set_data_layout m s = Llvm.set_data_layout s m
  let dump = Llvm.dump_module
  let set_inline_asm = Llvm.set_module_inline_asm
  let context = Llvm.module_context
  let type_by_name m s = maybe (new Type.c) (Llvm.type_by_name m s)
  (* Missing?
  let define_type_name m s ty = define_type_name m s ty#ptr
  *)
end

module GlobalValue = struct
  type t = llvalue
  type linkage = Llvm.Linkage.t =
    | External | Available_externally | Link_once | Link_once_odr
    | Weak | Weak_odr | Appending | Internal | Private
    | Dllimport | Dllexport | External_weak
    | Ghost | Common | Linker_private
  type visibility = Llvm.Visibility.t =
    | Default | Hidden | Protected
  class c p = object
    inherit Constant.c p
    method parent = Llvm.global_parent p
    method is_declaration = Llvm.is_declaration p
    method linkage = Llvm.linkage p
    method set_linkage l = Llvm.set_linkage l p
    method section = Llvm.section p
    method set_section s = Llvm.set_section s p
    method visibility = Llvm.visibility p
    method set_visibility v = Llvm.set_visibility v p
    method alignment = Llvm.alignment p
    method set_alignment i = Llvm.set_alignment i p
    (* TODO is_constant, set_constant *)
  end
  let test v =
    let open Llvm.ValueKind in
    match v#classify with
    | GlobalVariable | Function | GlobalAlias -> true
    | _ -> false
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.GlobalValue")
end

module GlobalVariable = struct
  type t = llvalue
  class c p = object
    inherit GlobalValue.c p
    method delete = Llvm.delete_global p
    method is_constant = Llvm.is_global_constant p
    method set_constant b = Llvm.set_global_constant b p
    method get_initializer = new Constant.c (Llvm.global_initializer p)
    method set_initializer (v : Constant.c) = Llvm.set_initializer p v#ptr
    method remove_initializer = Llvm.remove_initializer p
    method is_thread_local = Llvm.is_thread_local p
    method set_thread_local b = Llvm.set_thread_local b p
  end
  let declare ?(addrspace = 0) m ty ~name =
    if addrspace = 0 then new c (Llvm.declare_global ty#ptr name m)
    else new c (Llvm.declare_qualified_global ty#ptr name addrspace m)
  let define ?(addrspace = 0) m ~name ~init =
    if addrspace = 0 then new c (Llvm.define_global name init#ptr m)
    else new c (Llvm.define_qualified_global name init#ptr addrspace m)
  let lookup m ~name = maybe (new c) (Llvm.lookup_global name m)
  module Pos = struct
    type collection = Module.t
    type item = c
    let first m = wrapr (new c) (Llvm.global_begin m)
    let last m = rev_wrapr (new c) (Llvm.global_end m)
    let succ gv = wrapr (new c) (Llvm.global_succ gv#ptr)
    let pred gv = rev_wrapr (new c) (Llvm.global_pred gv#ptr)
  end
  include Iterable(Pos)
  let test v = v#classify = Llvm.ValueKind.GlobalVariable
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.GlobalVariable")
end

module Function = struct
  type t = llvalue
  module CallConv = struct
    let c = 0
    let fast = 8
    let cold = 9
    let x86_stdcall = 64
    let x86_fastcall = 65
  end
  class c p = object
    inherit GlobalValue.c p
    method delete = Llvm.delete_function p
    method is_intrinsic = Llvm.is_intrinsic p
    method call_conv = Llvm.function_call_conv p
    method set_call_conv i = Llvm.set_function_call_conv i p
    method gc = Llvm.gc p
    method set_gc s = Llvm.set_gc s p
    method attrs = Llvm.function_attr p
    method add_attr at = Llvm.add_function_attr p at
    method remove_attr at = Llvm.remove_function_attr p at
  end
  let declare m ~name ty = new c (Llvm.declare_function name ty#ptr m)
  let define m ~name ty = new c (Llvm.define_function name ty#ptr m)
  let lookup m ~name = maybe (new c) (Llvm.lookup_function name m)
  (* Missing?
  let get_or_insert m ~name ty =
    new Constant.c (get_or_insert_function m name ty#ptr)
  *)
  module Pos = struct
    type collection = Module.t
    type item = c
    let first m = wrapr (new c) (Llvm.function_begin m)
    let last m = rev_wrapr (new c) (Llvm.function_end m)
    let succ gv = wrapr (new c) (Llvm.function_succ gv#ptr)
    let pred gv = rev_wrapr (new c) (Llvm.function_pred gv#ptr)
  end
  include Iterable(Pos)
  let test v = v#classify = Llvm.ValueKind.Function
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.Function")
end

module GlobalAlias = struct
  type t = llvalue
  class c p = object inherit GlobalValue.c p end
  let make m ty ~aliasee ~name = new c (Llvm.add_alias m ty#ptr aliasee#ptr name)
  let test v = v#classify = Llvm.ValueKind.GlobalAlias
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.GlobalAlias")
end

module Argument = struct
  type t = llvalue
  class c p = object
    inherit Value.c p
    method parent = new Function.c (Llvm.param_parent p)
    method attrs = Llvm.param_attr p
    method add_attr at = Llvm.add_param_attr p at
    method remove_attr at = Llvm.remove_param_attr p at
    method set_alignment i = Llvm.set_param_alignment p i
  end
  let params fn = Array.map (new c) (Llvm.params fn#ptr)
  let param fn i = new c (Llvm.param fn#ptr i)
  module Pos = struct
    type collection = Function.c
    type item = c
    let first fn =
      Llvm.param_begin fn#ptr |> wrapr (new c) |> wrapl (new Function.c)
    let last fn =
      Llvm.param_end fn#ptr |> rev_wrapr (new c) |> rev_wrapl (new Function.c)
    let succ arg =
      Llvm.param_succ arg#ptr |> wrapr (new c) |> wrapl (new Function.c)
    let pred arg =
      Llvm.param_pred arg#ptr |> rev_wrapr (new c) |> rev_wrapl (new Function.c)
  end
  include Iterable(Pos)
  let test v = v#classify = Llvm.ValueKind.Argument
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.Argument")
end

module BasicBlock = struct
  type t = Llvm.llbasicblock
  class c p = object
    inherit Value.c (Llvm.value_of_block p)
    method parent = new Function.c (Llvm.block_parent p)
    method delete = Llvm.delete_block p
  end
  let blocks fn = Array.map (new c) (Llvm.basic_blocks fn#ptr)
  let entry fn = new c (Llvm.entry_block fn#ptr)
  let append fn ctx ~name = new c (Llvm.append_block ctx name fn#ptr)
  let insert ~before:bb ctx ~name =
    new c (Llvm.insert_block ctx name (Llvm.block_of_value bb#ptr))
  module Pos = struct
    type collection = Function.c
    type item = c
    let first fn =
      Llvm.block_begin fn#ptr |> wrapr (new c) |> wrapl (new Function.c)
    let last fn =
      Llvm.block_end fn#ptr |> rev_wrapr (new c) |> rev_wrapl (new Function.c)
    let succ bb =
      Llvm.block_succ (Llvm.block_of_value bb#ptr)
        |> wrapr (new c)
        |> wrapl (new Function.c)
    let pred bb =
      Llvm.block_pred (Llvm.block_of_value bb#ptr)
        |> rev_wrapr (new c)
        |> rev_wrapl (new Function.c)
  end
  include Iterable(Pos)
  let test v = Llvm.value_is_block v#ptr
  let from v =
    if test v then new c (Llvm.block_of_value v#ptr)
    else raise (Cast_failure "Llvm_safe.BasicBlock")
end

module BlockAddress = struct
  type t = llvalue
  class c p = object inherit Constant.c p end
  let make fn bb =
    new c (Llvm.block_address fn#ptr (Llvm.block_of_value bb#ptr))
  let test v = v#classify = Value.BlockAddress
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.BlockAddress")
end

module Instruction = struct
  type t = llvalue
  class c p = object
    inherit User.c p
    method delete = Llvm.delete_instruction p
    method parent = new BasicBlock.c (Llvm.instr_parent p)
    method has_metadata = Llvm.has_metadata p
    method metadata ~kind = maybe (new MDNode.c) (Llvm.metadata p kind)
    method set_metadata ~kind (md : MDNode.c) = Llvm.set_metadata p kind md#ptr
    method clear_metadata ~kind = Llvm.clear_metadata p kind
    method opcode = Llvm.instr_opcode p
    method icmp_predicate = Llvm.icmp_predicate p
  end
  module Pos = struct
    type collection = BasicBlock.c
    type item = c
    let first bb =
      Llvm.instr_begin (Llvm.block_of_value bb#ptr)
        |> wrapr (new c)
        |> wrapl (new BasicBlock.c)
    let last bb =
      Llvm.instr_end (Llvm.block_of_value bb#ptr)
        |> rev_wrapr (new c)
        |> rev_wrapl (new BasicBlock.c)
    let succ ins =
      Llvm.instr_succ ins#ptr
        |> wrapr (new c)
        |> wrapl (new BasicBlock.c)
    let pred ins =
      Llvm.instr_pred ins#ptr
        |> rev_wrapr (new c)
        |> rev_wrapl (new BasicBlock.c)
  end
  include Iterable(Pos)
  let test v = match v#classify with Value.Instruction _ -> true | _ -> false
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.Instruction")
end

module TerminatorInst = struct
  type t = llvalue
  class c p = object inherit Instruction.c p end
  let block_terminator bb =
    maybe (new c) (Llvm.block_terminator (Llvm.block_of_value bb#ptr))
  let test v =
    let open Opcode in
    match v#classify with
    | Value.Instruction (
        Ret | Br | Switch | IndirectBr | Invoke | Invalid2 | Unreachable
      ) -> true
    | _ -> false
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.TerminatorInst")
end

module CallInst = struct
  type t = llvalue
  class c p = object
    inherit Instruction.c p
    method call_conv = Llvm.instruction_call_conv p
    method set_call_conv i = Llvm.set_instruction_call_conv i p
    method add_param_attr i a = Llvm.add_instruction_param_attr p i a
    method remove_param_attr i a = Llvm.remove_instruction_param_attr p i a
    method is_tail_call = Llvm.is_tail_call p
    method set_tail_call tc = Llvm.set_tail_call tc p
  end
  let test v =
    match v#classify with
    | Value.Instruction Opcode.Call -> true
    | _ -> false
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.CallInst")
end

module InvokeInst = struct
  type t = llvalue
  class c p = object
    inherit Instruction.c p
    method call_conv = Llvm.instruction_call_conv p
    method set_call_conv i = Llvm.set_instruction_call_conv i p
    method add_param_attr i a = Llvm.add_instruction_param_attr p i a
    method remove_param_attr i a = Llvm.remove_instruction_param_attr p i a
  end
  let test v =
    match v#classify with
    | Value.Instruction Opcode.Invoke -> true
    | _ -> false
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.InvokeInst")
end

module PHINode = struct
  type t = llvalue
  class c p = object
    inherit Instruction.c p
    method incoming =
      List.map (fun (v, bb) ->
        (new Value.c v, new BasicBlock.c bb)
      ) (Llvm.incoming p)
    method add_incoming ((v : Value.c), (bb : BasicBlock.c)) =
      Llvm.add_incoming (v#ptr, (Llvm.block_of_value bb#ptr)) p
  end
  let test v =
    match v#classify with
    | Value.Instruction Opcode.PHI -> true
    | _ -> false
  let from v =
    if test v then new c v#ptr
    else raise (Cast_failure "Llvm_safe.PHINode")
end

module Builder = struct
  type t = llbuilder

  let create = Llvm.builder
  let position_at b (ip : (BasicBlock.c, Instruction.c) pos) =
    let pos = ip
      |> wrapl (fun x -> Llvm.block_of_value x#ptr)
      |> wrapr (fun x -> x#ptr)
    in
    Llvm.position_builder pos b
  let at ctx ip =
    let builder = create ctx in
    position_at builder ip;
    builder
  let before ctx ins = at ctx (Llvm.Before ins)
  let at_end ctx bb = at ctx (Llvm.At_end bb)
  let position_before b ins = position_at b (Llvm.Before ins)
  let position_at_end b bb = position_at b (Llvm.At_end bb)

  let insertion_block b = new BasicBlock.c (Llvm.insertion_block b)
  let insert_into_builder b ins name = Llvm.insert_into_builder ins#ptr name b

  let debug_location b = maybe (new MDNode.c) (Llvm.current_debug_location b)
  let set_debug_location b md = Llvm.set_current_debug_location b md#ptr
  (* TODO clear_debug_location *)
  let set_inst_debug_location b ins = Llvm.set_inst_debug_location b ins#ptr

  let ret_void b = new Instruction.c (Llvm.build_ret_void b)
  let ret b v = new Instruction.c (Llvm.build_ret v#ptr b)
  let aggregate_ret b vs =
    let vps = Array.map (fun x -> x#ptr) vs in
      new Instruction.c (Llvm.build_aggregate_ret vps b)
  let br b bb = new Instruction.c (Llvm.build_br (Llvm.block_of_value bb#ptr) b)
  let cond_br b v tbb fbb =
    new Instruction.c (Llvm.build_cond_br v#ptr (Llvm.block_of_value tbb#ptr) (Llvm.block_of_value fbb#ptr) b)
  let switch b v bb i = new Instruction.c (Llvm.build_switch v#ptr (Llvm.block_of_value bb#ptr) i b)
  (* TODO malloc, array_malloc, free, add_case, switch_default_dest *)
  let indirect_br b v i = new Instruction.c (Llvm.build_indirect_br v#ptr i b)
  (* TODO add_destination *)
  let invoke b fn args after catch name =
    let argps = Array.map (fun x -> x#ptr) args in
      new Instruction.c
        (Llvm.build_invoke fn#ptr argps (Llvm.block_of_value after#ptr) (Llvm.block_of_value catch#ptr) name b)
  (* TODO landingpad, set_cleanup, add_clause *)
  (* TODO resume *)
  let unreachable b = new Instruction.c (Llvm.build_unreachable b)

  let unary b op arg name =
    new Value.c (let open Operator in match op with
      | Neg -> Llvm.build_neg arg#ptr name b
      | NSWNeg -> Llvm.build_nsw_neg arg#ptr name b
      | NUWNeg -> Llvm.build_nuw_neg arg#ptr name b
      | FNeg -> Llvm.build_fneg arg#ptr name b
      | Not -> Llvm.build_not arg#ptr name b)
  let binary b op ~lhs ~rhs name =
    new Value.c (let open Operator in match op with
      | Add -> Llvm.build_add lhs#ptr rhs#ptr name b
      | NSWAdd -> Llvm.build_nsw_add lhs#ptr rhs#ptr name b
      | NUWAdd -> Llvm.build_nuw_add lhs#ptr rhs#ptr name b
      | FAdd -> Llvm.build_fadd lhs#ptr rhs#ptr name b
      | Sub -> Llvm.build_sub lhs#ptr rhs#ptr name b
      | NSWSub -> Llvm.build_nsw_sub lhs#ptr rhs#ptr name b
      | NUWSub -> Llvm.build_nuw_sub lhs#ptr rhs#ptr name b
      | FSub -> Llvm.build_fsub lhs#ptr rhs#ptr name b
      | Mul -> Llvm.build_mul lhs#ptr rhs#ptr name b
      | NSWMul -> Llvm.build_nsw_mul lhs#ptr rhs#ptr name b
      | NUWMul -> Llvm.build_nuw_mul lhs#ptr rhs#ptr name b
      | FMul -> Llvm.build_fmul lhs#ptr rhs#ptr name b
      | UDiv -> Llvm.build_udiv lhs#ptr rhs#ptr name b
      | SDiv -> Llvm.build_sdiv lhs#ptr rhs#ptr name b
      | FDiv -> Llvm.build_fdiv lhs#ptr rhs#ptr name b
      | ExactSDiv -> Llvm.build_exact_sdiv lhs#ptr rhs#ptr name b
      | URem -> Llvm.build_urem lhs#ptr rhs#ptr name b
      | SRem -> Llvm.build_srem lhs#ptr rhs#ptr name b
      | FRem -> Llvm.build_frem lhs#ptr rhs#ptr name b
      | Shl -> Llvm.build_shl lhs#ptr rhs#ptr name b
      | LShr -> Llvm.build_lshr lhs#ptr rhs#ptr name b
      | AShr -> Llvm.build_ashr lhs#ptr rhs#ptr name b
      | And -> Llvm.build_and lhs#ptr rhs#ptr name b
      | Or -> Llvm.build_or lhs#ptr rhs#ptr name b
      | Xor -> Llvm.build_xor lhs#ptr rhs#ptr name b)
  let cast b op arg ty name =
    new Value.c (let open Operator in match op with
      | Trunc -> Llvm.build_trunc arg#ptr ty#ptr name b
      | ZExt -> Llvm.build_zext arg#ptr ty#ptr name b
      | SExt -> Llvm.build_sext arg#ptr ty#ptr name b
      | FPToUI -> Llvm.build_fptoui arg#ptr ty#ptr name b
      | FPToSI -> Llvm.build_fptosi arg#ptr ty#ptr name b
      | UIToFP -> Llvm.build_uitofp arg#ptr ty#ptr name b
      | SIToFP -> Llvm.build_sitofp arg#ptr ty#ptr name b
      | FPTrunc -> Llvm.build_fptrunc arg#ptr ty#ptr name b
      | FPExt -> Llvm.build_fpext arg#ptr ty#ptr name b
      | PtrToInt -> Llvm.build_ptrtoint arg#ptr ty#ptr name b
      | IntToPtr -> Llvm.build_inttoptr arg#ptr ty#ptr name b
      | BitCast -> Llvm.build_bitcast arg#ptr ty#ptr name b
      | ZExtOrBitCast -> Llvm.build_zext_or_bitcast arg#ptr ty#ptr name b
      | SExtOrBitCast -> Llvm.build_sext_or_bitcast arg#ptr ty#ptr name b
      | TruncOrBitCast -> Llvm.build_trunc_or_bitcast arg#ptr ty#ptr name b
      | PointerCast -> Llvm.build_pointercast arg#ptr ty#ptr name b
      | IntCast -> Llvm.build_intcast arg#ptr ty#ptr name b
      | FPCast -> Llvm.build_fpcast arg#ptr ty#ptr name b)

  let alloca b ty name = new Instruction.c (Llvm.build_alloca ty#ptr name b)
  let array_alloca b ty v name =
    new Instruction.c (Llvm.build_array_alloca ty#ptr v#ptr name b)
  let load b v name = new Instruction.c (Llvm.build_load v#ptr name b)
  let store b loc v = new Instruction.c (Llvm.build_store loc#ptr v#ptr b)
  let gep ?(inbounds = false) b arg idxs name =
    let idxps = Array.map (fun x -> x#ptr) idxs in
      if not inbounds then new Value.c (Llvm.build_gep arg#ptr idxps name b)
      else new Value.c (Llvm.build_in_bounds_gep arg#ptr idxps name b)
  let struct_gep b v i name =
    new Value.c (Llvm.build_struct_gep v#ptr i name b)
  let global_string b s name =
    new Value.c (Llvm.build_global_string s name b)
  let global_stringptr b s name =
    new Value.c (Llvm.build_global_stringptr s name b)

  let icmp b pred ~lhs ~rhs name =
    new Value.c (Llvm.build_icmp pred lhs#ptr rhs#ptr name b)
  let fcmp b pred ~lhs ~rhs name =
    new Value.c (Llvm.build_fcmp pred lhs#ptr rhs#ptr name b)
  let phi b l name =
    let l' = List.map (fun (v, bb) -> (v#ptr, Llvm.block_of_value bb#ptr)) l in
      new Value.c (Llvm.build_phi l' name b)
  let call b fn args name =
    let argps = Array.map (fun x -> x#ptr) args in
      new Value.c (Llvm.build_call fn#ptr argps name b)
  let select b c t f name =
    new Value.c (Llvm.build_select c#ptr t#ptr f#ptr name b)
  let va_arg b v ty name =
    new Value.c (Llvm.build_va_arg v#ptr ty#ptr name b)
  let extractelement b ~vec ~idx name =
    new Value.c (Llvm.build_extractelement vec#ptr idx#ptr name b)
  let insertelement b ~vec ~elt ~idx name =
    new Value.c (Llvm.build_insertelement vec#ptr elt#ptr idx#ptr name b)
  let shufflevector b ~v1 ~v2 ~mask name =
    new Value.c (Llvm.build_shufflevector v1#ptr v2#ptr mask#ptr name b)
  let extractvalue b ~agg ~idx name =
    new Value.c (Llvm.build_extractvalue agg#ptr idx name b)
  let insertvalue b ~agg ~elt ~idx name =
    new Value.c (Llvm.build_insertvalue agg#ptr elt#ptr idx name b)
  let is_null b v name = new Value.c (Llvm.build_is_null v#ptr name b)
  let is_not_null b v name =
    new Value.c (Llvm.build_is_not_null v#ptr name b)
  let ptrdiff b v1 v2 name =
    new Value.c (Llvm.build_ptrdiff v1#ptr v2#ptr name b)
end

module MemoryBuffer = struct
  type t = llmemorybuffer
  let of_file = Llvm.MemoryBuffer.of_file
  let of_stdin = Llvm.MemoryBuffer.of_stdin
  let dispose = Llvm.MemoryBuffer.dispose
end

module PassManager = struct
  type 'a t
  type module_pass
  type function_pass
  external create
    : unit -> module_pass t
    = "llvm_passmanager_create"
  external create_function
    : llmodule -> function_pass t
    = "LLVMCreateFunctionPassManager"
  external run_module
    : module_pass t -> llmodule -> bool
    = "llvm_passmanager_run_module"
  external initialize
    : function_pass t -> bool
    = "llvm_passmanager_initialize"
  external run_function
    : function_pass t -> llvalue -> bool
    = "llvm_passmanager_run_function"
  external finalize
    : function_pass t -> bool
    = "llvm_passmanager_finalize"
  external dispose
    : 'a t -> unit
    = "llvm_passmanager_dispose"
  let run_function pm fn = run_function pm fn#ptr
end
