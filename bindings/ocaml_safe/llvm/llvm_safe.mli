(*===-- llvm/llvm.mli - LLVM OCaml Interface -------------------------------===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*)

(** Core API.

    This interface provides an OCaml API for the LLVM intermediate
    representation, the classes in the VMCore library.

    Unlike the default bindings that expose LLVM's unsafe C interface, these
    bindings re-establish the subtype hierarchy of LLVM's C++ code, using the
    advanced object and module systems available in OCaml.
*)


(** {6 Iteration} *)

(** {7 C++ style iteration with [pos] and [rev_pos]} *)

(** [Before b] and [At_end a] specify positions from the start of the ['b] list
    of ['a]. [pos] is used to specify positions in and for forward iteration
    through the various value lists maintained by the LLVM IR. *)
type ('a, 'b) pos = ('a, 'b) Llvm.llpos = At_end of 'a | Before of 'b

(** [After b] and [At_start a] specify positions from the end of the ['b] list
    of ['a]. [rev_pos] is used for reverse iteration through the various value
    lists maintained by the LLVM IR. *)
type ('a, 'b) rev_pos = ('a, 'b) Llvm.llrev_pos = At_start of 'a | After of 'b

(** {7 Higher order functional iterators} *)

(** Given a module with position functions over [(collection, item) pos] and
    [(collection, item) rev_pos], there are also the more convenient
    functional iterators: [iter], [rev_iter], [fold_left], and [fold_right].

    These functions can be used to traverse:
    - Global variables in a module: see {!GlobalVariable}
    - Functions in a module: see {!Function}
    - Arguments in a function: see {!Argument}
    - Basic blocks in a function: see {!BasicBlock}
    - Instructions in a basic block: see {!Instruction}

    The convention here is: for iterators over [Foo.c] items in some
    collection, look in module [Foo]. *)


(** {6 Exceptions} *)

(** Raised when creation of a {!MemoryBuffer} fails *)
exception IoError of string

(** Raised by the [from] functions if downcasting their argument would fail *)
exception Cast_failure of string


(** {6 Enumerations} *)

(** The opcodes for LLVM instructions and constant expressions *)
module Opcode : sig
  type t = Llvm.Opcode.t =
    | Invalid (** Not an instruction *)

    | Ret (** Terminator instructions *)
    | Br
    | Switch
    | IndirectBr
    | Invoke
    | Invalid2
    | Unreachable

    | Add (** Standard binary operators *)
    | FAdd
    | Sub
    | FSub
    | Mul
    | FMul
    | UDiv
    | SDiv
    | FDiv
    | URem
    | SRem
    | FRem

    | Shl (** Logical operators *)
    | LShr
    | AShr
    | And
    | Or
    | Xor

    | Alloca (** Memory operators *)
    | Load
    | Store
    | GetElementPtr

    | Trunc (** Cast operators *)
    | ZExt
    | SExt
    | FPToUI
    | FPToSI
    | UIToFP
    | SIToFP
    | FPTrunc
    | FPExt
    | PtrToInt
    | IntToPtr
    | BitCast

    | ICmp (** Other operators *)
    | FCmp
    | PHI
    | Call
    | Select
    | UserOp1
    | UserOp2
    | VAArg
    | ExtractElement
    | InsertElement
    | ShuffleVector
    | ExtractValue
    | InsertValue
    | Fence
    | AtomicCmpXchg
    | AtomicRMW
    | Resume
    | LandingPad
    | Unwind
end

(** The various operators available when constructing unary/binary/cast
    instructions or constant expressions through this interface *)
module Operator : sig
  type unary =
    | Neg (** Arithmetic negation *)
    | NSWNeg (** Arithmetic negation with no signed wrapping;
                 undefined if negation overflows *)
    | NUWNeg (** Arithmetic negation with no unsigned wrapping;
                 undefined if negation overflows *)
    | FNeg (** Arithmetic negation of floats *)
    | Not (** Bitwise inverse *)

  type binary =
    | Add (** Sum *)
    | NSWAdd (** Sum with no signed wrapping;
                 undefined if sum overflows *)
    | NUWAdd (** Sum with no unsigned wrapping;
                 undefined if sum overflows *)
    | FAdd (** Sum of floats *)
    | Sub (** Difference *)
    | NSWSub (** Difference with no signed wrapping;
                 undefined if difference overflows *)
    | NUWSub (** Difference with no unsigned wrapping;
                 undefined if difference overflows *)
    | FSub  (** Difference of floats *)
    | Mul (** Product *)
    | NSWMul (** Product with no signed wrapping;
                 undefined if product overflows *)
    | NUWMul (** Product with no unsigned wrapping;
                 undefined if product overflows *)
    | FMul (** Product of floats *)
    | UDiv (** Quotient *)
    | SDiv (** Signed quotient *)
    | ExactSDiv (** Signed quotient, undefined if rounded or overflows *)
    | FDiv (** Quotient of floats *)
    | URem (** Remainder *)
    | SRem (** Signed remainder *)
    | FRem (** Remainder of floats *)
    | Shl (** Bitwise shift left *)
    | LShr (** Bitwise logical shift right, with zero extension *)
    | AShr (** Bitwise arithmetic shift right, with sign extension *)
    | And (** Bitwise and *)
    | Or (** Bitwise or *)
    | Xor (** Bitwise exclusive or *)

  type cast =
    | Trunc (** Integer truncation *)
    | ZExt (** Zero extension *)
    | SExt (** Sign extension *)
    | FPToUI (** Float to unsigned integer *)
    | FPToSI (** Float to signed integer *)
    | UIToFP (** Unsigned integer to float *)
    | SIToFP (** Signed integer to float *)
    | FPTrunc (** Float truncation *)
    | FPExt (** Float extension *)
    | PtrToInt (** Pointer to integer *)
    | IntToPtr (** Integer to pointer *)
    | BitCast (** Bitwise conversion (true cast) *)
    | ZExtOrBitCast | SExtOrBitCast | TruncOrBitCast
    | PointerCast | IntCast | FPCast
end

(** Comparison predicates for [icmp] and [fcmp] instructions *)
module Predicate : sig
  (** The predicate for an integer comparison ([icmp]) instruction. See the
      [llvm::ICmpInst::Predicate] enumeration. *)
  type icmp =
    | Eq (** Equal *)
    | Ne (** Not equal *)
    | Ugt (** Unsigned greater than *)
    | Uge (** Unsigned greater or equal *)
    | Ult (** Unsigned less than *)
    | Ule (** Unsigned less or equal *)
    | Sgt (** Signed greater than *)
    | Sge (** Signed greater or equal *)
    | Slt (** Signed less than *)
    | Sle (** Signed less or equal *)

  (** The predicate for a floating-point comparison ([fcmp]) instruction. See
      the [llvm::FCmpInst::Predicate] enumeration. *)
  type fcmp =
    | False (** Always false (always folded) *)
    | Oeq (** True if ordered and equal *)
    | Ogt (** True if ordered and greater than *)
    | Oge (** True if ordered and greater than or equal *)
    | Olt (** True if ordered and less than *)
    | Ole (** True if ordered and less than or equal *)
    | One (** True if ordered and operands are unequal *)
    | Ord (** True if ordered (no NaNs) *)
    | Uno (** True if unordered: isNaN(X) | isNaN(Y) *)
    | Ueq (** True if unordered or equal *)
    | Ugt (** True if unordered or greater than *)
    | Uge (** True if unordered, greater than, or equal *)
    | Ult (** True if unordered or less than *)
    | Ule (** True if unordered, less than, or equal *)
    | Une (** True if unordered or not equal *)
    | True (** Always true (always folded) *)
end

(** Attributes of functions and their arguments *)
module Attribute : sig
  type t =
    | Zext | Sext | Noreturn | Inreg | Structret | Nounwind | Noalias | Byval
    | Nest | Readnone | Readonly | Noinline | Alwaysinline | Optsize
    | Ssp | Sspreq
    | Alignment of int
    | Nocapture | Noredzone | Noimplicitfloat | Naked | Inlinehint
    | Stackalignment of int
    | ReturnsTwice | UWTable | NonLazyBind
end


(** {6 LLVM Contexts} *)

(** The top-level container for all LLVM global data. See the
    [llvm::LLVMContext] class. *)
module Context : sig
  type t

  (** [Context.create ()] creates a context for storing the "global" state in
      LLVM. See the constructor [llvm::LLVMContext]. *)
  val create : unit -> t

  (** [Context.destroy ctx] destroys a context. See the destructor
      [llvm::LLVMContext::~LLVMContext]. *)
  val dispose : t -> unit

  (** See the function [llvm::getGlobalContext]. *)
  val global : unit -> t

  (** [Context.mdkind_id ctx name] returns the MDKind ID that corresponds to
      the name [name] in the context [ctx]. See the function
      [llvm::LLVMContext::getMDKindID]. *)
  val mdkind_id : t -> string -> int
end


(** {6 LLVM Types} *)

(**
  Inheritance hierarchy of Type subclasses:

  {v
  Type
  | IntegerType
  | FunctionType
  | StructType
  | SequentialType
    | ArrayType
    | PointerType
    | VectorType
  v}
*)

(** Each {!Value.c} in the LLVM IR has a type, an instance of {!Type.c}. See
    the [llvm::Type] class. *)
module Type : sig
  type t

  (** The kind of a [Type], the result of [ty#classify]. See the
      [llvm::Type::TypeID] enumeration. *)
  type kind = Llvm.TypeKind.t =
    | Void | Half | Float | Double | X86fp80 | Fp128 | Ppc_fp128 | Label
    | Integer | Function | Struct | Array | Pointer | Vector | Metadata

  class c : t -> object
    method ptr : t

    (** [ty#classify] returns the kind corresponding to the type [ty]. See the
        method [llvm::Type::getTypeID]. *)
    method classify : kind

    (** [ty#is_sized] returns whether the type has a size or not. If it doesn't
        then it is not safe to call the [DataLayout::] methods on it. *)
    method is_sized : bool

    (** [ty#context] returns the {!Context.t} corresponding to the type [ty].
        See the method [llvm::Type::getContext]. *)
    method context : Context.t

    (** [ty#to_string] returns a string describing the type [ty]. *)
    method to_string : string
  end

  (** [Type.void c] creates a type of a function which does not return any
      value in the context [c]. See [llvm::Type::VoidTy]. *)
  val void : Context.t -> c
 
  (* TODO Half, Metadata *)

  (** [Type.float c] returns the IEEE 32-bit floating point type in the context
      [c]. See [llvm::Type::FloatTy]. *)
  val float : Context.t -> c

  (** [Type.double c] returns the IEEE 64-bit floating point type in the
      context [c]. See [llvm::Type::DoubleTy]. *)
  val double : Context.t -> c

  (** [Type.x86fp80 c] returns the x87 80-bit floating point type in the
      context [c]. See [llvm::Type::X86_FP80Ty]. *)
  val x86fp80 : Context.t -> c

  (** [Type.fp128 c] returns the IEEE 128-bit floating point type in the
      context [c]. See [llvm::Type::FP128Ty]. *)
  val fp128 : Context.t -> c

  (** [Type.ppc_fp128 c] returns the PowerPC 128-bit floating point type in the
      context [c]. See [llvm::Type::PPC_FP128Ty]. *)
  val ppc_fp128 : Context.t -> c

  (** [Type.label c] creates a type of a basic block in the context [c]. See
      [llvm::Type::LabelTy]. *)
  val label : Context.t -> c
end

(**

*)

(** Methods of integer types: {!IntegerType.c} *)
module IntegerType : sig
  type t
  class c : t -> object
    inherit Type.c
    
    (** [ity#bitwidth] returns the number of bits in the integer type [ty]. See
        the method [llvm::IntegerType::getBitWidth]. *)
    method bitwidth : int
  end

  (** [IntegerType.i1 c] returns an integer type of bitwidth 1 in the context
      [c]. See [llvm::Type::Int1Ty]. *)
  val i1 : Context.t -> c

  (** [IntegerType.i8 c] returns an integer type of bitwidth 8 in the context
      [c]. See [llvm::Type::Int8Ty]. *)
  val i8 : Context.t -> c

  (** [IntegerType.i16 c] returns an integer type of bitwidth 16 in the context
      [c]. See [llvm::Type::Int16Ty]. *)
  val i16 : Context.t -> c

  (** [IntegerType.i32 c] returns an integer type of bitwidth 32 in the context
      [c]. See [llvm::Type::Int32Ty]. *)
  val i32 : Context.t -> c

  (** [IntegerType.i64 c] returns an integer type of bitwidth 64 in the context
      [c]. See [llvm::Type::Int64Ty]. *)
  val i64 : Context.t -> c

  (** [IntegerType.make c n] returns an integer type of bitwidth [n] in the
      context [c]. See the method [llvm::IntegerType::get]. *)
  val make : Context.t -> bits:int -> c

  (** [IntegerType.test ty] checks whether the {!Type.c} [ty] can be safely
      downcast to [IntegerType.c]. *)
  val test : Type.c -> bool

  (** [IntegerType.from ty] performs a checked downcast of the {!Type.c} [ty]
      to [IntegerType.c]. *)
  val from : Type.c -> c
end

(** Methods of function types: {!FunctionType.c} *)
module FunctionType : sig
  type t
  class c : t -> object
    inherit Type.c

    (** [fty#is_var_arg] returns [true] if [fty] is a varargs function type,
        [false] otherwise. See the method [llvm::FunctionType::isVarArg]. *)
    method is_var_arg : bool

    (** [fty#return_type] gets the return type of the function type [fty]. See
        the method [llvm::FunctionType::getReturnType]. *)
    method return_type : Type.c

    (** [fty#param_types] gets the parameter types of the function type [fty].
        See the method [llvm::FunctionType::getParamType]. *)
    method param_types : Type.c array
  end

  (** [FunctionType.make ret_ty param_tys] returns the function type returning
      [ret_ty] and taking [param_tys] as parameters. If [vararg] is true, the
      function type will also accept a variable number of arguments. See the
      method [llvm::FunctionType::get]. *)
  val make : ?vararg:bool -> ret:Type.c -> params:Type.c array -> c

  (** [FunctionType.test ty] checks whether the {!Type.c} [ty] can be safely
      downcast to [FunctionType.c]. *)
  val test : Type.c -> bool

  (** [FunctionType.from ty] performs a checked downcast of the {!Type.c} [ty]
      to [FunctionType.c]. *)
  val from : Type.c -> c
end

(** Methods of structure types: {!StructType.c} *)
module StructType : sig
  type t
  class c : t -> object
    inherit Type.c

    (** [sty#name] returns the name of the named structure type [ty], or None
        if the structure type is not named. *)
    method name : string option

    (** [sty#set_body elts] sets the body of the named struct [ty] to the
        [elts] elements. See the method [llvm::StructType::setBody]. *)
    method set_body : ?packed:bool -> elts:Type.c array -> unit

    (** [sty#element_types] returns the constituent types of the struct type
        [sty]. See the method [llvm::StructType::getElementType]. *)
    method element_types : Type.c array

    (** [sty#is_packed] returns [true] if the structure type [sty] is packed,
        [false] otherwise. See the method [llvm::StructType::isPacked]. *)
    method is_packed : bool

    (** [sty#is_opaque] returns [true] if the structure type [sty] is opaque.
        [false] otherwise. See the method [llvm::StructType::isOpaque]. *)
    method is_opaque : bool
  end

  (** [StructType.make context tys] returns the structure type in the context
      [context] containing in the types in the array [tys]. See the method
      [llvm::StructType::get]. *)
  val make : ?packed:bool -> Context.t -> elts:Type.c array -> c

  (** [StructType.named context name] returns the named structure type [name]
      in the context [context]. See the method [llvm::StructType::get]. *)
  val named : Context.t -> string -> c

  (** [StructType.test ty] checks whether the {!Type.c} [ty] can be safely
      downcast to [StructType.c]. *)
  val test : Type.c -> bool

  (** [StructType.from ty] performs a checked downcast of the {!Type.c} [ty] to
      [StructType.c]. *)
  val from : Type.c -> c
end


(** {7 Operations on pointer, vector, and array types} *)

(** Methods of all sequential types: {!SequentialType.c} *)
module SequentialType : sig
  type t
  class c : t -> object
    inherit Type.c

    (** [seqty#element_type] returns the element type of the pointer, vector,
        or array type [seqty]. See the method [llvm::SequentialType::get]. *)
    method element_type : Type.t
  end
end

(** Methods of array types: {!ArrayType.c} *)
module ArrayType : sig
  type t
  class c : t -> object
    inherit SequentialType.c

    (** [aty#length] returns the element count of the array type [aty]. See the
        method [llvm::ArrayType::getNumElements]. *)
    method length : int
  end

  (** [ArrayType.make ty n] returns the array type containing [n] elements of
      type [ty]. See the method [llvm::ArrayType::get]. *)
  val make : Type.c -> len:int -> c

  (** [ArrayType.test ty] checks whether the {!Type.c} [ty] can be safely
      downcast to [ArrayType.c]. *)
  val test : Type.c -> bool

  (** [ArrayType.from ty] performs a checked downcast of the {!Type.c} [ty] to
      [ArrayType.c]. *)
  val from : Type.c -> c
end

(** Methods of pointer types: {!PointerType.c} *)
module PointerType : sig
  type t
  class c : t -> object
    inherit SequentialType.c

    (** [pty#address_space] returns the address space qualifier of the pointer
        type [pty]. See the method [llvm::PointerType::getAddressSpace]. *)
    method address_space : int
  end

  (** [PointerType.make ty] returns the pointer type referencing objects of
      type [ty] in the specified address space (default 0). See the method
      [llvm::PointerType::getUnqual]. *)
  val make : ?addrspace:int -> Type.c -> c

  (** [PointerType.test ty] checks whether the {!Type.c} [ty] can be safely
      downcast to [PointerType.c]. *)
  val test : Type.c -> bool

  (** [PointerType.from ty] performs a checked downcast of the {!Type.c} [ty]
      to [PointerType.c]. *)
  val from : Type.c -> c
end

(** Methods of vector types: {!VectorType.c} *)
module VectorType : sig
  type t
  class c : t -> object
    inherit SequentialType.c

    (** [vty#size] returns the element count of the vector type [ty]. See the
        method [llvm::VectorType::getNumElements]. *)
    method size : int
  end

  (** [VectorType.make ty n] returns the array type containing [n] elements of
      the primitive type [ty]. See the method [llvm::ArrayType::get]. *)
  val make : Type.c -> size:int -> c

  (** [VectorType.test ty] checks whether the {!Type.c} [ty] can be safely
      downcast to [VectorType.c]. *)
  val test : Type.c -> bool

  (** [VectorType.from ty] performs a checked downcast of the {!Type.c} [ty] to
      [VectorType.c]. *)
  val from : Type.c -> c
end


(** {6 LLVM Modules} *)

(** The top-level container for all other LLVM Intermediate Representation (IR)
    objects. See the [llvm::Module] class. *)
module Module : sig
  type t

  (** [Module.create context id] creates a module with the supplied module ID
      in the context [context].  Modules are not garbage collected; it is
      mandatory to call {!dispose} to free memory. See the constructor
      [llvm::Module::Module]. *)
  val create : Context.t -> string -> t

  (** [Module.dispose m] destroys a module [m] and all of the IR objects it
      contained. All references to subordinate objects are invalidated;
      referencing them will invoke undefined behavior. See the destructor
      [llvm::Module::~Module]. *)
  val dispose : t -> unit

  (** [Module.target_triple m] is the target specifier for the module [m],
      something like [i686-apple-darwin8]. See the method
      [llvm::Module::getTargetTriple]. *)
  val target_triple : t -> string

  (** [Module.set_target_triple m triple] changes the target specifier for the
      module [m] to the string [triple]. See the method
      [llvm::Module::setTargetTriple]. *)
  val set_target_triple : t -> string -> unit

  (** [Module.data_layout m] is the data layout specifier for the module [m],
      something like
      [e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-...-a0:0:64-f80:128:128]. See the
      method [llvm::Module::getDataLayout]. *)
  val data_layout : t -> string

  (** [Module.set_data_layout m s] changes the data layout specifier for the
      module [m] to the string [s]. See the method
      [llvm::Module::setDataLayout]. *)
  val set_data_layout : t -> string -> unit

  (** [Module.dump m] prints the .ll representation of the module [m] to
      standard error. See the method [llvm::Module::dump]. *)
  val dump : t -> unit

  (** [Module.set_inline_asm m asm] sets the inline assembler for the module.
      See the method [llvm::Module::setModuleInlineAsm]. *)
  val set_inline_asm : t -> string -> unit

  (** [Module.context m] returns the context of the specified module.
      See the method [llvm::Module::getContext] *)
  val context : t -> Context.t

  (** [Module.type_by_name m name] returns the specified type from the current
      module if it exists. See the method [llvm::Module::getTypeByName] *)
  val type_by_name : t -> string -> Type.c option
end


(** {6 LLVM Values} *)

(**
  Inheritance hierarchy of Value subclasses:

  {v
  Value
  | Argument
  | BasicBlock
  | InlineAsm
  | MDNode
  | MDString
  | Use
  | User
    | Constant
    | | ConstantInt
    | | ConstantFP
    | | ConstantPointerNull
    | | ConstantArray
    | | ConstantStruct
    | | ConstantVector
    | | ConstantExpr
    | | BlockAddress
    | | GlobalValue
    |   | GlobalVariable
    |   | GlobalAlias
    |   | Function
    | Instruction
      | TerminatorInst
      | CallInst
      | InvokeInst
      | PHINode
  v}
*)

(** Methods of all values: {!Value.c} *)
module Value : sig
  type t

  (** The kind of a {!Value.c}, the result of [v#classify]. See the various
      [LLVMIsA*] functions. *)
  type kind = Llvm.ValueKind.t =
    | NullValue | Argument | BasicBlock | InlineAsm
    | MDNode | MDString | BlockAddress
    | ConstantAggregateZero | ConstantArray | ConstantExpr
    | ConstantFP | ConstantInt | ConstantPointerNull
    | ConstantStruct | ConstantVector
    | Function | GlobalAlias | GlobalVariable
    | UndefValue
    | Instruction of Opcode.t

  class c : t -> object
    method ptr : t

    (** [v#type_of] returns the type of the value [v].
        See the method [llvm::Value::getType]. *)
    method type_of : Type.c

    method classify : kind

    (** [v#name] returns the name of the value [v]. For global values, this is
        the symbol name. For instructions and basic blocks, it is the SSA
        register name. It is meaningless for constants. See the method
        [llvm::Value::getName]. *)
    method name : string

    (** [v#set_name n] sets the name of the value [v] to [n]. See the method
        [llvm::Value::setName]. *)
    method set_name : string -> unit

    (** [v#dump] prints the .ll representation of the value [v] to standard
        error. See the method [llvm::Value::dump]. *)
    method dump : unit

    (** [old#replace_all_uses_with new] replaces all uses of the value [old]
        with the value [new]. See the method
        [llvm::Value::replaceAllUsesWith]. *)
    method replace_all_uses_with : c -> unit
  end
end


(** A value that uses other values as its operands. See the [llvm::User]
    class. *)
(** Methods: {!User.c} *)
module User : sig
  type t
  class c : t -> object
    inherit Value.c

    (** [v#operand i] returns the operand at index [i] for the value [v]. See
        the method [llvm::User::getOperand]. *)
    method operand : int -> Value.c

    (** [set_operand v i o] sets the operand of the value [v] at the index [i]
        to the value [o].
        See the method [llvm::User::setOperand]. *)
    method set_operand : int -> Value.c -> unit

    (** [v#num_operands] returns the number of operands for the value [v].
        See the method [llvm::User::getNumOperands]. *)
    method num_operands : int
  end
end


(** Used to store users and usees of values. See the [llvm::Use] class. *)
module Use : sig
  type t

  (** [Use.user u] returns the user of the use [u].
      See the method [llvm::Use::getUser]. *)
  val user : t -> User.c

  (** [Use.used u] returns the usee of the use [u].
      See the method [llvm::Use::getUsedValue]. *)
  val used : t -> Value.c

  (** [Use.first v] returns the first position in the use list for the value
      [v]. [first] and [succ] can be used to iterate over the use list in
      order. See the method [llvm::Value::use_begin]. *)
  val first : Value.c -> t option

  (** [Use.succ u] returns the use list position succeeding [u].
      See the method [llvm::use_value_iterator::operator++]. *)
  val succ : t -> t option

  (** [Uses.iter f v] applies function [f] to each of the users of the value
      [v] in order. Tail recursive. *)
  val iter : (t -> unit) -> Value.c -> unit

  (** [Uses.fold_left f init v] is [f (... (f init u1) ...) uN] where
      [u1,...,uN] are the users of the value [v]. Tail recursive. *)
  val fold_left : ('a -> t -> 'a) -> 'a -> Value.c -> 'a

  (** [Uses.fold_right f v init] is [f u1 (... (f uN init) ...)] where
      [u1,...,uN] are the users of the value [v]. NOT tail recursive. *)
  val fold_right : (t -> 'a -> 'a) -> Value.c -> 'a -> 'a
end


(** {6 LLVM Constants} *)

(** Methods of all constants: {!Constant.c} *)
module Constant : sig
  type t
  class c : t -> object
    inherit User.c

    (** [v#is_null] returns [true] if the value [v] is the null (zero) value.
        See the method [llvm::Constant::isNullValue]. *)
    method is_null : bool

    (** [v#is_undef] returns [true] if the value [v] is an undefined value,
        [false] otherwise. Similar to [llvm::isa<UndefValue>]. *)
    method is_undef : bool
  end

  (** [Constant.null ty] returns the constant null (zero) of the type [ty]. See
      the method [llvm::Constant::getNullValue]. *)
  val null : Type.c -> c

  (** [Constant.undef ty] returns the undefined value of the type [ty]. See the
      method [llvm::UndefValue::get]. *)
  val undef : Type.c -> c

  (** [Constant.test v] checks whether the {!Value.c} [v] can be safely
      downcast to [Constant.c]. *)
  val test : Value.c -> bool

  (** [Constant.from v] performs a checked downcast of the {!Value.c} [v] to
      [Constant.c]. *)
  val from : Value.c -> c
end


(** {7 Operations on scalar constants} *)

(** Methods of constant integers: {!ConstantInt.c} *)
module ConstantInt : sig
  type t
  class c : t -> object
    inherit Constant.c

    (** [int64_of_const c] returns the int64 value of the [c] constant integer.
        None is returned if the bitwidth exceeds 64. See the method
        [llvm::ConstantInt::getSExtValue]. *)
    method int64_value : Int64.t option
  end

  (** [ConstantInt.all_ones ty] returns the constant '-1' of the integer or
      vector type [ty]. See the method [llvm::Constant::getAllOnesValue]. *)
  val all_ones : IntegerType.c -> c

  (** [ConstantInt.of_int ty i] returns the integer constant of type [ty] and
      value [i]. See the method [llvm::ConstantInt::get]. *)
  val of_int : IntegerType.c -> int -> c

  (** [ConstantInt.of_int64 ty i] returns the integer constant of type [ty] and
      value [i]. See the method [llvm::ConstantInt::get]. *)
  val of_int64 : ?signext:bool -> IntegerType.c -> Int64.t -> c

  (** [ConstantInt.of_string ty s r] returns the integer constant of type [ty]
      and value [s], with the radix [r]. See the method
      [llvm::ConstantInt::get]. *)
  val of_string : IntegerType.c -> string -> radix:int -> c

  (** [ConstantInt.test v] checks whether the {!Value.c} [v] can be safely
      downcast to [ConstantInt.c]. *)
  val test : Value.c -> bool

  (** [ConstantInt.from v] performs a checked downcast of the {!Value.c} [v] to
      [ConstantInt.c]. *)
  val from : Value.c -> c
end

(** Methods of floating point constants: {!ConstantFP.c} *)
module ConstantFP : sig
  type t
  class c : t -> object
    inherit Constant.c
  end

  (** [ConstantFP.of_float ty n] returns the floating point constant of type
      [ty] and value [n]. See the method [llvm::ConstantFP::get]. *)
  val of_float : Type.c -> float -> c

  (** [ConstantFP.of_string ty s] returns the floating point constant of type
      [ty] and value [s]. See the method [llvm::ConstantFP::get]. *)
  val of_string : Type.c -> string -> c

  (** [ConstantFP.test v] checks whether the {!Value.c} [v] can be safely
      downcast to [ConstantFP.c]. *)
  val test : Value.c -> bool

  (** [ConstantFP.from v] performs a checked downcast of the {!Value.c} [v] to
      [ConstantFP.c]. *)
  val from : Value.c -> c
end

(** Methods of constant null pointers: {!ConstantPointerNull.c} *)
module ConstantPointerNull : sig
  type t
  class c : t -> object inherit Constant.c end

  (** [ConstantPointerNull.make ty] returns the constant null (zero) pointer of
      the pointer type [ty]. See the method
      [llvm::ConstantPointerNull::get]. *)
  val make : PointerType.c -> c

  (** [ConstantPointerNull.test v] checks whether the {!Value.c} [v] can be
      safely downcast to [ConstantPointerNull.c]. *)
  val test : Value.c -> bool

  (** [ConstantPointerNull.from v] performs a checked downcast of the
      {!Value.c} [v] to [ConstantPointerNull.c]. *)
  val from : Value.c -> c
end


(** {7 Operations on composite constants} *)

(** Methods of constant arrays (including strings): {!ConstantArray.c} *)
module ConstantArray : sig
  type t
  class c : t -> object inherit Constant.c end

  (** [ConstantArray.of_string c s] returns the constant [i8] array with the
      values of the characters in the string [s] in the context [c]. This value
      can in turn be used as the initializer for a global variable. See the
      method [llvm::ConstantArray::get]. *)
  val of_string : ?nullterm:bool -> Context.t -> string -> c

  (** [ConstantArray.make ty elts] returns the constant array of type
      [array_type ty (Array.length elts)] and containing the values [elts].
      This value can in turn be used as the initializer for a global variable.
      See the method [llvm::ConstantArray::get]. *)
  val make : Type.c -> Constant.c array -> c

  (** [ConstantArray.test v] checks whether the {!Value.c} [v] can be safely
      downcast to [ConstantArray.c]. *)
  val test : Value.c -> bool

  (** [ConstantArray.from v] performs a checked downcast of the {!Value.c} [v]
      to [ConstantArray.c]. *)
  val from : Value.c -> c
end

(** Methods of structured constants: {!ConstantStruct.c} *)
module ConstantStruct : sig
  type t
  class c : t -> object inherit Constant.c end

  (** [ConstantStruct.make context elts] returns the structured constant of
      type [struct_type (Array.map type_of elts)] and containing the values
      [elts] in the context [context]. This value can in turn be used as the
      initializer for a global variable. See the method
      [llvm::ConstantStruct::getAnon]. *)
  val make : ?packed:bool -> Context.t -> Constant.c array -> c

  (** [ConstantStruct.named namedty elts] returns the structured constant of
      type [namedty] (which must be a named structure type) and containing the
      values [elts]. This value can in turn be used as the initializer for a
      global variable. See the method [llvm::ConstantStruct::get]. *)
  val named : StructType.c -> Constant.c array -> c

  (** [ConstantStruct.test v] checks whether the {!Value.c} [v] can be safely
      downcast to [ConstantStruct.c]. *)
  val test : Value.c -> bool

  (** [ConstantStruct.from v] performs a checked downcast of the {!Value.c} [v]
      to [ConstantStruct.c]. *)
  val from : Value.c -> c
end

(** Methods of vector constants: {!ConstantVector.c} *)
module ConstantVector : sig
  type t
  class c : t -> object inherit Constant.c end

  (** [ConstantVector.all_ones ty] returns the constant '-1' of the vector type
      [ty]. See the method [llvm::Constant::getAllOnesValue]. *)
  val all_ones : VectorType.c -> c

  (** [ConstantVector elts] returns the vector constant of type [vector_type
      (type_of elts.(0)) (Array.length elts)] and containing the values [elts].
      See the method [llvm::ConstantVector::get]. *)
  val make : Constant.c array -> c

  (** [ConstantVector.test v] checks whether the {!Value.c} [v] can be safely
      downcast to [ConstantVector.c]. *)
  val test : Value.c -> bool

  (** [ConstantVector.from v] performs a checked downcast of the {!Value.c} [v]
      to [ConstantVector.c]. *)
  val from : Value.c -> c
end


(** {7 Operations on constant expressions} *)

(** Methods of constant expressions: {!ConstantExpr.c} *)
module ConstantExpr : sig
  type t
  class c : t -> object
    inherit Constant.c
    method opcode : Opcode.t
  end

  (** The constructor functions below correspond to methods
      [llvm::ConstantExpr::get*]. *)

  (** [ConstantExpr.align_of ty] returns the alignof constant for the type
      [ty]. This is equivalent to [const_ptrtoint (const_gep (const_null
      (pointer_type {i8,ty})) (const_int i32_type 0) (const_int i32_type 1))
      i32_type], but considerably more readable. See the method
      [llvm::ConstantExpr::getAlignOf]. *)
  val align_of : Type.c -> c

  (** [ConstantExpr.size_of ty] returns the sizeof constant for the type [ty].
      This is equivalent to [const_ptrtoint (const_gep (const_null
      (pointer_type ty)) (const_int i32_type 1)) i64_type], but considerably
      more readable. See the method [llvm::ConstantExpr::getSizeOf]. *)
  val size_of : Type.c -> c

  (** Refer to the comments on {!Operator.unary}. *)
  val unary : Operator.unary -> Constant.c -> c

  (** Refer to the comments on {!Operator.binary}. *)
  val binary : Operator.binary -> lhs:Constant.c -> rhs:Constant.c -> c

  (** Refer to the comments on {!Operator.cast}. *)
  val cast : Operator.cast -> Constant.c -> Type.c -> c

  (** [ConstantExpr.icmp pred lhs rhs] returns the constant comparison of two
      integer constants, [lhs pred rhs]. See the method
      [llvm::ConstantExpr::getICmp]. *)
  val icmp : Predicate.icmp -> lhs:Constant.c -> rhs:Constant.c -> c

  (** [ConstantExpr.fcmp pred lhs rhs] returns the constant comparison of two
      floating point constants, [lhs pred rhs]. See the method
      [llvm::ConstantExpr::getFCmp]. *)
  val fcmp : Predicate.fcmp -> lhs:Constant.c -> rhs:Constant.c -> c

  (** [ConstantExpr.gep pc indices] returns the constant [getElementPtr] of
      [pc] with the constant integer indices from the array [indices]. See the
      method [llvm::ConstantExpr::getGetElementPtr]. *)
  val gep : ?inbounds:bool -> Constant.c -> Constant.c array -> c

  (** [ConstantExpr.select cond t f] returns the constant conditional which
      returns value [t] if the boolean constant [cond] is true and the value
      [f] otherwise. See the method [llvm::ConstantExpr::getSelect]. *)
  val select : cond:Constant.c -> t:Constant.c -> f:Constant.c -> c

  (** [ConstantExpr.extract_element vec i] returns the constant [i]th element
      of constant vector [vec]. [i] must be a constant [i32] value unsigned
      less than the size of the vector. See the method
      [llvm::ConstantExpr::getExtractElement]. *)
  val extract_element : vec:Constant.c -> idx:Constant.c -> c

  (** [ConstantExpr.insert_element vec v i] returns the constant vector with
      the same elements as constant vector [v] but the [i]th element replaced
      by the constant [v]. [v] must be a constant value with the type of the
      vector elements. [i] must be a constant [i32] value unsigned less than
      the size of the vector. See the method
      [llvm::ConstantExpr::getInsertElement]. *)
  val insert_element : vec:Constant.c -> elt:Constant.c -> idx:Constant.c -> c

  (** [ConstantExpr.shufflevector v1 v2 mask] returns a constant
      [shufflevector]. See the LLVM Language Reference for details on the
      [shufflevector] instruction. See the method
      [llvm::ConstantExpr::getShuffleVector]. *)
  val shuffle_vector : v1:Constant.c -> v2:Constant.c -> mask:Constant.c -> c

  (** [ConstantExpr.extractvalue agg idxs] returns the constant [idxs]th value
      of constant aggregate [agg]. Each [idxs] must be less than the size of
      the aggregate. See the method [llvm::ConstantExpr::getExtractValue]. *)
  val extract_value : agg:Constant.c -> idxs:int array -> c

  (** [ConstantExpr.insertvalue agg elt idxs] inserts the value [elt] in the
      specified indexes [idxs] in the aggegate [agg]. Each [idxs] must be less
      than the size of the aggregate. See the method
      [llvm::ConstantExpr::getInsertValue]. *)
  val insert_value : agg:Constant.c -> elt:Constant.c -> idxs:int array -> c

  (** [ConstantExpr.test v] checks whether the {Value.c} [v] can be safely
      downcast to [ConstantExpr.c]. *)
  val test : Value.c -> bool

  (** [ConstantExpr.from v] performs a checked downcast of the {Value.c} [v] to
      [ConstantExpr.c]. *)
  val from : Value.c -> c
end

module InlineAsm : sig
  type t
  class c : t -> object inherit Value.c end

  (** [InlineAsm.make ty asm con side align] inserts a inline assembly string.
      See the method [llvm::InlineAsm::get]. *)
  val make : Type.c -> asm:string -> constraints:string -> effects:bool ->
             align_stack:bool -> c

  (** [InlineAsm.test v] checks whether the {Value.c} [v] can be safely
      downcast to [InlineAsm.c]. *)
  val test : Value.c -> bool

  (** [InlineAsm.from v] performs a checked downcast of the {Value.c} [v]
      to [InlineAsm.c]. *)
  val from : Value.c -> c
end


(** {7 Operations on metadata} *)

module MDNode : sig
  type t
  class c : t -> object inherit Value.c end

  (** [MDNode.make c elts] returns the MDNode containing the values [elts] in
      the context [c]. See the method [llvm::MDNode::get]. *)
  val make : Context.t -> Value.c array -> c

  (** [MDNode.named_metadata m name] return all the MDNodes belonging to the named
      metadata (if any). See the method [llvm::NamedMDNode::getOperand]. *)
  val named_metadata : Module.t -> string -> c array
end

module MDString : sig
  type t
  class c : t -> object
    inherit Value.c

    (** [v#get] returns the string contents of the MDString [v].
        See the method [llvm::MDString::getString] *)
    method get : string option
  end

  (** [MDString.make c s] returns the MDString of the string [s] in the context
      [c]. See the method [llvm::MDString::get]. *)
  val make : Context.t -> string -> c
end


(** {6 LLVM GlobalValues (global variables, aliases, and functions)} *)

module GlobalValue : sig
  type t

  (** The linkage of a global value, accessed with [gv#linkage] and
      [gv#set_linkage]. See [llvm::GlobalValue::LinkageTypes]. *)
  type linkage =
    | External
    | Available_externally
    | Link_once
    | Link_once_odr
    | Weak
    | Weak_odr
    | Appending
    | Internal
    | Private
    | Dllimport
    | Dllexport
    | External_weak
    | Ghost
    | Common
    | Linker_private

  (** The linker visibility of a global value, accessed with [gv#visibility]
      and [gv#set_visibility]. See [llvm::GlobalValue::VisibilityTypes]. *)
  type visibility =
    | Default
    | Hidden
    | Protected

  class c : t -> object
    inherit Constant.c

    (** [g#parent] is the enclosing module of the global value [g].
        See the method [llvm::GlobalValue::getParent]. *)
    method parent : Module.t

    (** [g#is_declaration] returns [true] if the global value [g] is a declaration
        only. Returns [false] otherwise.
        See the method [llvm::GlobalValue::isDeclaration]. *)
    method is_declaration : bool

    (** [g#linkage] returns the linkage of the global value [g].
        See the method [llvm::GlobalValue::getLinkage]. *)
    method linkage : linkage

    (** [g#set_linkage l] sets the linkage of the global value [g] to [l].
        See the method [llvm::GlobalValue::setLinkage]. *)
    method set_linkage : linkage -> unit

    (** [g#section] returns the linker section of the global value [g].
        See the method [llvm::GlobalValue::getSection]. *)
    method section : string

    (** [g#set_section s] sets the linker section of the global value [g] to [s].
        See the method [llvm::GlobalValue::setSection]. *)
    method set_section : string -> unit

    (** [g#visibility] returns the linker visibility of the global value [g].
        See the method [llvm::GlobalValue::getVisibility]. *)
    method visibility : visibility

    (** [g#set_visibility v] sets the linker visibility of the global value [g] to
        [v]. See the method [llvm::GlobalValue::setVisibility]. *)
    method set_visibility : visibility -> unit

    (** [g#alignment] returns the required alignment of the global value [g].
        See the method [llvm::GlobalValue::getAlignment]. *)
    method alignment : int

    (** [g#set_alignment g] sets the required alignment of the global value [g] to
        [n] bytes. See the method [llvm::GlobalValue::setAlignment]. *)
    method set_alignment : int -> unit
  end

  (** [GlobalValue.test v] checks whether the {Value.c} [v] can be safely
      downcast to [GlobalValue.c]. *)
  val test : Value.c -> bool

  (** [GlobalValue.from v] performs a checked downcast of the {Value.c} [v]
      to [GlobalValue.c]. *)
  val from : Value.c -> c
end


module GlobalVariable : sig
  type t
  class c : t -> object
    inherit GlobalValue.c

    (** [gv#delete] destroys the global variable [gv].
        See the method [llvm::GlobalVariable::eraseFromParent]. *)
    method delete : unit

    (** [gv#is_global_constant] returns [true] if the global variable [gv] is a
        constant. Returns [false] otherwise.
        See the method [llvm::GlobalVariable::isConstant]. *)
    method is_constant : bool

    (** [gv#set_global_constant c] sets the global variable [gv] to be a constant if
        [c] is [true] and not if [c] is [false].
        See the method [llvm::GlobalVariable::setConstant]. *)
    method set_constant : bool -> unit

    (** [gv#get_initializer] returns the initializer for the global variable
        [gv]. See the method [llvm::GlobalVariable::getInitializer]. *)
    method get_initializer : Constant.c

    (** [gv#set_initializer c] sets the initializer for the global variable
        [gv] to the constant [c].
        See the method [llvm::GlobalVariable::setInitializer]. *)
    method set_initializer : Constant.c -> unit

    (** [gv#remove_initializer] unsets the initializer for the global variable
        [gv].
        See the method [llvm::GlobalVariable::setInitializer]. *)
    method remove_initializer : unit

    (** [gv#is_thread_local] returns [true] if the global variable [gv] is
        thread-local and [false] otherwise.
        See the method [llvm::GlobalVariable::isThreadLocal]. *)
    method is_thread_local : bool

    (** [gv#set_thread_local c] sets the global variable [gv] to be thread local if
        [c] is [true] and not otherwise.
        See the method [llvm::GlobalVariable::setThreadLocal]. *)
    method set_thread_local : bool -> unit
  end

  (** [GlobalVariable.declare ty name m] returns a new global variable of type [ty] and
      with name [name] in module [m] in the address space [addrspace] (default
      0). If such a global variable already exists, it is returned. If the type
      of the existing global differs, then a bitcast to [ty] is returned. *)
  val declare : ?addrspace:int -> Module.t -> Type.c -> name:string -> c

  (** [GlobalVariable.define name init m] returns a new global with name [name] and
      initializer [init] in module [m] in the address space [addrspace]
      (default 0). If the named global already exists, it is renamed. See the
      constructor of [llvm::GlobalVariable]. *)
  val define : ?addrspace:int -> Module.t -> name:string ->
               init:Constant.c -> c

  (** [GlobalVariable.lookup name m] returns [Some g] if a global variable with name
      [name] exists in module [m]. If no such global exists, returns [None].
      See the [llvm::GlobalVariable] constructor. *)
  val lookup : Module.t -> name:string -> c option

  (** [GlobalVariable.test v] checks whether the {Value.c} [v] can be safely
      downcast to [GlobalVariable.c]. *)
  val test : Value.c -> bool

  (** [GlobalVariable.from v] performs a checked downcast of the {Value.c} [v]
      to [GlobalVariable.c]. *)
  val from : Value.c -> c

  (** Iterators over [GlobalVariable]s in a [Module] *)

  module Pos : sig
    type collection = Module.t
    type item = c
    val first : collection -> (collection, item) pos
    val last : collection -> (collection, item) rev_pos
    val succ : item -> (collection, item) pos
    val pred : item -> (collection, item) rev_pos
  end

  val iter       : Module.t -> (c -> unit) -> unit
  val rev_iter   : Module.t -> (c -> unit) -> unit
  val fold_left  : 'a -> Module.t -> ('a -> c -> 'a) -> 'a
  val fold_right : Module.t -> 'a -> (c -> 'a -> 'a) -> 'a
end


module GlobalAlias : sig
  type t
  class c : t -> object inherit GlobalValue.c end

  (** [GlobalAlias.make m t a n] inserts an alias in the module [m] with the
      type [t] and the aliasee [a] with the name [n]. See the constructor for
      [llvm::GlobalAlias]. *)
  val make : Module.t -> Type.c -> aliasee:Constant.c -> name:string -> c

  (** [GlobalAlias.test v] checks whether the {Value.c} [v] can be safely
      downcast to [GlobalAlias.c]. *)
  val test : Value.c -> bool

  (** [GlobalAlias.from v] performs a checked downcast of the {Value.c} [v]
      to [GlobalAlias.c]. *)
  val from : Value.c -> c
end


module Function : sig
  type t

  (** The following calling convention values may be accessed with
      [fn#call_conv] and [fn#set_call_conv]. Calling conventions are
      open-ended. *)
  module CallConv : sig
    val c : int             (** [c] is the C calling convention. *)
    val fast : int          (** [fast] is the calling convention to allow LLVM
                                maximum optimization opportunities. Use only with
                                internal linkage. *)
    val cold : int          (** [cold] is the calling convention for
                                callee-save. *)
    val x86_stdcall : int   (** [x86_stdcall] is the familiar stdcall calling
                                convention from C. *)
    val x86_fastcall : int  (** [x86_fastcall] is the familiar fastcall calling
                                convention from C. *)
  end

  class c : t -> object
    inherit GlobalValue.c

    (** [f#delete] destroys the function [f].
        See the method [llvm::Function::eraseFromParent]. *)
    method delete : unit

    (** [f#is_intrinsic] returns true if the function [f] is an intrinsic.
        See the method [llvm::Function::isIntrinsic]. *)
    method is_intrinsic : bool

    (** [f#call_conv] returns the calling convention of the function [f].
        See the method [llvm::Function::getCallingConv]. *)
    method call_conv : int

    (** [f#set_call_conv cc] sets the calling convention of the function
        [f] to the calling convention numbered [cc].
        See the method [llvm::Function::setCallingConv]. *)
    method set_call_conv : int -> unit

    (** [f#gc] returns [Some name] if the function [f] has a garbage
        collection algorithm specified and [None] otherwise.
        See the method [llvm::Function::getGC]. *)
    method gc : string option

    (** [f#set_gc gc] sets the collection algorithm for the function [f] to
        [gc]. See the method [llvm::Function::setGC]. *)
    method set_gc : string option -> unit

    (** [f#attrs] returns the function attribute for the function [f].
     * See the method [llvm::Function::getAttributes] *)
    method attrs : Attribute.t list

    (** [f#add_attr f a] adds attribute [a] to the return type of function
        [f]. *)
    method add_attr : Attribute.t -> unit

    (** [f#remove_attr a] removes attribute [a] from the return type of
        function [f]. *)
    method remove_attr : Attribute.t -> unit
  end

  (** [Function.declare name ty m] returns a new function of type [ty] and
      with name [name] in module [m]. If such a function already exists,
      it is returned. If the type of the existing function differs, then a bitcast
      to [ty] is returned. *)
  val declare : Module.t -> name:string -> Type.c -> c

  (** [Function.define name ty m] creates a new function with name [name] and
      type [ty] in module [m]. If the named function already exists, it is
      renamed. An entry basic block is created in the function.
      See the constructor of [llvm::GlobalVariable]. *)
  val define : Module.t -> name:string -> Type.c -> c

  (** [Function.lookup name m] returns [Some f] if a function with name
      [name] exists in module [m]. If no such function exists, returns [None].
      See the method [llvm::Module] constructor. *)
  val lookup : Module.t -> name:string -> c option

  (** [Function.test v] checks whether the {Value.c} [v] can be safely
      downcast to [Function.c]. *)
  val test : Value.c -> bool

  (** [Function.from v] performs a checked downcast of the {Value.c} [v]
      to [Function.c]. *)
  val from : Value.c -> c

  (** Iterators over [Function]s in a [Module] *)

  module Pos : sig
    type collection = Module.t
    type item = c
    val first : collection -> (collection, item) pos
    val last : collection -> (collection, item) rev_pos
    val succ : item -> (collection, item) pos
    val pred : item -> (collection, item) rev_pos
  end

  val iter       : Module.t -> (c -> unit) -> unit
  val rev_iter   : Module.t -> (c -> unit) -> unit
  val fold_left  : 'a -> Module.t -> ('a -> c -> 'a) -> 'a
  val fold_right : Module.t -> 'a -> (c -> 'a -> 'a) -> 'a
end


(** {7 Operations on parameters} *)

module Argument : sig
  type t
  class c : t -> object
    inherit Value.c

    (** [p#parent] returns the parent function that owns the parameter.
        See the method [llvm::Argument::getParent]. *)
    method parent : Function.c

    (** [p#attrs] returns the attributes of parameter [p].
        See the methods [llvm::Function::getAttributes] and
        [llvm::Attributes::getParamAttributes] *)
    method attrs : Attribute.t list

    (** [p#add_attr a] adds attribute [a] to parameter [p]. *)
    method add_attr : Attribute.t -> unit

    (** [p#remove_attr a] removes attribute [a] from parameter [p]. *)
    method remove_attr : Attribute.t -> unit

    (** [p#set_alignment a] set the alignment of parameter [p] to [a]. *)
    method set_alignment : int -> unit
  end

  (** [Argument.params f] returns the parameters of function [f].
      See the method [llvm::Function::getArgumentList]. *)
  val params : Function.c -> c array

  (** [Argument.param f n] returns the [n]th parameter of function [f].
      See the method [llvm::Function::getArgumentList]. *)
  val param : Function.c -> int -> c

  (** [Argument.test v] checks whether the {Value.c} [v] can be safely
      downcast to [Argument.c]. *)
  val test : Value.c -> bool

  (** [Argument.from v] performs a checked downcast of the {Value.c} [v]
      to [Argument.c]. *)
  val from : Value.c -> c

  (** Iterators over [Argument]s in a [Function] *)

  module Pos : sig
    type collection = Function.c
    type item = c
    val first : collection -> (collection, item) pos
    val last : collection -> (collection, item) rev_pos
    val succ : item -> (collection, item) pos
    val pred : item -> (collection, item) rev_pos
  end

  val iter       : Function.c -> (c -> unit) -> unit
  val rev_iter   : Function.c -> (c -> unit) -> unit
  val fold_left  : 'a -> Function.c -> ('a -> c -> 'a) -> 'a
  val fold_right : Function.c -> 'a -> (c -> 'a -> 'a) -> 'a
end


(** {7 Operations on basic blocks} *)

module BasicBlock : sig
  type t
  class c : t -> object
    inherit Value.c

    (** [bb#parent] returns the parent function that owns the basic block.
        See the method [llvm::BasicBlock::getParent]. *)
    method parent : Function.c

    (** [bb#delete] deletes the basic block [bb].
        See the method [llvm::BasicBlock::eraseFromParent]. *)
    method delete : unit
  end

  (** [BasicBlock.blocks fn] returns the basic blocks of the function [f].
      See the method [llvm::Function::getBasicBlockList]. *)
  val blocks : Function.c -> c array

  (** [BasicBlock.entry fn] returns the entry basic block of the function [f].
      See the method [llvm::Function::getEntryBlock]. *)
  val entry : Function.c -> c

  (** [BasicBlock.append c name f] creates a new basic block named [name] at the end of
      function [f] in the context [c].
      See the constructor of [llvm::BasicBlock]. *)
  val append : Function.c -> Context.t -> name:string -> c

  (** [BasicBlock.insert c name bb] creates a new basic block named [name] before the
      basic block [bb] in the context [c].
      See the constructor of [llvm::BasicBlock]. *)
  val insert : before:c -> Context.t -> name:string -> c

  (** [BasicBlock.test v] checks whether the {Value.c} [v] can be safely
      downcast to [BasicBlock.c]. *)
  val test : Value.c -> bool

  (** [BasicBlock.from v] performs a checked downcast of the {Value.c} [v]
      to [BasicBlock.c]. *)
  val from : Value.c -> c

  (** Iterators over [BasicBlock]s in a [Function] *)

  module Pos : sig
    type collection = Function.c
    type item = c
    val first : collection -> (collection, item) pos
    val last : collection -> (collection, item) rev_pos
    val succ : item -> (collection, item) pos
    val pred : item -> (collection, item) rev_pos
  end

  val iter       : Function.c -> (c -> unit) -> unit
  val rev_iter   : Function.c -> (c -> unit) -> unit
  val fold_left  : 'a -> Function.c -> ('a -> c -> 'a) -> 'a
  val fold_right : Function.c -> 'a -> (c -> 'a -> 'a) -> 'a
end

module BlockAddress : sig
  type t
  class c : t -> object inherit Constant.c end

  (** [block_address f bb] returns the address of the basic block [bb] in the
      function [f]. See the method [llvm::BasicBlock::get]. *)
  val make : Function.c -> BasicBlock.c -> c

  (** [BlockAddress.test v] checks whether the {Value.c} [v] can be safely
      downcast to [BlockAddress.c]. *)
  val test : Value.c -> bool

  (** [BlockAddress.from v] performs a checked downcast of the {Value.c} [v]
      to [BlockAddress.c]. *)
  val from : Value.c -> c
end


(** {6 LLVM Instructions} *)

module Instruction : sig
  type t
  class c : t -> object
    inherit User.c

    method opcode : Opcode.t

    method icmp_predicate : Predicate.icmp option

    (** [i#delete] deletes the instruction [i].
        See the method [llvm::Instruction::eraseFromParent]. *)
    method delete : unit

    (** [#instr_parent] is the enclosing basic block of the instruction [i].
        See the method [llvm::Instruction::getParent]. *)
    method parent : BasicBlock.c

    (** [#has_metadata] returns whether or not the instruction [i] has any
        metadata attached to it. See the function
        [llvm::Instruction::hasMetadata]. *)
    method has_metadata : bool

    (** [i#metadata kind] optionally returns the metadata associated with the
        kind [kind] in the instruction [i] See the function
        [llvm::Instruction::getMetadata]. *)
    method metadata : kind:int -> MDNode.c option

    (** [i#set_metadata kind md] sets the metadata [md] of kind [kind] in the
        instruction [i]. See the function [llvm::Instruction::setMetadata]. *)
    method set_metadata : kind:int -> MDNode.c -> unit

    (** [i#clear_metadata kind] clears the metadata of kind [kind] in the
        instruction [i]. See the function [llvm::Instruction::setMetadata]. *)
    method clear_metadata : kind:int -> unit
  end

  (** [Instruction.test v] checks whether the {Value.c} [v] can be safely
      downcast to [Instruction.c]. *)
  val test : Value.c -> bool

  (** [Instruction.from v] performs a checked downcast of the {Value.c} [v]
      to [Instruction.c]. *)
  val from : Value.c -> c

  (** Iterators over [Instruction]s in a [BasicBlock] *)

  module Pos : sig
    type collection = BasicBlock.c
    type item = c
    val first : collection -> (collection, item) pos
    val last : collection -> (collection, item) rev_pos
    val succ : item -> (collection, item) pos
    val pred : item -> (collection, item) rev_pos
  end

  val iter       : BasicBlock.c -> (c -> unit) -> unit
  val rev_iter   : BasicBlock.c -> (c -> unit) -> unit
  val fold_left  : 'a -> BasicBlock.c -> ('a -> c -> 'a) -> 'a
  val fold_right : BasicBlock.c -> 'a -> (c -> 'a -> 'a) -> 'a
end

module TerminatorInst : sig
  type t
  class c : t -> object inherit Instruction.c end

  val block_terminator : BasicBlock.c -> c option

  (** [TerminatorInst.test v] checks whether the {Value.c} [v] can be safely
      downcast to [TerminatorInst.c]. *)
  val test : Value.c -> bool

  (** [TerminatorInst.from v] performs a checked downcast of the {Value.c} [v]
      to [TerminatorInst.c]. *)
  val from : Value.c -> c
end


(** {7 Operations on call sites} *)

module CallInst : sig
  type t
  class c : t -> object
    inherit Instruction.c

    (** [ci#call_conv] is the calling convention for the call instruction [ci],
        which may be one of the values from the module {!Function.CallConv}.
        See the method [llvm::CallInst::getCallingConv]. *)
    method call_conv : int

    (** [ci#set_call_conv cc] sets the calling convention for the call
        instruction [ci] to the integer [cc], which can be one of the values
        from the module {!Function.CallConv}. See the method
        [llvm::CallInst::setCallingConv]. *)
    method set_call_conv : int -> unit

    (** [ci#add_param_attr i a] adds attribute [a] to the [i]th parameter of
        the call instruction [ci]. [i]=0 denotes the return value. *)
    method add_param_attr : int -> Attribute.t -> unit

    (** [ci#remove_param_attr i a] removes attribute [a] from the [i]th
        parameter of the call instruction [ci]. [i]=0 denotes the return value. *)
    method remove_param_attr : int -> Attribute.t -> unit

    (** [ci#is_tail_call] is [true] if the call instruction [ci] is flagged as
        eligible for tail call optimization, [false] otherwise. See the method
        [llvm::CallInst::isTailCall]. *)
    method is_tail_call : bool

    (** [ci#set_tail_call tc] flags the call instruction [ci] as eligible for
        tail call optimization if [tc] is [true], clears otherwise. See the
        method [llvm::CallInst::setTailCall]. *)
    method set_tail_call : bool -> unit
  end

  (** [CallInst.test v] checks whether the {Value.c} [v] can be safely
      downcast to [CallInst.c]. *)
  val test : Value.c -> bool

  (** [CallInst.from v] performs a checked downcast of the {Value.c} [v]
      to [CallInst.c]. *)
  val from : Value.c -> c
end

module InvokeInst : sig
  type t
  class c : t -> object
    inherit Instruction.c

    (** [ci#call_conv] is the calling convention for the call or invoke
        instruction [ci], which may be one of the values from the module
        {!Function.CallConv}. See the method [llvm::CallInst::getCallingConv]
        and [llvm::InvokeInst::getCallingConv]. *)
    method call_conv : int

    (** [ci#set_call_conv cc] sets the calling convention for the call
        or invoke instruction [ci] to the integer [cc], which can be one of the
        values from the module {!Function.CallConv}.
        See the method [llvm::CallInst::setCallingConv]
        and [llvm::InvokeInst::setCallingConv]. *)
    method set_call_conv : int -> unit

    (** [ci#add_param_attr i a] adds attribute [a] to the [i]th
        parameter of the call or invoke instruction [ci]. [i]=0 denotes the return
        value. *)
    method add_param_attr : int -> Attribute.t -> unit

    (** [ci#remove_param_attr i a] removes attribute [a] from the
        [i]th parameter of the call or invoke instruction [ci]. [i]=0 denotes the
        return value. *)
    method remove_param_attr : int -> Attribute.t -> unit
  end

  (** [InvokeInst.test v] checks whether the {Value.c} [v] can be safely
      downcast to [InvokeInst.c]. *)
  val test : Value.c -> bool

  (** [InvokeInst.from v] performs a checked downcast of the {Value.c} [v]
      to [InvokeInst.c]. *)
  val from : Value.c -> c
end


(** {7 Operations on phi nodes} *)

module PHINode : sig
  type t
  class c : t -> object
    inherit Instruction.c

    (** [ph#incoming] returns the list of value-block pairs for phi node [pn].
        See the method [llvm::PHINode::getIncomingValue]. *)
    method incoming : (Value.c * BasicBlock.c) list

    (** [pn#add_incoming (v, bb)] adds the value [v] to the phi node [pn] for use
        with branches from [bb]. See the method [llvm::PHINode::addIncoming]. *)
    method add_incoming : (Value.c * BasicBlock.c) -> unit
  end

  (** [PHINode.test v] checks whether the {Value.c} [v] can be safely
      downcast to [PHINode.c]. *)
  val test : Value.c -> bool

  (** [PHINode.from v] performs a checked downcast of the {Value.c} [v]
      to [PHINode.c]. *)
  val from : Value.c -> c
end


(** {6 Instruction Builders} *)

(** Used to generate instructions in the LLVM IR. See the [llvm::LLVMBuilder]
    class. *)
module Builder : sig
  type t

  val create : Context.t -> t

  (** [builder_at ip] creates an instruction builder positioned at [ip].
      See the constructor for [llvm::LLVMBuilder]. *)
  val at : Context.t -> (BasicBlock.c, Instruction.c) pos -> t

  (** [builder_before ins] creates an instruction builder positioned before the
      instruction [isn]. See the constructor for [llvm::LLVMBuilder]. *)
  val before : Context.t -> Instruction.c -> t

  (** [builder_at_end bb] creates an instruction builder positioned at the end of
      the basic block [bb]. See the constructor for [llvm::LLVMBuilder]. *)
  val at_end : Context.t -> BasicBlock.c -> t


  (** [position_at ip bb] moves the instruction builder [bb] to the position
      [ip].
      See the constructor for [llvm::LLVMBuilder]. *)
  val position_at : t -> (BasicBlock.c, Instruction.c) pos -> unit

  (** [position_before ins b] moves the instruction builder [b] to before the
      instruction [isn]. See the method [llvm::LLVMBuilder::SetInsertPoint]. *)
  val position_before : t -> Instruction.c -> unit

  (** [position_at_end bb b] moves the instruction builder [b] to the end of the
      basic block [bb]. See the method [llvm::LLVMBuilder::SetInsertPoint]. *)
  val position_at_end : t -> BasicBlock.c -> unit


  (** [insertion_block b] returns the basic block that the builder [b] is
      positioned to insert into. Raises [Not_Found] if the instruction builder is
      uninitialized.
      See the method [llvm::LLVMBuilder::GetInsertBlock]. *)
  val insertion_block : t -> BasicBlock.c

  (** [insert_into_builder i name b] inserts the specified instruction [i] at the
      position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::Insert]. *)
  val insert_into_builder : t -> Instruction.c -> string -> unit


  (** {7 Metadata} *)

  (** [current_debug_location b] returns the current debug location, or None
      if none is currently set.
      See the method [llvm::IRBuilder::GetDebugLocation]. *)
  val debug_location : t -> MDNode.c option

  (*(** [set_current_debug_location b md] sets the current debug location [md] in
      the builder [b].
      See the method [llvm::IRBuilder::SetDebugLocation]. *)
  val set_current_debug_location : t -> MDNode.c -> unit

  (** [clear_current_debug_location b] clears the current debug location in the
      builder [b]. *)
  val clear_current_debug_location : t -> unit*)

  (** [set_inst_debug_location b i] sets the current debug location of the builder
      [b] to the instruction [i].
      See the method [llvm::IRBuilder::SetInstDebugLocation]. *)
  val set_inst_debug_location : t -> Value.c -> unit


  (** {7 Terminators} *)

  (** [ret_void b] creates a
      [ret void]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateRetVoid]. *)
  val ret_void : t -> Instruction.c

  (** [ret v b] creates a
      [ret %v]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateRet]. *)
  val ret : t -> Value.c -> Instruction.c

  (** [aggregate_ret vs b] creates a
      [ret {...} { %v1, %v2, ... } ]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateAggregateRet]. *)
  val aggregate_ret : t -> Value.c array -> Instruction.c


  (** [br bb b] creates a
      [br %bb]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateBr]. *)
  val br : t -> BasicBlock.c -> Instruction.c

  (** [cond_br cond tbb fbb b] creates a
      [br %cond, %tbb, %fbb]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateCondBr]. *)
  val cond_br : t -> Value.c -> BasicBlock.c -> BasicBlock.c -> Instruction.c

  (** [switch case elsebb count b] creates an empty
      [switch %case, %elsebb]
      instruction at the position specified by the instruction builder [b] with
      space reserved for [count] cases.
      See the method [llvm::LLVMBuilder::CreateSwitch]. *)
  val switch : t -> Value.c -> BasicBlock.c -> int -> Instruction.c

  (*(** [malloc ty name b] creates an [malloc]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::CallInst::CreateMalloc]. *)
  val malloc : t -> Type.c -> string -> Instruction.c

  (** [array_malloc ty val name b] creates an [array malloc]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::CallInst::CreateArrayMalloc]. *)
  val array_malloc : t -> Type.c -> Value.c -> string -> t -> Instruction.c*)

  (*(** [free p b] creates a [free]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::CallInst::CreateFree]. *)
  val free : t -> Value.c -> Instruction.c*)

  (* TODO doesnt belong here *)
  (** [add_case sw onval bb] causes switch instruction [sw] to branch to [bb]
      when its input matches the constant [onval].
      See the method [llvm::SwitchInst::addCase]. **)
  (*val add_case : Value.c -> Value.c -> BasicBlock.c -> unit*)

  (* TODO doesnt belong here *)
  (** [switch_default_dest sw] returns the default destination of the [switch]
   * instruction.
   * See the method [llvm:;SwitchInst::getDefaultDest]. **)
  (*val switch_default_dest : Value.c -> BasicBlock.c*)

  (** [indirect_br addr count b] creates a
      [indirectbr %addr]
      instruction at the position specified by the instruction builder [b] with
      space reserved for [count] destinations.
      See the method [llvm::LLVMBuilder::CreateIndirectBr]. *)
  val indirect_br : t -> Value.c -> int -> Instruction.c

  (* TODO doesnt belong here *)
  (** [add_destination br bb] adds the basic block [bb] as a possible branch
      location for the indirectbr instruction [br].
      See the method [llvm::IndirectBrInst::addDestination]. **)
  (*val add_destination : Value.c -> BasicBlock.c -> unit*)


  (** [invoke fn args tobb unwindbb name b] creates an
      [%name = invoke %fn(args) to %tobb unwind %unwindbb]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateInvoke]. *)
  val invoke : t -> Value.c -> Value.c array -> BasicBlock.c ->
               BasicBlock.c -> string -> Instruction.c

  (* TODO *)
  (** [landingpad ty persfn numclauses name b] creates an
      [landingpad]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateLandingPad]. *)
  (*val landingpad : Type.c -> Value.c -> int -> string -> t ->
                           Value.c*)

  (*(** [set_cleanup lp] sets the cleanup flag in the [landingpad]instruction.
      See the method [llvm::LandingPadInst::setCleanup]. *)
  val set_cleanup : Value.c -> bool -> unit

  (** [add_clause lp clause] adds the clause to the [landingpad]instruction.
      See the method [llvm::LandingPadInst::addClause]. *)
  val add_clause : Value.c -> Value.c -> unit*)

  (* TODO 
  (* [resume exn b] builds a [resume exn] instruction
   * at the position specified by the instruction builder [b].
   * See the method [llvm::LLVMBuilder::CreateResume] *)
  val resume : Value.c -> t -> Value.c *)

  (** [unreachable b] creates an
      [unreachable]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateUnwind]. *)
  val unreachable : t -> Instruction.c


  (** {7 Unary, binary, and cast instructions} *)

  val unary : t -> Operator.unary -> Value.c -> string -> Value.c
  val binary : t -> Operator.binary -> lhs:Value.c -> rhs:Value.c ->
               string -> Value.c
  val cast : t -> Operator.cast -> Value.c -> Type.c -> string -> Value.c

  (** {7 Memory} *)

  (** [alloca ty name b] creates a
      [%name = alloca %ty]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateAlloca]. *)
  val alloca : t -> Type.c -> string -> Instruction.c


  (** [array_alloca ty n name b] creates a
      [%name = alloca %ty, %n]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateAlloca]. *)
  val array_alloca : t -> Type.c -> Value.c -> string -> Instruction.c

  (** [load v name b] creates a
      [%name = load %v]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateLoad]. *)
  val load : t -> Value.c -> string -> Instruction.c


  (** [store v p b] creates a
      [store %v, %p]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateStore]. *)
  val store : t -> Value.c -> Value.c -> Instruction.c


  (** [gep p indices name b] creates a
      [%name = getelementptr %p, indices...]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateGetElementPtr]. *)
  val gep : ?inbounds:bool -> t -> Value.c -> Value.c array -> string -> Value.c

  (** [struct_gep p idx name b] creates a
      [%name = getelementptr %p, 0, idx]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateStructGetElementPtr]. *)
  val struct_gep : t -> Value.c -> int -> string -> Value.c

  (** [global_string str name b] creates a series of instructions that adds
      a global string at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateGlobalString]. *)
  val global_string : t -> string -> string -> Value.c


  (** [global_stringptr str name b] creates a series of instructions that
      adds a global string pointer at the position specified by the instruction
      builder [b].
      See the method [llvm::LLVMBuilder::CreateGlobalStringPtr]. *)
  val global_stringptr : t -> string -> string -> Value.c



  (** {7 Comparisons} *)

  (** [icmp pred x y name b] creates a
      [%name = icmp %pred %x, %y]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateICmp]. *)
  val icmp : t -> Predicate.icmp -> lhs:Value.c -> rhs:Value.c -> string -> Value.c

  (** [fcmp pred x y name b] creates a
      [%name = fcmp %pred %x, %y]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateFCmp]. *)
  val fcmp : t -> Predicate.fcmp -> lhs:Value.c -> rhs:Value.c -> string -> Value.c


  (** {7 Miscellaneous instructions} *)

  (** [phi incoming name b] creates a
      [%name = phi %incoming]
      instruction at the position specified by the instruction builder [b].
      [incoming] is a list of [(Value.c, BasicBlock.c)] tuples.
      See the method [llvm::LLVMBuilder::CreatePHI]. *)
  val phi : t -> (Value.c * BasicBlock.c) list -> string -> Value.c

  (** [call fn args name b] creates a
      [%name = call %fn(args...)]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateCall]. *)
  val call : t -> Value.c -> Value.c array -> string -> Value.c


  (** [select cond thenv elsev name b] creates a
      [%name = select %cond, %thenv, %elsev]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateSelect]. *)
  val select : t -> Value.c -> Value.c -> Value.c -> string -> Value.c

  (** [va_arg valist argty name b] creates a
      [%name = va_arg %valist, %argty]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateVAArg]. *)
  val va_arg : t -> Value.c -> Type.c -> string -> Value.c


  (** [extractelement vec i name b] creates a
      [%name = extractelement %vec, %i]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateExtractElement]. *)
  val extractelement : t -> vec:Value.c -> idx:Value.c -> string -> Value.c

  (** [insertelement vec elt i name b] creates a
      [%name = insertelement %vec, %elt, %i]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateInsertElement]. *)
  val insertelement : t -> vec:Value.c -> elt:Value.c -> idx:Value.c -> string -> Value.c

  (** [shufflevector veca vecb mask name b] creates a
      [%name = shufflevector %veca, %vecb, %mask]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateShuffleVector]. *)
  val shufflevector : t -> v1:Value.c -> v2:Value.c -> mask:Value.c -> string -> Value.c

  (** [insertvalue agg idx name b] creates a
      [%name = extractvalue %agg, %idx]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateExtractValue]. *)
  val extractvalue : t -> agg:Value.c -> idx:int -> string -> Value.c


  (** [insertvalue agg val idx name b] creates a
      [%name = insertvalue %agg, %val, %idx]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateInsertValue]. *)
  val insertvalue : t -> agg:Value.c -> elt:Value.c -> idx:int -> string -> Value.c

  (** [is_null val name b] creates a
      [%name = icmp eq %val, null]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateIsNull]. *)
  val is_null : t -> Value.c -> string -> Value.c


  (** [is_not_null val name b] creates a
      [%name = icmp ne %val, null]
      instruction at the position specified by the instruction builder [b].
      See the method [llvm::LLVMBuilder::CreateIsNotNull]. *)
  val is_not_null : t -> Value.c -> string -> Value.c


  (** [ptrdiff lhs rhs name b] creates a series of instructions that measure
      the difference between two pointer values at the position specified by the
      instruction builder [b].
      See the method [llvm::LLVMBuilder::CreatePtrDiff]. *)
  val ptrdiff : t -> Value.c -> Value.c -> string -> Value.c
end


(** {6 Memory Buffers} *)

(** Used to efficiently handle large buffers of read-only binary data.
    See the [llvm::MemoryBuffer] class. *)
module MemoryBuffer : sig
  type t

  (** [of_file p] is the memory buffer containing the contents of the file at
      path [p]. If the file could not be read, then [IoError msg] is
      raised. *)
  val of_file : string -> t
  
  (** [stdin ()] is the memory buffer containing the contents of standard input.
      If standard input is empty, then [IoError msg] is raised. *)
  val of_stdin : unit -> t
  
  (** Disposes of a memory buffer. *)
  val dispose : t -> unit
end


(** {6 Pass Managers} *)

(** Control of module and function pass pipelines *)
module PassManager : sig
  type 'a t
  type module_pass
  type function_pass
  
  (** [PassManager.create ()] constructs a new whole-module pass pipeline. This
      type of pipeline is suitable for link-time optimization and whole-module
      transformations.
      See the constructor of [llvm::PassManager]. *)
  val create : unit -> module_pass t
  
  (** [PassManager.create_function m] constructs a new function-by-function
      pass pipeline over the module [m]. It does not take ownership of [m].
      This type of pipeline is suitable for code generation and JIT compilation
      tasks.
      See the constructor of [llvm::FunctionPassManager]. *)
  val create_function : Module.t -> function_pass t 

  
  (** [run_module m pm] initializes, executes on the module [m], and finalizes
      all of the passes scheduled in the pass manager [pm]. Returns [true] if
      any of the passes modified the module, [false] otherwise.
      See the [llvm::PassManager::run] method. *)
  val run_module : module_pass t -> Module.t -> bool

  
  (** [initialize fpm] initializes all of the function passes scheduled in the
      function pass manager [fpm]. Returns [true] if any of the passes modified
      the module, [false] otherwise.
      See the [llvm::FunctionPassManager::doInitialization] method. *)
  val initialize : function_pass t -> bool
  
  (** [run_function f fpm] executes all of the function passes scheduled in the
      function pass manager [fpm] over the function [f]. Returns [true] if any
      of the passes modified [f], [false] otherwise.
      See the [llvm::FunctionPassManager::run] method. *)
  val run_function : function_pass t -> Function.c  -> bool

  
  (** [finalize fpm] finalizes all of the function passes scheduled in in the
      function pass manager [fpm]. Returns [true] if any of the passes
      modified the module, [false] otherwise.
      See the [llvm::FunctionPassManager::doFinalization] method. *)
  val finalize : function_pass t -> bool
  
  (** Frees the memory of a pass pipeline. For function pipelines, does not free
      the module.
      See the destructor of [llvm::BasePassManager]. *)
  val dispose : 'a t -> unit
end
