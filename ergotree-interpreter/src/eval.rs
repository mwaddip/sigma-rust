//! Interpreter
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Display;
use ergotree_ir::ergo_tree::ErgoTree;
use ergotree_ir::ergo_tree::ErgoTreeVersion;
use ergotree_ir::mir::constant::TryExtractInto;
use ergotree_ir::sigma_protocol::sigma_boolean::SigmaProp;
use snumeric::numeric_method_evalfn;

use ergotree_ir::mir::expr::Expr;
use ergotree_ir::mir::value::{Lambda, Value};
use ergotree_ir::sigma_protocol::sigma_boolean::SigmaBoolean;

use ergotree_ir::types::smethod::SMethod;
use ergotree_ir::types::stype::SType;

use self::env::Env;
use ergotree_ir::chain::context::Context;

/// Environment for
pub mod env;

pub(crate) mod and;
pub(crate) mod apply;
pub(crate) mod atleast;
pub(crate) mod bin_op;
pub(crate) mod block;
pub(crate) mod bool_to_sigma;
pub(crate) mod byte_array_to_bigint;
pub(crate) mod byte_array_to_long;
pub(crate) mod calc_blake2b256;
pub(crate) mod calc_sha256;
pub(crate) mod coll_append;
pub(crate) mod coll_by_index;
pub(crate) mod coll_exists;
pub(crate) mod coll_filter;
pub(crate) mod coll_fold;
pub(crate) mod coll_forall;
pub(crate) mod coll_map;
pub(crate) mod coll_shape;
pub(crate) mod coll_size;
pub(crate) mod coll_slice;
pub(crate) mod collection;
pub(crate) mod cost_accum;
pub(crate) mod create_prove_dh_tuple;
pub(crate) mod create_provedlog;
pub(crate) mod data_value_comparer;
pub(crate) mod decode_point;
mod deserialize_context;
mod deserialize_register;
pub(crate) mod downcast;
mod error;
pub(crate) mod exponentiate;
pub(crate) mod expr;
pub(crate) mod extract_amount;
pub(crate) mod extract_bytes;
pub(crate) mod extract_bytes_with_no_ref;
pub(crate) mod extract_creation_info;
pub(crate) mod extract_id;
pub(crate) mod extract_reg_as;
pub(crate) mod extract_script_bytes;
pub(crate) mod func_value;
pub(crate) mod get_var;
pub(crate) mod global_vars;
pub(crate) mod if_op;
pub(crate) mod logical_not;
pub(crate) mod long_to_byte_array;
pub(crate) mod method_call;
pub(crate) mod method_cost;
pub(crate) mod multiply_group;
pub(crate) mod negation;
pub(crate) mod option_get;
pub(crate) mod option_get_or_else;
pub(crate) mod option_is_defined;
pub(crate) mod or;
pub(crate) mod property_call;
pub(crate) mod runtime_type;
pub(crate) mod savltree;
pub(crate) mod sbox;
pub(crate) mod scoll;
pub(crate) mod scontext;
pub(crate) mod select_field;
pub(crate) mod sglobal;
pub(crate) mod sgroup_elem;
pub(crate) mod sheader;
pub(crate) mod sigma_and;
pub(crate) mod sigma_or;
pub(crate) mod sigma_prop_bytes;
pub(crate) mod snumeric;
pub(crate) mod soption;
pub(crate) mod spreheader;
pub(crate) mod subst_const;
pub(crate) mod tree_lookup;
pub(crate) mod tuple;
pub(crate) mod unevaluated;
pub(crate) mod upcast;
pub(crate) mod val_use;
pub(crate) mod xor;
pub(crate) mod xor_of;

pub use error::EvalError;

/// Diagnostic information about the reduction (pretty printed expr and/or env)
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct ReductionDiagnosticInfo {
    /// environment after the evaluation
    pub env: Env<'static>,
    /// expression pretty-printed
    pub pretty_printed_expr: Option<String>,
}

impl Display for ReductionDiagnosticInfo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(expr_str) = &self.pretty_printed_expr {
            writeln!(f, "Pretty printed expr:\n{}", expr_str)?;
        }
        write!(f, "Env:\n{}", self.env)
    }
}

/// Result of expression reduction procedure (see `reduce_to_crypto`).
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct ReductionResult {
    /// value of SigmaProp type which represents a statement verifiable via sigma protocol.
    pub sigma_prop: SigmaBoolean,
    /// estimated cost of expression evaluation
    pub cost: u64,
    /// Diagnostic information about the reduction (pretty printed expr and/or env)
    pub diag: ReductionDiagnosticInfo,
}

/// JIT cost for a script that trivially reduces to a SigmaProp constant (e.g.
/// bare P2PK). Scala's `EvalSigmaPropConstant` charges 50 JitCost.
const EVAL_SIGMA_PROP_CONSTANT: u64 = 50;

/// `AddToEnvironmentDesc` cost (Scala `values.scala`): charged once per lambda-arg
/// binding — i.e. once per invocation of a collection HOF's lambda.
pub(crate) const ADD_TO_ENV_COST: u64 = 5;

/// Short-circuit for trees whose proposition is a plain SigmaProp constant.
/// Returns `Some(sigma_bool)` for both forms:
///
/// * non-segregated P2PK where the root is already `Expr::Const(SSigmaProp)`,
/// * segregated P2PK where the root is `Expr::ConstPlaceholder` whose
///   SSigmaProp constant is resolved on the fly from `ctx.constants`
///   (lazy resolution).
///
/// Returns `None` when full evaluation is required.
fn trivial_reduce<'ctx>(expr: &Expr, ctx: &Context<'ctx>) -> Option<SigmaBoolean> {
    let constant = match expr {
        Expr::Const(c) if c.tpe == SType::SSigmaProp => c.clone(),
        Expr::ConstPlaceholder(cp) if cp.tpe == SType::SSigmaProp => ctx
            .constants
            .and_then(|cs| cs.get(cp.id as usize))
            .cloned()?,
        _ => return None,
    };
    constant
        .try_extract_into::<SigmaProp>()
        .ok()
        .map(|sp| sp.into())
}

/// Reject a self-box `ContextExtension` whose key set leaves the signed-`Byte`
/// range (any key `>= 0x80`), mirroring the JVM's `ErgoLikeContext.toSigmaContext`
/// (`ErgoLikeContext.scala:140-146`): it builds `contextVars` as
/// `new Array(maxKey + 1)` indexed by the SIGNED key, so a key `>= 0x80`
/// (signed-negative) crashes context construction — `NegativeArraySizeException`
/// (all keys negative ⇒ negative array size) or `ArrayIndexOutOfBoundsException`
/// (a negative index) — BEFORE any bytecode runs, and the spend never validates.
/// sigma-rust stores keys as unsigned `u8` and resolves them lazily at `GetVar`,
/// so without this guard it would ACCEPT a `>= 0x80` key where the JVM rejects —
/// a consensus fork on the attacker-supplied spending extension. The rejection is
/// on key PRESENCE (a `GetVar`-free tree still rejects), not on the read. This is
/// the top-level self extension only; the per-input `getVarFromInput` path masks
/// the id with `& 0xff` by design and is intentionally not affected.
fn validate_self_extension_key_domain(ctx: &Context) -> Result<(), EvalError> {
    if let Some(&bad_key) = ctx.extension.values.keys().find(|&&k| k > 0x7f) {
        return Err(EvalError::Misc(format!(
            "ContextExtension key {bad_key} is outside the signed-Byte range 0..=127 \
             that context construction accepts (>= 0x80 crashes the JVM vars array)"
        )));
    }
    Ok(())
}

/// sigmastate converts every self context variable as it builds the script context
/// (`ErgoLikeContext.scala:158-161`), which `CErgoTreeEvaluator.eval` does for every tree it
/// evaluates (`CErgoTreeEvaluator.scala:563`), whether a script reads the variable or not: a
/// variable that does not convert fails the reduction. A tree whose proposition is a
/// `SigmaProp` constant is not evaluated, so nothing is converted (`Interpreter.scala:211`).
fn convert_self_extension(ctx: &Context) -> Result<(), EvalError> {
    ctx.extension
        .values
        .values()
        .try_for_each(|value| value.to_script_value().map(|_| ()))
        .map_err(EvalError::from)
}

/// sigmastate's `fitsIn256Bits` (v6.0.6 `Extensions.scala:199`): `bitLength() <= 255`, the range
/// `-2^255 ..= 2^255 - 1` of a `BigInt` as data. A runtime `BigInt` can be wider below tree
/// version 3 (`CBigInt.scala:18-20`)
pub(crate) fn fits_256_bits(v: &num_bigint::BigInt) -> bool {
    match v.bits() {
        0..=255 => true,
        // a magnitude of 256 bits fits only as -2^255
        256 => v.sign() == num_bigint::Sign::Minus && v.trailing_zeros() == Some(255),
        _ => false,
    }
}

/// sigmastate's `Value.checkType` (v6.0.7 `values.scala:251-259`) through `SType.isValueOfType`
/// (`SType.scala:187-213`), run wherever sigmastate evaluates a node whose value its static type
/// does not guarantee. Shallow: only the type's root constructor is checked against the value, so
/// any collection passes `Coll[Int]`. A node's static type and its value can disagree: an `If` is
/// typed by its true branch and neither parser checks that the branches agree
/// (`trees.scala:1351`), so `if (false) 1 else 2L` is an `Int` node holding a `Long`.
pub(crate) fn check_value_of_type(
    tpe: &SType,
    value: &Value,
    tree_version: ErgoTreeVersion,
) -> Result<(), EvalError> {
    let is_value_of_type = match tpe {
        SType::SBoolean => matches!(value, Value::Boolean(_)),
        SType::SByte => matches!(value, Value::Byte(_)),
        SType::SShort => matches!(value, Value::Short(_)),
        SType::SInt => matches!(value, Value::Int(_)),
        SType::SLong => matches!(value, Value::Long(_)),
        SType::SBigInt => matches!(value, Value::BigInt(_)),
        SType::SUnsignedBigInt if tree_version >= ErgoTreeVersion::V3 => {
            matches!(value, Value::UnsignedBigInt(_))
        }
        SType::SGroupElement => matches!(value, Value::GroupElement(_)),
        SType::SSigmaProp => matches!(value, Value::SigmaProp(_)),
        SType::SBox => matches!(value, Value::CBox(_)),
        // a tuple of other than two items is a `Coll[Any]` in sigmastate (`TupleData`,
        // `package.scala:67`, `Evaluation.scala:99-102`) and here, so it passes as a collection
        SType::SColl(_) => matches!(value, Value::Coll(_)),
        SType::SOption(_) => matches!(value, Value::Opt(_)),
        SType::STuple(t) if t.items.len() == 2 => matches!(value, Value::Pair(_)),
        SType::STuple(_) => return Err(EvalError::Misc(format!("Unsupported tuple type {tpe:?}"))),
        SType::SFunc(f) if f.t_dom.len() == 1 => {
            matches!(value, Value::Lambda(l) if l.args.len() == 1)
        }
        SType::SFunc(_) => {
            return Err(EvalError::Misc(format!(
                "Unsupported function type {tpe:?}"
            )))
        }
        SType::SContext => matches!(value, Value::Context),
        SType::SAvlTree => matches!(value, Value::AvlTree(_)),
        SType::SGlobal => matches!(value, Value::Global),
        SType::SHeader => matches!(value, Value::Header(_)),
        SType::SPreHeader => matches!(value, Value::PreHeader(_)),
        SType::SUnit => matches!(value, Value::Unit),
        SType::SUnsignedBigInt
        | SType::SAny
        | SType::STypeVar(_)
        | SType::SString
        | SType::SNoType => return Err(EvalError::Misc(format!("Unknown type {tpe:?}"))),
    };
    if is_value_of_type {
        Ok(())
    } else {
        Err(EvalError::Misc(format!(
            "Invalid type returned by evaluator: expected type {tpe:?}, resulting value {value:?}"
        )))
    }
}

/// Evaluate the given expression by reducing it to SigmaBoolean value.
pub fn reduce_to_crypto(tree: &ErgoTree, ctx: &Context) -> Result<ReductionResult, EvalError> {
    // The JVM rejects a `>= 0x80` self-extension key at context construction,
    // before any bytecode; mirror that here at the reduction boundary.
    validate_self_extension_key_domain(ctx)?;
    // Track cost as a delta from the caller's accumulator state so the per-call cost
    // reported in ReductionResult stays meaningful while the ctx accumulator grows
    // cumulatively across repeated reduce_to_crypto invocations on the same Context
    // (required for per-tx jit_cost_limit enforcement — see tx_context::validate).
    fn inner<'ctx>(
        expr: &'ctx Expr,
        ctx: &Context<'ctx>,
        cost_before: u64,
    ) -> Result<ReductionResult, EvalError> {
        let mut env_mut = Env::empty();
        expr.eval(&mut env_mut, ctx)
            .and_then(|v| -> Result<ReductionResult, EvalError> {
                let cost = (ctx.jit_cost_value() - cost_before) / 10; // convert JitCost to block cost
                match v {
                    Value::Boolean(b) => Ok(ReductionResult {
                        sigma_prop: SigmaBoolean::TrivialProp(b),
                        cost,
                        diag: ReductionDiagnosticInfo {
                            env: env_mut.to_static(),
                            pretty_printed_expr: None,
                        },
                    }),
                    Value::SigmaProp(sp) => Ok(ReductionResult {
                        sigma_prop: sp.value().clone(),
                        cost,
                        diag: ReductionDiagnosticInfo {
                            env: env_mut.to_static(),
                            pretty_printed_expr: None,
                        },
                    }),
                    _ => Err(EvalError::InvalidResultType),
                }
            })
    }

    // Snapshot the caller's accumulator so the per-call cost returned in
    // ReductionResult stays a delta even as the outer ctx.jit_cost grows
    // cumulatively across repeated reduce_to_crypto invocations (required
    // for per-tx jit_cost_limit enforcement — see tx_context::validate).
    let cost_before = ctx.jit_cost_value();

    // The JVM interpreter wraps reduction in `withVersions(.., ergoTree.version)`;
    // mirror that by setting the eval context's tree version from the tree being
    // reduced. Otherwise version-gated ops (e.g. BigInt downcast, gated on
    // `tree_version >= V3`) see the `Cell`'s `V0` default and reject valid V3
    // trees the JVM accepts — a consensus divergence on any caller (the node)
    // that doesn't set it.
    let tree_version = tree.header()?.version();
    // From activated 2 the JVM's `VersionContext` refuses a tree version above the activated
    // one (`Interpreter.scala:207`, `VersionContext.scala:17-21`). `verify` and `prove` refuse
    // such a tree before they reduce it (`check_soft_fork_condition`); a reduction on its
    // own, as `ReducingInterpreter.reduce` makes, fails here. The JVM compares signed bytes:
    // a negative activated version is below 2 there, and reads as 0 here.
    let activated_version = ctx.activated_script_version();
    if activated_version >= ErgoTreeVersion::V2 && tree_version > activated_version {
        return Err(EvalError::TreeVersionAboveActivated {
            tree_version,
            activated_version: u8::from(activated_version) as i8,
        });
    }
    ctx.tree_version.set(tree_version);

    // Deserialize trees need an owned Expr for substitute_deserialize.
    // This is the rare path — most scripts don't have deserialize nodes.
    if tree.has_deserialize() {
        // The JVM charges the deserialize-substitution pass proportionally to
        // the tree's size: `ergoTree.bytes.length * CostPerTreeByte(2)`
        // block cost (`Interpreter.reductionWithDeserialize`, `Interpreter.scala:246`).
        // `ergoTree.bytes` is the bytes the tree was read from, whatever their
        // encoding (`ErgoTree.scala:123-130`), not the tree written again. The charge is
        // limit-checked in all eras; since V6 activation it is also included
        // in the reported cost, while pre-V6 the JVM excludes it from the
        // result (it passes the un-bumped context on) — mirror by rolling the
        // accumulator back after the limit check.
        let subst_cost_jit = tree.bytes()?.len() as u64 * 20; // 2 block × 10 jit/block
        ctx.add_jit_cost(subst_cost_jit)?;
        if ctx.activated_script_version() < ErgoTreeVersion::V3 {
            ctx.jit_cost
                .set(ctx.jit_cost.get().saturating_sub(subst_cost_jit));
        }
        let expr = tree.proposition()?;
        let expr = to_valid_script_type(expr.substitute_deserialize(ctx)?)?;
        // sigmastate evaluates the substituted tree whatever it is, a `SigmaProp` constant
        // included, building the script context first (`Interpreter.scala:171-177`)
        convert_self_extension(ctx)?;
        let res = inner(&expr, ctx, cost_before);
        return match res {
            Ok(reduction) if reduction.sigma_prop == SigmaBoolean::TrivialProp(false) => {
                let (_, printed_expr_str) = expr
                    .pretty_print()
                    .map_err(|e| EvalError::Misc(e.to_string()))?;
                Ok(ReductionResult {
                    sigma_prop: SigmaBoolean::TrivialProp(false),
                    cost: reduction.cost,
                    diag: ReductionDiagnosticInfo {
                        env: reduction.diag.env,
                        pretty_printed_expr: Some(printed_expr_str),
                    },
                })
            }
            Ok(reduction) => Ok(reduction),
            // A cost-limit error must NOT trigger the diagnostic retry — the
            // retry re-evaluates and re-charges, making the charged cost (and
            // near the budget even the verdict) path-dependent. `enrich_err`
            // wraps every error — including `CostError` — in `Spanned` at
            // each eval node boundary, so match through the wrappers.
            Err(e) if e.is_cost_error() => Err(e),
            Err(_) => {
                let (spanned_expr, printed_expr_str) = expr
                    .pretty_print()
                    .map_err(|e| EvalError::Misc(e.to_string()))?;
                ctx.jit_cost.set(cost_before);
                inner(&spanned_expr, ctx, cost_before)
                    .map_err(|e| e.wrap_spanned_with_src(printed_expr_str.to_string()))
            }
        };
    }

    // Common path: lazy constant resolution — no clone, no tree walk.
    // ConstPlaceholder nodes are resolved on-demand during evaluation
    // by looking up ctx.constants[placeholder.id].
    // NB: with_constants clones the Cell<u64> accumulator, so the cost
    // charged inside `inner` lives on `ctx_with_c` — we sync it back to
    // the caller's ctx after `inner` returns so the per-tx limit in
    // tx_context::validate sees the right running total.
    let root = tree.root_expr()?;
    let constants = tree.constants()?;
    let ctx_with_c = ctx.with_constants(constants);
    // Trivial short-circuit: plain SigmaProp constants (bare P2PK, both the
    // non-segregated Expr::Const(SSigmaProp) form and the segregated
    // Expr::ConstPlaceholder resolving to a SigmaProp via ctx.constants) are
    // priced at a flat 50 JitCost, matching Scala's EvalSigmaPropConstant.
    // Without this path, segregated P2PK pays only the 5 JitCost
    // ConstPlaceholder cost — a 10× undercharge on every P2PK input.
    if let Some(sigma_bool) = trivial_reduce(root, &ctx_with_c) {
        ctx.add_jit_cost(EVAL_SIGMA_PROP_CONSTANT)?;
        return Ok(ReductionResult {
            sigma_prop: sigma_bool,
            cost: (ctx.jit_cost_value() - cost_before) / 10,
            diag: ReductionDiagnosticInfo {
                env: Env::empty().to_static(),
                pretty_printed_expr: None,
            },
        });
    }
    convert_self_extension(&ctx_with_c)?;
    let res = inner(root, &ctx_with_c, cost_before);
    ctx.jit_cost.set(ctx_with_c.jit_cost_value());
    match res {
        Ok(reduction) if reduction.sigma_prop == SigmaBoolean::TrivialProp(false) => {
            // Diagnostic path: use proposition() for fully-resolved pretty-printing.
            // This clones, but only on the rare false-reduction diagnostic path.
            let resolved = tree.proposition()?;
            let (_, printed_expr_str) = resolved
                .pretty_print()
                .map_err(|e| EvalError::Misc(e.to_string()))?;
            Ok(ReductionResult {
                sigma_prop: SigmaBoolean::TrivialProp(false),
                cost: reduction.cost,
                diag: ReductionDiagnosticInfo {
                    env: reduction.diag.env,
                    pretty_printed_expr: Some(printed_expr_str),
                },
            })
        }
        Ok(reduction) => Ok(reduction),
        // A cost-limit error must NOT trigger the diagnostic retry: the retry
        // re-evaluates the constant-substituted `proposition()` tree, which
        // charges on a different lattice (substituted `Constant`s cost 5
        // JitCost where placeholders cost 1), making the charged cost — and
        // near the budget even the verdict — path-dependent. `enrich_err`
        // wraps every error — including `CostError` — in `Spanned` at each
        // eval node boundary, so match through the wrappers.
        Err(e) if e.is_cost_error() => Err(e),
        Err(_) => {
            // Error path: use proposition() for fully-resolved spanned re-evaluation.
            let resolved = tree.proposition()?;
            let (spanned_expr, printed_expr_str) = resolved
                .pretty_print()
                .map_err(|e| EvalError::Misc(e.to_string()))?;
            // Roll the accumulator back to the pre-reduce state so the diagnostic
            // retry doesn't double-count and can't spuriously trip jit_cost_limit.
            ctx.jit_cost.set(cost_before);
            inner(&spanned_expr, ctx, cost_before)
                .map_err(|e| e.wrap_spanned_with_src(printed_expr_str.to_string()))
        }
    }
}

/// sigmastate's `toValidScriptTypeJITC` after the substitution (v6.0.7 `Interpreter.scala:598-602`):
/// a Boolean root is wrapped in `BoolToSigmaProp`, a node evaluated and charged; a SigmaProp
/// root is kept; any other fails the spend. sigmastate's read of the root's type can throw (a
/// `Filter` over a non-collection); `tpe()` answers NoType, or a type built around it, where it
/// would, so such a root fails here too.
fn to_valid_script_type(expr: Expr) -> Result<Expr, EvalError> {
    match expr.tpe() {
        SType::SBoolean => Ok(ergotree_ir::mir::bool_to_sigma::BoolToSigmaProp {
            input: alloc::boxed::Box::new(expr),
        }
        .into()),
        SType::SSigmaProp => Ok(expr),
        tpe => Err(EvalError::Misc(format!(
            "Context-dependent pre-processing should produce tree of type Boolean or SigmaProp \
             but was {tpe:?}"
        ))),
    }
}

/// Expects SigmaProp constant value and returns it's value. Otherwise, returns an error.
pub fn extract_sigma_boolean(expr: &Expr) -> Result<SigmaBoolean, EvalError> {
    match expr {
        Expr::Const(c) => Ok(c.clone().try_extract_into::<SigmaProp>()?.into()),
        _ => Err(EvalError::InvalidResultType),
    }
}

/// Expression evaluation.
/// Should be implemented by every node that can be evaluated.
pub(crate) trait Evaluable {
    /// Evaluation routine to be implement by each node
    fn eval<'ctx>(
        &self,
        env: &mut Env<'ctx>,
        ctx: &Context<'ctx>,
        // JIT costing is handled via ctx.add_jit_cost()
    ) -> Result<Value<'ctx>, EvalError>;
}

/// Per-lambda invoker mirroring the JVM's closure semantics: `FuncValue.eval`
/// returns a closure over the *defining* environment, and each application
/// evaluates the body in that captured env extended with the argument
/// bindings (`env1 = env + (argId -> value)`) — the caller's environment at
/// application time plays no role.
///
/// The invoker starts from the captured bindings, shared. Its first argument
/// binding copies them while the lambda still holds them: once per invoker,
/// so once per `Apply` and per `Map`/`Filter`/… call, and again for an element
/// whose body keeps a closure over the invoker's bindings (the JVM's
/// persistent map extends in O(log n) instead). Each `invoke` overwrites the
/// argument slot(s), which matches Scala's per-call extension because an
/// argument binding always shadows a same-id captured binding.
pub(crate) struct LambdaInvoker<'l, 'ctx> {
    lambda: &'l Lambda<'ctx>,
    env: Env<'ctx>,
}

impl<'l, 'ctx> LambdaInvoker<'l, 'ctx> {
    pub(crate) fn new(lambda: &'l Lambda<'ctx>) -> Self {
        Self {
            lambda,
            env: Env::from_bindings(lambda.captured.clone()),
        }
    }

    /// Bind `args` positionally to the lambda's parameters and evaluate the
    /// body in the captured environment.
    pub(crate) fn invoke(
        &mut self,
        ctx: &Context<'ctx>,
        args: Vec<Value<'ctx>>,
    ) -> Result<Value<'ctx>, EvalError> {
        for (arg, v) in self.lambda.args.iter().zip(args) {
            // `FuncValue.eval`'s closure checks the argument against its type before it binds
            // it, and the body's result after (`values.scala:1074`, `:1080`)
            check_value_of_type(&arg.tpe, &v, ctx.tree_version())?;
            // ADD_TO_ENV_COST per argument binding — Scala charges
            // AddToEnvironment inside the closure on every invocation
            // (`FuncValue.eval`); previously each invocation site charged
            // this around its own env-insert dance.
            ctx.add_jit_cost(ADD_TO_ENV_COST)?;
            self.env.insert(arg.idx, v);
        }
        let res = self.lambda.body.eval(&mut self.env, ctx)?;
        check_value_of_type(&self.lambda.body.tpe(), &res, ctx.tree_version())?;
        Ok(res)
    }
}

/// A method call as its evaluator sees it: the (specialised) method and the call's argument
/// expressions. sigmastate's `_eval` handlers receive the `MethodCall` node itself
/// (`getOrElse_eval` reads `mc.args.last.tpe`, v6.0.7 `methods.scala:842`). It derefs to the
/// method, which is all most evaluators need.
pub(crate) struct MethodCallSite<'a> {
    pub(crate) method: &'a SMethod,
    /// the call's argument expressions; empty for a `PropertyCall`
    pub(crate) args: &'a [Expr],
}

impl core::ops::Deref for MethodCallSite<'_> {
    type Target = SMethod;
    fn deref(&self) -> &SMethod {
        self.method
    }
}

type EvalFn = for<'ctx> fn(
    mc: &MethodCallSite<'_>,
    env: &mut Env<'ctx>,
    ctx: &Context<'ctx>,
    Value<'ctx>,
    Vec<Value<'ctx>>,
) -> Result<Value<'ctx>, EvalError>;

fn smethod_eval_fn(method: &SMethod) -> Result<EvalFn, EvalError> {
    use ergotree_ir::types::*;
    Ok(match method.obj_type.type_code() {
        savltree::TYPE_CODE => match method.method_id() {
            savltree::DIGEST_METHOD_ID => self::savltree::DIGEST_EVAL_FN,
            savltree::UPDATE_DIGEST_METHOD_ID => self::savltree::UPDATE_DIGEST_EVAL_FN,
            savltree::ENABLED_OPERATIONS_METHOD_ID => self::savltree::ENABLED_OPERATIONS_EVAL_FN,
            savltree::KEY_LENGTH_METHOD_ID => self::savltree::KEY_LENGTH_EVAL_FN,
            savltree::VALUE_LENGTH_OPT_METHOD_ID => self::savltree::VALUE_LENGTH_OPT_EVAL_FN,
            savltree::IS_INSERT_ALLOWED_METHOD_ID => self::savltree::IS_INSERT_ALLOWED_EVAL_FN,
            savltree::IS_UPDATE_ALLOWED_METHOD_ID => self::savltree::IS_UPDATE_ALLOWED_EVAL_FN,
            savltree::IS_REMOVE_ALLOWED_METHOD_ID => self::savltree::IS_REMOVE_ALLOWED_EVAL_FN,
            savltree::UPDATE_OPERATIONS_METHOD_ID => self::savltree::UPDATE_OPERATIONS_EVAL_FN,
            savltree::GET_METHOD_ID => self::savltree::GET_EVAL_FN,
            savltree::GET_MANY_METHOD_ID => self::savltree::GET_MANY_EVAL_FN,
            savltree::INSERT_METHOD_ID => self::savltree::INSERT_EVAL_FN,
            savltree::CONTAINS_METHOD_ID => self::savltree::CONTAINS_EVAL_FN,
            savltree::REMOVE_METHOD_ID => self::savltree::REMOVE_EVAL_FN,
            savltree::UPDATE_METHOD_ID => self::savltree::UPDATE_EVAL_FN,
            savltree::INSERT_OR_UPDATE_METHOD_ID => self::savltree::INSERT_OR_UPDATE_EVAL_FN,
            method_id => {
                return Err(EvalError::NotFound(format!(
                    "Eval fn: unknown method id in SAvlTree: {:?}",
                    method_id
                )))
            }
        },
        scontext::TYPE_CODE => match method.method_id() {
            scontext::DATA_INPUTS_PROPERTY_METHOD_ID => self::scontext::DATA_INPUTS_EVAL_FN,
            scontext::SELF_BOX_INDEX_PROPERTY_METHOD_ID => self::scontext::SELF_BOX_INDEX_EVAL_FN,
            scontext::HEADERS_PROPERTY_METHOD_ID => self::scontext::HEADERS_EVAL_FN,
            scontext::PRE_HEADER_PROPERTY_METHOD_ID => self::scontext::PRE_HEADER_EVAL_FN,
            scontext::INPUTS_PROPERTY_METHOD_ID => self::scontext::INPUTS_EVAL_FN,
            scontext::OUTPUTS_PROPERTY_METHOD_ID => self::scontext::OUTPUTS_EVAL_FN,
            scontext::HEIGHT_PROPERTY_METHOD_ID => self::scontext::HEIGHT_EVAL_FN,
            scontext::SELF_PROPERTY_METHOD_ID => self::scontext::SELF_EVAL_FN,
            scontext::LAST_BLOCK_UTXO_ROOT_HASH_PROPERTY_METHOD_ID => {
                self::scontext::LAST_BLOCK_UTXO_ROOT_HASH_EVAL_FN
            }
            scontext::MINER_PUBKEY_PROPERTY_METHOD_ID => self::scontext::MINER_PUBKEY_EVAL_FN,
            scontext::GET_VAR_FROM_INPUT_METHOD_ID => self::scontext::GET_VAR_FROM_INPUT_EVAL_FN,
            method_id => {
                return Err(EvalError::NotFound(format!(
                    "Eval fn: unknown method id in SContext: {:?}",
                    method_id
                )))
            }
        },
        sbox::TYPE_CODE => match method.method_id() {
            sbox::VALUE_METHOD_ID => self::sbox::VALUE_EVAL_FN,
            sbox::PROPOSITION_BYTES_METHOD_ID => self::sbox::PROPOSITION_BYTES_EVAL_FN,
            sbox::BYTES_METHOD_ID => self::sbox::BYTES_EVAL_FN,
            sbox::BYTES_WITHOUT_REF_METHOD_ID => self::sbox::BYTES_WITHOUT_REF_EVAL_FN,
            sbox::ID_METHOD_ID => self::sbox::ID_EVAL_FN,
            sbox::CREATION_INFO_METHOD_ID => self::sbox::CREATION_INFO_EVAL_FN,
            sbox::GET_REG_METHOD_ID => self::sbox::GET_REG_EVAL_FN,
            sbox::TOKENS_METHOD_ID => self::sbox::TOKENS_EVAL_FN,
            method_id => {
                return Err(EvalError::NotFound(format!(
                    "Eval fn: unknown method id in SBox: {:?}",
                    method_id
                )))
            }
        },
        scoll::TYPE_CODE => match method.method_id() {
            scoll::INDEX_OF_METHOD_ID => self::scoll::INDEX_OF_EVAL_FN,
            scoll::FLATMAP_METHOD_ID => self::scoll::flatmap_eval,
            scoll::ZIP_METHOD_ID => self::scoll::ZIP_EVAL_FN,
            scoll::INDICES_METHOD_ID => self::scoll::INDICES_EVAL_FN,
            scoll::PATCH_METHOD_ID => self::scoll::PATCH_EVAL_FN,
            scoll::UPDATED_METHOD_ID => self::scoll::UPDATED_EVAL_FN,
            scoll::UPDATE_MANY_METHOD_ID => self::scoll::UPDATE_MANY_EVAL_FN,
            scoll::REVERSE_METHOD_ID => self::scoll::REVERSE_EVAL_FN,
            scoll::STARTS_WITH_METHOD_ID => self::scoll::STARTS_WITH_EVAL_FN,
            scoll::ENDS_WITH_METHOD_ID => self::scoll::ENDS_WITH_EVAL_FN,
            scoll::GET_METHOD_ID => self::scoll::GET_EVAL_FN,
            scoll::SIZE_METHOD_ID => self::scoll::SIZE_EVAL_FN,
            scoll::GET_OR_ELSE_METHOD_ID => self::scoll::GET_OR_ELSE_EVAL_FN,
            scoll::MAP_METHOD_ID => self::scoll::MAP_EVAL_FN,
            scoll::APPLY_METHOD_ID => self::scoll::APPLY_EVAL_FN,
            method_id => {
                return Err(EvalError::NotFound(format!(
                    "Eval fn: unknown method id in SCollection: {:?}",
                    method_id
                )))
            }
        },
        sgroup_elem::TYPE_CODE => match method.method_id() {
            sgroup_elem::GET_ENCODED_METHOD_ID => self::sgroup_elem::GET_ENCODED_EVAL_FN,
            sgroup_elem::NEGATE_METHOD_ID => self::sgroup_elem::NEGATE_EVAL_FN,
            sgroup_elem::EXPONENTIATE_METHOD_ID => self::sgroup_elem::EXPONENTIATE_EVAL_FN,
            sgroup_elem::MULTIPLY_METHOD_ID => self::sgroup_elem::MULTIPLY_EVAL_FN,
            sgroup_elem::EXPONENTIATE_UNSIGNED_METHOD_ID => {
                self::sgroup_elem::EXPONENTIATE_UNSIGNED_EVAL_FN
            }
            method_id => {
                return Err(EvalError::NotFound(format!(
                    "Eval fn: unknown method id in SGroupElement: {:?}",
                    method_id
                )))
            }
        },
        soption::TYPE_CODE => match method.method_id() {
            soption::MAP_METHOD_ID => self::soption::map_eval,
            soption::FILTER_METHOD_ID => self::soption::filter_eval,
            soption::IS_DEFINED_METHOD_ID => self::soption::IS_DEFINED_EVAL_FN,
            soption::GET_METHOD_ID => self::soption::GET_EVAL_FN,
            method_id => {
                return Err(EvalError::NotFound(format!(
                    "Eval fn: unknown method id in SOption: {:?}",
                    method_id
                )))
            }
        },
        sheader::TYPE_CODE => match method.method_id() {
            sheader::VERSION_METHOD_ID => self::sheader::VERSION_EVAL_FN,
            sheader::ID_METHOD_ID => self::sheader::ID_EVAL_FN,
            sheader::PARENT_ID_METHOD_ID => self::sheader::PARENT_ID_EVAL_FN,
            sheader::AD_PROOFS_ROOT_METHOD_ID => self::sheader::AD_PROOFS_ROOT_EVAL_FN,
            sheader::STATE_ROOT_METHOD_ID => self::sheader::STATE_ROOT_EVAL_FN,
            sheader::TRANSACTIONS_ROOT_METHOD_ID => self::sheader::TRANSACTION_ROOT_EVAL_FN,
            sheader::EXTENSION_ROOT_METHOD_ID => self::sheader::EXTENSION_ROOT_EVAL_FN,
            sheader::TIMESTAMP_METHOD_ID => self::sheader::TIMESTAMP_EVAL_FN,
            sheader::N_BITS_METHOD_ID => self::sheader::N_BITS_EVAL_FN,
            sheader::HEIGHT_METHOD_ID => self::sheader::HEIGHT_EVAL_FN,
            sheader::MINER_PK_METHOD_ID => self::sheader::MINER_PK_EVAL_FN,
            sheader::POW_ONETIME_PK_METHOD_ID => self::sheader::POW_ONETIME_PK_EVAL_FN,
            sheader::POW_DISTANCE_METHOD_ID => self::sheader::POW_DISTANCE_EVAL_FN,
            sheader::POW_NONCE_METHOD_ID => self::sheader::POW_NONCE_EVAL_FN,
            sheader::VOTES_METHOD_ID => self::sheader::VOTES_EVAL_FN,
            sheader::CHECK_POW_METHOD_ID => self::sheader::CHECK_POW_EVAL_FN,
            method_id => {
                return Err(EvalError::NotFound(format!(
                    "Eval fn: method {:?} with method id {:?} not found in SHeader",
                    method.name(),
                    method_id,
                )))
            }
        },
        spreheader::TYPE_CODE => match method.method_id() {
            spreheader::VERSION_METHOD_ID => self::spreheader::VERSION_EVAL_FN,
            spreheader::PARENT_ID_METHOD_ID => self::spreheader::PARENT_ID_EVAL_FN,
            spreheader::TIMESTAMP_METHOD_ID => self::spreheader::TIMESTAMP_EVAL_FN,
            spreheader::N_BITS_METHOD_ID => self::spreheader::N_BITS_EVAL_FN,
            spreheader::HEIGHT_METHOD_ID => self::spreheader::HEIGHT_EVAL_FN,
            spreheader::MINER_PK_METHOD_ID => self::spreheader::MINER_PK_EVAL_FN,
            spreheader::VOTES_METHOD_ID => self::spreheader::VOTES_EVAL_FN,
            method_id => {
                return Err(EvalError::NotFound(format!(
                    "Eval fn: method {:?} with method id {:?} not found in SPreHeader",
                    method.name(),
                    method_id,
                )))
            }
        },
        sglobal::TYPE_CODE => match method.method_id() {
            sglobal::GROUP_GENERATOR_METHOD_ID => self::sglobal::GROUP_GENERATOR_EVAL_FN,
            sglobal::XOR_METHOD_ID => self::sglobal::XOR_EVAL_FN,
            sglobal::FROM_BIGENDIAN_BYTES_METHOD_ID => {
                self::sglobal::SGLOBAL_FROM_BIGENDIAN_BYTES_EVAL_FN
            }
            sglobal::DESERIALIZE_METHOD_ID => self::sglobal::DESERIALIZE_EVAL_FN,
            sglobal::SERIALIZE_METHOD_ID => self::sglobal::SERIALIZE_EVAL_FN,
            sglobal::SOME_METHOD_ID => self::sglobal::SGLOBAL_SOME_EVAL_FN,
            sglobal::NONE_METHOD_ID => self::sglobal::SGLOBAL_NONE_EVAL_FN,
            sglobal::ENCODE_NBITS_METHOD_ID => self::sglobal::ENCODE_NBITS_EVAL_FN,
            sglobal::DECODE_NBITS_METHOD_ID => self::sglobal::DECODE_NBITS_EVAL_FN,
            sglobal::POW_HIT_METHOD_ID => self::sglobal::POW_HIT_EVAL_FN,
            method_id => {
                return Err(EvalError::NotFound(format!(
                    "Eval fn: method {:?} with method id {:?} not found in SGlobal",
                    method.name(),
                    method_id,
                )))
            }
        },
        snumeric::sbyte::TYPE_CODE
        | snumeric::sshort::TYPE_CODE
        | snumeric::sint::TYPE_CODE
        | snumeric::slong::TYPE_CODE
        | snumeric::sbigint::TYPE_CODE
        | snumeric::sunsignedbigint::TYPE_CODE => numeric_method_evalfn(method)?,
        type_id => {
            return Err(EvalError::NotFound(format!(
                "Eval fn: unknown type id {:?}",
                type_id
            )))
        }
    })
}

#[doc(hidden)]
#[allow(missing_docs)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::todo)]
#[cfg(feature = "arbitrary")]
pub mod test_util {

    use super::env::Env;
    use super::*;
    use ergotree_ir::mir::constant::TryExtractFrom;
    use ergotree_ir::mir::constant::TryExtractInto;
    use ergotree_ir::serialization::sigma_byte_reader::from_bytes;
    use ergotree_ir::serialization::sigma_byte_reader::SigmaByteRead;
    use ergotree_ir::serialization::SigmaSerializable;
    use sigma_test_util::force_any_val;

    thread_local! {
        static TEST_CTX: Context<'static> = force_any_val::<Context>();
    }

    /// The root of SANTA's eval tree (a v3 sized tree whose root is the expression, read
    /// without the tree's root-type check), and its value and JIT cost at activated version 3
    #[cfg(test)]
    pub(crate) fn eval_v3_body(tree_hex: &str) -> (Expr, Result<Value<'static>, EvalError>, u64) {
        eval_v3_body_in(force_any_val::<Context>(), tree_hex)
    }

    /// [`eval_v3_body`] in the given context
    #[cfg(test)]
    pub(crate) fn eval_v3_body_in(
        mut ctx: Context<'static>,
        tree_hex: &str,
    ) -> (Expr, Result<Value<'static>, EvalError>, u64) {
        let bytes = base16::decode(tree_hex.as_bytes()).unwrap();
        assert_eq!(bytes[0], 0x0b, "{tree_hex}");
        // the size: one VLQ byte, or two
        let root = if bytes[1] < 0x80 { 2 } else { 3 };
        assert!(bytes[root - 1] < 0x80, "{tree_hex}");
        let expr = from_bytes(&bytes[root..])
            .with_tree_version(ErgoTreeVersion::V3, Expr::sigma_parse)
            .unwrap();
        ctx.activated_script_version_byte = 3;
        ctx.tree_version.set(ErgoTreeVersion::V3);
        ctx.jit_cost.set(0);
        let mut env = Env::empty();
        let res = expr.eval(&mut env, &ctx).map(|v| v.to_static());
        (expr, res, ctx.jit_cost_value())
    }

    #[cfg(test)]
    pub(crate) fn assert_accepts(tree_hex: &str, expected: Value<'static>, cost: u64) {
        let (_, res, c) = eval_v3_body(tree_hex);
        assert_eq!(res.unwrap(), expected, "{tree_hex}");
        assert_eq!(c, cost, "{tree_hex}");
    }

    /// The value and JIT cost of a SANTA eval tree at activated version 3: a v3 sized tree
    /// (`0b`, [`eval_v3_body`]) or a v0 tree (`00`), whose root is the expression
    #[cfg(test)]
    pub(crate) fn eval_tree(tree_hex: &str) -> (Result<Value<'static>, EvalError>, u64) {
        if tree_hex.starts_with("0b") {
            let (_, res, cost) = eval_v3_body(tree_hex);
            return (res, cost);
        }
        assert!(tree_hex.starts_with("00"), "{tree_hex}");
        eval_body(&tree_hex[2..], ErgoTreeVersion::V0)
    }

    /// The value and JIT cost of an expression's bytes, parsed and evaluated at the given tree
    /// version, at activated version 3
    #[cfg(test)]
    pub(crate) fn eval_body(
        body_hex: &str,
        version: ErgoTreeVersion,
    ) -> (Result<Value<'static>, EvalError>, u64) {
        let bytes = base16::decode(body_hex.as_bytes()).unwrap();
        let expr = from_bytes(&bytes)
            .with_tree_version(version, Expr::sigma_parse)
            .unwrap();
        let mut ctx = force_any_val::<Context>();
        ctx.activated_script_version_byte = 3;
        ctx.tree_version.set(version);
        ctx.jit_cost.set(0);
        let mut env = Env::empty();
        let res = expr.eval(&mut env, &ctx).map(|v| v.to_static());
        (res, ctx.jit_cost_value())
    }

    /// The constant `BigInt.MIN` (-2^255)
    #[cfg(test)]
    pub(crate) const BIGINT_MIN_HEX: &str =
        "06208000000000000000000000000000000000000000000000000000000000000000";

    /// The constant `BigInt.MAX` (2^255 - 1)
    #[cfg(test)]
    pub(crate) const BIGINT_MAX_HEX: &str =
        "06207fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    /// `BigInt.MIN / -1`: 2^255 below tree version 3 (sigmastate's `divide` does not check its
    /// result there), a throw from version 3
    #[cfg(test)]
    pub(crate) fn bigint_x() -> alloc::string::String {
        alloc::format!("9d{BIGINT_MIN_HEX}0601ff")
    }

    /// `6 / -1`: the in-range twin of [`bigint_x`], -6
    #[cfg(test)]
    pub(crate) fn bigint_y() -> alloc::string::String {
        "9d0601060601ff".into()
    }

    /// Each tree's value and cost, or that it errors where no value is given
    #[cfg(test)]
    pub(crate) fn assert_trees(rows: &[(&str, Option<(Value<'static>, u64)>)]) {
        for (tree_hex, expected) in rows {
            let (res, cost) = eval_tree(tree_hex);
            match expected {
                Some((value, expected_cost)) => {
                    assert_eq!(res.as_ref().ok(), Some(value), "{tree_hex}: {res:?}");
                    assert_eq!(cost, *expected_cost, "{tree_hex}");
                }
                None => assert!(res.is_err(), "{tree_hex}: {res:?}"),
            }
        }
    }

    /// the error under the span wrapper each eval node adds
    #[cfg(test)]
    pub(crate) fn innermost(e: &EvalError) -> &EvalError {
        match e {
            EvalError::Spanned(s) => innermost(&s.error),
            e => e,
        }
    }

    /// The error of a tree that must not evaluate, under the span wrapper each eval node adds,
    /// and the JIT cost counter where the evaluation stopped
    #[cfg(test)]
    pub(crate) fn eval_tree_err(tree_hex: &str) -> (EvalError, u64) {
        let (res, cost) = eval_tree(tree_hex);
        assert!(res.is_err(), "{tree_hex}: {res:?}");
        (innermost(&res.unwrap_err()).clone(), cost)
    }

    /// The error of an expression's bytes that must not evaluate at tree version 3, under the
    /// span wrapper each eval node adds, and the JIT cost counter where the evaluation stopped
    #[cfg(test)]
    pub(crate) fn eval_body_err(body_hex: &str) -> (EvalError, u64) {
        let (res, cost) = eval_body(body_hex, ErgoTreeVersion::V3);
        assert!(res.is_err(), "{body_hex}: {res:?}");
        (innermost(&res.unwrap_err()).clone(), cost)
    }

    /// fails at eval in an evaluator, not for want of one
    #[cfg(test)]
    pub(crate) fn assert_evaluator_refuses(tree_hex: &str) {
        let (_, res, _) = eval_v3_body(tree_hex);
        let err = res.unwrap_err();
        assert!(
            !matches!(innermost(&err), EvalError::NotFound(_)),
            "{tree_hex}: {err:?}"
        );
    }

    pub fn eval_out_wo_ctx<T: TryExtractFrom<Value<'static>> + 'static>(expr: &Expr) -> T {
        TEST_CTX.with(|ctx| eval_out(expr, ctx))
    }

    pub fn eval_out<T: TryExtractFrom<Value<'static>> + 'static>(
        expr: &Expr,
        ctx: &Context<'static>,
    ) -> T {
        let mut env = Env::empty();
        expr.eval(&mut env, ctx)
            .unwrap()
            .to_static()
            .try_extract_into::<T>()
            .unwrap()
    }

    pub fn try_eval_out<'ctx, T: TryExtractFrom<Value<'static>> + 'static>(
        expr: &Expr,
        ctx: &'ctx Context<'ctx>,
    ) -> Result<T, EvalError> {
        let mut env = Env::empty();
        expr.eval(&mut env, ctx).and_then(|v| {
            v.to_static()
                .try_extract_into::<T>()
                .map_err(EvalError::TryExtractFrom)
        })
    }

    /// Eval expr, performing deserialize node substitution before evaluation
    pub fn try_eval_with_deserialize<'ctx, T: TryExtractFrom<Value<'static>> + 'static>(
        expr: &Expr,
        ctx: &'ctx Context<'ctx>,
    ) -> Result<T, EvalError> {
        // The eval-tier mirror of `reduce_to_crypto`'s context-construction guard:
        // a self extension key >= 0x80 crashes JVM context construction before any
        // bytecode, so reject it here too. This entry is the one the conformance
        // runner drives, so it must surface the same key-domain verdict the node
        // path does (without it the runner would never exercise the rejection).
        super::validate_self_extension_key_domain(ctx)?;
        // Mirror `Interpreter.fullReduction`: a deserialize-bearing segregated
        // tree is reduced from its constants-substituted proposition, not the
        // lazy placeholder form used for ordinary trees. Values are identical
        // either way; the distinction becomes observable once JIT costing
        // lands (Constant vs ConstantPlaceholder visit costs).
        let expr = match ctx.constants {
            Some(constants) if expr.has_deserialize() => {
                expr.clone().substitute_constants(constants)?
            }
            _ => expr.clone(),
        };
        let expr = expr.substitute_deserialize(ctx)?;
        super::convert_self_extension(ctx)?;
        try_eval_out(&expr, ctx)
    }

    // Evaluate with activated version
    pub fn try_eval_out_with_version<'ctx, T: TryExtractFrom<Value<'static>> + 'static>(
        expr: &Expr,
        ctx: &'ctx Context<'ctx>,
        tree_version: u8,
        activated_version: u8,
    ) -> Result<T, EvalError> {
        let mut ctx = ctx.clone();
        ctx.activated_script_version_byte = activated_version as i8;
        ctx.tree_version.set(tree_version.into());
        // roundtrip expr to test methodcall versioning
        from_bytes(&expr.sigma_serialize_bytes()?)
            .with_tree_version(ctx.tree_version(), Expr::sigma_parse)?;
        let mut env = Env::empty();
        expr.eval(&mut env, &ctx).and_then(|v| {
            v.to_static()
                .try_extract_into::<T>()
                .map_err(EvalError::TryExtractFrom)
        })
    }

    pub fn try_eval_out_wo_ctx<T: TryExtractFrom<Value<'static>> + 'static>(
        expr: &Expr,
    ) -> Result<T, EvalError> {
        TEST_CTX.with(|ctx| try_eval_out(expr, ctx))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod test {
    use ergotree_ir::{
        chain::context::Context,
        ergo_tree::ErgoTree,
        mir::{
            bin_op::{BinOp, BinOpKind, RelationOp},
            block::BlockValue,
            expr::Expr,
            val_def::ValDef,
            val_use::ValUse,
        },
        sigma_protocol::sigma_boolean::{SigmaBoolean, SigmaProp},
        types::stype::SType,
    };
    use expect_test::expect;
    use sigma_test_util::force_any_val;

    use crate::eval::reduce_to_crypto;
    use crate::eval::EvalError;

    #[test]
    fn diag_on_reduced_to_false() {
        let bin_op: Expr = BinOp {
            kind: BinOpKind::Relation(RelationOp::Eq),
            left: Box::new(
                ValUse {
                    val_id: 1.into(),
                    tpe: SType::SInt,
                }
                .into(),
            ),
            right: Box::new(0i32.into()),
        }
        .into();
        let block: ErgoTree = Expr::BlockValue(
            BlockValue {
                items: vec![ValDef {
                    id: 1.into(),
                    tpe_args: vec![],
                    rhs: Box::new(Expr::Const(1i32.into())),
                }
                .into()],
                result: Box::new(bin_op),
            }
            .into(),
        )
        .try_into()
        .unwrap();
        let ctx = force_any_val::<Context>();
        let res = reduce_to_crypto(&block, &ctx).unwrap();
        assert!(res.sigma_prop == SigmaBoolean::TrivialProp(false));
        expect![[r#"
            Pretty printed expr:
            {
              val v1 = 1
              v1 == 0
            }

            Env:
            v1: 1
        "#]]
        .assert_eq(&res.diag.to_string());
    }

    // The JVM rejects a self-box ContextExtension with any key >= 0x80 at context
    // construction (the signed-key vars array crashes), before any bytecode. Mirror:
    // both the node entry (reduce_to_crypto) and the eval-tier entry
    // (try_eval_with_deserialize) reject on key PRESENCE — a GetVar-free `true` tree
    // still rejects — with 0x7f the accepted boundary and 0x80/0xff rejected.
    #[test]
    fn self_extension_key_above_0x7f_rejected_at_construction() {
        use crate::eval::test_util::{try_eval_out, try_eval_with_deserialize};
        use ergotree_ir::chain::context_extension::ContextExtension;
        use ergotree_ir::mir::get_var::GetVar;
        use ergotree_ir::mir::value::Value;

        // GetVar-free trivial tree: reduces to TrivialProp(true) when the context is
        // accepted, so any rejection is purely the key-domain guard.
        let true_expr: Expr = Expr::Const(true.into());
        let tree: ErgoTree = true_expr.clone().try_into().unwrap();

        let ctx_with_key = |key: u8| -> Context<'static> {
            let mut ext = ContextExtension::empty();
            ext.values.insert(key, 1i32.into());
            let mut ctx = force_any_val::<Context>();
            ctx.extension = Box::leak(Box::new(ext));
            ctx
        };

        // 0x7f (127) — the boundary — accepts on both the node and eval-tier entries.
        assert_eq!(
            reduce_to_crypto(&tree, &ctx_with_key(0x7f))
                .unwrap()
                .sigma_prop,
            SigmaBoolean::TrivialProp(true)
        );
        assert_eq!(
            try_eval_with_deserialize::<Value>(&true_expr, &ctx_with_key(0x7f)).unwrap(),
            Value::Boolean(true)
        );
        // 0x80 (128) and 0xff (255) — out of the signed-Byte range — reject on both.
        assert!(reduce_to_crypto(&tree, &ctx_with_key(0x80)).is_err());
        assert!(reduce_to_crypto(&tree, &ctx_with_key(0xff)).is_err());
        assert!(try_eval_with_deserialize::<Value>(&true_expr, &ctx_with_key(0x80)).is_err());
        assert!(try_eval_with_deserialize::<Value>(&true_expr, &ctx_with_key(0xff)).is_err());
        // An empty self extension is unaffected.
        let mut ctx_empty = force_any_val::<Context>();
        ctx_empty.extension = Box::leak(Box::new(ContextExtension::empty()));
        assert!(reduce_to_crypto(&tree, &ctx_empty).is_ok());

        // Guard-rail: the READ is not over-rejected — GetVar of a >= 0x80 id with the
        // key ABSENT (so it never reaches the construction check) stays None.
        let get_var_0x80: Expr = GetVar {
            var_id: 0x80,
            var_tpe: SType::SInt,
        }
        .into();
        let mut ctx_no_var = force_any_val::<Context>();
        ctx_no_var.extension = Box::leak(Box::new(ContextExtension::empty()));
        assert_eq!(
            try_eval_out::<Value>(&get_var_0x80, &ctx_no_var).unwrap(),
            Value::Opt(None)
        );
    }

    #[test]
    fn reduce_to_crypto_with_constant_segregation() {
        // Build a simple script: { 1 == 1 } with constant segregation enabled
        use ergotree_ir::ergo_tree::ErgoTreeHeader;
        use ergotree_ir::mir::bool_to_sigma::BoolToSigmaProp;

        let expr: Expr = Expr::BoolToSigmaProp(BoolToSigmaProp {
            input: Box::new(
                BinOp {
                    kind: BinOpKind::Relation(RelationOp::Eq),
                    left: Box::new(Expr::Const(1i32.into())),
                    right: Box::new(Expr::Const(1i32.into())),
                }
                .into(),
            ),
        });
        let tree = ErgoTree::new(ErgoTreeHeader::v1(true), &expr).unwrap();
        // Verify this tree actually uses constant segregation
        assert!(
            tree.header().unwrap().is_constant_segregation(),
            "tree must use constant segregation"
        );
        // The root should contain ConstPlaceholder nodes, not Const nodes
        let root = tree.root_expr().unwrap();
        let has_placeholders = format!("{:?}", root).contains("ConstPlaceholder");
        assert!(
            has_placeholders,
            "root should contain ConstPlaceholder nodes"
        );

        let ctx = force_any_val::<Context>();
        let res = reduce_to_crypto(&tree, &ctx).unwrap();
        assert_eq!(res.sigma_prop, SigmaBoolean::TrivialProp(true));
    }

    #[test]
    fn jit_cost_trivial_prop() {
        // try_from puts a Boolean Const in a segregated v0(true) tree, so the
        // root becomes a ConstPlaceholder = JitCost(1) => block cost 0.
        let tree = ErgoTree::try_from(Expr::Const(true.into())).unwrap();
        let ctx = force_any_val::<Context>();
        let res = reduce_to_crypto(&tree, &ctx).unwrap();
        assert_eq!(res.sigma_prop, SigmaBoolean::TrivialProp(true));
        assert_eq!(res.cost, 0); // JitCost 1 / 10 = 0
    }

    #[test]
    fn jit_cost_self_value() {
        // SELF.value > 0 => Self(10) + ExtractAmount(8) + Constant(5) + GT(20) + BoolToSigmaProp(15)
        // = JitCost(58) => block cost 5
        use ergotree_ir::mir::bool_to_sigma::BoolToSigmaProp;
        use ergotree_ir::mir::extract_amount::ExtractAmount;
        use ergotree_ir::mir::global_vars::GlobalVars;

        let self_value: Expr = ExtractAmount {
            input: Box::new(GlobalVars::SelfBox.into()),
        }
        .into();
        let tree = ErgoTree::try_from(Expr::BoolToSigmaProp(BoolToSigmaProp {
            input: Box::new(
                BinOp {
                    kind: BinOpKind::Relation(RelationOp::Gt),
                    left: Box::new(self_value),
                    right: Box::new(Expr::Const(0i64.into())),
                }
                .into(),
            ),
        }))
        .unwrap();
        let ctx = force_any_val::<Context>();
        let res = reduce_to_crypto(&tree, &ctx).unwrap();
        assert_eq!(res.cost, 5); // 58 / 10 = 5
    }

    #[test]
    fn jit_cost_limit_exceeded() {
        // Set a zero cost limit and verify that evaluation returns CostError.
        // The tree is segregated (Boolean Const → v0(true)), so its single
        // ConstPlaceholder eval charges 1 JitCost; any positive limit would
        // accept it. Limit 0 catches the very first cost addition.
        let tree = ErgoTree::try_from(Expr::Const(true.into())).unwrap();
        let mut ctx = force_any_val::<Context>();
        ctx.jit_cost_limit = Some(0);
        let res = reduce_to_crypto(&tree, &ctx);
        assert!(res.is_err());
        let is_cost_error = match res.unwrap_err() {
            EvalError::CostError(_) => true,
            EvalError::Spanned(e) => matches!(*e.error, EvalError::CostError(_)),
            _ => false,
        };
        assert!(is_cost_error, "Expected CostError");
    }

    // Bug 1 regression: in a constant-segregated tree, every reference to a
    // constant is a ConstPlaceholder node, not an inline Const. Scala prices
    // these differently: ConstPlaceholder = 1 JitCost, Const = 5 JitCost.
    // Pre-fix, `ErgoTree::proposition()` substituted placeholders into Const
    // nodes before eval, so segregated trees paid 5 JIT per constant — a
    // 5× overcharge on every reference. Post-fix, the root is evaluated with
    // placeholders intact (lazy resolution from `ctx.constants`) and the
    // eval arm charges 1 JIT per placeholder.
    #[test]
    fn segregated_constants_charge_1_not_5_per_placeholder() {
        // Segregated: a Boolean Const goes through `ErgoTree::try_from`'s
        // v0(true) branch, so the root is a ConstPlaceholder.
        let segregated = ErgoTree::try_from(Expr::Const(true.into())).unwrap();
        let ctx_seg = force_any_val::<Context>();
        let before_seg = ctx_seg.jit_cost_value();
        reduce_to_crypto(&segregated, &ctx_seg).unwrap();
        assert_eq!(
            ctx_seg.jit_cost_value() - before_seg,
            1,
            "segregated ConstPlaceholder eval must charge JitCost(1); pre-fix \
             charged JitCost(5) via the substituted Const.",
        );

        // Non-segregated: a SigmaProp Const goes through try_from's v0(false)
        // branch. It's also short-circuited by trivial_reduce (Phase 8), which
        // pays EVAL_SIGMA_PROP_CONSTANT = 50 JitCost. The contrast proves the
        // 1-vs-5 per-node distinction isn't masked by something collapsing the
        // placeholder-aware path with the Const path.
        use ergotree_ir::sigma_protocol::sigma_boolean::ProveDlog;
        let sp = SigmaProp::from(force_any_val::<ProveDlog>());
        let non_segregated = ErgoTree::try_from(Expr::Const(sp.into())).unwrap();
        let ctx_ns = force_any_val::<Context>();
        let before_ns = ctx_ns.jit_cost_value();
        reduce_to_crypto(&non_segregated, &ctx_ns).unwrap();
        assert_eq!(
            ctx_ns.jit_cost_value() - before_ns,
            50,
            "non-segregated SigmaProp Const must short-circuit to \
             EVAL_SIGMA_PROP_CONSTANT = 50.",
        );
    }

    // Bug 2 regression: a tree whose proposition is a plain SigmaProp constant
    // (e.g. bare P2PK) must be priced at Scala's EvalSigmaPropConstant = 50
    // JitCost via the trivial_reduce short-circuit. Pre-fix, it went through
    // the generic Expr::Const arm and paid only 5 JitCost — 10× undercharge
    // on every P2PK input.
    #[test]
    fn p2pk_trivial_reduce_charges_50() {
        use ergotree_ir::sigma_protocol::sigma_boolean::ProveDlog;

        let pd = force_any_val::<ProveDlog>();
        let sp = SigmaProp::from(pd.clone());
        let expr: Expr = Expr::Const(sp.into());
        let tree = ErgoTree::try_from(expr).unwrap();
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();

        let res = reduce_to_crypto(&tree, &ctx).unwrap();

        // JitCost delta must be exactly 50 (EvalSigmaPropConstant), not 5
        // (the Expr::Const generic cost that the pre-fix path would pay).
        assert_eq!(
            ctx.jit_cost_value() - before,
            50,
            "P2PK trivial reduce must charge JitCost(50), not the generic \
             Expr::Const(5). Got JitCost delta {}.",
            ctx.jit_cost_value() - before,
        );
        // Returned block cost = 50 / 10 = 5.
        assert_eq!(res.cost, 5);
        // SigmaProp round-trips back out through reduction.
        assert_eq!(res.sigma_prop, SigmaBoolean::from(pd));
    }

    // A cost-limit trip must NOT trigger the diagnostic retry: the retry
    // resets the accumulator and re-evaluates, re-charging work already
    // counted. `enrich_err` wraps the `CostError` in `Spanned` at every eval
    // node boundary, so the pre-fix bare `EvalError::CostError` match arm
    // never fired and every limit trip took the retry. Assert the error
    // stays recognizable through the wrappers and the accumulator lands
    // exactly on the single-evaluation total.
    #[test]
    fn cost_limit_error_skips_diagnostic_retry() {
        let eq: Expr = BinOp {
            kind: BinOpKind::Relation(RelationOp::Eq),
            left: Box::new(Expr::Const(1i32.into())),
            right: Box::new(Expr::Const(1i32.into())),
        }
        .into();
        let tree = ErgoTree::try_from(eq).unwrap();

        // Learn the total with no limit.
        let ctx = force_any_val::<Context>();
        ctx.jit_cost.set(0);
        reduce_to_crypto(&tree, &ctx).unwrap();
        let total = ctx.jit_cost_value();

        // Trip the limit on the final charge.
        let mut ctx = force_any_val::<Context>();
        ctx.jit_cost.set(0);
        ctx.jit_cost_limit = Some(total - 1);
        let err = reduce_to_crypto(&tree, &ctx).unwrap_err();
        assert!(
            err.is_cost_error(),
            "limit trip must surface as a cost error through the span wrappers, got {err:?}"
        );
        assert_eq!(
            ctx.jit_cost_value(),
            total,
            "accumulator must stop at the single-evaluation total"
        );
    }

    // Regression: `inner` returns a bare `InvalidResultType` (eval.rs match arm
    // for non-Boolean/non-SigmaProp Value) when the expression reduces to e.g.
    // an Int. That bare variant enters the Err(_) diagnostic retry arm, the
    // spanned re-eval also returns bare, and `wrap_spanned_with_src` used to
    // panic. The fix in error.rs wraps any variant — assert reduce_to_crypto
    // returns Err instead of panicking.
    #[test]
    fn reduce_to_crypto_non_sigma_prop_result_returns_err_not_panic() {
        use super::EvalError;
        let tree = ErgoTree::try_from(Expr::Const(42i32.into())).unwrap();
        let ctx = force_any_val::<Context>();
        let res = reduce_to_crypto(&tree, &ctx);
        assert!(
            matches!(res, Err(EvalError::SpannedWithSource(_))),
            "expected Err(SpannedWithSource) after wrap_spanned_with_src \
             fix, got {:?}",
            res
        );
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod context_conversion_tests {
    //! JVM parity: sigmastate converts every self context variable as it builds the script
    //! context (`ErgoLikeContext.scala:158-161`), which it does for every tree it evaluates
    //! (`CErgoTreeEvaluator.scala:563`) and not for a tree whose proposition is a `SigmaProp`
    //! constant (`Interpreter.scala:211`). A variable that does not convert fails the spend.
    use super::*;
    use ergotree_ir::chain::context_extension::ContextExtension;
    use ergotree_ir::chain::evaluated_value::EvaluatedValue;
    use ergotree_ir::ergo_tree::ErgoTreeHeader;
    use ergotree_ir::mir::deserialize_context::DeserializeContext;
    use ergotree_ir::serialization::SigmaSerializable;
    use sigma_test_util::force_any_val;

    fn value(hex: &str) -> EvaluatedValue {
        EvaluatedValue::sigma_parse_bytes(&base16::decode(hex).unwrap()).unwrap()
    }

    fn reduce(
        tree: &ErgoTree,
        vars: Vec<(u8, EvaluatedValue)>,
    ) -> Result<ReductionResult, EvalError> {
        let ext = ContextExtension {
            values: vars.into_iter().collect(),
        };
        let mut ctx = force_any_val::<Context>().with_extension(&ext);
        ctx.activated_script_version_byte = 3;
        reduce_to_crypto(tree, &ctx)
    }

    #[test]
    fn an_evaluated_tree_fails_on_a_variable_that_does_not_convert() {
        // SANTA V1: `Tuple(1, HEIGHT)` in var 0 under `sigmaProp(true)`, which is evaluated; V2:
        // the same under a `SigmaProp` constant, which is not; then `Tuple(1, 2)`, which converts
        let evaluated = ErgoTree::sigma_parse_bytes(&[0x00, 0xd1, 0x01, 0x01]).unwrap();
        let constant = ErgoTree::sigma_parse_bytes(&[0x00, 0x08, 0xd3]).unwrap();
        assert!(reduce(&evaluated, vec![(0, value("86020402a3"))]).is_err());
        assert!(reduce(&constant, vec![(0, value("86020402a3"))]).is_ok());
        assert!(reduce(&evaluated, vec![(0, value("860204020404"))]).is_ok());
    }

    #[test]
    fn a_deserialized_sigma_prop_constant_is_evaluated() {
        // SANTA D4: `executeFromVar[SigmaProp](1)`, var 1 deserializing to `TrueProp`. sigmastate
        // evaluates the substituted tree, whatever it is (`Interpreter.scala:171-177`), so its
        // context conversion runs: var 0, `Tuple(1, HEIGHT)`, fails the spend
        let tree = ErgoTree::new(
            ErgoTreeHeader::v1(false),
            &DeserializeContext {
                tpe: SType::SSigmaProp,
                id: 1,
            }
            .into(),
        )
        .unwrap();
        let true_prop_bytes = || (1u8, value("0e0208d3"));
        assert!(reduce(&tree, vec![(0, value("86020402a3")), true_prop_bytes()]).is_err());
        // D4's twin, without var 0: the tree's bytes x 20 JitCost (substitution, since v6), 2
        // bytes x 20 (deserialization), and a `Constant`'s 5, not the flat 50 of a tree that is a
        // `SigmaProp` constant (SANTA: 12112, where sigma-rust charged 12117)
        let ext = ContextExtension {
            values: [true_prop_bytes()].into_iter().collect(),
        };
        let mut ctx = force_any_val::<Context>().with_extension(&ext);
        ctx.activated_script_version_byte = 3;
        ctx.jit_cost.set(0);
        reduce_to_crypto(&tree, &ctx).unwrap();
        let tree_len = tree.sigma_serialize_bytes().unwrap().len() as u64;
        assert_eq!(ctx.jit_cost_value(), tree_len * 20 + 2 * 20 + 5);
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod version_context_tests {
    //! JVM parity: sigmastate reduces a tree inside `VersionContext.withVersions(activated,
    //! ergoTree.version)` (v6.0.6 `Interpreter.scala:207`), and from activated 2 the context
    //! refuses a tree version above the activated one (`VersionContext.scala:17-21`). `verify`
    //! and `prove` refuse such a tree before they reduce it; a reduction on its own meets the
    //! context: `ReducingInterpreter.reduce` (`ReducingInterpreter.scala:33-45`) and the
    //! multi-signature hints (`ProverUtils.scala:21`, `:83`). SANTA measured `fullReduction`
    //! on the JVM (2026-10-02): a v4 tree at activated 3 and a v3 tree at activated 2 throw, a
    //! v2 tree at activated 1 and a v4 tree at activated 4 reduce. It has no vector, as no
    //! tier reduces without a spend.
    use super::*;
    use ergotree_ir::serialization::SigmaSerializable;
    use sigma_test_util::force_any_val;

    /// A size-flagged tree of `version`, read with no context: `SigmaProp(true)`
    fn reduce(version: u8, block_version: u8) -> Result<ReductionResult, EvalError> {
        let tree = ErgoTree::sigma_parse_bytes(&[0x08 | version, 0x02, 0x08, 0xd3]).unwrap();
        let mut ctx = force_any_val::<Context>();
        ctx.activated_script_version_byte = (block_version as i8).wrapping_sub(1);
        reduce_to_crypto(&tree, &ctx)
    }

    fn is_reduced(res: Result<ReductionResult, EvalError>) -> bool {
        matches!(res, Ok(reduction) if reduction.sigma_prop == SigmaBoolean::TrivialProp(true))
    }

    #[test]
    fn from_activated_2_a_tree_above_the_activated_version_is_not_reduced() {
        for activated in 0..=7u8 {
            for version in 0..=7u8 {
                let res = reduce(version, activated + 1);
                if activated >= 2 && version > activated {
                    assert_eq!(
                        res.err(),
                        Some(EvalError::TreeVersionAboveActivated {
                            tree_version: version.into(),
                            activated_version: activated as i8,
                        }),
                        "tree {version} at {activated}"
                    );
                } else {
                    assert!(is_reduced(res), "tree {version} at {activated}");
                }
            }
        }
    }

    #[test]
    fn a_negative_activated_version_is_below_2() {
        // sigmastate's versions are signed bytes: at block version 0 the activated version is
        // -1, and above block version 128 it is negative, so the context compares nothing
        for block_version in [0u8, 0x81, 0xff] {
            for version in 0..=7u8 {
                assert!(
                    is_reduced(reduce(version, block_version)),
                    "tree {version} at block version {block_version}"
                );
            }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod value_check_tests {
    //! JVM parity: a pair tuple expression's data is a collection typed as a pair
    //! (`values.scala:818-822`), which fails `Value.checkType` where sigmastate checks a value
    //! against its type: a lambda's argument and result (`values.scala:1074`, `:1080`), a
    //! `ValDef`'s value and a block's result (`:1027`, `:1034`), and a `ValUse` (`:991`).
    use super::*;
    use crate::eval::test_util::try_eval_out;
    use crate::eval::test_util::{try_eval_out_with_version, try_eval_out_wo_ctx};
    use ergo_chain_types::{EcPoint, Header, PreHeader};
    use ergotree_ir::bigint256::BigInt256;
    use ergotree_ir::chain::context_extension::ContextExtension;
    use ergotree_ir::chain::ergo_box::ErgoBox;
    use ergotree_ir::chain::evaluated_value::EvaluatedValue;
    use ergotree_ir::mir::apply::Apply;
    use ergotree_ir::mir::avl_tree_data::AvlTreeData;
    use ergotree_ir::mir::bin_op::{BinOp, RelationOp};
    use ergotree_ir::mir::block::BlockValue;
    use ergotree_ir::mir::coll_by_index::ByIndex;
    use ergotree_ir::mir::coll_fold::Fold;
    use ergotree_ir::mir::coll_size::SizeOf;
    use ergotree_ir::mir::coll_value::CollValue;
    use ergotree_ir::mir::collection::Collection;
    use ergotree_ir::mir::constant::{Constant, Literal};
    use ergotree_ir::mir::func_value::{FuncArg, FuncValue};
    use ergotree_ir::mir::get_var::GetVar;
    use ergotree_ir::mir::if_op::If;
    use ergotree_ir::mir::method_call::MethodCall;
    use ergotree_ir::mir::option_get::OptionGet;
    use ergotree_ir::mir::option_get_or_else::OptionGetOrElse;
    use ergotree_ir::mir::select_field::SelectField;
    use ergotree_ir::mir::tuple::Tuple;
    use ergotree_ir::mir::unary_op::OneArgOpTryBuild;
    use ergotree_ir::mir::val_def::ValDef;
    use ergotree_ir::mir::val_use::ValUse;
    use ergotree_ir::serialization::SigmaSerializable;
    use ergotree_ir::types::sfunc::SFunc;
    use ergotree_ir::types::soption;
    use ergotree_ir::types::stuple::STuple;
    use ergotree_ir::types::stype_param::STypeVar;
    use ergotree_ir::unsignedbigint256::UnsignedBigInt;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    fn pair_type() -> SType {
        SType::STuple(STuple::pair(SType::SInt, SType::SInt).into())
    }

    /// var 0 = `hex`: the tuple expression `Tuple(1, 2)`, or its constant twin `58 02 04`
    fn ctx_with_var0(hex: &str) -> Context<'static> {
        let value = EvaluatedValue::sigma_parse_bytes(&base16::decode(hex).unwrap()).unwrap();
        let ext = ContextExtension {
            values: [(0u8, value)].into_iter().collect(),
        };
        Context {
            extension: Box::leak(Box::new(ext)),
            ..force_any_val::<Context>()
        }
    }

    /// `getVar[(Int, Int)](0).get`
    fn pair_var0() -> Expr {
        let var0: Expr = GetVar {
            var_id: 0,
            var_tpe: pair_type(),
        }
        .into();
        OptionGet::try_build(var0).unwrap().into()
    }

    fn check(expr: &Expr) {
        assert!(try_eval_out::<i32>(expr, &ctx_with_var0("860204020404")).is_err());
        assert_eq!(
            try_eval_out::<i32>(expr, &ctx_with_var0("580204")).unwrap(),
            1
        );
    }

    #[test]
    fn a_lambda_checks_its_argument() {
        // SANTA T3's lambda, `{ (p: (Int, Int)) => 1 }`, applied to the pair
        let lambda: Expr = FuncValue::new(
            vec![FuncArg {
                idx: 1.into(),
                tpe: pair_type(),
            }],
            Expr::Const(1i32.into()),
        )
        .into();
        check(&Apply::new(lambda, vec![pair_var0()]).unwrap().into());
    }

    #[test]
    fn a_lambda_checks_its_result() {
        // `{ (x: Int) => getVar[(Int, Int)](0).get }(1)._1`, from source
        let lambda: Expr = FuncValue::new(
            vec![FuncArg {
                idx: 1.into(),
                tpe: SType::SInt,
            }],
            pair_var0(),
        )
        .into();
        let applied: Expr = Apply::new(lambda, vec![Expr::Const(1i32.into())])
            .unwrap()
            .into();
        let first =
            ergotree_ir::mir::select_field::SelectField::new(applied, 1u8.try_into().unwrap())
                .unwrap();
        // the result check fires before `SelectField` would
        assert!(matches!(
            try_eval_out::<i32>(&first.into(), &ctx_with_var0("860204020404")),
            Err(e) if format!("{e:?}").contains("Invalid type returned by evaluator")
        ));
    }

    #[test]
    fn a_block_checks_its_val_defs() {
        // `{ val p = getVar[(Int, Int)](0).get; 1 }`, from source
        let block: Expr = BlockValue {
            items: vec![ValDef {
                id: 1.into(),
                tpe_args: vec![],
                rhs: Box::new(pair_var0()),
            }
            .into()],
            result: Box::new(Expr::Const(1i32.into())),
        }
        .into();
        check(&block);
    }

    #[test]
    fn a_val_use_checks_its_value() {
        // a `ValUse` of a pair bound to the collection, from source
        let use_pair: Expr = ValUse {
            val_id: 1.into(),
            tpe: pair_type(),
        }
        .into();
        let coll = EvaluatedValue::sigma_parse_bytes(&base16::decode("860204020404").unwrap())
            .unwrap()
            .to_script_value()
            .unwrap()
            .v;
        let mut env = Env::empty();
        env.insert(1.into(), coll);
        let ctx = force_any_val::<Context>();
        assert!(use_pair.eval(&mut env, &ctx).is_err());
    }

    #[test]
    fn a_collection_checks_its_items() {
        // SANTA T5, `Coll(getVar[(Int, Int)](0).get).size == 1`, as the size
        let coll: Expr = Collection::new(pair_type(), vec![pair_var0()])
            .unwrap()
            .into();
        check(&SizeOf::try_build(coll).unwrap().into());
    }

    #[test]
    fn a_tuple_checks_its_items() {
        // `(getVar[(Int, Int)](0).get, 1)._2`, from source (`values.scala:828-833`)
        let tuple: Expr = Tuple::new(vec![pair_var0(), Expr::Const(1i32.into())])
            .unwrap()
            .into();
        check(
            &SelectField::new(tuple, 2u8.try_into().unwrap())
                .unwrap()
                .into(),
        );
    }

    #[test]
    fn an_if_checks_its_branch() {
        // `(if (true) getVar[(Int, Int)](0).get else (0, 0))._1`, from source
        // (`trees.scala:1358-1365`): the branch's check fails before `SelectField` would
        let zeros = Constant::sigma_parse_bytes(&[0x58, 0x00, 0x00]).unwrap();
        let if_expr: Expr = If {
            condition: Box::new(Expr::Const(true.into())),
            true_branch: Box::new(pair_var0()),
            false_branch: Box::new(zeros.into()),
        }
        .into();
        check_first_item(if_expr);
    }

    /// `expr._1` fails with the tuple expression on the value check itself, before
    /// `SelectField` would, and reads 1 with the constant twin
    fn check_first_item(expr: Expr) {
        let first: Expr = SelectField::new(expr, 1u8.try_into().unwrap())
            .unwrap()
            .into();
        assert!(matches!(
            try_eval_out::<i32>(&first, &ctx_with_var0("860204020404")),
            Err(e) if format!("{e:?}").contains("Invalid type returned by evaluator")
        ));
        assert_eq!(
            try_eval_out::<i32>(&first, &ctx_with_var0("580204")).unwrap(),
            1
        );
    }

    #[test]
    fn a_fold_checks_its_zero() {
        // `Coll[Int]().fold(getVar[(Int, Int)](0).get, { (a: ((Int, Int), Int)) => a._1 })._1`,
        // from source (`transformers.scala:224-228`)
        let acc_item = SType::STuple(STuple::pair(pair_type(), SType::SInt).into());
        let acc: Expr = ValUse {
            val_id: 1.into(),
            tpe: acc_item.clone(),
        }
        .into();
        let op: Expr = FuncValue::new(
            vec![FuncArg {
                idx: 1.into(),
                tpe: acc_item,
            }],
            SelectField::new(acc, 1u8.try_into().unwrap())
                .unwrap()
                .into(),
        )
        .into();
        let empty: Expr = Collection::new(SType::SInt, vec![]).unwrap().into();
        check_first_item(Fold::new(empty, pair_var0(), op).unwrap().into());
    }

    #[test]
    fn a_by_index_checks_its_default() {
        // `Coll[(Int, Int)]().getOrElse(0, getVar[(Int, Int)](0).get)._1`, from source
        // (`transformers.scala:258-276`)
        let empty: Expr = Collection::new(pair_type(), vec![]).unwrap().into();
        let at0 =
            ByIndex::new(empty, Expr::Const(0i32.into()), Some(Box::new(pair_var0()))).unwrap();
        check_first_item(at0.into());
    }

    // `isValueOfType` at every site (v6.0.7 `SType.scala:187-213`, through `Value.checkType`,
    // `values.scala:251-259`). `X` below is `int_holding_long()`.

    /// `If(false, 1, 2L)`: typed `SInt`, as an `If` is typed by its true branch
    /// (`trees.scala:1351`), and evaluates to `2L`
    fn int_holding_long() -> Expr {
        If {
            condition: Box::new(Expr::Const(false.into())),
            true_branch: Box::new(Expr::Const(1i32.into())),
            false_branch: Box::new(Expr::Const(2i64.into())),
        }
        .into()
    }

    fn int(v: i32) -> Expr {
        Expr::Const(v.into())
    }

    fn relation(op: RelationOp, left: Expr, right: Expr) -> Expr {
        BinOp {
            kind: op.into(),
            left: Box::new(left),
            right: Box::new(right),
        }
        .into()
    }

    fn lambda(arg_tpe: SType, body: Expr) -> Expr {
        FuncValue::new(
            vec![FuncArg {
                idx: 1.into(),
                tpe: arg_tpe,
            }],
            body,
        )
        .into()
    }

    /// `func(arg)` as the parser builds it, with no check of the argument's type
    fn apply(func: Expr, arg: Expr) -> Expr {
        Apply {
            func: Box::new(func),
            args: vec![arg],
        }
        .into()
    }

    fn is_value_check_error<T: core::fmt::Debug>(res: &Result<T, EvalError>) -> bool {
        matches!(res, Err(e) if format!("{e:?}").contains("Invalid type returned by evaluator"))
    }

    fn rejects_another_kind(expr: &Expr) {
        let res = try_eval_out_wo_ctx::<Value>(expr);
        assert!(
            is_value_check_error(&res),
            "expected a value check to fail, got {res:?}"
        );
    }

    #[test]
    fn the_check_is_the_jvm_table() {
        let v3 = ErgoTreeVersion::V3;
        let a_lambda = Value::Lambda(Lambda {
            args: vec![FuncArg {
                idx: 1.into(),
                tpe: SType::SInt,
            }],
            body: alloc::sync::Arc::new(int(1)),
            captured: Default::default(),
        });
        let rows: Vec<(SType, Value<'static>)> = vec![
            (SType::SBoolean, Value::Boolean(true)),
            (SType::SByte, Value::Byte(1)),
            (SType::SShort, Value::Short(1)),
            (SType::SInt, Value::Int(1)),
            (SType::SLong, Value::Long(1)),
            (SType::SBigInt, Value::from(BigInt256::from(1i64))),
            (
                SType::SUnsignedBigInt,
                Value::UnsignedBigInt(UnsignedBigInt::from(1u64)),
            ),
            (
                SType::SGroupElement,
                Value::from(force_any_val::<EcPoint>()),
            ),
            (
                SType::SSigmaProp,
                Value::SigmaProp(Box::new(force_any_val::<SigmaProp>())),
            ),
            (SType::SBox, Value::CBox(force_any_val::<ErgoBox>().into())),
            (SType::SAvlTree, Value::from(force_any_val::<AvlTreeData>())),
            (SType::SColl(SType::SInt.into()), Value::from(vec![1i32])),
            (
                SType::SOption(SType::SInt.into()),
                Value::Opt(Some(Box::new(Value::Int(1)))),
            ),
            (pair_type(), Value::pair(Value::Int(1), Value::Int(2))),
            (
                SType::SFunc(SFunc::new(vec![SType::SInt], SType::SInt).into()),
                a_lambda.clone(),
            ),
            (SType::SContext, Value::Context),
            (SType::SGlobal, Value::Global),
            (
                SType::SHeader,
                Value::Header(Box::new(force_any_val::<Header>())),
            ),
            (
                SType::SPreHeader,
                Value::PreHeader(Box::new(force_any_val::<PreHeader>())),
            ),
            (SType::SUnit, Value::Unit),
        ];
        for (tpe, value) in &rows {
            assert!(
                check_value_of_type(tpe, value, v3).is_ok(),
                "{tpe:?} refused {value:?}"
            );
            let other = if *tpe == SType::SInt {
                Value::Long(1)
            } else {
                Value::Int(1)
            };
            assert!(
                check_value_of_type(tpe, &other, v3).is_err(),
                "{tpe:?} took {other:?}"
            );
        }
        // no arm matches: "Unknown type", whatever the value
        for tpe in [
            SType::SAny,
            SType::STypeVar(STypeVar::t()),
            SType::SString,
            SType::SNoType,
        ] {
            for (_, value) in &rows {
                assert!(
                    check_value_of_type(&tpe, value, v3).is_err(),
                    "{tpe:?} took {value:?}"
                );
            }
        }
        // a tuple other than a pair and a function of other than one argument throw
        let triple = SType::STuple(STuple::triple(SType::SInt, SType::SInt, SType::SInt).into());
        // a tuple of three as it is held: a collection of `Any`
        let three = Value::Coll(CollValue::generic(
            SType::SAny,
            vec![Value::Int(1), Value::Int(2), Value::Int(3)],
        ));
        assert!(check_value_of_type(&triple, &three, v3).is_err());
        let binary = SType::SFunc(SFunc::new(vec![SType::SInt, SType::SInt], SType::SInt).into());
        assert!(check_value_of_type(&binary, &a_lambda, v3).is_err());
        // a pair needs a `Tuple2`, a unary function a `Function1`
        assert!(check_value_of_type(&pair_type(), &three, v3).is_err());
        let two_args = Value::Lambda(Lambda {
            args: vec![
                FuncArg {
                    idx: 1.into(),
                    tpe: SType::SInt,
                },
                FuncArg {
                    idx: 2.into(),
                    tpe: SType::SInt,
                },
            ],
            body: alloc::sync::Arc::new(int(1)),
            captured: Default::default(),
        });
        let unary = SType::SFunc(SFunc::new(vec![SType::SInt], SType::SInt).into());
        assert!(check_value_of_type(&unary, &two_args, v3).is_err());
    }

    #[test]
    fn a_tuple_of_three_is_a_collection() {
        // sigmastate's data for a tuple of three or more items is a `Coll[Any]` (`TupleData`,
        // `package.scala:67`; `Evaluation.toDslTuple`, `Evaluation.scala:99-102`), so it passes
        // the collection arm (`SType.scala:198`); a pair is a `Tuple2`, which does not
        let coll_type = SType::SColl(SType::SInt.into());
        // a tuple of three as it is held: a collection of `Any`
        let three = Value::Coll(CollValue::generic(
            SType::SAny,
            vec![Value::Int(1), Value::Int(2), Value::Int(3)],
        ));
        let pair = Value::pair(Value::Int(1), Value::Int(2));
        assert!(check_value_of_type(&coll_type, &three, ErgoTreeVersion::V3).is_ok());
        assert!(check_value_of_type(&coll_type, &pair, ErgoTreeVersion::V3).is_err());
        // `{ (x: Coll[Int]) => true }((1, 2, 3))`: neither parser checks the argument's type
        let items: Vec<Literal> = vec![1i32.into(), 2i32.into(), 3i32.into()];
        let triple: Expr = Constant {
            tpe: SType::STuple(STuple::triple(SType::SInt, SType::SInt, SType::SInt).into()),
            v: Literal::Tup(items.try_into().unwrap()),
        }
        .into();
        let applied = apply(lambda(coll_type, Expr::Const(true.into())), triple);
        assert!(try_eval_out_wo_ctx::<bool>(&applied).unwrap());
    }

    #[test]
    fn unsigned_big_int_passes_from_tree_version_3() {
        // `case SUnsignedBigInt if VersionContext.current.isV3OrLaterErgoTreeVersion`
        // (`SType.scala:194`); below it no arm matches
        let value = Value::UnsignedBigInt(UnsignedBigInt::from(1u64));
        assert!(check_value_of_type(&SType::SUnsignedBigInt, &value, ErgoTreeVersion::V3).is_ok());
        assert!(check_value_of_type(&SType::SUnsignedBigInt, &value, ErgoTreeVersion::V2).is_err());
    }

    proptest! {
        #[test]
        fn every_constant_passes_against_its_own_type(c in any::<Constant>()) {
            let res = check_value_of_type(&c.tpe, &Value::from(c.v.clone()), ErgoTreeVersion::V3);
            match &c.tpe {
                // the JVM's `ConstantPlaceholder` throws on these too
                SType::STuple(t) if t.items.len() != 2 => prop_assert!(res.is_err()),
                _ => prop_assert!(res.is_ok(), "{:?} refused its own value: {:?}", c.tpe, res),
            }
        }
    }

    #[test]
    fn a_val_def_rejects_another_kind() {
        // `{ val x = X; true }` (`values.scala:1027`)
        rejects_another_kind(
            &BlockValue {
                items: vec![ValDef {
                    id: 1.into(),
                    tpe_args: vec![],
                    rhs: Box::new(int_holding_long()),
                }
                .into()],
                result: Box::new(Expr::Const(true.into())),
            }
            .into(),
        );
    }

    #[test]
    fn a_block_result_rejects_another_kind() {
        // `{ val y = 1; X } == 2` (`values.scala:1034`)
        let block: Expr = BlockValue {
            items: vec![ValDef {
                id: 1.into(),
                tpe_args: vec![],
                rhs: Box::new(int(1)),
            }
            .into()],
            result: Box::new(int_holding_long()),
        }
        .into();
        rejects_another_kind(&relation(RelationOp::Eq, block, int(2)));
    }

    #[test]
    fn eq_and_neq_check_both_operands() {
        // `X == 2`, `2 == X`, `X != 3` (`trees.scala:1206-1208`, `:1226-1228`)
        rejects_another_kind(&relation(RelationOp::Eq, int_holding_long(), int(2)));
        rejects_another_kind(&relation(RelationOp::Eq, int(2), int_holding_long()));
        rejects_another_kind(&relation(RelationOp::NEq, int_holding_long(), int(3)));
    }

    #[test]
    fn an_if_branch_rejects_another_kind() {
        // `(if (true) X else 3) == 2` (`trees.scala:1360`)
        let outer: Expr = If {
            condition: Box::new(Expr::Const(true.into())),
            true_branch: Box::new(int_holding_long()),
            false_branch: Box::new(int(3)),
        }
        .into();
        rejects_another_kind(&relation(RelationOp::Eq, outer, int(2)));
    }

    #[test]
    fn tuple_items_reject_another_kind() {
        // `(X, 3)._1 == 2`, `(3, X)._2 == 2` (`values.scala:830`, `:833`)
        for (items, field) in [
            (vec![int_holding_long(), int(3)], 1u8),
            (vec![int(3), int_holding_long()], 2u8),
        ] {
            let tuple: Expr = Tuple::new(items).unwrap().into();
            let item: Expr = SelectField::new(tuple, field.try_into().unwrap())
                .unwrap()
                .into();
            rejects_another_kind(&relation(RelationOp::Eq, item, int(2)));
        }
    }

    #[test]
    fn collection_items_reject_another_kind() {
        // `Coll(X).size == 1` (`values.scala:894`)
        let coll: Expr = Collection::new(SType::SInt, vec![int_holding_long()])
            .unwrap()
            .into();
        let size: Expr = SizeOf::try_build(coll).unwrap().into();
        rejects_another_kind(&relation(RelationOp::Eq, size, int(1)));
    }

    #[test]
    fn a_lambda_argument_rejects_another_kind() {
        // `{ (x: Int) => true }(X)` (`values.scala:1074`); `{ (x: Long) => true }(X)` takes
        // the `2L`
        let yes = || Expr::Const(true.into());
        rejects_another_kind(&apply(lambda(SType::SInt, yes()), int_holding_long()));
        assert!(try_eval_out_wo_ctx::<bool>(&apply(
            lambda(SType::SLong, yes()),
            int_holding_long()
        ))
        .unwrap());
    }

    #[test]
    fn a_lambda_result_rejects_another_kind() {
        // `{ (x: Int) => X }(1) == 2` (`values.scala:1080`)
        let applied = apply(lambda(SType::SInt, int_holding_long()), int(1));
        rejects_another_kind(&relation(RelationOp::Eq, applied, int(2)));
    }

    #[test]
    fn a_fold_zero_rejects_another_kind() {
        // `Coll[Int]().fold(X, { (p: (Int, Int)) => p._1 }) == 2` (`transformers.scala:227`)
        let int_pair = SType::STuple(STuple::pair(SType::SInt, SType::SInt).into());
        let p: Expr = ValUse {
            val_id: 1.into(),
            tpe: int_pair.clone(),
        }
        .into();
        let op = lambda(
            int_pair,
            SelectField::new(p, 1u8.try_into().unwrap()).unwrap().into(),
        );
        let empty: Expr = Collection::new(SType::SInt, vec![]).unwrap().into();
        let fold: Expr = Fold::new(empty, int_holding_long(), op).unwrap().into();
        rejects_another_kind(&relation(RelationOp::Eq, fold, int(2)));
    }

    /// at activated 3 and the given tree version
    fn eval_at(expr: &Expr, ctx: &Context<'static>, tree_version: u8) -> Result<i32, EvalError> {
        try_eval_out_with_version::<i32>(expr, ctx, tree_version, 3)
    }

    #[test]
    fn by_index_checks_its_default_by_tree_version() {
        // lazy from tree version 3, eager below (`transformers.scala:258-276`)
        let ctx = force_any_val::<Context>();
        let empty: Expr = Collection::new(SType::SInt, vec![]).unwrap().into();
        let five: Expr = Collection::new(SType::SInt, vec![int(5)]).unwrap().into();
        let at0 = |coll: Expr| -> Expr {
            ByIndex::new(coll, int(0), Some(Box::new(int_holding_long())))
                .unwrap()
                .into()
        };
        assert!(is_value_check_error(&eval_at(&at0(empty), &ctx, 3)));
        assert_eq!(eval_at(&at0(five.clone()), &ctx, 3).unwrap(), 5);
        assert!(is_value_check_error(&eval_at(&at0(five), &ctx, 2)));
    }

    #[test]
    fn option_get_or_else_checks_its_default_by_tree_version() {
        // lazy from tree version 3, eager below (`transformers.scala:627-643`); var 0 is the
        // `Int` 5, var 1 is absent
        let ctx = ctx_with_var0("040a");
        let var = |id: u8| -> Expr {
            GetVar {
                var_id: id,
                var_tpe: SType::SInt,
            }
            .into()
        };
        let or_x = |input: Expr| -> Expr {
            OptionGetOrElse::new(input, int_holding_long())
                .unwrap()
                .into()
        };
        assert!(is_value_check_error(&eval_at(&or_x(var(1)), &ctx, 3)));
        assert_eq!(eval_at(&or_x(var(0)), &ctx, 3).unwrap(), 5);
        assert!(is_value_check_error(&eval_at(&or_x(var(0)), &ctx, 2)));
    }

    #[test]
    fn an_option_map_lambda_checks_its_argument() {
        // `If(false, Some(1), Some(2L)).map({ (x: Int) => x })`: a method's lambda checks its
        // argument as any lambda does (`values.scala:1074`)
        let opt: Expr = If {
            condition: Box::new(Expr::Const(false.into())),
            true_branch: Box::new(Expr::Const(Some(1i32).into())),
            false_branch: Box::new(Expr::Const(Some(2i64).into())),
        }
        .into();
        let map = |arg_tpe: SType| -> Expr {
            MethodCall {
                obj: Box::new(opt.clone()),
                method: soption::MAP_METHOD
                    .clone()
                    .with_concrete_types(
                        &[
                            (STypeVar::iv(), arg_tpe.clone()),
                            (STypeVar::ov(), arg_tpe.clone()),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    )
                    .unwrap(),
                args: vec![lambda(
                    arg_tpe.clone(),
                    ValUse {
                        val_id: 1.into(),
                        tpe: arg_tpe,
                    }
                    .into(),
                )],
                explicit_type_args: Default::default(),
            }
            .into()
        };
        rejects_another_kind(&map(SType::SInt));
        assert_eq!(
            try_eval_out_wo_ctx::<Value>(&map(SType::SLong)).unwrap(),
            Value::Opt(Some(Box::new(Value::Long(2))))
        );
    }
}

#[cfg(test)]
#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod root_wrap_tests {
    //! JVM parity: after the substitution, `toValidScriptTypeJITC` wraps a Boolean root in
    //! `BoolToSigmaProp`, keeps a SigmaProp root and throws for any other (v6.0.7
    //! `Interpreter.scala:598-602`). The wrap is a node, so it is evaluated and charged.
    use super::*;
    use ergotree_ir::chain::ergo_box::{ErgoBox, NonMandatoryRegisters};
    use ergotree_ir::serialization::SigmaSerializable;
    use sigma_test_util::force_any_val;

    /// `reduce_to_crypto` of the tree `hex` under activated version 3, with a self box without
    /// registers: the result and the JIT cost charged
    fn reduce(hex: &str) -> (Result<SigmaBoolean, EvalError>, u64) {
        let tree = ErgoTree::sigma_parse_bytes(&base16::decode(hex.as_bytes()).unwrap()).unwrap();
        let self_box =
            force_any_val::<ErgoBox>().with_additional_registers(NonMandatoryRegisters::empty());
        let mut ctx = Context {
            self_box: &self_box,
            ..force_any_val::<Context>()
        };
        ctx.activated_script_version_byte = 3;
        ctx.jit_cost_limit = None;
        let before = ctx.jit_cost_value();
        let res = reduce_to_crypto(&tree, &ctx).map(|r| r.sigma_prop);
        (res, ctx.jit_cost_value() - before)
    }

    #[test]
    fn a_boolean_root_after_the_substitution_is_wrapped() {
        // SANTA #20: the root DeserializeRegister(R4, SigmaProp, default true), R4 absent
        let (res, wrapped) = reduce("00d50408010101");
        assert_eq!(res, Ok(SigmaBoolean::TrivialProp(true)));
        // the default written as sigmaProp(true): the same BoolToSigmaProp evaluated, in a tree
        // one byte longer (20 jit), so the wrap is charged as that node is
        let (res, explicit) = reduce("00d5040801d10101");
        assert_eq!(res, Ok(SigmaBoolean::TrivialProp(true)));
        assert_eq!(wrapped + 20, explicit);
    }

    #[test]
    fn a_root_of_another_type_fails() {
        // the default Int 1 is neither Boolean nor SigmaProp
        assert!(reduce("00d50408010402").0.is_err());
    }
}
