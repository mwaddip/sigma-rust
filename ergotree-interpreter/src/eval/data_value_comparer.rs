//! Per-type equality comparison with JIT costing.
//!
//! Mirrors Scala sigmastate-interpreter's `DataValueComparer`: every equality
//! check charges a type-specific cost before (or instead of) structural
//! comparison. Constants are drawn from the Scala reference and validated by
//! PR 846 (arkadianet/jit-costing) against 19,549 mainnet transactions.

use ergotree_ir::chain::context::Context;
use ergotree_ir::mir::coll_value::CollValue;
use ergotree_ir::mir::value::Value;
use ergotree_ir::sigma_protocol::sigma_boolean::{
    SigmaBoolean, SigmaConjecture, SigmaProofOfKnowledgeTree,
};
use ergotree_ir::types::stype::SType;

use super::EvalError;

// --- Fixed per-type equality costs (JitCost units) ---
const EQ_PRIM_COST: u64 = 3;
const EQ_BIGINT_COST: u64 = 5;
const EQ_GROUP_ELEMENT_COST: u64 = 172;
const EQ_TUPLE_COST: u64 = 4;
const EQ_OPTION_COST: u64 = 4;
const EQ_AVL_TREE_COST: u64 = 6;
const EQ_BOX_COST: u64 = 6;
const EQ_PREHEADER_COST: u64 = 4;
const EQ_HEADER_COST: u64 = 6;

// Scala's MatchType dispatch cost (`CostOf_MatchType` = 1), charged on each
// type-match step: the collection dispatch (before its length-mismatch
// short-circuit, so it is always paid), the SigmaProp dispatch, and once per
// SigmaBoolean tree node.
const MATCH_TYPE_COST: u64 = 1;

// Per-element collection equality costs as (base, per_chunk, chunk_size),
// matching `Context::add_per_item_jit_cost`'s argument shape.
const EQ_COLL_BYTE_PER_ITEM: (u32, u32, u32) = (15, 2, 128);
const EQ_COLL_SHORT_PER_ITEM: (u32, u32, u32) = (15, 2, 96);
const EQ_COLL_INT_PER_ITEM: (u32, u32, u32) = (15, 2, 64);
const EQ_COLL_LONG_PER_ITEM: (u32, u32, u32) = (15, 2, 48);
const EQ_COLL_BOOLEAN_PER_ITEM: (u32, u32, u32) = (15, 2, 128);
const EQ_COLL_BIGINT_PER_ITEM: (u32, u32, u32) = (15, 7, 5);
const EQ_COLL_GROUP_ELEMENT_PER_ITEM: (u32, u32, u32) = (15, 5, 1);
const EQ_COLL_AVL_TREE_PER_ITEM: (u32, u32, u32) = (15, 5, 2);
const EQ_COLL_BOX_PER_ITEM: (u32, u32, u32) = (15, 5, 1);
const EQ_COLL_PREHEADER_PER_ITEM: (u32, u32, u32) = (15, 3, 1);
const EQ_COLL_HEADER_PER_ITEM: (u32, u32, u32) = (15, 5, 1);
const EQ_COLL_DEFAULT_PER_ITEM: (u32, u32, u32) = (10, 2, 1);

/// A `Byte`, `Short`, `Int` or `Long` value as an `i64`: what Scala's `==` compares for two boxed
/// integers (`BoxesRunTime.equalsNumNum`), whatever their widths. `None` for any other value
pub(crate) fn integral(v: &Value<'_>) -> Option<i64> {
    match v {
        Value::Byte(x) => Some(i64::from(*x)),
        Value::Short(x) => Some(i64::from(*x)),
        Value::Int(x) => Some(i64::from(*x)),
        Value::Long(x) => Some(*x),
        _ => None,
    }
}

/// sigmastate's `equalDataValues` ends in `sys.error("Cannot compare $l and $r: unknown type")`
/// for a left value it has no case for (v6.0.6 `DataValueComparer.scala:412-413`)
fn no_case(kind: &str) -> EvalError {
    EvalError::Misc(format!("Cannot compare a {kind} value: unknown type"))
}

/// Compare two values for equality, charging per-type costs to `ctx`: sigmastate's
/// `DataValueComparer.equalDataValues` (v6.0.6 `DataValueComparer.scala:310-416`), the one
/// comparison behind `EQ`, `NEQ` and `Coll.indexOf`.
///
/// It dispatches on the class of the LEFT value alone: each case charges its own cost, and then
/// a right value of another class is unequal. So there is one arm per [`Value`] variant, in
/// sigmastate's order, and no wildcard arm: a new variant must be given its row here.
///
/// Returns `Ok(true)` / `Ok(false)`, an error for a left value sigmastate has no case for, or
/// an `EvalError::CostError` if the cumulative JIT cost limit is exceeded during charging.
pub(crate) fn eq_with_cost<'ctx>(
    lv: &Value<'ctx>,
    rv: &Value<'ctx>,
    ctx: &Context<'ctx>,
) -> Result<bool, EvalError> {
    match lv {
        // case 1 (`:313-316`): Scala's `==` on two boxed values. Two integers are equal when
        // their values are, whatever their widths (SANTA `Coll.specialisation_blind_ops` X1,
        // X2); a `Boolean` equals only a `Boolean`; anything else is unequal
        Value::Boolean(_) | Value::Byte(_) | Value::Short(_) | Value::Int(_) | Value::Long(_) => {
            ctx.add_jit_cost(EQ_PRIM_COST)?;
            Ok(match (integral(lv), integral(rv)) {
                (Some(l), Some(r)) => l == r,
                _ => lv == rv,
            })
        }

        // case 2 (`:318-329`): the MatchType dispatch cost is always paid, also against a right
        // value that is not a collection
        Value::Coll(l_coll) => {
            ctx.add_jit_cost(MATCH_TYPE_COST)?;
            let r_coll = match rv {
                Value::Coll(r_coll) => r_coll,
                _ => return Ok(false),
            };
            let n = l_coll.len();
            // Scala short-circuits on a length or item-type mismatch without charging
            // per-item or base cost (bug 5; `DataValueComparer.scala:318-326`, `coll1.tItem !=
            // coll2.tItem`). The item types decide even for two empty collections.
            if n != r_coll.len() || l_coll.elem_tpe() != r_coll.elem_tpe() {
                return Ok(false);
            }
            // Both JVM coll-eq paths short-circuit the *per-item cost* on the
            // first mismatch. `addSeqCost(costKind){ block }` charges
            // `costKind.cost(i)`, where the block returns `i` = the number of
            // items actually compared (CErgoTreeEvaluator.scala:412;
            // `equalCOA_Prim`/`equalColls` both loop `while (i < len && okEqual)`
            // and return `i`). So the per-item charge is on the *compared count*
            // (first-mismatch index + 1, or the full length when equal), NOT the
            // full operand length. (h28931 colleq cost divergence.)
            match coll_eq_cost(l_coll) {
                Some((base, per_chunk, chunk_size)) => {
                    // COA leaf colls: JVM `equalCOA_*` bulk-compares and returns
                    // the compared count. Derive it from the first mismatch; the
                    // result stays the proven `lv == rv`.
                    let compared = coll_eq_compared_count(l_coll, r_coll);
                    ctx.add_per_item_jit_cost(base, per_chunk, chunk_size, compared)?;
                    Ok(lv == rv)
                }
                None => {
                    // Composite element colls: JVM generic `equalColls` recurses
                    // per element (nested costs accrue during the loop) and
                    // charges `EQ_Coll` on the compared count *after* the loop.
                    let (base, per_chunk, chunk_size) = EQ_COLL_DEFAULT_PER_ITEM;
                    let l_items = l_coll.as_vec();
                    let r_items = r_coll.as_vec();
                    let mut compared: u32 = 0;
                    let mut all_equal = true;
                    for (l, r) in l_items.iter().zip(r_items.iter()) {
                        compared += 1;
                        if !eq_with_cost(l, r, ctx)? {
                            all_equal = false;
                            break;
                        }
                    }
                    ctx.add_per_item_jit_cost(base, per_chunk, chunk_size, compared)?;
                    Ok(all_equal)
                }
            }
        }

        // case 3 (`:331-338`): the second items are compared only if the first are equal
        Value::Pair(l) => {
            ctx.add_jit_cost(EQ_TUPLE_COST)?;
            match rv {
                Value::Pair(r) => {
                    Ok(eq_with_cost(&l.0, &r.0, ctx)? && eq_with_cost(&l.1, &r.1, ctx)?)
                }
                _ => Ok(false),
            }
        }

        // case 4 (`:340-341`)
        Value::GroupElement(_) => {
            ctx.add_jit_cost(EQ_GROUP_ELEMENT_COST)?;
            Ok(lv == rv)
        }

        // case 5 (`:343-351`): a `BigInt` never equals an `UnsignedBigInt`
        Value::BigInt(_) | Value::UnsignedBigInt(_) => {
            ctx.add_jit_cost(EQ_BIGINT_COST)?;
            Ok(lv == rv)
        }

        // `:353-361`: one MatchType for the dispatch, then `equalSigmaBoolean` walks both trees
        Value::SigmaProp(l) => {
            ctx.add_jit_cost(MATCH_TYPE_COST)?;
            match rv {
                Value::SigmaProp(r) => eq_sigma_bool_with_cost(l.value(), r.value(), ctx),
                _ => Ok(false),
            }
        }

        // case 6 (`:363-366`)
        Value::AvlTree(_) => {
            ctx.add_jit_cost(EQ_AVL_TREE_COST)?;
            Ok(lv == rv)
        }

        // case 7 (`:368-384`)
        Value::Opt(l_opt) => {
            ctx.add_jit_cost(EQ_OPTION_COST)?;
            match rv {
                Value::Opt(r_opt) => match (l_opt.as_deref(), r_opt.as_deref()) {
                    (None, None) => Ok(true),
                    (Some(l), Some(r)) => eq_with_cost(l, r, ctx),
                    _ => Ok(false),
                },
                _ => Ok(false),
            }
        }

        // case 8 (`:385-388`)
        Value::PreHeader(_) => {
            ctx.add_jit_cost(EQ_PREHEADER_COST)?;
            Ok(lv == rv)
        }

        // case 9 (`:389-392`)
        Value::Header(_) => {
            ctx.add_jit_cost(EQ_HEADER_COST)?;
            Ok(lv == rv)
        }

        // case 10 (`:393-396`)
        Value::CBox(_) => {
            ctx.add_jit_cost(EQ_BOX_COST)?;
            Ok(lv == rv)
        }

        // `:397-408`: one MatchType; two strings of different lengths are unequal with nothing
        // more charged, two of the same length are charged `EQ_COA_Short` over that length
        // whatever the answer. `String.length` counts UTF-16 code units
        Value::String(l) => {
            ctx.add_jit_cost(MATCH_TYPE_COST)?;
            match rv {
                Value::String(r) => {
                    let len = l.encode_utf16().count();
                    if len != r.encode_utf16().count() {
                        return Ok(false);
                    }
                    let (base, per_chunk, chunk_size) = EQ_COLL_SHORT_PER_ITEM;
                    ctx.add_per_item_jit_cost(base, per_chunk, chunk_size, len as u32)?;
                    Ok(l == r)
                }
                _ => Ok(false),
            }
        }

        // `:409-410`: the `Unit` case charges nothing (SANTA 6.0.7: 3 less per unit than
        // `EQ_Prim`)
        Value::Unit => Ok(matches!(rv, Value::Unit)),

        // no case in sigmastate: it throws before any cost (`:412-413`)
        Value::Context => Err(no_case("Context")),
        Value::Global => Err(no_case("Global")),
        Value::Lambda(_) => Err(no_case("function")),
    }
}

/// Compare two `SigmaBoolean` trees, charging cost per Scala `equalSigmaBoolean`
/// (DataValueComparer.scala:253): `MATCH_TYPE_COST` once per node + the
/// `EQ_GROUP_ELEMENT_COST` per EcPoint. Short-circuits exactly as Scala's `&&`
/// and length checks do, so an unequal pair is charged only up to the first
/// mismatch.
fn eq_sigma_bool_with_cost(
    l: &SigmaBoolean,
    r: &SigmaBoolean,
    ctx: &Context<'_>,
) -> Result<bool, EvalError> {
    use SigmaBoolean::{ProofOfKnowledge, SigmaConjecture as Conj, TrivialProp};
    use SigmaConjecture::{Cand, Cor, Cthreshold};
    use SigmaProofOfKnowledgeTree::{ProveDhTuple, ProveDlog};
    ctx.add_jit_cost(MATCH_TYPE_COST)?; // once per node
    match (l, r) {
        (ProofOfKnowledge(ProveDlog(x)), ProofOfKnowledge(ProveDlog(y))) => {
            ctx.add_jit_cost(EQ_GROUP_ELEMENT_COST)?;
            Ok(x.h == y.h)
        }
        (ProofOfKnowledge(ProveDhTuple(x)), ProofOfKnowledge(ProveDhTuple(y))) => {
            // Four `equalECPoint`s, &&-short-circuited: each one reached charges
            // EQ_GROUP_ELEMENT_COST (including the mismatching one); a mismatch
            // stops the rest.
            for (lp, rp) in [(&x.g, &y.g), (&x.h, &y.h), (&x.u, &y.u), (&x.v, &y.v)] {
                ctx.add_jit_cost(EQ_GROUP_ELEMENT_COST)?;
                if lp != rp {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (TrivialProp(a), TrivialProp(b)) => Ok(a == b),
        (Conj(Cand(x)), Conj(Cand(y))) => eq_sigma_bools_with_cost(&x.items, &y.items, ctx),
        (Conj(Cor(x)), Conj(Cor(y))) => eq_sigma_bools_with_cost(&x.items, &y.items, ctx),
        (Conj(Cthreshold(x)), Conj(Cthreshold(y))) => Ok(x.k == y.k
            && eq_sigma_bools_with_cost(x.children.as_slice(), y.children.as_slice(), ctx)?),
        // Mismatched node kinds, dispatched on the LEFT per Scala's guard
        // structure: a leaf left (ProveDlog / ProveDhTuple / TrivialProp)
        // falls into that arm's inner `case _ => false`; the conjecture arms
        // are *guarded* (`case CAND(children) if r.isInstanceOf[CAND]`), so a
        // conjecture left vs a different-kind right falls through to the
        // top-level `case _ => sys.error(...)` — eval throws. The node's
        // MatchType is charged above either way, matching the JVM's
        // charge-then-throw shape.
        _ => match l {
            ProofOfKnowledge(_) | TrivialProp(_) => Ok(false),
            Conj(_) => Err(EvalError::Misc(format!(
                "Cannot compare SigmaBoolean values {l:?} and {r:?}: unknown type"
            ))),
        },
    }
}

/// `equalSigmaBooleans` (DataValueComparer.scala:241): length check (no per-item
/// charge on mismatch), then per-child `eq_sigma_bool_with_cost`, short-circuiting.
fn eq_sigma_bools_with_cost(
    xs: &[SigmaBoolean],
    ys: &[SigmaBoolean],
    ctx: &Context<'_>,
) -> Result<bool, EvalError> {
    if xs.len() != ys.len() {
        return Ok(false);
    }
    for (x, y) in xs.iter().zip(ys.iter()) {
        if !eq_sigma_bool_with_cost(x, y, ctx)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Per-item equality cost for a collection (keyed on element type) **and**
/// whether the collection is a COA (CollOverArray) leaf — the single source of
/// truth for the leaf-vs-composite partition. `Some((base, per_chunk,
/// chunk_size))` = COA leaf, which JVM `equalColls_Dispatch` bulk-compares
/// without recursion. `None` = a composite element type (Coll/Tuple/Option/
/// SigmaProp/...), which pays `EQ_COLL_DEFAULT_PER_ITEM` and recurses per
/// element via JVM's generic `equalColls`. Mirrors Scala's per-type array cost
/// kinds.
fn coll_eq_cost(coll: &CollValue<'_>) -> Option<(u32, u32, u32)> {
    // keyed on the item type alone, as the JVM's dispatch on `tItem` is: the collection's form
    // never changes the cost
    match coll.elem_tpe() {
        SType::SByte => Some(EQ_COLL_BYTE_PER_ITEM),
        SType::SShort => Some(EQ_COLL_SHORT_PER_ITEM),
        SType::SInt => Some(EQ_COLL_INT_PER_ITEM),
        SType::SLong => Some(EQ_COLL_LONG_PER_ITEM),
        SType::SBoolean => Some(EQ_COLL_BOOLEAN_PER_ITEM),
        SType::SBigInt | SType::SUnsignedBigInt => Some(EQ_COLL_BIGINT_PER_ITEM),
        SType::SGroupElement => Some(EQ_COLL_GROUP_ELEMENT_PER_ITEM),
        SType::SAvlTree => Some(EQ_COLL_AVL_TREE_PER_ITEM),
        SType::SBox => Some(EQ_COLL_BOX_PER_ITEM),
        SType::SPreHeader => Some(EQ_COLL_PREHEADER_PER_ITEM),
        SType::SHeader => Some(EQ_COLL_HEADER_PER_ITEM),
        _ => None,
    }
}

/// JVM coll-eq per-item cost short-circuits on the first mismatch: the count it
/// charges is the number of items examined before the result is decided —
/// `first_mismatch_index + 1`, or the full length when the operands are equal
/// (`equalCOA_Prim`/`equalColls` return this `i`). Returns that count for a COA
/// (leaf) collection pair already known to have equal length. Mixed collection
/// kinds cannot occur for a COA element type (`Coll[Byte]` is always native,
/// every other leaf type always wrapped with a matching element type); the `n`
/// fallback keeps the prior full-length charge for that unreachable shape.
fn coll_eq_compared_count(l: &CollValue<'_>, r: &CollValue<'_>) -> u32 {
    let first_mismatch = match (l, r) {
        (CollValue::Bytes(la), CollValue::Bytes(ra)) => {
            la.iter().zip(ra.iter()).position(|(a, b)| a != b)
        }
        (CollValue::Generic { items: li, .. }, CollValue::Generic { items: ri, .. }) => {
            li.iter().zip(ri.iter()).position(|(a, b)| a != b)
        }
        // two forms of the same item type: compare item by item
        _ => (0..l.len()).position(|i| l.get_val(i) != r.get_val(i)),
    };
    match first_mismatch {
        Some(idx) => idx as u32 + 1,
        None => l.len() as u32,
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use alloc::vec;
    use sigma_test_util::force_any_val;

    #[test]
    fn strings_compare_as_the_text_the_jvm_decodes() {
        // sigma-state 6.0.7 (the round's `measure/rows/eval.txt`), the same at tree versions 0
        // and 3. S is `ED A0 80`, an encoded surrogate: ONE U+FFFD on the JVM. R is a real
        // U+FFFD. An operand of `==` that is of type `String` does not evaluate (sigmastate
        // checks the operand's value against its type, and that check has no `String` case:
        // the last row), so two strings are compared inside a collection, a pair or an option
        use crate::eval::test_util::eval_body;
        use ergotree_ir::ergo_tree::ErgoTreeVersion;
        let item = |text: &str| {
            let len = text.len() / 2;
            if len < 0x80 {
                alloc::format!("{len:02x}{text}")
            } else {
                alloc::format!("{:02x}{:02x}{text}", (len & 0x7f) | 0x80, len >> 7)
            }
        };
        let coll = |text: &str| alloc::format!("0c6601{}", item(text));
        let pair = |text: &str| alloc::format!("3c6604{}02", item(text));
        let (s, r) = ("eda080", "efbfbd");
        let rows = [
            (alloc::format!("93{}{}", coll(r), coll(r)), true, 41),
            (alloc::format!("93{}{}", coll("41"), coll("42")), false, 41),
            (alloc::format!("93{}{}", coll(s), coll(r)), true, 41),
            (
                alloc::format!("93{}{}", coll(s), coll(&r.repeat(3))),
                false,
                24,
            ),
            (alloc::format!("93{}{}", coll(s), coll(s)), true, 41),
            (
                alloc::format!("93{}{}", coll(&s.repeat(40)), coll(&s.repeat(40))),
                true,
                41,
            ),
            (
                alloc::format!("93{}{}", coll(&r.repeat(40)), coll(&r.repeat(40))),
                true,
                41,
            ),
            (
                alloc::format!("93{}{}", coll(&r.repeat(120)), coll(&r.repeat(120))),
                true,
                43,
            ),
            (
                alloc::format!("93{}{}", coll("e08080"), coll(&r.repeat(3))),
                true,
                41,
            ),
            (alloc::format!("93{}{}", coll("e08080"), coll(r)), false, 24),
            (alloc::format!("94{}{}", coll(s), coll(r)), false, 41),
            (alloc::format!("93{}{}", pair(s), pair(r)), true, 35),
            (
                alloc::format!("93{}{}", pair(s), pair(&r.repeat(3))),
                false,
                15,
            ),
            (alloc::format!("93{}{}", pair(r), pair(r)), true, 35),
        ];
        for version in [ErgoTreeVersion::V0, ErgoTreeVersion::V3] {
            for (body, value, cost) in &rows {
                let (res, c) = eval_body(body, version);
                assert_eq!(res.unwrap(), Value::Boolean(*value), "{body}");
                assert_eq!(c, *cost, "{body}");
            }
            let (res, c) = eval_body("936603efbfbd6603efbfbd", version);
            assert!(res.is_err(), "{res:?}");
            assert_eq!(c, 5);
        }
    }

    #[test]
    fn strings_of_other_producers_compare_as_the_text_the_jvm_decodes() {
        // sigma-state 6.0.7 (the round's `measure/review/rows.txt`), S and R as above, each
        // compared with R and with three R: an `Option[String]` constant (from tree version
        // 3), `deserializeTo[Coll[String]]` of bytes the script holds (a version 3 method),
        // and a `String` constant of a tree that `substConstants` writes again
        use crate::eval::test_util::eval_body;
        use ergotree_ir::ergo_tree::ErgoTreeVersion;
        let (s, r, rrr) = ("03eda080", "03efbfbd", "09efbfbdefbfbdefbfbd");
        let option = |l: &str, r: &str| alloc::format!("93246601{l}246601{r}");
        let deserialized = |l: &str, r: &str| {
            alloc::format!("93dc6a04dd010e{:02x}01{l}0c660c6601{r}", l.len() / 2 + 1)
        };
        let substituted = |l: &str, r: &str| {
            let tree = |text: &str| alloc::format!("0e{:02x}100166{text}7300", text.len() / 2 + 5);
            alloc::format!("93740e09100166{l}730010001000{}", tree(r))
        };
        let v3 = [ErgoTreeVersion::V3];
        let both = [ErgoTreeVersion::V0, ErgoTreeVersion::V3];
        let rows = [
            (option(s, r), true, 32, &v3[..]),
            (option(s, rrr), false, 15, &v3[..]),
            (option(r, r), true, 32, &v3[..]),
            (deserialized(s, r), true, 182, &v3[..]),
            (deserialized(s, rrr), false, 165, &v3[..]),
            (deserialized(r, r), true, 182, &v3[..]),
            (substituted(s, r), true, 238, &both[..]),
            (substituted(s, rrr), false, 221, &both[..]),
            (substituted(r, r), true, 238, &both[..]),
        ];
        for (body, value, cost, versions) in &rows {
            for version in *versions {
                let (res, c) = eval_body(body, *version);
                assert_eq!(res.unwrap(), Value::Boolean(*value), "{body}");
                assert_eq!(c, *cost, "{body}");
            }
        }
    }

    #[test]
    fn primitive_eq_charges_prim_cost() {
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        let lv: Value<'_> = Value::Int(42);
        let rv: Value<'_> = Value::Int(42);
        assert!(eq_with_cost(&lv, &rv, &ctx).unwrap());
        assert_eq!(ctx.jit_cost_value() - before, EQ_PRIM_COST);
    }

    /// sigmastate's `case _: Unit => okEqual = r.isInstanceOf[Unit]` charges nothing (v6.0.6
    /// `DataValueComparer.scala:409-410`). SANTA 6.0.7 prices it end to end in its third
    /// reference-store reply: R9 @147 and R7m @167, 3 less per unit than `EQ_PRIM_COST` gives
    #[test]
    fn unit_eq_charges_nothing() {
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        assert!(eq_with_cost(&Value::Unit, &Value::Unit, &ctx).unwrap());
        assert!(!eq_with_cost(&Value::Unit, &Value::Int(1), &ctx).unwrap());
        assert_eq!(ctx.jit_cost_value() - before, 0);
        // the dispatch is on the LEFT value: a unit on the right is the left's case
        assert!(!eq_with_cost(&Value::Int(1), &Value::Unit, &ctx).unwrap());
        assert_eq!(ctx.jit_cost_value() - before, EQ_PRIM_COST);
    }

    #[test]
    fn forms_compare_equal_at_same_cost() {
        // A specialised and a generic collection with the same items are equal on the JVM
        // (SANTA `Coll.specialisation_blind_ops` Q1-Q3), at the cost of their item type
        use ergotree_ir::mir::coll_value::CollValue;
        let cost = |l: Value<'static>, r: Value<'static>| {
            let ctx = force_any_val::<Context>();
            let before = ctx.jit_cost_value();
            let eq = eq_with_cost(&l, &r, &ctx).unwrap();
            (eq, ctx.jit_cost_value() - before)
        };
        let bytes = || Value::Coll(CollValue::Bytes([1i8, 2, 3].as_slice().into()));
        let gen_bytes = Value::Coll(CollValue::generic(
            SType::SByte,
            vec![Value::Byte(1), Value::Byte(2), Value::Byte(3)],
        ));
        assert_eq!(cost(bytes(), gen_bytes.clone()), cost(bytes(), bytes()));
        assert!(cost(gen_bytes, bytes()).0);
        let ints = || Value::Coll(CollValue::Ints([1, 2, 3].as_slice().into()));
        let gen_ints = Value::Coll(CollValue::generic(
            SType::SInt,
            vec![Value::Int(1), Value::Int(2), Value::Int(3)],
        ));
        assert_eq!(
            cost(ints(), gen_ints.clone()),
            cost(gen_ints.clone(), gen_ints)
        );
        assert!(cost(ints(), ints()).0);
    }

    #[test]
    fn coll_eq_charges_match_type_plus_per_item() {
        // Bug 4 regression: equal Coll[Int] of length 3 must pay MatchType(1)
        // + per-item SInt cost (base=15, per_chunk=2, chunk_size=64 → 1 chunk
        // of 3 items = 15 + 2 = 17). Total = 1 + 17 = 18.
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        let items: Arc<[Value<'_>]> = Arc::from(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        let lv: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SInt,
            items: items.clone(),
        });
        let rv: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SInt,
            items,
        });
        assert!(eq_with_cost(&lv, &rv, &ctx).unwrap());
        assert_eq!(ctx.jit_cost_value() - before, 18);
    }

    #[test]
    fn nested_coll_eq_recurses_element_cost() {
        // B4: a Coll with a composite element type (here Coll[Coll[Int]]) must
        // recurse eq_with_cost per element -- charging the nested MatchType +
        // per-item -- matching JVM's generic `equalColls`; leaf-element colls
        // (the inner Coll[Int]) still bulk-compare. For [[1,2,3],[4,5,6]] vs
        // itself: outer = MatchType 1 + default per-item (10,2,1) over 2 =
        // 1+14 = 15; each inner Coll[Int] len 3 = MatchType 1 + (15,2,64) over
        // 3 = 1+17 = 18; total = 15 + 18 + 18 = 51 (pre-fix, no recursion: 15).
        let ctx = force_any_val::<Context>();
        let inner_a: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SInt,
            items: Arc::from(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
        });
        let inner_b: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SInt,
            items: Arc::from(vec![Value::Int(4), Value::Int(5), Value::Int(6)]),
        });
        let outer: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SColl(Arc::new(SType::SInt)),
            items: Arc::from(vec![inner_a, inner_b]),
        });
        let before = ctx.jit_cost_value();
        assert!(eq_with_cost(&outer, &outer.clone(), &ctx).unwrap());
        assert_eq!(ctx.jit_cost_value() - before, 51);
    }

    #[test]
    fn coll_eq_length_mismatch_short_circuits() {
        // Bug 5 regression: Coll[Int] of different lengths must pay only the
        // MatchType(1) dispatch cost, NOT per-item cost (early false return).
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        let items_a: Arc<[Value<'_>]> = Arc::from(vec![Value::Int(1), Value::Int(2)]);
        let items_b: Arc<[Value<'_>]> = Arc::from(vec![Value::Int(1)]);
        let lv: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SInt,
            items: items_a,
        });
        let rv: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SInt,
            items: items_b,
        });
        assert!(!eq_with_cost(&lv, &rv, &ctx).unwrap());
        assert_eq!(ctx.jit_cost_value() - before, MATCH_TYPE_COST);
    }

    /// Charge for equality of two equal empty `Coll[elem_tpe]` (wrapped form):
    /// MatchType dispatch + per-item cost at n=0.
    fn empty_wrapped_coll_eq_cost(elem_tpe: SType) -> u64 {
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        let empty: Arc<[Value<'_>]> = Arc::from(Vec::<Value<'_>>::new());
        let lv: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: elem_tpe.clone(),
            items: empty.clone(),
        });
        let rv: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe,
            items: empty,
        });
        assert!(eq_with_cost(&lv, &rv, &ctx).unwrap());
        ctx.jit_cost_value() - before
    }

    #[test]
    fn empty_coll_eq_cs_ge_2_charges_one_chunk() {
        // n=0 regression (the mainnet 1,520,814 finding): for chunkSize>=2
        // element types an empty Coll must still pay one chunk, mirroring Scala
        // PerItemCost.chunks(0) = (0-1)/cs + 1 = 1 (signed, toward-zero div).
        // Cost = MatchType(1) + base(15) + 1*per_chunk. Before the fix Rust's
        // ceiling chunks(0)=0 charged only base -> undercharge vs the JVM.
        // Long(cs=48) / Int(cs=64) / Short(cs=96) / Boolean(cs=128): per_chunk=2.
        assert_eq!(empty_wrapped_coll_eq_cost(SType::SLong), 18);
        assert_eq!(empty_wrapped_coll_eq_cost(SType::SInt), 18);
        assert_eq!(empty_wrapped_coll_eq_cost(SType::SShort), 18);
        assert_eq!(empty_wrapped_coll_eq_cost(SType::SBoolean), 18);
        // BigInt: per_chunk=7, cs=5 -> 1 + 15 + 7 = 23.
        assert_eq!(empty_wrapped_coll_eq_cost(SType::SBigInt), 23);
        // AvlTree: per_chunk=5, cs=2 -> 1 + 15 + 5 = 21.
        assert_eq!(empty_wrapped_coll_eq_cost(SType::SAvlTree), 21);
    }

    #[test]
    fn empty_coll_byte_eq_charges_one_chunk() {
        // Coll[Byte] is a NativeColl (cs=128, per_chunk=2). Empty -> one chunk:
        // MatchType(1) + base(15) + 2 = 18.
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        let empty_bytes: Arc<[i8]> = Arc::from(Vec::<i8>::new());
        let lv: Value<'_> = Value::Coll(CollValue::Bytes(empty_bytes.clone()));
        let rv: Value<'_> = Value::Coll(CollValue::Bytes(empty_bytes));
        assert!(eq_with_cost(&lv, &rv, &ctx).unwrap());
        assert_eq!(ctx.jit_cost_value() - before, 18);
    }

    #[test]
    fn empty_coll_cs_eq_1_charges_base_only() {
        // chunkSize==1 types stay at base only: Scala chunks(0) = (0-1)/1 + 1 = 0.
        // Cost = MatchType(1) + base(15) + 0 = 16. These already matched the JVM
        // before the fix; assert the n=0 change does NOT move them.
        assert_eq!(empty_wrapped_coll_eq_cost(SType::SBox), 16);
        assert_eq!(empty_wrapped_coll_eq_cost(SType::SGroupElement), 16);
        assert_eq!(empty_wrapped_coll_eq_cost(SType::SHeader), 16);
        assert_eq!(empty_wrapped_coll_eq_cost(SType::SPreHeader), 16);
    }

    #[test]
    fn nonempty_coll_long_path_byte_identical() {
        // Prove n>=1 is byte-identical to the old ceiling formula across a chunk
        // boundary. Long: base=15, per_chunk=2, cs=48. Total = MatchType(1) +
        // base + chunks(n)*per_chunk, where chunks(n) == ceil(n/48) for n>=1.
        //   len 1  -> chunks=1 -> 1 + 15 + 2  = 18
        //   len 48 -> chunks=1 -> 1 + 15 + 2  = 18  (last item still in chunk 1)
        //   len 49 -> chunks=2 -> 1 + 15 + 4  = 20  (spills into chunk 2)
        let cost = |len: usize| -> u64 {
            let ctx = force_any_val::<Context>();
            let before = ctx.jit_cost_value();
            let items: Arc<[Value<'_>]> =
                Arc::from((0..len as i64).map(Value::Long).collect::<Vec<Value<'_>>>());
            let lv: Value<'_> = Value::Coll(CollValue::Generic {
                elem_tpe: SType::SLong,
                items: items.clone(),
            });
            let rv: Value<'_> = Value::Coll(CollValue::Generic {
                elem_tpe: SType::SLong,
                items,
            });
            assert!(eq_with_cost(&lv, &rv, &ctx).unwrap());
            ctx.jit_cost_value() - before
        };
        assert_eq!(cost(1), 18);
        assert_eq!(cost(48), 18);
        assert_eq!(cost(49), 20);
    }

    /// CONSENSUS PARITY: `SigmaProp == SigmaProp` matches Scala
    /// `DataValueComparer.equalDataValues`'s dedicated `SigmaProp` case
    /// (DataValueComparer.scala:353) → `equalSigmaBoolean`: `MatchType(1)` for the
    /// dispatch + `MatchType(1)` per tree node + `EQ_GroupElement(172)` per EcPoint.
    ///   ProveDlog    == ProveDlog:    1 + 1 + 172     = 174
    ///   ProveDHTuple == ProveDHTuple: 1 + 1 + 4 * 172 = 690
    #[test]
    fn sigmaprop_eq_matches_scala_equalsigmaboolean() {
        use ergotree_ir::sigma_protocol::sigma_boolean::{
            ProveDhTuple, ProveDlog, SigmaBoolean, SigmaProp,
        };
        let cost_of = |sp: SigmaProp| -> u64 {
            let ctx = force_any_val::<Context>();
            let v: Value<'_> = Value::sigma_prop(sp);
            let rv = v.clone();
            let before = ctx.jit_cost_value();
            assert!(eq_with_cost(&v, &rv, &ctx).unwrap());
            ctx.jit_cost_value() - before
        };
        let dlog = SigmaProp::new(SigmaBoolean::from(force_any_val::<ProveDlog>()));
        let dht = SigmaProp::new(SigmaBoolean::from(force_any_val::<ProveDhTuple>()));
        assert_eq!(
            cost_of(dlog),
            174,
            "ProveDlog==ProveDlog: Scala charges MatchType*2 + EQ_GroupElement(172)"
        );
        assert_eq!(
            cost_of(dht),
            690,
            "ProveDHTuple==ProveDHTuple: Scala charges MatchType*2 + 4*EQ_GroupElement(172)"
        );
    }

    /// CONSENSUS PARITY: a node-kind mismatch dispatches on the LEFT
    /// (`equalSigmaBoolean`'s guard structure): a leaf left is `false` (the
    /// leaf arm's inner `case _`); a conjecture left falls through to the
    /// top-level `case _ => sys.error` and THROWS. Either way the node's
    /// MatchType is charged first (charge-then-throw), so both directions
    /// cost SigmaProp dispatch (1) + node (1) = 2.
    #[test]
    fn sigmaprop_eq_kind_mismatch_dispatches_on_left() {
        use ergotree_ir::sigma_protocol::sigma_boolean::cand::Cand;
        use ergotree_ir::sigma_protocol::sigma_boolean::{ProveDlog, SigmaBoolean, SigmaProp};

        let dlog = SigmaBoolean::from(force_any_val::<ProveDlog>());
        let cand = SigmaBoolean::SigmaConjecture(SigmaConjecture::Cand(Cand {
            items: vec![dlog.clone(), dlog.clone()],
        }));
        let case = |l: SigmaBoolean, r: SigmaBoolean| -> (Result<bool, EvalError>, u64) {
            let ctx = force_any_val::<Context>();
            let lv: Value<'_> = Value::sigma_prop(SigmaProp::new(l));
            let rv: Value<'_> = Value::sigma_prop(SigmaProp::new(r));
            let before = ctx.jit_cost_value();
            let res = eq_with_cost(&lv, &rv, &ctx);
            (res, ctx.jit_cost_value() - before)
        };
        // Conjecture left vs leaf right: throws after charging.
        let (res, cost) = case(cand.clone(), dlog.clone());
        assert!(matches!(res, Err(EvalError::Misc(_))));
        assert_eq!(cost, 2, "MatchType dispatch + node precede the throw");
        // Leaf left vs conjecture right: the leaf arm's inner `case _` → false.
        let (res, cost) = case(dlog, cand);
        assert!(!res.unwrap());
        assert_eq!(cost, 2);
    }

    #[test]
    fn coll_eq_coa_early_mismatch_charges_compared_count() {
        // h28931 class: an equal-length COA eq that mismatches early charges the
        // per-item cost on the *compared count* (first-mismatch index + 1), not
        // the full length. Coll[Long]: base=15, per_chunk=2, cs=48.
        //   mismatch @ idx 0  -> compared 1  -> chunks(1)=1  -> 1 + 15 + 2 = 18
        //   mismatch @ idx 47 -> compared 48 -> chunks(48)=1 -> 1 + 15 + 2 = 18
        //   mismatch @ idx 48 -> compared 49 -> chunks(49)=2 -> 1 + 15 + 4 = 20
        // Pre-fix every case charged chunks(full len)=chunks(49)=2 -> 20, so the
        // early-mismatch cases over-charged by 2.
        let cost = |len: usize, mismatch_at: usize| -> (bool, u64) {
            let ctx = force_any_val::<Context>();
            let l: Arc<[Value<'_>]> =
                Arc::from((0..len as i64).map(Value::Long).collect::<Vec<Value<'_>>>());
            let mut r_vec: Vec<Value<'_>> = (0..len as i64).map(Value::Long).collect();
            r_vec[mismatch_at] = Value::Long(-1); // -1 is never equal to its index
            let r: Arc<[Value<'_>]> = Arc::from(r_vec);
            let lv = Value::Coll(CollValue::Generic {
                elem_tpe: SType::SLong,
                items: l,
            });
            let rv = Value::Coll(CollValue::Generic {
                elem_tpe: SType::SLong,
                items: r,
            });
            let before = ctx.jit_cost_value();
            let eq = eq_with_cost(&lv, &rv, &ctx).unwrap();
            (eq, ctx.jit_cost_value() - before)
        };
        assert_eq!(cost(49, 0), (false, 18));
        assert_eq!(cost(49, 47), (false, 18));
        assert_eq!(cost(49, 48), (false, 20)); // last item: compared == full length
    }

    #[test]
    fn coll_eq_generic_early_mismatch_charges_compared_count() {
        // Composite-element coll: the generic `EQ_Coll` per-item cost is charged
        // on the compared count *after* recursing, and recursion short-circuits
        // at the first unequal element. Outer Coll[Coll[Int]] of length 3 whose
        // element 0 already differs (elements 1 and 2 are never reached):
        //   outer MatchType                                            = 1
        //   recurse element 0: inner Coll[Int] [1,2,3] vs [9,2,3],
        //     mismatch @ idx 0 -> compared 1 -> 15+2: MatchType 1 + 17 = 18
        //   outer EQ_Coll on compared count 1 (base 10, per_chunk 2, cs 1):
        //     chunks(1)=1 -> 10 + 2                                    = 12
        //   total                                                      = 31
        // Pre-fix charged EQ_Coll on the full length 3 (=16) BEFORE recursing,
        // for a total of 35.
        let ctx = force_any_val::<Context>();
        let inner_l: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SInt,
            items: Arc::from(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
        });
        let inner_r: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SInt,
            items: Arc::from(vec![Value::Int(9), Value::Int(2), Value::Int(3)]),
        });
        let other: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SInt,
            items: Arc::from(vec![Value::Int(7), Value::Int(8), Value::Int(9)]),
        });
        let outer_l: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SColl(Arc::new(SType::SInt)),
            items: Arc::from(vec![inner_l, other.clone(), other.clone()]),
        });
        let outer_r: Value<'_> = Value::Coll(CollValue::Generic {
            elem_tpe: SType::SColl(Arc::new(SType::SInt)),
            items: Arc::from(vec![inner_r, other.clone(), other]),
        });
        let before = ctx.jit_cost_value();
        assert!(!eq_with_cost(&outer_l, &outer_r, &ctx).unwrap());
        assert_eq!(ctx.jit_cost_value() - before, 31);
    }

    // sigmastate returns false for two collections whose item types differ, after its one
    // `MatchType` charge and before any item (v6.0.7 `DataValueComparer.scala:318-326`)

    fn empty_coll(elem_tpe: SType) -> Value<'static> {
        Value::Coll(CollValue::Generic {
            elem_tpe,
            items: Arc::from(vec![]),
        })
    }

    fn int_pair() -> SType {
        SType::STuple(ergotree_ir::types::stuple::STuple::pair(SType::SInt, SType::SInt).into())
    }

    #[test]
    fn empty_colls_of_different_tuple_item_types_are_unequal() {
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        let long_pair = SType::STuple(
            ergotree_ir::types::stuple::STuple::pair(SType::SLong, SType::SLong).into(),
        );
        assert!(!eq_with_cost(&empty_coll(int_pair()), &empty_coll(long_pair), &ctx).unwrap());
        assert_eq!(ctx.jit_cost_value() - before, MATCH_TYPE_COST);
    }

    #[test]
    fn empty_colls_of_different_nested_item_types_are_unequal() {
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        let ints = SType::SColl(Arc::new(SType::SInt));
        let longs = SType::SColl(Arc::new(SType::SLong));
        assert!(!eq_with_cost(&empty_coll(ints), &empty_coll(longs), &ctx).unwrap());
        assert_eq!(ctx.jit_cost_value() - before, MATCH_TYPE_COST);
    }

    #[test]
    fn equal_tuple_colls_stay_equal() {
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        let pairs = || -> Value<'static> {
            Value::Coll(CollValue::Generic {
                elem_tpe: int_pair(),
                items: Arc::from(vec![
                    Value::pair(Value::Int(1), Value::Int(2)),
                    Value::pair(Value::Int(3), Value::Int(4)),
                ]),
            })
        };
        assert!(eq_with_cost(&pairs(), &pairs(), &ctx).unwrap());
        // the cost on `368af3be`, before the item-type comparison
        assert_eq!(ctx.jit_cost_value() - before, 35);
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn comparing_a_lambda_fails_as_in_sigmastate() {
        use crate::eval::test_util::try_eval_out_wo_ctx;
        use alloc::boxed::Box;
        use ergotree_ir::mir::bin_op::{BinOp, RelationOp};
        use ergotree_ir::mir::collection::Collection;
        use ergotree_ir::mir::expr::Expr;
        use ergotree_ir::mir::func_value::{FuncArg, FuncValue};
        use ergotree_ir::mir::val_use::ValUse;
        use ergotree_ir::types::sfunc::SFunc;
        // sigmastate's `equalDataValues` dispatches on the left value and has no case for a
        // closure: it ends in `sys.error("Cannot compare $l and $r: unknown type")` (v6.0.7
        // `DataValueComparer.scala`, the last case). Rust compared the two structurally, which
        // also walked every captured lambda once per path to it
        let identity = || -> Expr {
            FuncValue::new(
                vec![FuncArg {
                    idx: 1.into(),
                    tpe: SType::SInt,
                }],
                ValUse {
                    val_id: 1.into(),
                    tpe: SType::SInt,
                }
                .into(),
            )
            .into()
        };
        let relation = |op: RelationOp, left: Expr, right: Expr| -> Expr {
            BinOp {
                kind: op.into(),
                left: Box::new(left),
                right: Box::new(right),
            }
            .into()
        };
        let compare = |e: &Expr| try_eval_out_wo_ctx::<bool>(e);
        assert!(compare(&relation(RelationOp::Eq, identity(), identity())).is_err());
        assert!(compare(&relation(RelationOp::NEq, identity(), identity())).is_err());
        // a collection compares its items the same way
        let lambdas = || -> Expr {
            Collection::new(
                SType::SFunc(SFunc::new(vec![SType::SInt], SType::SInt).into()),
                vec![identity()],
            )
            .unwrap()
            .into()
        };
        assert!(compare(&relation(RelationOp::Eq, lambdas(), lambdas())).is_err());
        // a lambda on the right only: the left value's case decides, `l == r` for a number
        assert!(!compare(&relation(RelationOp::Eq, 1i32.into(), identity())).unwrap());
    }

    // --- The left dispatch (queue item 11). The JVM column is `equalDataValues` itself, called
    // from Java on the sigma-state 6.0.7 jar with a fresh evaluator per row (the round's
    // `measure/jvm-6.0.7-output.txt`), unless a row says "by reading".

    /// The answer, or the error, and what the comparison charged
    fn compare(l: &Value<'static>, r: &Value<'static>) -> (Result<bool, EvalError>, u64) {
        let ctx = force_any_val::<Context>();
        let before = ctx.jit_cost_value();
        let res = eq_with_cost(l, r, &ctx);
        (res, ctx.jit_cost_value() - before)
    }

    fn some(v: Value<'static>) -> Value<'static> {
        Value::Opt(Some(alloc::boxed::Box::new(v)))
    }

    fn string(s: &str) -> Value<'static> {
        Value::String(s.into())
    }

    fn coll(elem_tpe: SType, items: Vec<Value<'static>>) -> Value<'static> {
        Value::Coll(CollValue::Generic {
            elem_tpe,
            items: Arc::from(items),
        })
    }

    /// One value of every kind `equalDataValues` has a case for: its name, its class (the four
    /// integer widths are one class, `Some` and `None` another), and what its case charges
    /// against a right value of another class
    fn kinds() -> Vec<(&'static str, &'static str, Value<'static>, u64)> {
        use alloc::boxed::Box;
        use ergo_chain_types::{Header, PreHeader};
        use ergotree_ir::chain::ergo_box::ErgoBox;
        use ergotree_ir::mir::avl_tree_data::AvlTreeData;
        use ergotree_ir::reference::Ref;
        use ergotree_ir::sigma_protocol::sigma_boolean::{SigmaBoolean, SigmaProp};
        use ergotree_ir::unsignedbigint256::UnsignedBigInt;
        vec![
            ("Boolean", "boolean", Value::Boolean(true), 3),
            ("Byte", "integer", Value::Byte(1), 3),
            ("Short", "integer", Value::Short(1), 3),
            ("Int", "integer", Value::Int(1), 3),
            ("Long", "integer", Value::Long(1), 3),
            ("BigInt", "bigint", Value::BigInt(1.into()), 5),
            (
                "UnsignedBigInt",
                "unsigned",
                Value::UnsignedBigInt(UnsignedBigInt::from(1u32)),
                5,
            ),
            (
                "GroupElement",
                "group element",
                Value::from(ergo_chain_types::ec_point::generator()),
                172,
            ),
            ("pair", "pair", Value::pair(Value::Int(1), Value::Int(2)), 4),
            ("Some", "option", some(Value::Int(1)), 4),
            ("None", "option", Value::Opt(None), 4),
            ("Coll", "coll", coll(SType::SInt, vec![Value::Int(1)]), 1),
            (
                "SigmaProp",
                "sigma prop",
                Value::sigma_prop(SigmaProp::new(SigmaBoolean::TrivialProp(true))),
                1,
            ),
            (
                "AvlTree",
                "avl tree",
                Value::from(force_any_val::<AvlTreeData>()),
                6,
            ),
            // by reading (`:385-396`): the harness builds none of these three
            (
                "PreHeader",
                "pre-header",
                Value::PreHeader(Box::new(force_any_val::<PreHeader>())),
                4,
            ),
            (
                "Header",
                "header",
                Value::Header(Box::new(force_any_val::<Header>())),
                6,
            ),
            (
                "Box",
                "box",
                Value::CBox(Ref::from(force_any_val::<ErgoBox>())),
                6,
            ),
            ("String", "string", string("a"), 1),
            ("Unit", "unit", Value::Unit, 0),
        ]
    }

    /// The dispatch is on the LEFT value alone: against a right value of another class every
    /// kind answers false at its own case's cost
    #[test]
    fn a_left_kind_charges_its_own_case_against_another_kind() {
        let kinds = kinds();
        let mut cells = 0;
        for (l_name, l_class, l, cost) in &kinds {
            for (r_name, r_class, r, _) in &kinds {
                if l_class == r_class {
                    continue;
                }
                let (res, charged) = compare(l, r);
                assert!(!res.unwrap(), "{l_name} == {r_name}");
                assert_eq!(charged, *cost, "{l_name} == {r_name}");
                cells += 1;
            }
        }
        // 19 kinds, less the same-class cells: 19 of a kind with itself, 12 among the four
        // integers, 2 between `Some` and `None`
        assert_eq!(cells, 19 * 19 - 19 - 12 - 2);
    }

    /// Every kind equals itself, at the cost of its case (`SigmaProp(true)`: one more
    /// MatchType for its one node; `Some(1)`, `(1, 2)`, `Coll(1)`: their items')
    #[test]
    fn a_kind_equals_itself_at_its_cost() {
        let own_cost = [
            ("Boolean", 3),
            ("Byte", 3),
            ("Short", 3),
            ("Int", 3),
            ("Long", 3),
            ("BigInt", 5),
            ("UnsignedBigInt", 5),
            ("GroupElement", 172),
            ("pair", 10),
            ("Some", 7),
            ("None", 4),
            ("Coll", 18),
            ("SigmaProp", 2),
            ("AvlTree", 6),
            ("PreHeader", 4),
            ("Header", 6),
            ("Box", 6),
            ("String", 18),
            ("Unit", 0),
        ];
        for ((name, _, v, _), (cost_name, cost)) in kinds().iter().zip(own_cost) {
            assert_eq!(*name, cost_name);
            let (res, charged) = compare(v, v);
            assert!(res.unwrap(), "{name}");
            assert_eq!(charged, cost, "{name}");
        }
    }

    /// Case 1 is Scala's `==` on two boxed numbers: two integers are equal when their values
    /// are, whatever their widths, and nothing is truncated or zero-extended
    #[test]
    fn integers_compare_by_value_across_widths() {
        let ones = [
            Value::Byte(1),
            Value::Short(1),
            Value::Int(1),
            Value::Long(1),
        ];
        let twos = [
            Value::Byte(2),
            Value::Short(2),
            Value::Int(2),
            Value::Long(2),
        ];
        for l in &ones {
            for r in &ones {
                assert_eq!(compare(l, r), (Ok(true), 3), "{l:?} == {r:?}");
            }
            for r in &twos {
                assert_eq!(compare(l, r), (Ok(false), 3), "{l:?} == {r:?}");
            }
        }
        let rows = [
            (Value::Long(4294967297), Value::Int(1), false),
            (Value::Int(1), Value::Long(4294967297), false),
            (Value::Byte(-1), Value::Long(-1), true),
            (Value::Long(-1), Value::Byte(-1), true),
            (Value::Byte(-1), Value::Int(255), false),
            (Value::Short(-1), Value::Long(65535), false),
            (Value::Short(-1), Value::Int(-1), true),
            (Value::Int(-1), Value::Short(-1), true),
            (Value::Long(i64::MIN), Value::Int(0), false),
            (Value::Int(i32::MIN), Value::Long(i64::from(i32::MIN)), true),
            // a `Boolean` is not a number
            (Value::Boolean(true), Value::Byte(1), false),
            (Value::Byte(1), Value::Boolean(true), false),
            (Value::Boolean(false), Value::Boolean(false), true),
            (Value::Boolean(true), Value::Boolean(false), false),
        ];
        for (l, r, expected) in rows {
            assert_eq!(compare(&l, &r), (Ok(expected), 3), "{l:?} == {r:?}");
        }
    }

    /// The items of a pair, an option and a `Coll[Any]` are compared by the same function, so
    /// by case 1; two collections compare their item TYPES first
    #[test]
    fn a_wrong_width_item_is_compared_by_value() {
        let pair = |a: Value<'static>, b: Value<'static>| Value::pair(a, b);
        let any = |items: Vec<Value<'static>>| coll(SType::SAny, items);
        let rows = [
            (
                pair(Value::Int(1), Value::Int(2)),
                pair(Value::Long(1), Value::Int(2)),
                true,
                10,
            ),
            (
                pair(Value::Long(1), Value::Int(2)),
                pair(Value::Int(1), Value::Int(2)),
                true,
                10,
            ),
            (some(Value::Int(1)), some(Value::Long(1)), true, 7),
            (some(Value::Long(1)), some(Value::Int(1)), true, 7),
            (
                any(vec![Value::Int(1), Value::Long(2), Value::Int(3)]),
                any(vec![Value::Long(1), Value::Int(2), Value::Int(3)]),
                true,
                26,
            ),
            (
                any(vec![Value::Int(1), Value::Long(2), Value::Int(3)]),
                any(vec![Value::Long(1), Value::Int(2), Value::Int(4)]),
                false,
                26,
            ),
            (
                coll(SType::SInt, vec![Value::Int(1)]),
                coll(SType::SLong, vec![Value::Long(1)]),
                false,
                1,
            ),
            // the left item's case decides inside a pair too
            (
                pair(
                    Value::from(ergo_chain_types::ec_point::generator()),
                    Value::Int(2),
                ),
                pair(Value::Long(1), Value::Int(2)),
                false,
                176,
            ),
            (
                pair(string("a"), Value::Int(1)),
                pair(Value::Long(1), Value::Int(1)),
                false,
                5,
            ),
            (
                pair(Value::Long(1), Value::Int(1)),
                pair(string("a"), Value::Int(1)),
                false,
                7,
            ),
        ];
        for (l, r, expected, cost) in rows {
            assert_eq!(compare(&l, &r), (Ok(expected), cost), "{l:?} == {r:?}");
        }
    }

    /// `:397-408`: MatchType, then two strings of the same length (in UTF-16 units, as
    /// `String.length`) are charged `EQ_COA_Short` over that length whatever the answer
    #[test]
    fn strings_charge_match_type_and_their_length() {
        let rows = [
            ("a", "a", true, 18),
            ("a", "b", false, 18),
            ("ab", "ab", true, 18),
            ("", "", true, 18),
            ("a", "ab", false, 1),
            // one UTF-16 unit, two UTF-8 bytes
            ("\u{e9}", "ab", false, 1),
            ("\u{e9}", "a", false, 18),
        ];
        for (l, r, expected, cost) in rows {
            assert_eq!(
                compare(&string(l), &string(r)),
                (Ok(expected), cost),
                "{l:?} == {r:?}"
            );
        }
        // `PerItemCost(15, 2, 96)`: 96 units are one chunk, 97 are two
        let long = |n: usize| string(&"a".repeat(n));
        assert_eq!(compare(&long(96), &long(96)), (Ok(true), 18));
        assert_eq!(compare(&long(97), &long(97)), (Ok(true), 20));
        // two UTF-16 units each (a surrogate pair), four UTF-8 bytes
        assert_eq!(
            compare(&string("\u{1f600}"), &string("ab")),
            (Ok(false), 18)
        );
        assert_eq!(
            compare(&string("\u{1f600}"), &string("abcd")),
            (Ok(false), 1)
        );
    }

    /// sigmastate has no case for a `Context` or a `Global` on the left: it throws before any
    /// cost (`:412-413`; `Global` measured, `Context` by reading). On the right they are the
    /// left value's case
    #[test]
    fn context_and_global_have_no_case() {
        for l in [Value::Context, Value::Global] {
            for r in [Value::Context, Value::Global, Value::Int(1)] {
                let (res, charged) = compare(&l, &r);
                assert!(
                    matches!(res, Err(EvalError::Misc(_))),
                    "{l:?} == {r:?}: {res:?}"
                );
                assert_eq!(charged, 0);
            }
            assert_eq!(compare(&Value::Int(1), &l), (Ok(false), 3));
        }
    }

    /// An item with no case fails the comparison only when it is reached, after what was
    /// charged on the way: a pair stops at its first unequal item
    #[test]
    fn an_item_without_a_case_fails_only_when_reached() {
        let pair = |a: Value<'static>, b: Value<'static>| Value::pair(a, b);
        let (res, charged) = compare(
            &pair(Value::Int(1), Value::Global),
            &pair(Value::Int(1), Value::Global),
        );
        assert!(matches!(res, Err(EvalError::Misc(_))), "{res:?}");
        assert_eq!(charged, 7);
        let (res, charged) = compare(&some(Value::Global), &some(Value::Global));
        assert!(matches!(res, Err(EvalError::Misc(_))), "{res:?}");
        assert_eq!(charged, 4);
        let bytes = Value::Coll(CollValue::Bytes([1i8].as_slice().into()));
        let rows = [
            (
                pair(Value::Int(1), Value::Global),
                pair(Value::Int(2), Value::Global),
                7,
            ),
            (some(Value::Int(1)), some(Value::Global), 7),
            (coll(SType::SInt, vec![Value::Int(1)]), Value::Global, 1),
            (bytes, Value::Int(1), 1),
            (pair(Value::Int(1), Value::Int(2)), Value::Global, 4),
        ];
        for (l, r, cost) in rows {
            assert_eq!(compare(&l, &r), (Ok(false), cost), "{l:?} == {r:?}");
        }
    }

    /// The comparer in the evaluator, through `EQ`, `NEQ` and `Coll.indexOf`. `lie(a, b)` is
    /// `If(false, a, b)`: typed as `a`, valued as `b`. X1 and X2 carry SANTA's costs; every
    /// other tree's JVM value and cost are the measured comparer placed in the tree by reading
    #[test]
    fn the_comparer_in_the_evaluator() {
        crate::eval::test_util::assert_trees(&[
            // SANTA `Coll.specialisation_blind_ops` X1, X2 (sigma-state 6.0.7)
            // X1: `Coll[Byte](1,2,3).indices.indexOf(lie Int:1L, 0)`
            (
                "0b15dc0c1adb0c0e0e0301020302950100040205020400",
                Some((Value::Int(1), 96)),
            ),
            // X2: `Coll[Int](0,1,2).indexOf(lie Int:1L, 0)`
            (
                "0b16dc0c1a83030404000402040402950100040205020400",
                Some((Value::Int(1), 100)),
            ),
            // `indexOf`: an integer item against an integer argument of another width
            // I3: `Coll(1L).indexOf(lie Long:1, 0)`
            (
                "0b12dc0c1a830105050202950100050204020400",
                Some((Value::Int(0), 87)),
            ),
            // I3c: `Coll(1L).indexOf(1L, 0)`
            ("0b0ddc0c1a83010505020205020400", Some((Value::Int(0), 72))),
            // I4: `Coll[Byte] const (1).indexOf(lie Byte:1L, 0)`
            (
                "0b10dc0c1a0e010102950100020105020400",
                Some((Value::Int(0), 67)),
            ),
            // `EQ`/`NEQ`: a wrong-width item behind the shallow operand check
            // E3: `lie((1,2),(1L,2)) == (1,2)`
            (
                "0b1693950100860204020404860205020404860204020404",
                Some((Value::Boolean(true), 75)),
            ),
            // E3c: `(1,2) == (1,2)`
            (
                "0b0d93860204020404860204020404",
                Some((Value::Boolean(true), 60)),
            ),
            // E4: `(1,2) == lie((1,2),(1L,2))`
            (
                "0b1693860204020404950100860204020404860205020404",
                Some((Value::Boolean(true), 75)),
            ),
            // E5: `lie((1,2),(1L,2)) != (1,2)`
            (
                "0b1694950100860204020404860205020404860204020404",
                Some((Value::Boolean(false), 75)),
            ),
            // E6: `lie((1,2),(2L,2)) == (1,2)`
            (
                "0b1693950100860204020404860205040404860204020404",
                Some((Value::Boolean(false), 72)),
            ),
            // E7: `lie((1b,2),(1L,2)) == (1b,2)`
            (
                "0b1693950100860202010404860205020404860202010404",
                Some((Value::Boolean(true), 75)),
            ),
            // E13: `lie(Coll(1).get(0), Coll(1L).get(0)) == Coll(1).get(0)`
            (
                "0b2593950100dc0c218301040402010400dc0c218301050502010400dc0c218301040402010400",
                Some((Value::Boolean(true), 150)),
            ),
            // E13c: `Coll(1).get(0) == Coll(1).get(0)`
            (
                "0b1793dc0c218301040402010400dc0c218301040402010400",
                Some((Value::Boolean(true), 135)),
            ),
            // a left value that is not a primitive charges its own case
            // E9: `(big1,2) == lie((big1,2),(1L,2))`
            (
                "0b18938602060101040495010086020601010404860205020404",
                Some((Value::Boolean(false), 74)),
            ),
            // E9c: `(big1,2) == (big1,2)`
            (
                "0b0f938602060101040486020601010404",
                Some((Value::Boolean(true), 62)),
            ),
            // `NEQ` hands its operands to the comparer in `EQ`'s order (v6.0.6
            // `trees.scala:1224-1230`): the four rows below are measured on the sigma-state 6.0.7
            // jars (a profiling evaluator on the expression's bytes), not by SANTA
            // N9: `(big1,2) != lie((big1,2),(1L,2))`
            (
                "0b18948602060101040495010086020601010404860205020404",
                Some((Value::Boolean(true), 74)),
            ),
            // N9m: `lie((big1,2),(1L,2)) != (big1,2)`
            (
                "0b18949501008602060101040486020502040486020601010404",
                Some((Value::Boolean(true), 72)),
            ),
            // N9c: `(big1,2) != (big1,2)`
            (
                "0b0f948602060101040486020601010404",
                Some((Value::Boolean(false), 62)),
            ),
            // E9m: `lie((big1,2),(1L,2)) == (big1,2)`
            (
                "0b18939501008602060101040486020502040486020601010404",
                Some((Value::Boolean(false), 72)),
            ),
            // E11: `(g,2) == lie((g,2),(1L,2))`
            (
                "0b149386028204049501008602820404860205020404",
                Some((Value::Boolean(false), 246)),
            ),
            // I1: `Coll[Box](SELF).indexOf(lie SELF:1L, 0)`
            (
                "0b10dc0c1a830163a702950100a705020400",
                Some((Value::Int(-1), 95)),
            ),
            // I1c: `Coll[Box](SELF).indexOf(SELF, 0)`
            ("0b0bdc0c1a830163a702a70400", Some((Value::Int(0), 85))),
            // I2: `Coll[BigInt](1).indexOf(lie big:1L, 0)`
            (
                "0b14dc0c1a8301060601010295010006010105020400",
                Some((Value::Int(-1), 89)),
            ),
            // I2c: `Coll[BigInt](1).indexOf(big1, 0)`
            (
                "0b0fdc0c1a830106060101020601010400",
                Some((Value::Int(0), 74)),
            ),
            // I5: `Coll[Coll[Int]](Coll(1)).indexOf(lie Coll(1):1L, 0)`
            (
                "0b18dc0c1a830110830104040202950100830104040205020400",
                Some((Value::Int(-1), 105)),
            ),
            // I5c: `Coll[Coll[Int]](Coll(1)).indexOf(Coll(1), 0)`
            (
                "0b13dc0c1a83011083010404020283010404020400",
                Some((Value::Int(0), 127)),
            ),
            // I6: `Coll((1,2)).indexOf(lie (1,2):1L, 0)`
            (
                "0b1adc0c1a8301588602040204040295010086020402040405020400",
                Some((Value::Int(-1), 108)),
            ),
            // I6c: `Coll((1,2)).indexOf((1,2), 0)`
            (
                "0b15dc0c1a830158860204020404028602040204040400",
                Some((Value::Int(0), 119)),
            ),
            // I7: `Coll[GroupElement](g).indexOf(lie g:1L, 0)`
            (
                "0b10dc0c1a83010782029501008205020400",
                Some((Value::Int(-1), 261)),
            ),
            // I7c: `Coll[GroupElement](g).indexOf(g, 0)`
            ("0b0bdc0c1a8301078202820400", Some((Value::Int(0), 251))),
            // a primitive left against another kind: unequal at `EQ_Prim`, as before
            // E8: `lie((true,2),(1,2)) == (true,2) Int vs Boolean: false @ same`
            (
                "0b1693950100860201010404860204020404860201010404",
                Some((Value::Boolean(false), 72)),
            ),
            // E10: `lie((big1,2),(1L,2)) == (big1,2) Long left, BigInt right: false, EQ_Prim 3`
            (
                "0b18939501008602060101040486020502040486020601010404",
                Some((Value::Boolean(false), 72)),
            ),
            // E12: `lie((g,2),(1L,2)) == (g,2)`
            (
                "0b149395010086028204048602050204048602820404",
                Some((Value::Boolean(false), 77)),
            ),
            // I8: `Coll(1).indexOf(lie 1:g, 0)`
            (
                "0b11dc0c1a8301040402029501000402820400",
                Some((Value::Int(-1), 92)),
            ),
            // I9: `Coll(true).indexOf(lie true:1, 0)`
            (
                "0b12dc0c1a830101010102950100010104020400",
                Some((Value::Int(-1), 87)),
            ),
            // I10: `Coll(1).indexOf(lie 1:true, 0)`
            (
                "0b12dc0c1a830104040202950100040201010400",
                Some((Value::Int(-1), 87)),
            ),
            // `String` items of pair constants
            // S1: `("a",1) == ("a",1)`
            (
                "0b0b934c660161024c66016102",
                Some((Value::Boolean(true), 35)),
            ),
            // S2: `("a",1) == ("b",1)`
            (
                "0b0b934c660161024c66016202",
                Some((Value::Boolean(false), 32)),
            ),
            // S3: `("a",1) == ("ab",1)`
            (
                "0b0c934c660161024c6602616202",
                Some((Value::Boolean(false), 15)),
            ),
            // S4: `("a",1) == lie(("a",1) const, (1L,1))`
            (
                "0b14934c660161029501004c66016102860205020402",
                Some((Value::Boolean(false), 50)),
            ),
            // S5: `lie(("a",1) const, (1L,1)) == ("a",1)`
            (
                "0b14939501004c660161028602050204024c66016102",
                Some((Value::Boolean(false), 52)),
            ),
        ]);
    }

    /// The trees the evaluator refuses around the comparer: each stops with the comparer's or
    /// a type check's error, at the cost the JVM's accumulator holds when it throws. The
    /// JVM figures are measured on the sigma-state 6.0.7 jars (a profiling evaluator on the
    /// expression's bytes, its accumulator read at the throw); a tree that reads `CONTEXT`
    /// needs a context that harness has not, so its figure is Rust's, by reading
    #[test]
    fn the_refusals_in_the_evaluator_stop_where_the_jvm_stops() {
        for (tree_hex, cost) in [
            // the operand check (`trees.scala:1204-1209`)
            // E1: `lie(1, 1L) == 1`, measured: `InterpreterException` @20
            ("0b0a93950100040205020402", 20),
            // the `Tuple` node's item check (`values.scala:828-833`), before the node's cost
            // E2: `(lie(1,1L), 2) == (1, 2)`, measured: `InterpreterException` @20
            ("0b12938602950100040205020404860204020404", 20),
            // E2m: `(2, lie(1,1L)) == (2, 1)`, the check of item 1, measured:
            // `InterpreterException` @25
            ("0b12938602040495010004020502860204040402", 25),
            // `Context` and `Global` have no case (`DataValueComparer.scala:413`)
            // K1: `CONTEXT == CONTEXT`, by reading
            ("0b0393fefe", 2),
            // K2: `Global == Global`, measured: `RuntimeException` @10
            ("0b0393dddd", 10),
            // K3: `(CONTEXT, 1) == (CONTEXT, 1)`, by reading: K4 with `CONTEXT`'s cost of 1 for
            // `Global`'s 5, twice
            ("0b0b938602fe04028602fe0402", 46),
            // K4: `(Global, 1) == (Global, 1)`, measured: `RuntimeException` @54
            ("0b0b938602dd04028602dd0402", 54),
            // `NEQ` negates an answer, not an error: `CONTEXT != CONTEXT`, by reading
            // (`Global != Global` is a `RuntimeException` @10 on the jars)
            ("0b0394fefe", 2),
        ] {
            let (e, stopped_at) = crate::eval::test_util::eval_tree_err(tree_hex);
            assert!(matches!(e, EvalError::Misc(_)), "{tree_hex}: {e:?}");
            assert_eq!(stopped_at, cost, "{tree_hex}");
        }
    }
}
