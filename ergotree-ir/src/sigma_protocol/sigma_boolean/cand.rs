//! AND conjunction for sigma proposition
use core::convert::TryInto;

use alloc::vec::Vec;
use bounded_vec::NonEmptyVec;

use super::SigmaBoolean;
use crate::has_opcode::HasStaticOpCode;
use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaSerializationError;
use crate::serialization::{SigmaParsingError, SigmaSerializable, SigmaSerializeResult};
use crate::sigma_protocol::sigma_boolean::SigmaConjecture;

/// AND conjunction for sigma proposition
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Cand {
    /// Items of the conjunctions
    pub items: Vec<SigmaBoolean>,
}

impl HasStaticOpCode for Cand {
    const OP_CODE: OpCode = OpCode::AND;
}

impl Cand {
    /// Connects the given sigma propositions into CAND proposition performing
    /// partial evaluation when some of them are trivial propositioins.
    /// The list is not empty, as sigmastate's `CAND.normalized` requires
    /// (`SigmaBoolean.scala:165`).
    pub fn normalized(items: NonEmptyVec<SigmaBoolean>) -> SigmaBoolean {
        let mut res: Vec<SigmaBoolean> = Vec::new();
        for it in items {
            match it {
                SigmaBoolean::TrivialProp(false) => return it,
                SigmaBoolean::TrivialProp(true) => (),
                _ => res.push(it),
            }
        }
        if res.is_empty() {
            true.into()
        } else if res.len() == 1 {
            #[allow(clippy::unwrap_used)]
            res.first().unwrap().clone()
        } else {
            SigmaBoolean::SigmaConjecture(SigmaConjecture::Cand(Cand { items: res }))
        }
    }
}

impl core::fmt::Display for Cand {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("(")?;
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                f.write_str(" && ")?;
            }
            item.fmt(f)?;
        }
        f.write_str(")")
    }
}

impl SigmaSerializable for Cand {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        // sigmastate's `putUShort` refuses a count above 65535
        let count: u16 = self.items.len().try_into().map_err(|_| {
            SigmaSerializationError::NotSupported("CAND with more than 65535 children".into())
        })?;
        w.put_u16(count)?;
        // child count is Scala `putUShort` => PutUnsignedNumericCost(3) under `Global.serialize`
        w.add_put_numeric_cost();
        self.items.iter().try_for_each(|i| i.sigma_serialize(w))
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let items_count = r.get_u16()?;
        let mut items = Vec::new();
        for _ in 0..items_count {
            items.push(SigmaBoolean::sigma_parse(r)?);
        }
        Ok(Cand { items })
    }
}

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arbitrary {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;

    impl Arbitrary for Cand {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            vec(any::<SigmaBoolean>(), 2..=4)
                .prop_map(|items| Cand { items })
                .boxed()
        }
    }
}

#[allow(clippy::panic)]
#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "arbitrary")]
mod tests {
    use super::*;
    use crate::serialization::sigma_serialize_roundtrip;
    use crate::sigma_protocol::sigma_boolean::ProveDlog;

    use alloc::vec;
    use core::convert::TryInto;
    use proptest::prelude::*;
    use sigma_test_util::force_any_val;

    #[test]
    fn trivial_true() {
        let cand = Cand::normalized(vec![true.into(), true.into()].try_into().unwrap());
        assert!(matches!(cand, SigmaBoolean::TrivialProp(true)));
    }

    #[test]
    fn trivial_false() {
        let cand = Cand::normalized(vec![false.into(), true.into()].try_into().unwrap());
        assert!(matches!(cand, SigmaBoolean::TrivialProp(false)));
    }

    #[test]
    fn pk_triv_true() {
        let pk = force_any_val::<ProveDlog>();
        let cand = Cand::normalized(vec![pk.clone().into(), true.into()].try_into().unwrap());
        let res: ProveDlog = cand.try_into().unwrap();
        assert_eq!(res, pk);
    }

    #[test]
    fn pk_triv_false() {
        let pk = force_any_val::<ProveDlog>();
        let cand = Cand::normalized(vec![pk.into(), false.into()].try_into().unwrap());
        assert!(matches!(cand, SigmaBoolean::TrivialProp(false)));
    }

    #[test]
    fn pk_pk() {
        let pk1 = force_any_val::<ProveDlog>();
        let pk2 = force_any_val::<ProveDlog>();
        let pks: Vec<SigmaBoolean> = vec![pk1.into(), pk2.into()];
        let cand = Cand::normalized(pks.clone().try_into().unwrap());
        assert!(matches!(
            cand,
            SigmaBoolean::SigmaConjecture(SigmaConjecture::Cand(Cand {items})) if items == pks
        ));
    }

    proptest! {

        #[test]
        fn sigma_proposition_ser_roundtrip(
            v in any_with::<Cand>(())) {
                prop_assert_eq![sigma_serialize_roundtrip(&v), v]
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod bounds_tests {
    //! JVM parity: sigmastate reads CAND's child count with `getUShort` and builds the node with
    //! no check (`SigmaBoolean.scala:80-86`), so 0 to 65535 children parse.
    use super::*;
    use crate::ergo_tree::ErgoTree;
    use alloc::vec;

    /// `CAND(n × TrueProp)`, the count as written
    fn conjecture_bytes(count: &[u8], children: usize) -> Vec<u8> {
        [&[0x96][..], count, &vec![0xd3; children][..]].concat()
    }

    #[test]
    fn no_children_and_256_children_parse() {
        // SANTA `conjecture_bounds` #3 and #7
        for (bytes, n) in [
            (conjecture_bytes(&[0x00], 0), 0),
            (conjecture_bytes(&[0x80, 0x02], 256), 256),
        ] {
            let parsed = Cand::sigma_parse_bytes(&bytes[1..]).unwrap();
            assert_eq!(parsed.items.len(), n);
            let parsed = SigmaBoolean::sigma_parse_bytes(&bytes).unwrap();
            assert_eq!(parsed.sigma_serialize_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn a_tree_whose_root_is_the_constant_parses() {
        // SANTA `tree_sigmaboolean_bounds` #2 and #3: unsized trees, `00 08` and the constant
        for conjecture in [
            conjecture_bytes(&[0x00], 0),
            conjecture_bytes(&[0x80, 0x02], 256),
        ] {
            let tree = [&[0x00, 0x08][..], &conjecture].concat();
            let parsed = ErgoTree::sigma_parse_bytes(&tree).unwrap();
            assert!(matches!(parsed, ErgoTree::Parsed(_)));
            assert_eq!(parsed.sigma_serialize_bytes().unwrap(), tree);
        }
    }

    #[cfg(feature = "json")]
    #[test]
    fn json_roundtrip_without_children() {
        // the JSON form is sigma-rust's own: the JVM node's decoder reads ProveDlog only
        let parsed = SigmaBoolean::sigma_parse_bytes(&conjecture_bytes(&[0x00], 0)).unwrap();
        let json = serde_json::to_string(&parsed).unwrap();
        assert_eq!(json, r#"{"op":"150","args":[]}"#);
        assert_eq!(serde_json::from_str::<SigmaBoolean>(&json).unwrap(), parsed);
    }

    #[test]
    fn more_than_65535_children_cannot_be_written() {
        // sigmastate's `putUShort` refuses a count above 65535 (`SigmaBoolean.scala:48`)
        let at_the_bound = Cand {
            items: vec![SigmaBoolean::TrivialProp(true); 65535],
        };
        assert!(at_the_bound.sigma_serialize_bytes().is_ok());
        let above = Cand {
            items: vec![SigmaBoolean::TrivialProp(true); 65536],
        };
        assert!(above.sigma_serialize_bytes().is_err());
    }
}
