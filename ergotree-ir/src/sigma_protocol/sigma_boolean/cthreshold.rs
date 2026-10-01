//! THRESHOLD conjunction for sigma proposition

use core::convert::TryInto;

use alloc::string::ToString;
use alloc::vec::Vec;

use super::cand::Cand;
use super::cor::Cor;
use super::SigmaBoolean;
use super::SigmaConjecture;
use super::SigmaConjectureItems;
use crate::has_opcode::HasStaticOpCode;
use crate::serialization::op_code::OpCode;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaSerializeResult;
use crate::serialization::{SigmaParsingError, SigmaSerializable};

// use crate::sigma_protocol::sigma_boolean::SigmaConjecture;

/// THRESHOLD conjunction for sigma proposition
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Cthreshold {
    /// Number of conjectures to be proven
    // Our polynomial arithmetic can take only byte inputs
    pub k: u8,
    /// Items of the proposal
    pub children: SigmaConjectureItems<SigmaBoolean>,
}

impl Cthreshold {
    /// Reduce all possible TrivialProps in the tree
    pub fn reduce(k: u8, children: SigmaConjectureItems<SigmaBoolean>) -> SigmaBoolean {
        if k == 0 {
            return true.into();
        }
        if k as usize > children.len() {
            return false.into();
        }

        let mut curr_k = k;
        let mut children_left = children.len();
        let mut res: Vec<SigmaBoolean> = Vec::new();

        for (i, ch) in children.iter().enumerate() {
            if curr_k == 1 {
                res.append(&mut children.as_vec()[i..children.len()].to_vec());
                // `res` now holds child `i` and those after it, so the unwrap is safe
                #[allow(clippy::unwrap_used)]
                return Cor::normalized(res.try_into().unwrap());
            }
            if curr_k as usize == children_left {
                res.append(&mut children.as_vec()[i..children.len()].to_vec());
                // `res` now holds child `i` and those after it, so the unwrap is safe
                #[allow(clippy::unwrap_used)]
                return Cand::normalized(res.try_into().unwrap());
            }
            match ch {
                &SigmaBoolean::TrivialProp(true) => {
                    children_left -= 1;
                    curr_k -= 1;
                }
                &SigmaBoolean::TrivialProp(false) => {
                    children_left -= 1;
                }
                sigma => {
                    res.push(sigma.clone());
                }
            }
        }

        // `res` holds 2 or more of the children here, so each conversion succeeds
        #[allow(clippy::unwrap_used)]
        match curr_k as usize {
            1 => Cor::normalized(res.try_into().unwrap()),
            ch if ch == children_left => Cand::normalized(res.try_into().unwrap()),
            _ => SigmaBoolean::SigmaConjecture(SigmaConjecture::Cthreshold(Cthreshold {
                k: curr_k,
                children: res.try_into().unwrap(),
            })),
        }
    }
}

impl core::fmt::Display for Cthreshold {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("atLeast(")?;
        f.write_str(self.k.to_string().as_str())?;
        f.write_str(", (")?;
        for (i, item) in self.children.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            item.fmt(f)?;
        }
        f.write_str(")")
    }
}

impl HasStaticOpCode for Cthreshold {
    const OP_CODE: OpCode = OpCode::ATLEAST;
}

impl SigmaSerializable for Cthreshold {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        // put_u16 is used in sigmastate
        // https://github.com/ScorexFoundation/sigmastate-interpreter/blob/e64ca7930ff818403bb3020eadd4b5d8c029d9b6/sigmastate/src/main/scala/sigmastate/Values.scala#L799-L799
        w.put_u16(self.k as u16)?;
        // k is Scala `putUShort` => PutUnsignedNumericCost(3) under `Global.serialize`
        w.add_put_numeric_cost();
        w.put_u16(self.children.len() as u16)?;
        // child count is Scala `putUShort` => PutUnsignedNumericCost(3) under `Global.serialize`
        w.add_put_numeric_cost();
        self.children.iter().try_for_each(|i| i.sigma_serialize(w))
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let k = r.get_u16()?;
        let items_count = r.get_u16()?;
        let mut items = Vec::new();
        for _ in 0..items_count {
            items.push(SigmaBoolean::sigma_parse(r)?);
        }
        // sigmastate's CTHRESHOLD requires 0 <= k <= n <= 255 once the children are read
        // (`SigmaBoolean.scala:223`)
        if items.len() > 255 || usize::from(k) > items.len() {
            return Err(SigmaParsingError::CthresholdOutOfBounds(k, items.len()));
        }
        Ok(Cthreshold {
            k: k as u8,
            children: items.try_into()?,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    //! JVM parity: sigmastate reads CTHRESHOLD's `k` and number of children with `getUShort`,
    //! reads the children, then requires `0 <= k <= n <= 255` (`SigmaBoolean.scala:94-100`,
    //! `:223`). A failed `require` rejects, even in a size-flagged tree.
    use super::*;
    use crate::ergo_tree::ErgoTree;
    use alloc::vec;

    /// `CTHRESHOLD(k, n × TrueProp)`, `k` and `n` as written
    fn cthreshold_bytes(k: &[u8], n: &[u8], children: usize) -> Vec<u8> {
        [&[0x98][..], k, n, &vec![0xd3; children][..]].concat()
    }

    #[test]
    fn k_and_n_within_bounds_parse() {
        // SANTA `conjecture_bounds` #1 and #5: 255 children, and k = 0
        for bytes in [
            cthreshold_bytes(&[0x01], &[0xff, 0x01], 255),
            cthreshold_bytes(&[0x00], &[0x01], 1),
        ] {
            let parsed = SigmaBoolean::sigma_parse_bytes(&bytes).unwrap();
            assert_eq!(parsed.sigma_serialize_bytes().unwrap(), bytes);
        }
    }

    #[test]
    fn k_or_n_out_of_bounds_rejects() {
        // SANTA `conjecture_bounds` #0 and #2, 256 children and k above n; then k = 256,
        // which a byte-wide read of k turned into 0
        for (bytes, k, n) in [
            (cthreshold_bytes(&[0x01], &[0x80, 0x02], 256), 1, 256),
            (cthreshold_bytes(&[0x02], &[0x01], 1), 2, 1),
            (cthreshold_bytes(&[0x80, 0x02], &[0x01], 1), 256, 1),
        ] {
            assert_eq!(
                SigmaBoolean::sigma_parse_bytes(&bytes),
                Err(SigmaParsingError::CthresholdOutOfBounds(k, n))
            );
        }
    }

    #[test]
    fn out_of_bounds_rejects_a_sized_tree() {
        // SANTA `tree_sigmaboolean_bounds` #0: a size-flagged tree whose root is the constant
        // `CTHRESHOLD(1, 256 × TrueProp)` does not degrade
        let body = [&[0x08][..], &cthreshold_bytes(&[0x01], &[0x80, 0x02], 256)].concat();
        assert_eq!(body.len(), 261);
        let tree = [&[0x08, 0x85, 0x02][..], &body].concat();
        assert_eq!(
            ErgoTree::sigma_parse_bytes(&tree),
            Err(SigmaParsingError::CthresholdOutOfBounds(1, 256))
        );
    }
}
