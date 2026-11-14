use core::fmt::Formatter;
use core::hash::Hash;

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::mir::expr::InvalidArgumentError;
use crate::serialization::jvm_string::jvm_string;
use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializeResult;

/// Type variable for generic signatures
#[derive(PartialEq, Eq, Clone, Hash)]
pub struct STypeVar {
    /// Type variable name (e.g. "T")
    name_bytes: Vec<u8>,
}

impl core::fmt::Debug for STypeVar {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.as_string().fmt(f)
    }
}

impl STypeVar {
    /// Creates a type variable from a UTF8 text string (name length is a `u8`, so 0..=255 bytes)
    pub fn new_from_str(name: &'static str) -> Result<Self, InvalidArgumentError> {
        Ok(Self {
            name_bytes: name.to_string().into_bytes(),
        })
    }

    /// Creates a type variable from bytes of a UTF8 text string (name length 0..=255).
    ///
    /// Mirrors the JVM `TypeSerializer` (`new String(bytes, UTF_8)`): a non-UTF8 name is
    /// decoded as the JVM decodes it -- malformed bytes become U+FFFD by the JDK's rule
    /// (`jvm_string`) -- and stored as that text's bytes, rather than rejected. (`Result`
    /// is kept for API stability; this no longer errors.)
    pub fn new_from_bytes(bytes: Vec<u8>) -> Result<Self, InvalidArgumentError> {
        Ok(Self::from_jvm_bytes(&bytes))
    }

    /// A type variable named by `bytes` as the JVM decodes them (`jvm_string`), which
    /// cannot fail
    pub(crate) fn from_jvm_bytes(bytes: &[u8]) -> Self {
        Self {
            name_bytes: jvm_string(bytes).into_bytes(),
        }
    }

    /// Returns text representation (e.g "T", etc.)
    pub fn as_string(&self) -> String {
        #[allow(clippy::unwrap_used)]
        String::from_utf8(self.name_bytes.clone()).unwrap()
    }

    /// "T" type variable
    pub fn t() -> Self {
        #[allow(clippy::unwrap_used)]
        STypeVar::new_from_str("T").unwrap()
    }

    /// "IV"(Input Value) type variable
    pub fn iv() -> STypeVar {
        #[allow(clippy::unwrap_used)]
        STypeVar::new_from_str("IV").unwrap()
    }
    /// "OV"(Output Value) type variable
    pub fn ov() -> STypeVar {
        #[allow(clippy::unwrap_used)]
        STypeVar::new_from_str("OV").unwrap()
    }
}

impl SigmaSerializable for STypeVar {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        w.put_u8(self.name_bytes.len() as u8)?;
        // name length is one byte (PutByteCost) -- Scala `TypeSerializer`'s `putUByte`
        w.add_put_byte_cost();
        w.write_all(self.name_bytes.as_slice())?;
        // name bytes are written as one block -- Scala `putBytes` => PutChunkCost over the length
        w.add_put_chunk_cost(self.name_bytes.len());
        Ok(())
    }

    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let name_len = r.get_u8()?;
        let mut bytes = vec![0; name_len as usize];
        r.get_bytes_into(&mut bytes)?;
        Ok(STypeVar::from_jvm_bytes(&bytes))
    }
}

/// Type parameter
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct STypeParam {
    pub(crate) ident: STypeVar,
}

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arbitrary {
    use super::*;
    use proptest::prelude::*;

    impl Arbitrary for STypeVar {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            "[A-Z][A-Z0-9]{0,2}"
                .prop_map(|name| STypeVar::new_from_bytes(name.into_bytes()).unwrap())
                .boxed()
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // The JVM `TypeSerializer.deserialize` reads the type-var name length via
    // `getUByte()` (0..=255, no bound) then `new String(getBytes(len))`, so an
    // empty name (len 0) and a 255-byte name both deserialize. The previous
    // `BoundedVec<u8, 1, 254>` over-rejected both -- a divergence from sigma-state.
    // See sigma-state core/.../serialization/TypeSerializer.scala:202-206.

    #[test]
    fn parse_name_length_0() {
        // u8 length 0x00, zero name bytes -> STypeVar("")
        let tv = STypeVar::sigma_parse_bytes(&[0x00u8]).unwrap();
        assert_eq!(tv.as_string(), "");
        assert_eq!(tv.sigma_serialize_bytes().unwrap(), vec![0x00u8]);
    }

    #[test]
    fn parse_name_length_255() {
        // u8 length 0xff (255), then 255 'a' bytes -> STypeVar("a" * 255)
        let mut bytes = vec![0xffu8];
        bytes.extend_from_slice(&[b'a'; 255]);
        let tv = STypeVar::sigma_parse_bytes(&bytes).unwrap();
        assert_eq!(tv.as_string(), "a".repeat(255));
        assert_eq!(tv.sigma_serialize_bytes().unwrap(), bytes);
    }

    #[test]
    fn name_jvm_lossy_substitution() {
        // A non-UTF8 name must canonicalize exactly as the JVM `new String(bytes, UTF_8)`
        // does, so the re-serialized name (-> tree bytes -> script hash) matches. Oracle of
        // the first five rows: sigma-state 6.0.3 / Java 17 (SANTA byte-exactness table).
        // `from_utf8_lossy` would differ on the surrogate `ed a0 80` (3 U+FFFD vs the JVM's 1).
        // (input bytes, JVM canonical name bytes)
        let cases: &[(&[u8], &[u8])] = &[
            (&[0xff], &[0xef, 0xbf, 0xbd]),
            (&[0xe2, 0x82], &[0xef, 0xbf, 0xbd]),
            (&[0xc0, 0x80], &[0xef, 0xbf, 0xbd, 0xef, 0xbf, 0xbd]),
            (&[0xed, 0xa0, 0x80], &[0xef, 0xbf, 0xbd]),
            (&[0x61, 0xff, 0x62], &[0x61, 0xef, 0xbf, 0xbd, 0x62]),
            // an overlong form and a value above U+10FFFF: each byte is replaced (OpenJDK 8, 17
            // and 21, and sigma-state 6.0.7 writing the name back)
            (
                &[0xe0, 0x80, 0x80],
                &[0xef, 0xbf, 0xbd, 0xef, 0xbf, 0xbd, 0xef, 0xbf, 0xbd],
            ),
            (
                &[0xf0, 0x80, 0x80, 0x80],
                &[
                    0xef, 0xbf, 0xbd, 0xef, 0xbf, 0xbd, 0xef, 0xbf, 0xbd, 0xef, 0xbf, 0xbd,
                ],
            ),
            (
                &[0xf4, 0x90, 0x80, 0x80],
                &[
                    0xef, 0xbf, 0xbd, 0xef, 0xbf, 0xbd, 0xef, 0xbf, 0xbd, 0xef, 0xbf, 0xbd,
                ],
            ),
        ];
        for (input, expected_name) in cases {
            let tv = STypeVar::new_from_bytes(input.to_vec()).unwrap();
            let mut expected_wire = vec![expected_name.len() as u8];
            expected_wire.extend_from_slice(expected_name);
            assert_eq!(
                tv.sigma_serialize_bytes().unwrap(),
                expected_wire,
                "input {:02x?}",
                input
            );
        }
    }

    #[test]
    fn two_names_are_one_type_as_their_text_is() {
        // `EQ` of two `Coll[T]()` builds only for operands of one type, and a name is its text.
        // sigma-state 6.0.7 (the round's `measure/rows/tv.txt`): the names, and the names
        // written back where the expression parses
        use crate::ergo_tree::ErgoTreeVersion;
        use crate::mir::expr::Expr;
        use crate::serialization::sigma_byte_reader::{from_bytes, SigmaByteRead};
        let coll = |name: &str| alloc::format!("830067{:02x}{name}", name.len() / 2);
        let (r1, r3, r4) = ("efbfbd", "efbfbdefbfbdefbfbd", "efbfbdefbfbdefbfbdefbfbd");
        let rows = [
            ("54", "54", Some(("54", "54"))),
            ("54", "55", None),
            ("e08080", "ffffff", Some((r3, r3))),
            ("e08080", "ff", None),
            ("f4908080", "ffffffff", Some((r4, r4))),
            ("f4908080", "ff", None),
            ("eda080", "ff", Some((r1, r1))),
            ("eda080", "ffffff", None),
        ];
        for version in [ErgoTreeVersion::V0, ErgoTreeVersion::V3] {
            for (left, right, written) in &rows {
                let bytes =
                    base16::decode(alloc::format!("93{}{}", coll(left), coll(right)).as_bytes())
                        .unwrap();
                let parsed = from_bytes(&bytes).with_tree_version(version, Expr::sigma_parse);
                match written {
                    Some((l, r)) => assert_eq!(
                        base16::encode_lower(&parsed.unwrap().sigma_serialize_bytes().unwrap()),
                        alloc::format!("93{}{}", coll(l), coll(r)),
                        "{left} {right}"
                    ),
                    None => assert!(
                        matches!(parsed, Err(SigmaParsingError::ConstraintFailed(_))),
                        "{left} {right}: {parsed:?}"
                    ),
                }
            }
        }
    }

    #[test]
    fn name_valid_utf8_unchanged() {
        // a valid multi-byte name round-trips byte-for-byte (euro sign U+20AC = e2 82 ac)
        let tv = STypeVar::new_from_bytes(vec![0xe2, 0x82, 0xac]).unwrap();
        assert_eq!(tv.as_string(), "\u{20ac}");
        assert_eq!(
            tv.sigma_serialize_bytes().unwrap(),
            vec![0x03, 0xe2, 0x82, 0xac]
        );
    }
}
