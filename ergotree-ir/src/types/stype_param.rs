use core::fmt::Formatter;
use core::hash::Hash;

use alloc::format;
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
use crate::serialization::SigmaSerializationError;
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
    /// Creates a type variable from a UTF8 text string. A name of any length is taken, as
    /// sigmastate's `STypeVar(name)` takes it; one longer than 255 bytes cannot be written
    pub fn new_from_str(name: &'static str) -> Result<Self, InvalidArgumentError> {
        Ok(Self {
            name_bytes: name.to_string().into_bytes(),
        })
    }

    /// Creates a type variable from bytes of a UTF8 text string. A name of any length is
    /// taken; one whose text is longer than 255 bytes cannot be written.
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
    /// "R"(Result) type variable
    pub fn r() -> STypeVar {
        #[allow(clippy::unwrap_used)]
        STypeVar::new_from_str("R").unwrap()
    }
}

impl SigmaSerializable for STypeVar {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        // sigmastate writes the length with `putUByte`, which refuses one above 255 (v6.0.6
        // `TypeSerializer.scala:122-126`). The text of a name read from bytes can be longer
        // than the bytes were: a malformed byte becomes three
        let len = u8::try_from(self.name_bytes.len()).map_err(|_| {
            SigmaSerializationError::NotSupported(format!(
                "a type variable name of {} bytes is out of unsigned byte range",
                self.name_bytes.len()
            ))
        })?;
        // name length is one byte (PutByteCost) -- Scala `TypeSerializer`'s `putUByte`, which
        // looks at the length before its `put` adds the cost
        w.add_put_byte_cost()?;
        w.put_u8(len)?;
        // name bytes are written as one block -- Scala `putBytes` => PutChunkCost over the length
        w.add_put_chunk_cost(self.name_bytes.len())?;
        w.write_all(self.name_bytes.as_slice())?;
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod name_length_tests {
    //! sigmastate writes a name as `putUByte(bytes.length)` and the bytes of its TEXT, and
    //! `putUByte` throws `N is out of unsigned byte range` above 255 (v6.0.6
    //! `TypeSerializer.scala:122-126`). A text decoded from 255 bytes can be 765 bytes long.
    //! Measured on the sigma-state 6.0.7 jars (the round's `measure/rows/rt.txt`): at every type
    //! position that parses, a text of 255 bytes is written and a longer one throws.
    use super::*;
    use crate::ergo_tree::ErgoTreeVersion;
    use crate::mir::expr::Expr;
    use crate::serialization::sigma_byte_reader::{from_bytes, SigmaByteRead};
    use crate::serialization::SigmaSerializationError;

    /// A name of the rows: its bytes on the wire and the length of its text
    type Name = (&'static str, Vec<u8>, usize);
    /// A type position: the bytes before the type variable, those after it, the tree versions
    /// it parses at
    type Position = (&'static str, &'static str, &'static str, &'static [u8]);

    fn names() -> [Name; 6] {
        let with_ff = |a: usize| [vec![b'a'; a], vec![0xff]].concat();
        [
            ("a255", vec![b'a'; 255], 255),
            ("ff85", vec![0xff; 85], 255),
            ("mix255", with_ff(252), 255),
            ("mix256", with_ff(253), 256),
            ("ff86", vec![0xff; 86], 258),
            ("ff255", vec![0xff; 255], 765),
        ]
    }

    fn refused(res: &Result<Vec<u8>, SigmaSerializationError>) -> bool {
        matches!(res, Err(SigmaSerializationError::NotSupported(_)))
    }

    #[test]
    fn a_text_of_255_bytes_is_written_and_a_longer_one_refused() {
        for (name, wire, text_len) in names() {
            let tv = STypeVar::new_from_bytes(wire).unwrap();
            let text = tv.as_string();
            assert_eq!(text.len(), text_len, "{name}");
            let written = tv.sigma_serialize_bytes();
            if text_len <= 255 {
                let bytes = written.unwrap();
                assert_eq!(bytes[0] as usize, text_len, "{name}");
                assert_eq!(&bytes[1..], text.as_bytes(), "{name}");
            } else {
                assert!(refused(&written), "{name}: {written:?}");
            }
        }
    }

    #[test]
    fn a_constructor_takes_a_name_its_writer_refuses() {
        let tv = STypeVar::new_from_bytes(vec![b'a'; 256]).unwrap();
        assert_eq!(tv.as_string().len(), 256);
        assert!(refused(&tv.sigma_serialize_bytes()));
        // `new_from_str` takes a `&'static str`
        let long: &'static str = alloc::boxed::Box::leak("a".repeat(256).into_boxed_str());
        assert!(refused(
            &STypeVar::new_from_str(long)
                .unwrap()
                .sigma_serialize_bytes()
        ));
    }

    #[test]
    fn every_type_position_writes_the_text_or_refuses() {
        const BOTH: &[u8] = &[0, 3];
        const V3: &[u8] = &[3];
        let positions: [Position; 14] = [
            ("a collection node's item type", "8300", "", BOTH),
            ("its item type Coll[T]", "83000c", "", BOTH),
            ("its item type Option[T]", "830024", "", BOTH),
            ("its item type (T, Int)", "83004c", "", BOTH),
            ("a function type's type parameter", "83007001040401", "", V3),
            ("a function type's domain", "83007001", "0400", V3),
            ("getVar's type", "e301", "", BOTH),
            ("ExtractRegisterAs' type", "c6a704", "", BOTH),
            ("a lambda's argument", "d90101", "7201", BOTH),
            ("a FunDef's type argument", "d801d70101", "04020402", BOTH),
            ("executeFromVar's type", "d4", "01", BOTH),
            ("a method call's type argument", "dc6a04dd010e0100", "", V3),
            (
                "a method call's type argument Coll[T]",
                "dc6a04dd010e01000c",
                "",
                V3,
            ),
            ("a None constant of Option[T]", "24", "00", V3),
        ];
        let mut rows = 0;
        for (position, before, after, versions) in positions {
            for version in versions {
                for (name, wire, text_len) in names() {
                    let text = STypeVar::new_from_bytes(wire.clone()).unwrap().as_string();
                    let row = |name_len: usize, name_bytes: &[u8]| {
                        let hex = format!(
                            "{before}67{name_len:02x}{}{after}",
                            base16::encode_lower(name_bytes)
                        );
                        base16::decode(hex.as_bytes()).unwrap()
                    };
                    let expr = from_bytes(&row(wire.len(), &wire))
                        .with_tree_version(ErgoTreeVersion::from(*version), Expr::sigma_parse)
                        .unwrap();
                    let written = expr.sigma_serialize_bytes();
                    if text_len <= 255 {
                        // the bytes read, with the name's text in the name's place
                        assert_eq!(
                            written.unwrap(),
                            row(text_len, text.as_bytes()),
                            "{position}, v{version}, {name}"
                        );
                    } else {
                        assert!(refused(&written), "{position}, v{version}, {name}");
                    }
                    rows += 1;
                }
            }
        }
        assert_eq!(rows, 23 * 6);
    }
}
