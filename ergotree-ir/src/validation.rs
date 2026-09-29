//! Validation rules and their statuses.
//!
//! Port of the parts of `sigma.validation` / `org.ergoplatform.validation`
//! (sigmastate-interpreter v6.0.3) needed to parse and serialize the rule
//! statuses embedded in `ErgoValidationSettingsUpdate` extension payloads
//! (extension key `[0x00, 124]`): [`RuleStatus`] (JVM `RuleStatus.scala`) and
//! its consensus serialization (JVM `RuleStatusSerializer.scala`).

use alloc::vec::Vec;

use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;
use crate::serialization::SigmaParsingError;
use crate::serialization::SigmaSerializable;
use crate::serialization::SigmaSerializationError;
use crate::serialization::SigmaSerializeResult;
use sigma_ser::vlq_encode::WriteSigmaVlqExt;

/// The id of the first validation rule (JVM
/// `sigma.validation.ValidationRules.FirstRuleId`, `ValidationRules.scala:78`;
/// re-declared as `RuleStatusSerializer.FirstRuleId`,
/// `RuleStatusSerializer.scala:9`).
///
/// Rule ids never appear on the wire directly: both the `statusUpdates` keys of
/// `ErgoValidationSettingsUpdate` and the [`RuleStatus::ReplacedRule`] payload
/// store a VLQ-encoded unsigned-short *offset* from this base.
pub const FIRST_RULE_ID: i16 = 1000;

/// Status of a validation rule (JVM `sigma.validation.RuleStatus`,
/// `RuleStatus.scala`), alterable by soft-forks via block extension voting.
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum RuleStatus {
    /// Default status of a rule which is registered in the table and not yet
    /// altered by soft-forks (JVM `EnabledRule`, status code 1).
    EnabledRule,
    /// Rule disabled in the current version via block extensions and the
    /// voting process (JVM `DisabledRule`, status code 2).
    DisabledRule,
    /// Rule replaced by a new rule via soft-fork extensions. Like
    /// `DisabledRule`, but additionally requires the new rule to be enabled at
    /// the same time, i.e. atomically (JVM `ReplacedRule(newRuleId: Short)`,
    /// status code 3).
    ///
    /// The id is a JVM `Short`: parsing computes
    /// `(offset + FirstRuleId).toShort` (`RuleStatusSerializer.scala:50`), so
    /// offsets near the unsigned-short ceiling wrap into small or negative
    /// ids. Such wrapped ids (and any id below [`FIRST_RULE_ID`]) cannot be
    /// re-serialized — see [`SigmaSerializable::sigma_serialize`].
    ReplacedRule(i16),
    /// Rule whose parameters are changed via soft-fork extensions; the payload
    /// is the new value of the block extension value with key == rule id (JVM
    /// `ChangedRule(newValue: Array[Byte])`, status code 4).
    ChangedRule(Vec<u8>),
}

impl RuleStatus {
    /// Status code of [`RuleStatus::EnabledRule`] (JVM `RuleStatus.EnabledRuleCode`)
    pub const ENABLED_RULE_CODE: u8 = 1;
    /// Status code of [`RuleStatus::DisabledRule`] (JVM `RuleStatus.DisabledRuleCode`)
    pub const DISABLED_RULE_CODE: u8 = 2;
    /// Status code of [`RuleStatus::ReplacedRule`] (JVM `RuleStatus.ReplacedRuleCode`)
    pub const REPLACED_RULE_CODE: u8 = 3;
    /// Status code of [`RuleStatus::ChangedRule`] (JVM `RuleStatus.ChangedRuleCode`)
    pub const CHANGED_RULE_CODE: u8 = 4;

    /// Wire code of this status (JVM `RuleStatus.statusCode`)
    pub fn status_code(&self) -> u8 {
        match self {
            RuleStatus::EnabledRule => Self::ENABLED_RULE_CODE,
            RuleStatus::DisabledRule => Self::DISABLED_RULE_CODE,
            RuleStatus::ReplacedRule(_) => Self::REPLACED_RULE_CODE,
            RuleStatus::ChangedRule(_) => Self::CHANGED_RULE_CODE,
        }
    }
}

/// Consensus serialization of [`RuleStatus`] (JVM `RuleStatusSerializer`,
/// `RuleStatusSerializer.scala`).
///
/// The general format for rule statuses (`RuleStatusSerializer.scala:18-24`):
///
/// ```text
/// field      | format | #bytes         | description
/// -----------------------------------------------------------------------
/// dataSize   | UShort | 1..2 bytes     | number of bytes for dataBytes
/// statusCode | Byte   | 1 byte         | code of the status type
/// dataBytes  | Bytes  | dataSize bytes | serialized bytes of status value
/// ```
impl SigmaSerializable for RuleStatus {
    /// Mirror of `RuleStatusSerializer.serialize` (`RuleStatusSerializer.scala:25-39`).
    ///
    /// Statuses that the JVM writer rejects with an assertion error are
    /// rejected here with [`SigmaSerializationError::NotSupported`]:
    /// `ReplacedRule` ids below [`FIRST_RULE_ID`] (negative wire offset) and
    /// `ChangedRule` payloads longer than `0xFFFF` (both via scorex
    /// `putUShort`'s `[0, 0xFFFF]` range assert).
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        match self {
            RuleStatus::EnabledRule | RuleStatus::DisabledRule => {
                w.put_u16(0)?; // zero bytes for dataBytes
                w.put_u8(self.status_code())?;
                Ok(())
            }
            RuleStatus::ReplacedRule(new_rule_id) => {
                // id offset (JVM Int arithmetic, RuleStatusSerializer.scala:30)
                let ofs = *new_rule_id as i32 - FIRST_RULE_ID as i32;
                let ofs: u16 = u16::try_from(ofs).map_err(|_| {
                    SigmaSerializationError::NotSupported(format!(
                        "ReplacedRule id {} is below FIRST_RULE_ID {} (negative wire offset)",
                        new_rule_id, FIRST_RULE_ID
                    ))
                })?;
                // number of bytes to store the id offset
                // (JVM measureWrittenBytes, RuleStatusSerializer.scala:11-16,31)
                let mut ofs_vlq = Vec::new();
                ofs_vlq.put_u16(ofs)?;
                w.put_u16(ofs_vlq.len() as u16)?; // size of dataBytes
                w.put_u8(self.status_code())?;
                w.put_u16(ofs)?; // dataBytes
                Ok(())
            }
            RuleStatus::ChangedRule(data) => {
                let data_size: u16 = u16::try_from(data.len()).map_err(|_| {
                    SigmaSerializationError::NotSupported(format!(
                        "ChangedRule payload of {} bytes exceeds the UShort dataSize range",
                        data.len()
                    ))
                })?;
                w.put_u16(data_size)?;
                w.put_u8(self.status_code())?;
                w.write_all(data)?;
                Ok(())
            }
        }
    }

    /// Mirror of `RuleStatusSerializer.parse` (`RuleStatusSerializer.scala:41-59`).
    ///
    /// JVM quirks preserved exactly:
    /// - the `ReplacedRule` arm ignores `dataSize` and reads a VLQ UShort
    ///   offset regardless, wrapping `(offset + FirstRuleId)` to a signed
    ///   16-bit id (`.toShort`, line 50);
    /// - an unrecognized status code skips `dataSize` bytes and yields
    ///   `ReplacedRule(0)`, so old code processes it as a soft-fork
    ///   (lines 55-57).
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let data_size = r.get_u16()? as usize; // number of bytes occupied by status data
        let status_type = r.get_u8()?;
        match status_type {
            RuleStatus::ENABLED_RULE_CODE => Ok(RuleStatus::EnabledRule),
            RuleStatus::DISABLED_RULE_CODE => Ok(RuleStatus::DisabledRule), // the rule is explicitly disabled
            RuleStatus::REPLACED_RULE_CODE => {
                // store small offsets using a single byte
                let new_rule_id = r.get_u16()?.wrapping_add(FIRST_RULE_ID as u16) as i16;
                Ok(RuleStatus::ReplacedRule(new_rule_id)) // the rule is disabled, but we also have info about the new rule
            }
            RuleStatus::CHANGED_RULE_CODE => {
                // value bytes except statusType
                let mut data = vec![0u8; data_size];
                r.get_bytes_into(&mut data)?;
                Ok(RuleStatus::ChangedRule(data))
            }
            _ => {
                // Skip status bytes which we don't understand; an unrecognized
                // status code is processed as a soft-fork. The JVM skips via a
                // raw `r.position += dataSize`, which bounds-checks against the
                // buffer end but NOT against the soft position-limit window
                // (unlike `getBytes`) — hence `read_exact` here rather than the
                // window-checked `get_bytes_into`.
                let mut skipped = vec![0u8; data_size];
                r.read_exact(&mut skipped)?;
                Ok(RuleStatus::ReplacedRule(0))
            }
        }
    }
}

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arbitrary {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;

    impl Arbitrary for RuleStatus {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        /// Mirrors the JVM `statusGen` (`ObjectGenerators.scala:444-448`),
        /// constrained like `replacedRuleIdGen` to ids that can be
        /// re-serialized (non-negative wire offset).
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            prop_oneof![
                Just(RuleStatus::EnabledRule),
                Just(RuleStatus::DisabledRule),
                (FIRST_RULE_ID..=i16::MAX).prop_map(RuleStatus::ReplacedRule),
                vec(any::<u8>(), 0..10).prop_map(RuleStatus::ChangedRule),
            ]
            .boxed()
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use crate::serialization::sigma_byte_reader::from_bytes;
    use sigma_ser::vlq_encode::ReadSigmaVlqExt;

    fn parse(bytes: &[u8]) -> Result<RuleStatus, SigmaParsingError> {
        RuleStatus::sigma_parse_bytes(bytes)
    }

    #[test]
    fn enabled_and_disabled_rule_vectors() {
        assert_eq!(
            RuleStatus::EnabledRule.sigma_serialize_bytes().unwrap(),
            vec![0x00, 0x01]
        );
        assert_eq!(
            RuleStatus::DisabledRule.sigma_serialize_bytes().unwrap(),
            vec![0x00, 0x02]
        );
        assert_eq!(parse(&[0x00, 0x01]).unwrap(), RuleStatus::EnabledRule);
        assert_eq!(parse(&[0x00, 0x02]).unwrap(), RuleStatus::DisabledRule);
    }

    #[test]
    fn replaced_rule_vector() {
        // offset 16 from FIRST_RULE_ID = rule 1016; VLQ(16) is one byte
        assert_eq!(
            RuleStatus::ReplacedRule(1016)
                .sigma_serialize_bytes()
                .unwrap(),
            vec![0x01, 0x03, 0x10]
        );
        assert_eq!(
            parse(&[0x01, 0x03, 0x10]).unwrap(),
            RuleStatus::ReplacedRule(1016)
        );
        // two-byte VLQ offset: rule 1000 + 300; VLQ(300) = [0xAC, 0x02]
        assert_eq!(
            RuleStatus::ReplacedRule(1300)
                .sigma_serialize_bytes()
                .unwrap(),
            vec![0x02, 0x03, 0xAC, 0x02]
        );
        assert_eq!(
            parse(&[0x02, 0x03, 0xAC, 0x02]).unwrap(),
            RuleStatus::ReplacedRule(1300)
        );
    }

    #[test]
    fn changed_rule_vector() {
        assert_eq!(
            RuleStatus::ChangedRule(vec![0x0A, 0x14])
                .sigma_serialize_bytes()
                .unwrap(),
            vec![0x02, 0x04, 0x0A, 0x14]
        );
        assert_eq!(
            parse(&[0x02, 0x04, 0x0A, 0x14]).unwrap(),
            RuleStatus::ChangedRule(vec![0x0A, 0x14])
        );
        // empty payload round-trips
        assert_eq!(
            RuleStatus::ChangedRule(vec![])
                .sigma_serialize_bytes()
                .unwrap(),
            vec![0x00, 0x04]
        );
        assert_eq!(
            parse(&[0x00, 0x04]).unwrap(),
            RuleStatus::ChangedRule(vec![])
        );
    }

    /// Mirror of the JVM "parse unrecognized status" spec
    /// (`RuleStatusSerializerSpec.scala:21-31`): dataSize bytes are skipped,
    /// `ReplacedRule(0)` is returned and the reader continues right after the
    /// entry.
    #[test]
    fn unknown_status_code_skips_data_size_bytes() {
        let unknown_code = 100u8;
        let bytes = [1, unknown_code, 10, 20];
        let mut r = from_bytes(&bytes[..]);
        let s = RuleStatus::sigma_parse(&mut r).unwrap();
        assert_eq!(s, RuleStatus::ReplacedRule(0));
        assert_eq!(r.get_u8().unwrap(), 20);
    }

    /// JVM `(r.getUShort() + FirstRuleId).toShort` wraps around the signed
    /// 16-bit range (`RuleStatusSerializer.scala:50`); such ids then fail to
    /// re-serialize (negative wire offset), as on the JVM.
    #[test]
    fn replaced_rule_id_wraps_like_jvm_toshort() {
        // offset 64536: 64536 + 1000 = 65536 -> .toShort = 0
        let wrapped_to_zero = parse(&[0x03, 0x03, 0x98, 0xF8, 0x03]).unwrap();
        assert_eq!(wrapped_to_zero, RuleStatus::ReplacedRule(0));
        // offset 40000: 40000 + 1000 = 41000 -> .toShort = -24536
        let wrapped_negative = parse(&[0x03, 0x03, 0xC0, 0xB8, 0x02]).unwrap();
        assert_eq!(wrapped_negative, RuleStatus::ReplacedRule(-24536));
        assert!(wrapped_to_zero.sigma_serialize_bytes().is_err());
        assert!(wrapped_negative.sigma_serialize_bytes().is_err());
    }

    /// The ReplacedRule arm ignores dataSize and reads the offset regardless
    /// (`RuleStatusSerializer.scala:49-51` never touches dataSize) — an
    /// inconsistent dataSize is accepted, exactly like the JVM.
    #[test]
    fn replaced_rule_data_size_is_not_validated() {
        assert_eq!(
            parse(&[0x7F, 0x03, 0x10]).unwrap(),
            RuleStatus::ReplacedRule(1016)
        );
    }

    #[test]
    fn truncated_and_malformed_entries_are_rejected() {
        // empty input
        assert!(parse(&[]).is_err());
        // dataSize only, no status code
        assert!(parse(&[0x00]).is_err());
        // ChangedRule with dataSize=1 but no payload
        assert!(parse(&[0x01, 0x04]).is_err());
        // ChangedRule with dataSize beyond the remaining bytes
        assert!(parse(&[0x05, 0x04, 0xAA]).is_err());
        // ReplacedRule with the offset bytes missing
        assert!(parse(&[0x01, 0x03]).is_err());
        // unknown code whose dataSize overruns the buffer (JVM
        // `r.position += dataSize` throws when positioned past the end)
        assert!(parse(&[0x05, 0x63]).is_err());
    }

    #[test]
    fn replaced_rule_below_first_rule_id_fails_to_serialize() {
        // mirrors the JVM scorex `putUShort` assert on the negative offset
        assert!(RuleStatus::ReplacedRule(999)
            .sigma_serialize_bytes()
            .is_err());
        assert!(RuleStatus::ReplacedRule(0).sigma_serialize_bytes().is_err());
        assert!(RuleStatus::ReplacedRule(1000)
            .sigma_serialize_bytes()
            .is_ok());
    }

    /// Acceptance vector: the mainnet h=1,628,160 extension value for key
    /// `[0x00, 124]` (`ErgoValidationSettingsUpdate`). Walks the value exactly
    /// like `ErgoValidationSettingsUpdateSerializer.parse`
    /// (`ErgoValidationSettingsUpdate.scala:51-56`) and re-encodes it
    /// byte-identically.
    #[test]
    fn mainnet_1628160_status_updates_roundtrip() {
        let sample: [u8; 18] = [
            0x02, 0xD7, 0x01, 0x99, 0x03, // rulesToDisable: count 2, ids 215 and 409
            0x03, // statusUpdates: count 3
            0x0B, 0x01, 0x03, 0x10, // rule 1011 -> ReplacedRule(1016)
            0x07, 0x01, 0x03, 0x11, // rule 1007 -> ReplacedRule(1017)
            0x08, 0x01, 0x03, 0x12, // rule 1008 -> ReplacedRule(1018)
        ];

        let mut r = from_bytes(&sample[..]);
        let disabled_count = r.get_u32().unwrap();
        assert_eq!(disabled_count, 2);
        let disabled: Vec<u16> = (0..disabled_count).map(|_| r.get_u16().unwrap()).collect();
        assert_eq!(disabled, vec![215, 409]);

        let updates_count = r.get_u32().unwrap();
        assert_eq!(updates_count, 3);
        let updates: Vec<(i16, RuleStatus)> = (0..updates_count)
            .map(|_| {
                let rule_id = r.get_u16().unwrap().wrapping_add(FIRST_RULE_ID as u16) as i16;
                let status = RuleStatus::sigma_parse(&mut r).unwrap();
                (rule_id, status)
            })
            .collect();
        assert_eq!(
            updates,
            vec![
                (1011, RuleStatus::ReplacedRule(1016)),
                (1007, RuleStatus::ReplacedRule(1017)),
                (1008, RuleStatus::ReplacedRule(1018)),
            ]
        );
        // the sample is fully consumed
        assert!(r.get_u8().is_err());

        // re-encode byte-identically, mirroring
        // ErgoValidationSettingsUpdateSerializer.serialize
        let mut encoded = Vec::new();
        {
            use crate::serialization::sigma_byte_writer::SigmaByteWriter;
            let mut w = SigmaByteWriter::new(&mut encoded, None);
            w.put_u32(disabled_count).unwrap();
            for id in &disabled {
                w.put_u16(*id).unwrap();
            }
            w.put_u32(updates_count).unwrap();
            for (rule_id, status) in &updates {
                w.put_u16((*rule_id as i32 - FIRST_RULE_ID as i32) as u16)
                    .unwrap();
                status.sigma_serialize(&mut w).unwrap();
            }
        }
        assert_eq!(encoded, sample);
    }

    #[cfg(feature = "arbitrary")]
    mod proptests {
        use super::*;
        use crate::serialization::sigma_serialize_roundtrip;
        use proptest::prelude::*;

        proptest! {

            /// Mirror of the JVM "RuleStatusSerializer round trip" property
            /// (`RuleStatusSerializerSpec.scala:15-19`)
            #[test]
            fn ser_roundtrip(v in any::<RuleStatus>()) {
                prop_assert_eq![sigma_serialize_roundtrip(&v), v];
            }
        }
    }
}
