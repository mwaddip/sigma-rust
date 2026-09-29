//! ErgoTree header

use alloc::string::{String, ToString};
use derive_more::{From, Into};
use thiserror::Error;

use crate::serialization::sigma_byte_reader::SigmaByteRead;
use crate::serialization::sigma_byte_writer::SigmaByteWrite;

/// Currently we define meaning for only first byte, which may be extended in future versions.
///    7  6  5  4  3  2  1  0
///  -------------------------
///  |  |  |  |  |  |  |  |  |
///  -------------------------
///  Bit 7 == 1 if the header contains more than 1 byte (default == 0)
///  Bit 6 - reserved for GZIP compression (should be 0)
///  Bit 5 == 1 - reserved for context dependent costing (should be = 0)
///  Bit 4 == 1 if constant segregation is used for this ErgoTree (default = 0)
///  (see <https://github.com/ScorexFoundation/sigmastate-interpreter/issues/264>)
///  Bit 3 == 1 if size of the whole tree is serialized after the header byte (default = 0)
///  Bits 2-0 - language version (current version == 0)
///
///  Currently we don't specify interpretation for the second and other bytes of the header.
///  We reserve the possibility to extend header by using Bit 7 == 1 and chain additional bytes as in VLQ.
///  Once the new bytes are required, a new version of the language should be created and implemented.
///  That new language will give an interpretation for the new bytes.
///
///  The byte is kept as read. Like sigmastate's `ErgoTree.header` (`ErgoTree.scala:82`), only
///  the version, size and segregation bits are interpreted (`:255-261`), and the byte is written
///  back whole (`ErgoTreeSerializer.scala:81`), so bits 5-7 round-trip.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct ErgoTreeHeader {
    header_byte: u8,
}

impl ErgoTreeHeader {
    /// Serialization
    pub fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> Result<(), core3::io::Error> {
        w.put_u8(self.serialized())
    }
    /// Deserialization
    pub fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, ErgoTreeHeaderError> {
        let header_byte = r
            .get_u8()
            .map_err(|e| ErgoTreeHeaderError::IoError(e.to_string()))?;

        ErgoTreeHeader::new(header_byte)
    }
}

impl ErgoTreeHeader {
    const CONSTANT_SEGREGATION_FLAG: u8 = 0b0001_0000;
    const HAS_SIZE_FLAG: u8 = 0b0000_1000;

    /// Parse from byte
    pub fn new(header_byte: u8) -> Result<Self, ErgoTreeHeaderError> {
        let header = ErgoTreeHeader { header_byte };
        // JVM `CheckHeaderSizeBit` (ValidationRules Rule-1012, applied in
        // `ErgoTreeSerializer` right after the header byte is read): for any
        // version > 0 the size bit must be set. Reject otherwise, mirroring the
        // JVM `ValidationException` (a malformed v>0 header without the size slot).
        if header.version() != ErgoTreeVersion::V0 && !header.has_size() {
            return Err(ErgoTreeHeaderError::InvalidSizeBit(header.version().0));
        }
        Ok(header)
    }

    /// Serialize to byte
    pub fn serialized(&self) -> u8 {
        self.header_byte
    }

    /// Return a header with version set to 0 and constant segregation flag set to the given value
    pub fn v0(constant_segregation: bool) -> Self {
        Self::from_flags(ErgoTreeVersion::V0, constant_segregation, false)
    }

    /// Return a header with version set to 1 (with size flag set) and constant segregation flag set to the given value
    pub fn v1(constant_segregation: bool) -> Self {
        Self::from_flags(ErgoTreeVersion::V1, constant_segregation, true)
    }

    fn from_flags(version: ErgoTreeVersion, is_constant_segregation: bool, has_size: bool) -> Self {
        let mut header_byte: u8 = version.0;
        if is_constant_segregation {
            header_byte |= Self::CONSTANT_SEGREGATION_FLAG;
        }
        if has_size {
            header_byte |= Self::HAS_SIZE_FLAG;
        }
        ErgoTreeHeader { header_byte }
    }

    /// Returns true if constant segregation flag is set
    pub fn is_constant_segregation(&self) -> bool {
        self.header_byte & Self::CONSTANT_SEGREGATION_FLAG != 0
    }

    /// Returns true if size flag is set
    pub fn has_size(&self) -> bool {
        self.header_byte & Self::HAS_SIZE_FLAG != 0
    }

    /// Returns ErgoTree version
    pub fn version(&self) -> ErgoTreeVersion {
        ErgoTreeVersion::parse_version(self.header_byte)
    }
}

/// Header parsing error
#[derive(Error, PartialEq, Eq, Debug, Clone, From)]
pub enum ErgoTreeHeaderError {
    /// Invalid version
    #[error("Invalid version: {0}")]
    VersionError(ErgoTreeVersionError),
    /// IO error
    #[error("IO error: {0}")]
    IoError(String),
    /// Size bit not set for a version > 0 header (JVM Rule-1012 `CheckHeaderSizeBit`)
    #[error("For version greater than 0, size bit should be set (version {0})")]
    InvalidSizeBit(u8),
}

/// ErgoTree version 0..=7, should fit in 3 bits
#[derive(PartialOrd, Ord, PartialEq, Eq, Debug, Clone, Copy, From, Into, Default)]
pub struct ErgoTreeVersion(u8);

impl ErgoTreeVersion {
    /// Header mask to extract version bits.
    pub const VERSION_MASK: u8 = 0x07;

    /// Max version of ErgoTree supported by interpreter
    pub const MAX_SCRIPT_VERSION: Self = Self::V3;
    /// Version 0
    pub const V0: Self = ErgoTreeVersion(0);
    /// Version 1 (size flag is mandatory)
    pub const V1: Self = ErgoTreeVersion(1);
    /// Version 2 (JIT)
    pub const V2: Self = ErgoTreeVersion(2);
    /// Version 3 (v6.0/Evolution)
    pub const V3: Self = ErgoTreeVersion(3);

    /// Returns a value of the version bits from the given header byte.
    pub fn parse_version(header_byte: u8) -> Self {
        ErgoTreeVersion(header_byte & ErgoTreeVersion::VERSION_MASK)
    }
}

/// Version parsing error
#[derive(Error, PartialEq, Eq, Debug, Clone, From)]
pub enum ErgoTreeVersionError {
    /// Invalid version
    #[error("Invalid version: {0}")]
    InvalidVersion(u8),
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // JVM Rule-1012 (`CheckHeaderSizeBit`): a version > 0 header MUST set the
    // size bit (0x08); v0 has no such requirement. Header byte layout: low 3
    // bits = version, 0x08 = size, 0x10 = constant segregation.
    #[test]
    fn header_rule_1012_size_bit_required_for_version_gt_0() {
        // v0 without size bit: allowed
        assert!(ErgoTreeHeader::new(0x00).is_ok());
        // v>0 WITH size bit: allowed
        assert!(ErgoTreeHeader::new(0x09).unwrap().has_size()); // v1 + size
        assert!(ErgoTreeHeader::new(0x0b).unwrap().has_size()); // v3 + size
                                                                // v>0 WITHOUT size bit: rejected (Rule-1012)
        for hb in [0x01u8, 0x02, 0x03] {
            assert_eq!(
                ErgoTreeHeader::new(hb),
                Err(ErgoTreeHeaderError::InvalidSizeBit(hb & 0x07)),
                "header 0x{:02x} (version {}, no size bit) must be rejected",
                hb,
                hb & 0x07
            );
        }
    }

    // sigmastate reads only the version, size and segregation bits of the header byte
    // (`ErgoTree.scala:255-261`) and keeps the byte whole, so bits 5-7 come back as read.
    #[test]
    fn header_keeps_bits_5_to_7() {
        for hb in [0x28u8, 0x48, 0x88, 0xe8, 0xe0, 0xf8, 0x2b] {
            let header = ErgoTreeHeader::new(hb).unwrap();
            assert_eq!(header.serialized(), hb);
            assert_eq!(header.version(), ErgoTreeVersion::from(hb & 0x07));
            assert_eq!(header.has_size(), hb & 0x08 != 0);
            assert_eq!(header.is_constant_segregation(), hb & 0x10 != 0);
        }
    }
}
