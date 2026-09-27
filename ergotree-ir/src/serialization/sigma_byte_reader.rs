//! Sigma byte stream writer
use crate::ergo_tree::ErgoTreeVersion;

use super::constant_store::ConstantStore;
use super::val_def_type_store::ValDefTypeStore;
use super::SigmaParsingError;
use core3::io::Cursor;
use core3::io::Read;
use core3::io::Seek;
use sigma_ser::vlq_encode::PositionLimit;
use sigma_ser::vlq_encode::ReadSigmaVlqExt;

/// Maximum nesting level of value deserialization on one reader
/// (sigmastate `SigmaConstants.MaxTreeDepth`, the default `maxTreeDepth` of every reader)
pub const MAX_TREE_DEPTH: usize = 110;

/// Implementation of SigmaByteRead
pub struct SigmaByteReader<R> {
    inner: R,
    constant_store: ConstantStore,
    substitute_placeholders: bool,
    val_def_type_store: ValDefTypeStore,
    was_deserialize: bool,
    version: ErgoTreeVersion,
    position_limit: u64,
    level: usize,
}

impl<R: Read> SigmaByteReader<R> {
    /// Create new reader from PeekableReader
    pub fn new(pr: R, constant_store: ConstantStore) -> SigmaByteReader<R> {
        SigmaByteReader {
            inner: pr,
            constant_store,
            substitute_placeholders: false,
            val_def_type_store: ValDefTypeStore::new(),
            was_deserialize: false,
            version: ErgoTreeVersion::V0,
            position_limit: u64::MAX,
            level: 0,
        }
    }

    /// Make a new reader with underlying PeekableReader and constant_store to resolve constant
    /// placeholders
    pub fn new_with_substitute_placeholders(
        pr: R,
        constant_store: ConstantStore,
    ) -> SigmaByteReader<R> {
        SigmaByteReader {
            inner: pr,
            constant_store,
            substitute_placeholders: true,
            val_def_type_store: ValDefTypeStore::new(),
            was_deserialize: false,
            version: ErgoTreeVersion::MAX_SCRIPT_VERSION,
            position_limit: u64::MAX,
            level: 0,
        }
    }
}

/// Create SigmaByteReader from a byte array (with empty constant store)
pub fn from_bytes<T: AsRef<[u8]>>(bytes: T) -> SigmaByteReader<Cursor<T>> {
    SigmaByteReader::new(Cursor::new(bytes), ConstantStore::empty())
}

/// Sigma byte reader trait with a constant store to resolve segregated constants
pub trait SigmaByteRead: ReadSigmaVlqExt {
    /// Constant store with constants to resolve constant placeholder types
    fn constant_store(&mut self) -> &mut ConstantStore;

    /// Option to substitute ConstantPlaceholder with Constant from the store
    fn substitute_placeholders(&self) -> bool;

    /// Set new constant store
    fn set_constant_store(&mut self, constant_store: ConstantStore);

    /// ValDef types store (resolves tpe on ValUse parsing)
    fn val_def_type_store(&mut self) -> &mut ValDefTypeStore;

    /// Returns if value that was deserialized has deserialize nodes, such as DeserializeContext and DeserializeRegister
    fn was_deserialize(&self) -> bool;

    /// Set that deserialization node was read
    fn set_deserialize(&mut self, has_deserialize: bool);

    /// Get position of reader in buffer. This is functionally equivalent to [`std::io::Seek::stream_position`] but redefined here so it can be used in no_std contexts
    fn position(&mut self) -> core3::io::Result<u64> {
        #[cfg(feature = "std")]
        {
            <Self as Seek>::stream_position(self)
        }
        #[cfg(not(feature = "std"))]
        {
            self.seek(core3::io::SeekFrom::Current(0))
        }
    }

    /// Call `f` with reader's ErgoTree version set to `version` inside f's scope
    fn with_tree_version<T>(
        &mut self,
        version: ErgoTreeVersion,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T;

    /// Maximum ErgoTree version that deserializer can handle.
    fn tree_version(&self) -> ErgoTreeVersion;

    /// Current nesting level of value deserialization (sigmastate `CoreByteReader.level`)
    fn level(&self) -> usize;

    /// Set the nesting level. A level above [`MAX_TREE_DEPTH`] fails the parse
    /// (sigmastate v6.0.6 `CoreByteReader.level_=`, `CoreByteReader.scala:127-131`).
    fn set_level(&mut self, level: usize) -> Result<(), SigmaParsingError>;
}

impl<R: Read> Read for SigmaByteReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> core3::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl<R: Seek> Seek for SigmaByteReader<R> {
    fn seek(&mut self, pos: core3::io::SeekFrom) -> core3::io::Result<u64> {
        self.inner.seek(pos)
    }

    #[cfg(feature = "std")]
    fn rewind(&mut self) -> core3::io::Result<()> {
        self.inner.rewind()
    }

    #[cfg(feature = "std")]
    fn stream_position(&mut self) -> core3::io::Result<u64> {
        self.inner.stream_position()
    }
}

/// Real position-limit storage: this is the decorating reader the limit lives on
/// (the reference impl's `CoreByteReader.positionLimit`), while the wrapped inner
/// reader stays unchecked.
impl<R> PositionLimit for SigmaByteReader<R> {
    fn position_limit(&self) -> u64 {
        self.position_limit
    }
    fn set_position_limit(&mut self, limit: u64) {
        self.position_limit = limit;
    }
}

impl<R: ReadSigmaVlqExt> SigmaByteRead for SigmaByteReader<R> {
    fn constant_store(&mut self) -> &mut ConstantStore {
        &mut self.constant_store
    }

    fn substitute_placeholders(&self) -> bool {
        self.substitute_placeholders
    }

    fn set_constant_store(&mut self, constant_store: ConstantStore) {
        self.constant_store = constant_store;
    }

    fn val_def_type_store(&mut self) -> &mut ValDefTypeStore {
        &mut self.val_def_type_store
    }

    fn was_deserialize(&self) -> bool {
        self.was_deserialize
    }

    fn set_deserialize(&mut self, has_deserialize: bool) {
        self.was_deserialize = has_deserialize
    }

    fn with_tree_version<T>(
        &mut self,
        version: ErgoTreeVersion,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let tmp = self.version;
        self.version = version;
        let res = f(self);
        self.version = tmp;
        res
    }

    fn tree_version(&self) -> ErgoTreeVersion {
        self.version
    }

    fn level(&self) -> usize {
        self.level
    }

    fn set_level(&mut self, level: usize) -> Result<(), SigmaParsingError> {
        if level > MAX_TREE_DEPTH {
            return Err(SigmaParsingError::DeserializeCallDepthExceeded(level));
        }
        self.level = level;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod level_tests {
    use super::*;

    #[test]
    fn set_level_allows_max_tree_depth_and_rejects_above() {
        // `CoreByteReader.level_=` (`CoreByteReader.scala:127-131`): 110 is allowed,
        // 111 fails and leaves the level where it was.
        let mut r = from_bytes([0u8; 0]);
        assert_eq!(r.level(), 0);
        r.set_level(MAX_TREE_DEPTH).unwrap();
        assert!(matches!(
            r.set_level(MAX_TREE_DEPTH + 1),
            Err(SigmaParsingError::DeserializeCallDepthExceeded(111))
        ));
        assert_eq!(r.level(), MAX_TREE_DEPTH);
    }
}
