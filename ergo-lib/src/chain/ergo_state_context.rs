//! Blockchain state
use bounded_vec::BoundedVec;
use ergo_chain_types::{Header, PreHeader};

use super::parameters::Parameters;

/// Last block headers in descending order (first header is the newest one).
/// Between 1 and 10: the SDK signs and validates against an existing chain tip,
/// so at least the newest header is always available (the script context itself
/// allows fewer — see `ergotree_ir::chain::context::ContextHeaders`). A node
/// near genesis supplies as many real headers as exist instead of padding.
pub type Headers = BoundedVec<Header, 1, 10>;

/// Blockchain state (last headers, etc.)
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct ErgoStateContext {
    /// Block header with the current `spendingTransaction`, that can be predicted
    /// by a miner before it's formation
    pub pre_header: PreHeader,
    /// Last block headers in descending order (first header is the newest one)
    pub headers: Headers,
    /// Parameters that can be adjusted by voting
    pub parameters: Parameters,
}

impl ErgoStateContext {
    /// Create an ErgoStateContext instance
    /// # Parameters
    /// The parameters' block version activates scripts, for signing as for validation: the
    /// activated script version is that block version minus 1, whatever the pre-header's
    /// version. [Parameters::default()] is the table set at genesis, with block version 1.
    /// Under it only a version 0 tree is proved, the interpreter's version gates read as
    /// before the 5.0 protocol (`CONTEXT.selfBoxIndex` is -1, for one), in a reduction that is
    /// signed elsewhere too, and validation applies neither the rule on an output's creation
    /// height nor the one on a negative height. So pass the chain's block version at least,
    /// and its whole table to validate. The table has to hold a `BlockVersion` entry: signing
    /// and validation read it, and panic without one.
    pub fn new(
        pre_header: PreHeader,
        headers: Headers,
        parameters: Parameters,
    ) -> ErgoStateContext {
        ErgoStateContext {
            pre_header,
            headers,
            parameters,
        }
    }

    /// The block version this state judges by: the voted parameters', as a signed byte (ergo
    /// v6.0.6 `ErgoStateContext.scala:114`, `Parameters.scala:80`). A block's header need not
    /// carry it between epoch starts (`exBlockVersion` is checked when an epoch starts,
    /// `:241`, `:265`), so the pre-header's version is what a script reads and nothing more.
    pub(crate) fn block_version(&self) -> i8 {
        self.parameters.block_version() as i8
    }
}

#[cfg(feature = "arbitrary")]
#[allow(clippy::unwrap_used)]
mod arbitrary {
    use super::*;
    use proptest::{collection::vec, prelude::*};

    impl Arbitrary for ErgoStateContext {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            // TODO: parameters should implement arbitrary as well, based on minimum/maximum constraints of each parameter
            (any::<PreHeader>(), vec(any::<Header>(), 10))
                .prop_map(|(pre_header, headers)| {
                    Self::new(
                        pre_header,
                        headers.try_into().unwrap(),
                        Parameters::default(),
                    )
                })
                .boxed()
        }
    }
}
