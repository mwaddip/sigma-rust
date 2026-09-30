//! Transaction input

pub mod prover_result;

use ergotree_interpreter::sigma_protocol::prover::ProofBytes;
use ergotree_ir::chain::context_extension::ContextExtension;
use ergotree_ir::chain::ergo_box::BoxId;
use ergotree_ir::serialization::sigma_byte_reader::SigmaByteRead;
use ergotree_ir::serialization::sigma_byte_writer::SigmaByteWrite;
use ergotree_ir::serialization::SigmaParsingError;
use ergotree_ir::serialization::SigmaSerializable;
use ergotree_ir::serialization::SigmaSerializeResult;

use crate::wallet::box_selector::ErgoBoxId;
#[cfg(feature = "json")]
use ergotree_ir::ergo_tree::ErgoTreeVersion;
#[cfg(feature = "json")]
use serde::ser::SerializeStruct;
#[cfg(feature = "json")]
use serde::{Deserialize, Serialize};

use self::prover_result::ProverResult;

/// Unsigned (without proofs) transaction input
#[derive(PartialEq, Eq, Debug, Clone)]
#[cfg_attr(feature = "arbitrary", derive(proptest_derive::Arbitrary))]
#[cfg_attr(feature = "json", derive(Deserialize))]
pub struct UnsignedInput {
    /// id of the box to spent
    #[cfg_attr(feature = "json", serde(rename = "boxId"))]
    pub box_id: BoxId,
    /// user-defined variables to be put into context
    #[cfg_attr(feature = "json", serde(rename = "extension",))]
    pub extension: ContextExtension,
}

#[cfg(feature = "json")]
impl Serialize for UnsignedInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.json_at(ErgoTreeVersion::V0).serialize(serializer)
    }
}

#[cfg(feature = "json")]
impl UnsignedInput {
    /// JSON of the input, with its context extension values as a writer at ErgoTree `version`
    /// writes them
    pub(crate) fn json_at(&self, version: ErgoTreeVersion) -> impl Serialize + '_ {
        UnsignedInputJson {
            input: self,
            version,
        }
    }
}

/// [`UnsignedInput::json_at`]
#[cfg(feature = "json")]
struct UnsignedInputJson<'a> {
    input: &'a UnsignedInput,
    version: ErgoTreeVersion,
}

#[cfg(feature = "json")]
impl Serialize for UnsignedInputJson<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut s = serializer.serialize_struct("UnsignedInput", 2)?;
        s.serialize_field("boxId", &self.input.box_id)?;
        s.serialize_field("extension", &self.input.extension.json_at(self.version))?;
        s.end()
    }
}

impl UnsignedInput {
    /// Create new with empty ContextExtension
    pub fn new(box_id: BoxId, extension: ContextExtension) -> Self {
        UnsignedInput { box_id, extension }
    }

    /// Create new Input with empty proof (for UnsignedTransaction id calculation)
    pub fn input_to_sign(&self) -> Input {
        Input {
            box_id: self.box_id,
            spending_proof: ProverResult {
                proof: ProofBytes::Empty,
                extension: self.extension.clone(),
            },
        }
    }
}

impl<T: ErgoBoxId> From<T> for UnsignedInput {
    fn from(b: T) -> Self {
        UnsignedInput::new(b.box_id(), ContextExtension::empty())
    }
}

/// Fully signed transaction input
#[derive(PartialEq, Eq, Debug, Clone)]
#[cfg_attr(feature = "arbitrary", derive(proptest_derive::Arbitrary))]
#[cfg_attr(feature = "json", derive(Deserialize))]
pub struct Input {
    /// id of the box to spent
    #[cfg_attr(feature = "json", serde(rename = "boxId", alias = "id"))]
    pub box_id: BoxId,
    /// proof of spending correctness
    #[cfg_attr(
        feature = "json",
        serde(
            rename = "spendingProof",
            deserialize_with = "ergotree_ir::chain::json::t_as_string_or_struct"
        )
    )]
    pub spending_proof: ProverResult,
}

#[cfg(feature = "json")]
impl Serialize for Input {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.json_at(ErgoTreeVersion::V0).serialize(serializer)
    }
}

#[cfg(feature = "json")]
impl Input {
    /// JSON of the input, with its context extension values as a writer at ErgoTree `version`
    /// writes them
    pub(crate) fn json_at(&self, version: ErgoTreeVersion) -> impl Serialize + '_ {
        InputJson {
            input: self,
            version,
        }
    }
}

/// [`Input::json_at`]
#[cfg(feature = "json")]
struct InputJson<'a> {
    input: &'a Input,
    version: ErgoTreeVersion,
}

#[cfg(feature = "json")]
impl Serialize for InputJson<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut s = serializer.serialize_struct("Input", 2)?;
        s.serialize_field("boxId", &self.input.box_id)?;
        s.serialize_field(
            "spendingProof",
            &self.input.spending_proof.json_at(self.version),
        )?;
        s.end()
    }
}

impl Input {
    /// Create new
    pub fn new(box_id: BoxId, spending_proof: ProverResult) -> Self {
        Self {
            box_id,
            spending_proof,
        }
    }

    /// Create Input from UnsignedInput and a proof
    pub fn from_unsigned_input(unsigned_input: UnsignedInput, proof_bytes: ProofBytes) -> Self {
        Self::new(
            unsigned_input.box_id,
            ProverResult {
                proof: proof_bytes,
                extension: unsigned_input.extension,
            },
        )
    }

    /// input with an empty proof
    pub fn input_to_sign(&self) -> Input {
        Input {
            box_id: self.box_id,
            spending_proof: ProverResult {
                proof: ProofBytes::Empty,
                extension: self.spending_proof.extension.clone(),
            },
        }
    }
}

impl SigmaSerializable for Input {
    fn sigma_serialize<W: SigmaByteWrite>(&self, w: &mut W) -> SigmaSerializeResult {
        self.box_id.sigma_serialize(w)?;
        self.spending_proof.sigma_serialize(w)?;
        Ok(())
    }
    fn sigma_parse<R: SigmaByteRead>(r: &mut R) -> Result<Self, SigmaParsingError> {
        let box_id = BoxId::sigma_parse(r)?;
        let spending_proof = ProverResult::sigma_parse(r)?;
        Ok(Input {
            box_id,
            spending_proof,
        })
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use ergotree_ir::serialization::sigma_serialize_roundtrip;
    use proptest::prelude::*;

    proptest! {

        #[test]
        fn ser_roundtrip(v in any::<Input>()) {
            prop_assert_eq![sigma_serialize_roundtrip(&v), v];
        }
    }
}
