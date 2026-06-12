//! Code to implement `AutolykosSolution` JSON encoding

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::str::FromStr;
use num_bigint::BigUint;

use num_traits::FromPrimitive;
use serde::{Deserialize, Deserializer};

pub(crate) fn as_base16_string<S>(value: &[u8], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&base16::encode_lower(value))
}

pub(crate) fn from_base16_string<'de, D, T: From<Vec<u8>>>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    String::deserialize(deserializer)
        .and_then(|string| base16::decode(&string).map_err(|err| Error::custom(err.to_string())))
        .map(From::from)
}

/// Serialize `BigInt` as a string.
///
/// `None` is never reached in practice: the `d` field is
/// `skip_serializing_if = "Option::is_none"`, mirroring how an absent
/// distance round-trips (the JVM decoder fills `dForV2` for a missing
/// `d`, `AutolykosSolution.scala:53-55`).
pub(crate) fn bigint_as_str<S>(value: &Option<BigUint>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if let Some(value) = value {
        serializer.serialize_str(&value.to_string())
    } else {
        serializer.serialize_none()
    }
}

/// Deserialize a `BigInt` instance from either a String or from a `serde_json::Number` value.  We
/// need to do this because the JSON specification allows for arbitrarily-large numbers, a feature
/// that Autolykos makes use of to encode the PoW-distance (d) parameter. Note that we also need to
/// use `serde_json` with the `arbitrary_precision` feature for this to work.
///
/// An explicit `null` is accepted as `None`, matching the JVM decoder, where
/// `c.downField("d").as[Option[BigInt]]` (`AutolykosSolution.scala:53`) maps
/// both an absent and a `null` `d` to the `dForV2` default via circe's
/// `Option` decoder.
pub(crate) fn bigint_from_serde_json_number<'de, D>(
    deserializer: D,
) -> Result<Option<BigUint>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;

    match Option::<DeserializeBigIntFrom>::deserialize(deserializer) {
        Ok(None) => Ok(None),
        Ok(Some(s)) => match s {
            DeserializeBigIntFrom::String(s) => BigUint::from_str(&s)
                .map(Some)
                .map_err(|e| Error::custom(e.to_string())),
            DeserializeBigIntFrom::SerdeJsonNumber(n) => {
                let bigint = if n.is_f64() {
                    let n_f64 = n
                        .as_f64()
                        .ok_or_else(|| Error::custom("failed to convert JSON number to f64"))?;

                    BigUint::from_f64(n_f64).ok_or_else(|| {
                        Error::custom("failed to create BigInt from f64".to_string())
                    })
                } else {
                    BigUint::from_str(&n.to_string()).map_err(|e| Error::custom(e.to_string()))
                };

                bigint.map(Some)
            }
        },
        Err(e) => Err(Error::custom(e.to_string())),
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DeserializeBigIntFrom {
    String(String),
    SerdeJsonNumber(serde_json::Number),
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use serde::de::IntoDeserializer;
    use serde_json::Value;

    use super::bigint_from_serde_json_number;

    #[test]
    fn test_scientific_notion_deser() {
        let pow_d_parameter = r#"4.69094608138843e64"#;
        let j: Value = serde_json::from_str(pow_d_parameter).unwrap();
        let result = bigint_from_serde_json_number(j.into_deserializer()).unwrap();

        assert_eq!(
            result.unwrap().to_string(),
            "46909460813884301641411510982628556119846083366464832536248844288"
        )
    }
}
