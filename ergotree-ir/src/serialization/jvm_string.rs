//! Bytes into a `String`, as the JVM's `new String(bytes, UTF_8)`

use alloc::string::String;

/// Decodes `bytes` as the JVM's `new String(bytes, StandardCharsets.UTF_8)` does: sigmastate
/// reads `String` data (`CoreDataSerializer.scala:110`) and a type variable's name
/// (`TypeSerializer.scala:204`) with it, and writes both back from the decoded text, so the
/// text decides the bytes a constant or a type is written with, and the length a `String` is
/// compared by.
///
/// Well-formed UTF-8 is its own text. Anything else is read in attempts: a lead byte and the
/// bytes after it that can still continue it, each attempt that is no character becoming ONE
/// `U+FFFD`. A byte continues a lead as in the UTF-8 table, with one exception: after `ED`
/// the JDK takes any continuation byte, so an encoded surrogate (`ED A0..BF` and the
/// continuation byte after it, where there is one) is one attempt, where
/// `String::from_utf8_lossy` replaces each of its bytes.
pub(crate) fn jvm_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        // the length the lead byte announces, and the second bytes that may follow it
        let (len, second) = match bytes[i] {
            0x00..=0x7F => (1, 0x80..=0xBF),
            0xC2..=0xDF => (2, 0x80..=0xBF),
            0xE0 => (3, 0xA0..=0xBF),
            0xE1..=0xEF => (3, 0x80..=0xBF),
            0xF0 => (4, 0x90..=0xBF),
            0xF1..=0xF3 => (4, 0x80..=0xBF),
            0xF4 => (4, 0x80..=0x8F),
            // a continuation byte, `C0`, `C1` and `F5..=FF` lead nothing
            _ => (0, 0x80..=0xBF),
        };
        let mut j = i + 1;
        while j < bytes.len() && j - i < len {
            let continues = if j == i + 1 {
                second.contains(&bytes[j])
            } else {
                (0x80..=0xBF).contains(&bytes[j])
            };
            if !continues {
                break;
            }
            j += 1;
        }
        match core::str::from_utf8(&bytes[i..j]) {
            Ok(s) if j - i == len => out.push_str(s),
            _ => out.push('\u{FFFD}'),
        }
        i = j;
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::jvm_string;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;

    /// One byte or more of every class the decoder tells apart (`A` of the round's
    /// `measure/decoder/U.java`)
    const CLASS_BYTES: [u8; 29] = [
        0x41, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xE1, 0xEC,
        0xED, 0xEE, 0xEF, 0xF0, 0xF1, 0xF3, 0xF4, 0xF5, 0xF7, 0xF8, 0xFB, 0xFC, 0xFE, 0xFF,
    ];

    /// Twelve of them, for the strings of five bytes: a four-byte attempt cut short with a byte
    /// on each side of it (`A5` of `U.java`)
    const SHORT_BYTES: [u8; 12] = [
        0x41, 0x80, 0x9F, 0xA0, 0xBF, 0xC2, 0xE0, 0xED, 0xF0, 0xF1, 0xF4, 0xFF,
    ];

    /// (bytes, the text's UTF-16 units) on OpenJDK 17.0.20.1; 1.8.0_504 and 21.0.12.1 give the
    /// same (the round's `measure/decoder/class-rows-java17.txt`)
    const CLASS_ROWS: [(&str, &str); 61] = [
        ("eda080", "fffd"),
        ("eda041", "fffd0041"),
        ("eda0", "fffd"),
        ("41eda0", "0041fffd"),
        ("e08080", "fffdfffdfffd"),
        ("e080", "fffdfffd"),
        ("e09f41", "fffdfffd0041"),
        ("f0808080", "fffdfffdfffdfffd"),
        ("f08080", "fffdfffdfffd"),
        ("f080", "fffdfffd"),
        ("f4908080", "fffdfffdfffdfffd"),
        ("f490", "fffdfffd"),
        ("f09f98", "fffd"),
        ("f09f9841", "fffd0041"),
        ("c080", "fffdfffd"),
        ("e282", "fffd"),
        ("ff", "fffd"),
        ("f5808080", "fffdfffdfffdfffd"),
        ("f8888080", "fffdfffdfffdfffd"),
        ("edb080eda080", "fffdfffd"),
        ("80", "fffd"),
        ("bf", "fffd"),
        ("c0", "fffd"),
        ("c1", "fffd"),
        ("c2", "fffd"),
        ("c241", "fffd0041"),
        ("e1", "fffd"),
        ("e180", "fffd"),
        ("e18041", "fffd0041"),
        ("e180c2", "fffdfffd"),
        ("ed", "fffd"),
        ("edbf", "fffd"),
        ("eda0c2", "fffdfffd"),
        ("e0a0", "fffd"),
        ("e0a041", "fffd0041"),
        ("f1", "fffd"),
        ("f180", "fffd"),
        ("f18080", "fffd"),
        ("f1808041", "fffd0041"),
        ("f180c2", "fffdfffd"),
        ("f090", "fffd"),
        ("f09080", "fffd"),
        ("f08f", "fffdfffd"),
        ("f4", "fffd"),
        ("f48f", "fffd"),
        ("f48fbf", "fffd"),
        ("f5", "fffd"),
        ("f7", "fffd"),
        ("f8", "fffd"),
        ("fe", "fffd"),
        ("41c2", "0041fffd"),
        ("e282ac", "20ac"),
        ("f09f9880", "d83dde00"),
        ("f48fbfbf", "dbffdfff"),
        // an attempt cut short with a byte on each side of it
        ("41f1808041", "0041fffd0041"),
        ("41f0908041", "0041fffd0041"),
        ("41f48fbf41", "0041fffd0041"),
        ("41eda08041", "0041fffd0041"),
        ("f1808041f1808041", "fffd0041fffd0041"),
        ("e282acf1808041", "20acfffd0041"),
        ("41f18080c241", "0041fffdfffd0041"),
    ];

    fn units(text: &str) -> String {
        text.encode_utf16()
            .map(|unit| format!("{unit:04x}"))
            .collect()
    }

    /// Every string of `len` bytes of `alphabet`, in counting order (the last byte runs fastest)
    fn each_string(alphabet: &[u8], len: usize, f: &mut dyn FnMut(&[u8])) {
        let mut index = vec![0usize; len];
        let mut bytes = vec![0u8; len];
        loop {
            for (byte, i) in bytes.iter_mut().zip(&index) {
                *byte = alphabet[*i];
            }
            f(&bytes);
            let mut place = len;
            loop {
                if place == 0 {
                    return;
                }
                place -= 1;
                index[place] += 1;
                if index[place] < alphabet.len() {
                    break;
                }
                index[place] = 0;
            }
        }
    }

    #[test]
    fn class_rows_decode_as_on_the_jdk() {
        for (bytes, expected) in CLASS_ROWS {
            let text = jvm_string(&base16::decode(bytes.as_bytes()).unwrap());
            assert_eq!(units(&text), expected, "{bytes}");
        }
    }

    /// How many strings of those lengths `alphabet` has, and ONE FNV-1a 64 over their texts: per
    /// string the text's length in UTF-16 units as two bytes, low first, then each unit low
    /// byte first (`U.java`'s `digest` and `digest5`)
    fn digest(alphabet: &[u8], lengths: core::ops::RangeInclusive<usize>) -> (u32, u64) {
        let mut hash: u64 = 0xcbf29ce484222325;
        let mut mix = |byte: u64| hash = (hash ^ byte).wrapping_mul(0x100000001b3);
        let mut count = 0u32;
        for len in lengths {
            each_string(alphabet, len, &mut |bytes| {
                let text = jvm_string(bytes);
                let length = text.encode_utf16().count() as u64;
                mix(length & 0xff);
                mix((length >> 8) & 0xff);
                for unit in text.encode_utf16() {
                    mix(unit as u64 & 0xff);
                    mix(unit as u64 >> 8);
                }
                count += 1;
            });
        }
        (count, hash)
    }

    #[test]
    fn every_class_string_decodes_as_on_the_jdk() {
        // `U.java digest` prints this count and hash on the three JDKs;
        // `String::from_utf8_lossy` gives 21f5f333c0106d71
        assert_eq!(digest(&CLASS_BYTES, 1..=4), (732_540, 0x5d3c84deb6752215));
    }

    #[test]
    fn every_short_string_of_five_bytes_decodes_as_on_the_jdk() {
        // `U.java digest5`, the same on the three JDKs; `String::from_utf8_lossy` gives
        // ecd87d063bee1725
        assert_eq!(digest(&SHORT_BYTES, 5..=5), (248_832, 0x7e8e4a928c63e5e5));
    }

    #[test]
    fn only_an_encoded_surrogate_is_not_the_maximal_subpart_rule() {
        // `String::from_utf8_lossy` replaces by Unicode's maximal subparts. The JDK differs
        // exactly where `ED` is followed by `A0..=BF`
        for len in 1..=4 {
            each_string(&CLASS_BYTES, len, &mut |bytes| {
                let surrogate = bytes
                    .windows(2)
                    .any(|pair| pair[0] == 0xED && (0xA0..=0xBF).contains(&pair[1]));
                let same = jvm_string(bytes) == String::from_utf8_lossy(bytes);
                assert_eq!(same, !surrogate, "{bytes:02x?}");
            });
        }
    }

    #[cfg(feature = "arbitrary")]
    proptest::proptest! {
        #[test]
        fn well_formed_text_is_its_own(text in proptest::prelude::any::<String>()) {
            proptest::prop_assert_eq!(jvm_string(text.as_bytes()), text);
        }
    }
}
