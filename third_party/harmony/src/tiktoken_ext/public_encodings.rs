use std::{collections::HashMap, io::Cursor, path::PathBuf};

use base64::{prelude::BASE64_STANDARD, Engine as _};

use crate::tiktoken::{CoreBPE, Rank};
use sha2::{Digest as _, Sha256};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("the env var TIKTOKEN_ENCODINGS_BASE is not set, or invalid")]
    InvalidEncodingBaseDirEnvVar,

    #[error("unknown encoding name: {0}")]
    UnknownEncodingName(String),

    #[error("invalid tiktoken vocab file: {0}")]
    InvalidTiktokenVocabFile(#[source] std::io::Error),

    #[error("failed to create CoreBPE: {0}")]
    CoreBPECreationFailed(#[source] Box<dyn std::error::Error + Send + Sync>),
}

const TIKTOKEN_ENCODINGS_BASE_VAR: &str = "TIKTOKEN_ENCODINGS_BASE";

const O200K_BASE_DATA: &[u8] = include_bytes!("data/o200k_base.tiktoken.zst");
const CL100K_BASE_DATA: &[u8] = include_bytes!("data/cl100k_base.tiktoken.zst");

fn decompress_vocab(compressed: &[u8]) -> Vec<u8> {
    zstd::decode_all(Cursor::new(compressed)).expect("embedded vocab data is corrupt")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    O200kBase,
    O200kHarmony,
    Cl100kBase,
}

impl Encoding {
    pub fn all() -> &'static [Self] {
        &[Self::O200kBase, Self::O200kHarmony, Self::Cl100kBase]
    }

    pub fn from_name(name: impl AsRef<str>) -> Option<Self> {
        let name_str = name.as_ref();
        for encoding in Self::all() {
            if encoding.name() == name_str {
                return Some(*encoding);
            }
        }
        None
    }

    pub fn load_from_name(name: impl AsRef<str>) -> Result<CoreBPE, LoadError> {
        let name = name.as_ref();
        Self::from_name(name)
            .ok_or_else(|| LoadError::UnknownEncodingName(name.to_string()))?
            .load()
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::O200kBase => "o200k_base",
            Self::O200kHarmony => "o200k_harmony",
            Self::Cl100kBase => "cl100k_base",
        }
    }

    pub fn load(&self) -> Result<CoreBPE, LoadError> {
        let encoder = if let Ok(base_dir) = std::env::var(TIKTOKEN_ENCODINGS_BASE_VAR) {
            let path = PathBuf::from(base_dir).join(self.vocab_file_name());
            load_tiktoken_vocab_file(&path, Some(self.expected_hash()))
                .map_err(LoadError::InvalidTiktokenVocabFile)?
        } else {
            let raw = decompress_vocab(self.embedded_data());
            load_tiktoken_vocab_bytes(&raw, None).map_err(LoadError::InvalidTiktokenVocabFile)?
        };

        let specials = self.build_special_tokens();

        CoreBPE::new(encoder, specials, &self.pattern()).map_err(LoadError::CoreBPECreationFailed)
    }

    fn vocab_file_name(&self) -> &'static str {
        match self {
            Self::O200kBase | Self::O200kHarmony => "o200k_base.tiktoken",
            Self::Cl100kBase => "cl100k_base.tiktoken",
        }
    }

    fn embedded_data(&self) -> &'static [u8] {
        match self {
            Self::O200kBase | Self::O200kHarmony => O200K_BASE_DATA,
            Self::Cl100kBase => CL100K_BASE_DATA,
        }
    }

    fn expected_hash(&self) -> &'static str {
        match self {
            Self::O200kBase | Self::O200kHarmony => {
                "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
            }
            Self::Cl100kBase => "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        }
    }

    fn build_special_tokens(&self) -> Vec<(String, Rank)> {
        let base: Vec<(String, Rank)> = self
            .special_tokens()
            .iter()
            .map(|(s, r)| ((*s).to_string(), *r))
            .collect();

        match self {
            Self::O200kHarmony => {
                let mut specials = base;
                specials.extend((200014..=201088).map(|id| (format!("<|reserved_{id}|>"), id)));
                specials
            }
            Self::O200kBase => {
                let mut specials = base;
                specials.extend((199998..=201088).map(|id| (format!("<|reserved_{id}|>"), id)));
                specials
            }
            Self::Cl100kBase => base,
        }
    }

    fn special_tokens(&self) -> &'static [(&'static str, Rank)] {
        match self {
            Self::O200kBase => &[],
            Self::O200kHarmony => &[
                ("<|startoftext|>", 199998),
                ("<|endoftext|>", 199999),
                ("<|reserved_200000|>", 200000),
                ("<|reserved_200001|>", 200001),
                ("<|return|>", 200002),
                ("<|constrain|>", 200003),
                ("<|reserved_200004|>", 200004),
                ("<|channel|>", 200005),
                ("<|start|>", 200006),
                ("<|end|>", 200007),
                ("<|message|>", 200008),
                ("<|reserved_200009|>", 200009),
                ("<|reserved_200010|>", 200010),
                ("<|reserved_200011|>", 200011),
                ("<|call|>", 200012),
                ("<|reserved_200013|>", 200013),
            ],
            Self::Cl100kBase => &[
                ("<|endoftext|>", 100257),
                ("<|fim_prefix|>", 100258),
                ("<|fim_middle|>", 100259),
                ("<|fim_suffix|>", 100260),
                ("<|endofprompt|>", 100276),
            ],
        }
    }

    fn pattern(&self) -> String {
        match self {
            Self::O200kBase | Self::O200kHarmony => {
                [
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "\\p{N}{1,3}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
                    "\\s*[\\r\\n]+",
                    "\\s+(?!\\S)",
                    "\\s+",
                ].join("|")
            }
            Self::Cl100kBase => {
                "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".to_string()
            }
        }
    }
}

fn load_tiktoken_vocab<R>(
    mut reader: R,
    expected_hash: Option<&str>,
) -> std::result::Result<HashMap<Vec<u8>, Rank>, std::io::Error>
where
    R: std::io::BufRead,
{
    let mut hasher = expected_hash.map(|_| Sha256::new());
    let mut bpe_ranks = HashMap::new();
    let mut lin_no = 0;
    let mut line_buffer = String::new();
    while reader.read_line(&mut line_buffer)? > 0 {
        lin_no += 1;
        if let Some(hasher) = hasher.as_mut() {
            hasher.update(line_buffer.as_bytes());
        }
        let line = line_buffer.trim_end();
        let (token, rank) = line.split_once(' ').ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected token and rank, could not split on ' ' at line {lin_no}"),
            )
        })?;
        let bytes = BASE64_STANDARD.decode(token).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to decode base64 token at line {lin_no}: {e}",),
            )
        })?;
        let rank = rank.parse().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to parse rank at line {lin_no}: {e}"),
            )
        })?;
        bpe_ranks.insert(bytes, rank);
        line_buffer.clear();
    }
    if let Some(hasher) = hasher {
        let expected_hash = expected_hash.unwrap();
        let computed_hash = format!("{:x}", hasher.finalize());
        if computed_hash != expected_hash {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("hash mismatch: computed={computed_hash}, expected={expected_hash}"),
            ));
        }
    }
    Ok(bpe_ranks)
}

pub fn load_tiktoken_vocab_file<P>(
    path: P,
    expected_hash: Option<&str>,
) -> std::result::Result<HashMap<Vec<u8>, Rank>, std::io::Error>
where
    P: AsRef<std::path::Path>,
{
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    load_tiktoken_vocab(reader, expected_hash)
}

fn load_tiktoken_vocab_bytes(
    data: &[u8],
    expected_hash: Option<&str>,
) -> std::result::Result<HashMap<Vec<u8>, Rank>, std::io::Error> {
    let reader = std::io::BufReader::new(Cursor::new(data));
    load_tiktoken_vocab(reader, expected_hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_encodings() {
        for encoding in Encoding::all() {
            let _ = encoding.load().unwrap();
        }
    }

    #[test]
    fn test_embedded_data_integrity() {
        for encoding in &[Encoding::O200kBase, Encoding::Cl100kBase] {
            let raw = decompress_vocab(encoding.embedded_data());
            let hash = format!("{:x}", Sha256::digest(&raw));
            assert_eq!(hash, encoding.expected_hash());
        }
    }
}
