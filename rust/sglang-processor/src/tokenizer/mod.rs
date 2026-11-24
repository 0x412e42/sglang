//! Tokenizer loading and encoding over Dynamo tokenizers.

use crate::ProcessorError as Error;

mod sglang;

pub use sglang::{PromptAffixes, SglangTokenizer};

/// The `i32` ids `sglang-server` sends the scheduler, over [`SglangTokenizer`]
/// or a test double. `Send + Sync` so one instance is shared across pinned workers.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>, Error>;

    /// Python's `tokenizer.bos_token_id`.
    fn bos_token_id(&self) -> Option<i32> {
        None
    }

    /// Python's chat encode: the template without specials, then the continuation
    /// minus one leading BOS (`_append_assistant_prefix_to_prompt_ids`).
    fn encode_chat(
        &self,
        prompt: &[dynamo_tokenizers::EncodeSegment<'_>],
        continuation: &str,
    ) -> Result<Vec<i32>, Error> {
        let mut ids = self.encode_segments(prompt, false)?;
        if !continuation.is_empty() {
            let suffix = self.encode(continuation, true)?;
            let bos = self.bos_token_id();
            ids.extend_from_slice(
                bos.and_then(|bos| suffix.strip_prefix(&[bos]))
                    .unwrap_or(&suffix),
            );
        }
        Ok(ids)
    }

    fn encode_segments(
        &self,
        segments: &[dynamo_tokenizers::EncodeSegment<'_>],
        add_special_tokens: bool,
    ) -> Result<Vec<i32>, Error> {
        let text = segments
            .iter()
            .map(|segment| segment.text)
            .collect::<String>();
        self.encode(&text, add_special_tokens)
    }
}

#[cfg(test)]
mod tests {
    use dynamo_tokenizers::EncodeSegment;

    use super::*;

    /// With a configured, auto-added `<s>`, the template encodes without it and the
    /// continuation loses one leading BOS, as on Python.
    #[test]
    fn chat_encode_drops_one_configured_bos() {
        let dir = std::env::temp_dir().join(format!("processor-bos-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tokenizer = serde_json::json!({
            "added_tokens": [{"id": 0, "content": "<s>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true}],
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
                "special_tokens": {"<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]}}},
            "model": {"type": "WordLevel", "unk_token": "[UNK]",
                "vocab": {"<s>": 0, "[UNK]": 1, "Hi": 2, "It": 3}},
        });
        let file = dir.join("tokenizer.json");
        std::fs::write(&file, tokenizer.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), r#"{"bos_token": "<s>"}"#).unwrap();
        let tokenizer = SglangTokenizer::load(file.to_str().unwrap(), None).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(TextTokenizer::bos_token_id(&tokenizer), Some(0));

        let prompt = [EncodeSegment::new("Hi", true)];
        let encode_chat =
            |continuation| TextTokenizer::encode_chat(&tokenizer, &prompt, continuation);
        assert_eq!(encode_chat("It").unwrap(), vec![2, 3]);
        // Only one BOS is dropped, as Python's: the auto-added one here.
        assert_eq!(encode_chat("<s><s>It").unwrap(), vec![2, 0, 0, 3]);
    }
}
