//! Checked-in prompt-eval fixtures used by the `eval` command.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::Deserialize;

use crate::core::error::{Result, RustGptError};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct PromptEvalCase {
    pub name: String,
    pub prompt: String,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub must_contain: Vec<String>,
    #[serde(default)]
    pub must_not_contain: Vec<String>,
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
}

impl PromptEvalCase {
    fn validate(self, path: &Path, line_idx: usize) -> Result<Self> {
        if self.name.trim().is_empty() {
            return Err(RustGptError::Data(format!(
                "prompt eval case on line {} in {} is missing a non-empty `name`",
                line_idx + 1,
                path.display()
            )));
        }
        if self.prompt.trim().is_empty() {
            return Err(RustGptError::Data(format!(
                "prompt eval case {:?} on line {} in {} is missing a non-empty `prompt`",
                self.name,
                line_idx + 1,
                path.display()
            )));
        }
        Ok(self)
    }
}

pub fn load_prompt_eval_suite(path: &Path) -> Result<Vec<PromptEvalCase>> {
    let file = File::open(path).map_err(|source| RustGptError::io_with_path(path, source))?;
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| RustGptError::io_with_path(path, source))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let case = serde_json::from_str::<PromptEvalCase>(trimmed).map_err(|source| {
            RustGptError::Data(format!(
                "failed to parse prompt eval case on line {} in {}: {source}",
                line_idx + 1,
                path.display()
            ))
        })?;
        cases.push(case.validate(path, line_idx)?);
    }

    if cases.is_empty() {
        return Err(RustGptError::Data(format!(
            "prompt eval suite {} does not contain any cases",
            path.display()
        )));
    }

    Ok(cases)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::load_prompt_eval_suite;

    fn temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rustgpt-{unique}-{name}"))
    }

    #[test]
    fn prompt_eval_suite_loads_jsonl_cases() {
        let path = temp_path("prompt-eval.jsonl");
        fs::write(
            &path,
            concat!(
                "# comment\n",
                "{\"name\":\"stop-after-answer\",\"prompt\":\"User: hi\\nAssistant: \",\"must_not_contain\":[\"\\nUser:\"],\"max_new_tokens\":8}\n"
            ),
        )
        .unwrap();

        let suite = load_prompt_eval_suite(&path).unwrap();
        assert_eq!(suite.len(), 1);
        assert_eq!(suite[0].name, "stop-after-answer");
        assert_eq!(suite[0].must_not_contain, vec!["\nUser:".to_string()]);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn prompt_eval_suite_rejects_empty_prompt() {
        let path = temp_path("prompt-eval-invalid.jsonl");
        fs::write(&path, "{\"name\":\"bad\",\"prompt\":\"   \"}\n").unwrap();

        let error = load_prompt_eval_suite(&path).unwrap_err().to_string();
        assert!(error.contains("non-empty `prompt`"));

        fs::remove_file(path).unwrap();
    }
}
