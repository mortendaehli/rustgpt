//! Normalized data schemas used across all loaders.
//! Raw text, JSONL, and Parquet inputs are all converted into these records before
//! tokenization and training-example construction.

use std::borrow::Cow;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::config::ChatTemplateKind;
use crate::core::error::{Result, RustGptError};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MessageRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }

    pub fn prefix(self, template: ChatTemplateKind) -> &'static str {
        match template {
            ChatTemplateKind::SimpleTranscript => match self {
                Self::System => "System: ",
                Self::User => "User: ",
                Self::Assistant => "Assistant: ",
                Self::Tool => "Tool: ",
            },
            ChatTemplateKind::ChatMl => match self {
                Self::System => "<|system|>\n",
                Self::User => "<|user|>\n",
                Self::Assistant => "<|assistant|>\n",
                Self::Tool => "<|tool|>\n",
            },
        }
    }
}

impl FromStr for MessageRole {
    type Err = ();

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            "tool" => Ok(Self::Tool),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextRecord {
    pub text: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub meta: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatRecord {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub meta: Option<Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DatasetRecord {
    Text(TextRecord),
    Chat(ChatRecord),
}

impl DatasetRecord {
    pub fn rendered_text(&self) -> String {
        self.rendered_text_with_template(ChatTemplateKind::SimpleTranscript)
    }

    pub fn rendered_text_with_template(&self, template: ChatTemplateKind) -> String {
        match self {
            Self::Text(record) => record.text.clone(),
            Self::Chat(record) => render_messages(&record.messages, template),
        }
    }

    pub fn lowercase(&self) -> Self {
        match self {
            Self::Text(record) => Self::Text(TextRecord {
                text: record.text.to_lowercase(),
                source: record.source.clone(),
                meta: record.meta.clone(),
            }),
            Self::Chat(record) => Self::Chat(ChatRecord {
                messages: record
                    .messages
                    .iter()
                    .map(|message| Message {
                        role: message.role,
                        content: message.content.to_lowercase(),
                    })
                    .collect(),
                source: record.source.clone(),
                meta: record.meta.clone(),
            }),
        }
    }
}

pub fn render_messages(messages: &[Message], template: ChatTemplateKind) -> String {
    let mut rendered = String::new();
    for message in messages {
        rendered.push_str(message.role.prefix(template));
        rendered.push_str(&message.content);
        rendered.push('\n');
    }
    rendered
}

pub fn parse_text_value(value: Value) -> Result<TextRecord> {
    match value {
        Value::String(text) => Ok(TextRecord {
            text,
            source: None,
            meta: None,
        }),
        Value::Object(mut map) => {
            let text = map.remove("text").and_then(json_to_string).ok_or_else(|| {
                RustGptError::Data("text record is missing string field `text`".to_string())
            })?;
            let source = map.remove("source").and_then(json_to_string);
            let meta = if map.is_empty() {
                None
            } else {
                Some(Value::Object(map))
            };
            Ok(TextRecord { text, source, meta })
        }
        other => Err(RustGptError::Data(format!(
            "expected a text record object or string, got {}",
            json_type_name(&other)
        ))),
    }
}

pub fn parse_chat_value(value: Value) -> Result<ChatRecord> {
    let Value::Object(mut map) = value else {
        return Err(RustGptError::Data(
            "chat record must be a JSON object with a `messages` field".to_string(),
        ));
    };

    let messages_value = map
        .remove("messages")
        .ok_or_else(|| RustGptError::Data("chat record is missing `messages`".to_string()))?;
    let Value::Array(raw_messages) = messages_value else {
        return Err(RustGptError::Data(
            "chat record `messages` must be an array".to_string(),
        ));
    };

    let mut messages = Vec::with_capacity(raw_messages.len());
    for raw_message in raw_messages {
        messages.push(parse_message_value(raw_message)?);
    }
    if messages.is_empty() {
        return Err(RustGptError::Data(
            "chat record `messages` must not be empty".to_string(),
        ));
    }

    let source = map.remove("source").and_then(json_to_string);
    let meta = if map.is_empty() {
        None
    } else {
        Some(Value::Object(map))
    };
    Ok(ChatRecord {
        messages,
        source,
        meta,
    })
}

fn parse_message_value(value: Value) -> Result<Message> {
    let Value::Object(mut map) = value else {
        return Err(RustGptError::Data(
            "chat message must be a JSON object".to_string(),
        ));
    };

    let role_raw = map.remove("role").and_then(json_to_string).ok_or_else(|| {
        RustGptError::Data("chat message is missing string field `role`".to_string())
    })?;
    let role = MessageRole::from_str(&role_raw)
        .map_err(|_| RustGptError::Data(format!("unsupported chat role {role_raw:?}")))?;

    let content = map
        .remove("content")
        .and_then(json_to_string)
        .ok_or_else(|| {
            RustGptError::Data("chat message is missing string field `content`".to_string())
        })?;

    Ok(Message { role, content })
}

fn json_to_string(value: Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        Value::Null => None,
        Value::Array(_) | Value::Object(_) => None,
    }
}

fn json_type_name(value: &Value) -> Cow<'static, str> {
    match value {
        Value::Null => Cow::Borrowed("null"),
        Value::Bool(_) => Cow::Borrowed("bool"),
        Value::Number(_) => Cow::Borrowed("number"),
        Value::String(_) => Cow::Borrowed("string"),
        Value::Array(_) => Cow::Borrowed("array"),
        Value::Object(_) => Cow::Borrowed("object"),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::core::config::ChatTemplateKind;

    use super::{MessageRole, parse_chat_value, parse_text_value, render_messages};

    #[test]
    fn text_value_accepts_string_or_object() {
        assert_eq!(parse_text_value(json!("hello")).unwrap().text, "hello");
        assert_eq!(
            parse_text_value(json!({"text":"world","source":"demo"}))
                .unwrap()
                .source
                .as_deref(),
            Some("demo")
        );
    }

    #[test]
    fn chat_value_parses_messages() {
        let chat = parse_chat_value(json!({
            "messages": [
                {"role":"system","content":"brief"},
                {"role":"user","content":"hello"},
                {"role":"assistant","content":"hi"}
            ]
        }))
        .unwrap();
        assert_eq!(chat.messages[2].role, MessageRole::Assistant);
        assert_eq!(
            render_messages(&chat.messages, ChatTemplateKind::SimpleTranscript),
            "System: brief\nUser: hello\nAssistant: hi\n"
        );
    }

    #[test]
    fn chatml_renders_role_markers() {
        let rendered = render_messages(
            &[super::Message {
                role: MessageRole::Assistant,
                content: "hi".to_string(),
            }],
            ChatTemplateKind::ChatMl,
        );
        assert_eq!(rendered, "<|assistant|>\nhi\n");
    }
}
