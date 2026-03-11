use crate::core::config::ChatTemplateKind;
use crate::core::error::Result;
use crate::data::schema::{Message, MessageRole, render_messages};
use crate::data::tokenizer::Tokenizer;
use crate::engine::generate::StopCondition;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatDirective {
    Exit,
    Reset,
    History,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChatTemplate {
    kind: ChatTemplateKind,
    stop_texts: &'static [&'static str],
}

impl ChatTemplate {
    pub fn from_kind(kind: ChatTemplateKind) -> Self {
        match kind {
            ChatTemplateKind::SimpleTranscript => Self {
                kind,
                stop_texts: &["\nUser: ", "\nSystem: ", "\nTool: "],
            },
            ChatTemplateKind::ChatMl => Self {
                kind,
                stop_texts: &["\n<|user|>\n", "\n<|system|>\n", "\n<|tool|>\n"],
            },
        }
    }

    pub fn kind(self) -> ChatTemplateKind {
        self.kind
    }

    pub fn stop_texts(self) -> &'static [&'static str] {
        self.stop_texts
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedChatPrompt {
    pub prompt: String,
    pub prompt_tokens: Vec<usize>,
    pub stop_condition: StopCondition,
    pub dropped_turns: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatSession {
    template: ChatTemplate,
    system_prompt: String,
    turns: Vec<Message>,
}

impl ChatSession {
    pub fn new(template_kind: ChatTemplateKind, system_prompt: String) -> Self {
        Self {
            template: ChatTemplate::from_kind(template_kind),
            system_prompt,
            turns: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.turns.clear();
    }

    pub fn push_user_turn(&mut self, user_text: &str) {
        self.turns.push(Message {
            role: MessageRole::User,
            content: user_text.to_string(),
        });
    }

    pub fn push_assistant_turn(&mut self, assistant_text: &str) {
        self.turns.push(Message {
            role: MessageRole::Assistant,
            content: assistant_text.to_string(),
        });
    }

    pub fn prompt(&self) -> String {
        let mut rendered = self.history();
        if matches!(
            self.turns.last().map(|turn| turn.role),
            Some(MessageRole::User)
        ) {
            rendered.push_str(MessageRole::Assistant.prefix(self.template.kind()));
        }
        rendered
    }

    pub fn prepare_prompt(
        &self,
        tokenizer: &Tokenizer,
        block_size: usize,
        max_new_tokens: usize,
    ) -> Result<PreparedChatPrompt> {
        let reserve = max_new_tokens.min(block_size.saturating_sub(1));
        let max_prompt_tokens = usize::max(1, block_size.saturating_sub(reserve));
        let allowed_prompt_tokens = max_prompt_tokens.saturating_sub(1);
        let mut dropped_turns = 0;
        let system_message = (!self.system_prompt.is_empty()).then(|| Message {
            role: MessageRole::System,
            content: self.system_prompt.clone(),
        });

        loop {
            let mut messages = Vec::with_capacity(
                self.turns.len().saturating_sub(dropped_turns)
                    + usize::from(system_message.is_some()),
            );
            if let Some(system_message) = &system_message {
                messages.push(system_message.clone());
            }
            messages.extend(self.turns[dropped_turns..].iter().cloned());

            let mut prompt = render_messages(&messages, self.template.kind());
            if matches!(
                self.turns.last().map(|turn| turn.role),
                Some(MessageRole::User)
            ) {
                prompt.push_str(MessageRole::Assistant.prefix(self.template.kind()));
            }

            let prompt_tokens = tokenizer.encode_text(&prompt);
            if prompt_tokens.len() + 1 <= max_prompt_tokens
                || dropped_turns >= self.turns.len().saturating_sub(1)
            {
                let prompt_tokens = if prompt_tokens.len() <= allowed_prompt_tokens {
                    prompt_tokens
                } else {
                    prompt_tokens[prompt_tokens.len().saturating_sub(allowed_prompt_tokens)..]
                        .to_vec()
                };
                let prompt = tokenizer.decode(&prompt_tokens, false)?;
                return Ok(PreparedChatPrompt {
                    prompt,
                    prompt_tokens,
                    stop_condition: StopCondition::from_text_sequences(
                        tokenizer,
                        self.template.stop_texts(),
                    ),
                    dropped_turns,
                });
            }

            dropped_turns += 1;
        }
    }

    pub fn history(&self) -> String {
        let mut messages =
            Vec::with_capacity(self.turns.len() + usize::from(!self.system_prompt.is_empty()));
        if !self.system_prompt.is_empty() {
            messages.push(Message {
                role: MessageRole::System,
                content: self.system_prompt.clone(),
            });
        }
        messages.extend(self.turns.iter().cloned());
        render_messages(&messages, self.template.kind())
    }
}

pub fn parse_chat_directive(input: &str) -> Option<ChatDirective> {
    match input.trim() {
        "/exit" | "/quit" => Some(ChatDirective::Exit),
        "/reset" => Some(ChatDirective::Reset),
        "/history" => Some(ChatDirective::History),
        _ => None,
    }
}
