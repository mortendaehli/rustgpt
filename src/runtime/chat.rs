//! Reusable chat-session logic.
//! The terminal command owns stdin/stdout, while this module owns transcript rendering,
//! chat directives, and the prompt template used for generation.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatDirective {
    Exit,
    Reset,
    History,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRole {
    User,
    Assistant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChatTemplate {
    system_prefix: &'static str,
    user_prefix: &'static str,
    assistant_prefix: &'static str,
    stop_sequences: &'static [&'static str],
}

impl ChatTemplate {
    pub fn simple_transcript() -> Self {
        Self {
            system_prefix: "System: ",
            user_prefix: "User: ",
            assistant_prefix: "Assistant: ",
            stop_sequences: &["\nUser: ", "\nSystem: "],
        }
    }

    pub fn stop_sequences(self) -> &'static [&'static str] {
        self.stop_sequences
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ChatTurn {
    role: ChatRole,
    content: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatSession {
    template: ChatTemplate,
    system_prompt: String,
    turns: Vec<ChatTurn>,
}

impl ChatSession {
    pub fn new(system_prompt: String) -> Self {
        Self {
            template: ChatTemplate::simple_transcript(),
            system_prompt,
            turns: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.turns.clear();
    }

    pub fn push_user_turn(&mut self, user_text: &str) {
        self.turns.push(ChatTurn {
            role: ChatRole::User,
            content: user_text.to_string(),
        });
    }

    pub fn push_assistant_turn(&mut self, assistant_text: &str) {
        self.turns.push(ChatTurn {
            role: ChatRole::Assistant,
            content: assistant_text.to_string(),
        });
    }

    pub fn stop_sequences(&self) -> &'static [&'static str] {
        self.template.stop_sequences()
    }

    pub fn prompt(&self) -> String {
        let mut rendered = self.history();
        if matches!(
            self.turns.last().map(|turn| turn.role),
            Some(ChatRole::User)
        ) {
            rendered.push_str(self.template.assistant_prefix);
        }
        rendered
    }

    pub fn history(&self) -> String {
        let mut rendered = String::new();
        if !self.system_prompt.is_empty() {
            rendered.push_str(self.template.system_prefix);
            rendered.push_str(&self.system_prompt);
            rendered.push('\n');
        }
        for turn in &self.turns {
            match turn.role {
                ChatRole::User => rendered.push_str(self.template.user_prefix),
                ChatRole::Assistant => rendered.push_str(self.template.assistant_prefix),
            }
            rendered.push_str(&turn.content);
            rendered.push('\n');
        }
        rendered
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

#[cfg(test)]
mod tests {
    use super::{ChatDirective, ChatSession, parse_chat_directive};

    #[test]
    fn session_prompt_appends_assistant_prefix_after_user_turn() {
        let mut session = ChatSession::new("be terse".to_string());
        session.push_user_turn("hello");
        assert_eq!(
            session.prompt(),
            "System: be terse\nUser: hello\nAssistant: "
        );
    }

    #[test]
    fn session_history_contains_completed_turns_only() {
        let mut session = ChatSession::new(String::new());
        session.push_user_turn("hello");
        session.push_assistant_turn("world");
        assert_eq!(session.history(), "User: hello\nAssistant: world\n");
    }

    #[test]
    fn directives_are_parsed_from_trimmed_input() {
        assert_eq!(
            parse_chat_directive(" /history "),
            Some(ChatDirective::History)
        );
        assert_eq!(parse_chat_directive("hello"), None);
    }
}
