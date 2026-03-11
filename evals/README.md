Checked-in prompt-eval suites for `cargo run -- eval`.

Use the suite that matches the checkpoint chat template:

- [instruction_following_simple.jsonl](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/evals/instruction_following_simple.jsonl)
- [instruction_following_chatml.jsonl](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/evals/instruction_following_chatml.jsonl)
- [retention_simple.jsonl](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/evals/retention_simple.jsonl)
- [retention_chatml.jsonl](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/evals/retention_chatml.jsonl)
- [repetition_simple.jsonl](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/evals/repetition_simple.jsonl)
- [repetition_chatml.jsonl](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/evals/repetition_chatml.jsonl)
- [assistant_termination_simple.jsonl](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/evals/assistant_termination_simple.jsonl)
- [assistant_termination_chatml.jsonl](/Users/mortendaehliaslesen/RustroverProjects/rustgpt/evals/assistant_termination_chatml.jsonl)

Example:

```bash
cargo run -- eval \
  --checkpoint /tmp/rustgpt-multiturn.ckpt \
  --prompt-file evals/assistant_termination_chatml.jsonl \
  --temperature 0.0 \
  --top-k 1 \
  --max-new-tokens 24
```

Each line is a JSON object with:

- `name`
- `prompt`
- optional `notes`
- optional `must_contain`
- optional `must_not_contain`
- optional `max_new_tokens`
