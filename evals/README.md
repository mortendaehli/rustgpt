Checked-in prompt-eval suites for `cargo run -- eval`.

Use the suite that matches the checkpoint chat template:

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
