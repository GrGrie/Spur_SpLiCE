# AGENTS.md — Working with Astra

Source: [OpenAI Astra prompting guidance](https://developers.openai.com/api/docs/guides/latest-model), reviewed 2026-09-06. Condensed and adapted to this repository.

- Infer intent and scope from context. Complete authorized work; make reasonable assumptions for routine gaps.
- Treat action requests as instructions to act. For analysis-only or planning requests, deliver the artifact before stopping; do not implement unrequested changes.
- Ask focused questions when missing information materially affects the result. Continue independent work meanwhile.
- Prepare reviewable work before requesting necessary approval. Do not repeat authorization requests already resolved.
- Check relevant instruction files for ambiguity. Explicit user instructions override skill guidance, subject to higher-priority rules.
- When a skill blocks progress, link its file, quote the rule, and explain its application. Separate requirements from interpretation.
- Lead with the result. Use concise, plain paragraphs; lists and tables should clarify steps or comparisons. Avoid jargon and stock phrases.
- Delegation policy: use subagents only when explicitly requested by the user.
- Match verification to the change. Broaden or repeat checks only for changes, failures, or unresolved concerns.
- Incorporate corrections and answer side questions while preserving the larger task.
