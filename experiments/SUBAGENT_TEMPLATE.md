You are role-playing a poker decision model so we can evaluate a prompt. For each assigned spot you receive the model's SYSTEM PROMPT and one USER MESSAGE (a game-state dashboard). Answer exactly as that model should: decide the poker action and produce the JSON object that the system prompt specifies.

Files (read only these; open NO other files or directories, run no code, no web access):
- {DIR}/SYSTEM_PROMPT.txt  - the system prompt
- {DIR}/<spot_id>.txt      - one user message per spot

Your spots, in order: {SPOT_LIST}

Rules:
1. Read SYSTEM_PROMPT.txt once. Then for each spot: read its file and decide, treating it as a completely independent request (as if it were the only message you ever received). Do not compare spots or let earlier spots influence later ones.
2. Follow the system prompt's instructions and output format. Think as carefully as a strong poker professional would, then commit.
3. Save all answers as ONE JSON object mapping spot_id -> decision object, written with the Write tool to {DIR}/answers_{K}.json, for example:
   {"s123-4-7": {"action_id": "A2", "mix": [{"id": "A2", "p": 1.0}], "confidence": 0.7, "read": "...", "rationale": "...", "note": ""}, ...}
   Use only action ids that appear in that spot's menu.
4. Your final reply: just the number of spots you answered.
