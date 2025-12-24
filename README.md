# Agent Evaluator Example 🧪

**Open Source** pattern for building and evaluating AI agents with automated testing.

> Uses a Market Research Agent as the example implementation.

## Architecture

```
agent_core.py  →  Your agent logic (framework-agnostic)
evaluator.py   →  Automated test suite + LLM judge
```

## The Pattern

### 1. `agent_core.py` - Decoupled Agent Logic
Keep your core logic **framework-free**:
- Async generator yielding typed chunks (`text`, `tool_call`, `error`, etc.)
- Pure Python - testable without deployment infrastructure
- Easy to swap LLM providers or tools

### 2. `evaluator.py` - Automated Evaluation
Test suite with:
- **Heuristic checks** - deterministic validations (tool called? has table? has sources?)
- **LLM judge** - blind scoring on relevance, quality, safety, format
- **Test categories** - easy, medium, adversarial (prompt injection, gibberish input)
- **Scoring** - 40% heuristics + 60% LLM judge → pass/fail threshold

## Setup

```bash
pip install cycls requests openai python-dotenv
```

```env
OPENAI_API_KEY=sk-...
EXA_API_KEY=...
```

## Run

```bash
python evaluator.py
# Enter test subject (e.g. "Spotify") → runs 10 tests with full report
```

## Adapt for Your Agent

1. Replace `agent_core.py` with your agent logic
2. Update test cases in `get_test_cases()` 
3. Adjust `heuristic_checks()` for your domain
4. Modify LLM judge rubric in `llm_judge_blind()`

## License

Open source - use freely.
