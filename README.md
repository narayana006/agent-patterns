# agent-patterns

Six common AI agent patterns as small, runnable Python files. Each is under 200 lines, uses only the standard library, and runs offline against a scripted stand-in model so you can read the trace before wiring in a real one.

**Status: first working version.** All six planned patterns run today. They are teaching examples to read and copy, not a framework or an installable library.

| Pattern | File | What it shows |
| --- | --- | --- |
| Router | [`patterns/router.py`](patterns/router.py) | One classification call, then dispatch to a specialist handler, with a fallback for bad labels |
| Planner-executor | [`patterns/planner_executor.py`](patterns/planner_executor.py) | The model writes a whole plan as JSON; plain code validates and runs it |
| Evaluator-optimizer | [`patterns/evaluator_optimizer.py`](patterns/evaluator_optimizer.py) | Draft, critique, revise until it passes or a round limit is hit; rule checks before model checks |
| Human-in-the-loop | [`patterns/human_in_the_loop.py`](patterns/human_in_the_loop.py) | Risky tool calls wait for a person; a denial goes back to the model as an observation |
| Memory | [`patterns/memory.py`](patterns/memory.py) | A bounded window plus running summary for the conversation, and a fact store that outlives it |
| Tool retry | [`patterns/tool_retry.py`](patterns/tool_retry.py) | Backoff for transient failures; permanent errors returned to the model so it can correct itself |

## Run one

```bash
git clone https://github.com/narayana006/agent-patterns
cd agent-patterns
python patterns/evaluator_optimizer.py
```

```text
[draft 1 ] We are currently experiencing an incident impacting the authentication subsystem whereby a subset of requests to the token issuance endpoint are returning elevated error rates, and our engineers are actively investigating the root cause of the degradation.
[evaluate] REVISE: Too long: 37 words, the limit is 30. Does not mention 'sign'.
[draft 2 ] Some sign-ins are failing because of a fault in our auth token cache. A fix is being rolled out.
[evaluate] REVISE: 'auth token cache' is jargon. Say what customers see, and when it will be fixed.
[draft 3 ] Some customers cannot sign in right now. We have found the cause and expect a fix within 30 minutes.
[evaluate] PASS

result: approved after 3 rounds
model calls: 3 drafts, 2 critiques (round 1 failed on rules alone)
```

Python 3.10 or newer. There is nothing to install and no API key is needed.

## More traces

`python patterns/planner_executor.py`

```text
[goal    ] Budget a 3-night trip: return train, 3 hostel nights, one museum pass.
[plan    ] 6 steps: price_of -> multiply -> price_of -> multiply -> price_of -> add
[step 1  ] price_of(item='train ticket') -> 42.0
[step 2  ] multiply(a=42.0, b=2) -> 84.0
[step 3  ] price_of(item='hostel night') -> 31.5
[step 4  ] multiply(a=31.5, b=3) -> 94.5
[step 5  ] price_of(item='museum pass') -> 18.0
[step 6  ] add(values=[84.0, 94.5, 18.0]) -> 196.5
[answer  ] The trip costs 196.50 in total: 84.00 for trains, 94.50 for the hostel and 18.00 for the museum pass.

model calls: 2 (one to plan, one to answer; the six steps ran as plain code)
```

`python patterns/human_in_the_loop.py`

```text
[task    ] Customer says the lamp in order A-1042 arrived broken. Make it right.
[propose ] look_up_order({'order_id': 'A-1042'})
[observe ] order A-1042: 1 x desk lamp, 39.00, delivered damaged
[propose ] issue_refund({'order_id': 'A-1042', 'amount': 390.0})  <- needs approval
[human   ] denied: amount is 10x the order value; the order total is 39.00
[observe ] issue_refund was denied by a human reviewer: amount is 10x the order value; the order total is 39.00
[propose ] issue_refund({'order_id': 'A-1042', 'amount': 39.0})  <- needs approval
[human   ] approved
[observe ] refunded 39.00 on order A-1042
[final   ] Sorry about the damaged lamp. I have refunded 39.00 to your original payment method.

side effects: ['refund 39.00 on A-1042']
```

Add `--interactive` to that one to answer the approval prompts yourself.

`python patterns/tool_retry.py`

```text
[question] What is the weather in Lisbon and Oslo?
[call    ] live_weather({'city': 'Lisbon'})
[retry   ] attempt 1 failed (upstream timeout); waiting 0.5s
[retry   ] attempt 2 failed (upstream timeout); waiting 1.0s
[observe ] ok: 18C, clear
[call    ] live_weather({'city': 'Olso'})
[observe ] error: unknown city 'Olso'; known cities: Lisbon, Oslo
[call    ] live_weather({'city': 'Oslo'})
[retry   ] attempt 1 failed (upstream timeout); waiting 0.5s
[retry   ] attempt 2 failed (upstream timeout); waiting 1.0s
[observe ] error: still failing after 3 attempts (upstream timeout); try another tool
[call    ] cached_weather({'city': 'Oslo'})
[observe ] ok: 2C, cloudy (cached 2h ago)
[final   ] Lisbon is 18C and clear. Oslo was 2C and cloudy two hours ago; live data is unavailable.

live service calls: 7; model calls: 5; backoff waits: [0.5, 1.0, 0.5, 1.0] (not actually slept in the demo)
```

`router.py` and `memory.py` print traces in the same style.

## How the files are built

Every file has the same four parts, in this order:

1. A docstring that states the idea in a paragraph.
2. `Model = Callable[[str], str]`. A model is any function that takes a prompt and returns text. That one line is the whole provider interface.
3. The pattern itself, as a few plain functions with no base classes or decorators.
4. A `demo()` that feeds the pattern a `ScriptedModel`, which replays canned replies in order and records the prompts it was sent.

Because the replies are scripted, the traces above are the same on every run. They show the control flow of the pattern: what gets called, in what order, and what happens when something fails. They do not show how well any particular model would do.

Each file repeats the 12-line `ScriptedModel` instead of importing it, so a single file can be copied into another project and still run.

## Using a real model

Replace the `ScriptedModel` with any function of the same shape. For example, in a script saved inside `patterns/`:

```python
import router

def my_model(prompt: str) -> str:
    ...  # call whatever you use and return its text

router.route("The app will not sync my notes.", my_model)
```

[`examples/http_model.py`](examples/http_model.py) is one such function, written with `urllib`, for any OpenAI-compatible chat completions endpoint (hosted, or a model server on your own machine):

```bash
export MODEL_BASE_URL=http://localhost:8080/v1
export MODEL_NAME=my-model
python examples/http_model.py "The app will not sync my notes."
```

That sends the message through the router pattern. The adapter is tested against a local fake server only; it has not been run against any hosted service from this repository. Real models do not always return clean JSON or a bare label, which is why the patterns validate replies (see `make_plan` in the planner and `classify` in the router); expect to tighten prompts and parsing for the model you use.

## Patterns

- [x] Router / classifier dispatch
- [x] Planner-executor
- [x] Evaluator-optimizer loop
- [x] Human-in-the-loop approvals
- [x] Short- and long-term memory
- [x] Tool retry and error recovery
- [ ] Parallel fan-out and merge
- [ ] Multi-agent handoff
- [ ] Streaming and native tool-calling variants

## Limitations

- These are reference implementations. There is no concurrency, streaming, token counting, persistence beyond a JSON file, or security hardening.
- Tool calls are plain JSON in the prompt text, not a provider's native tool-calling API. That keeps the files provider-neutral but is less reliable with real models.
- The memory pattern recalls facts by word overlap. It is a placeholder for embedding search and will miss paraphrases.
- In `human_in_the_loop.py --interactive` the model's replies are still scripted, so its next proposal does not react to what you decide; only what runs does.
- `human_in_the_loop.py` assumes the model returns valid JSON with correct argument names. `tool_retry.py` shows how to handle it when it does not.

## Contributing

New patterns and clearer explanations are welcome. A pattern should be one file under 200 lines, standard library only, with a `demo()` that runs offline, and a test in `tests/`. To run the tests:

```bash
pip install pytest
pytest
```

Built and maintained by [Narayana Vemburaj](https://github.com/narayana006).

## License

[MIT](LICENSE)
