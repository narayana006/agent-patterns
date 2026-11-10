# agent-patterns

Small, runnable reference implementations of common AI agent patterns, each under 200 lines.

> **Project status: planning. There is no working code in this repository yet.**
> This README describes what the project will do. Nothing below can be installed or run today.
> Features are ticked off in the checklist as they are built.

## What it is

A set of runnable example scripts.

**The problem it solves:** Agent frameworks hide the core ideas. Small readable examples teach the patterns and are easy to copy.

## Who it is for

Developers learning how AI agents work who want small examples they can read in one sitting.

## How it will work

1. Each pattern is one short, self-contained file with comments explaining the idea.
2. You run a file directly to see the pattern work end to end.
3. You copy the pattern into your own project and adapt it.

## Planned features

Unticked items are not built yet.

- [ ] Router / classifier dispatch
- [ ] Planner-executor
- [ ] Evaluator-optimizer loop
- [ ] Human-in-the-loop approvals
- [ ] Short- and long-term memory
- [ ] Tool retry and error recovery

## Planned usage

This is the intended interface, shown so you can judge the design. **It does not work yet**, and names may change before the first release.

```bash
python patterns/evaluator_optimizer.py
```

## Contributing

Feedback on the design is the most useful contribution right now. Open an issue to suggest a feature, question an approach or describe your use case. Code contributions are welcome once the first version lands.

## License

[MIT](LICENSE)
