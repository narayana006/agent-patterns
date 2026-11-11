import json

import pytest

import evaluator_optimizer
import human_in_the_loop
import memory
import planner_executor
import router
import tool_retry


# --- router --------------------------------------------------------------------

def test_router_dispatches_and_normalises_label():
    model = router.ScriptedModel(["  Billing.\n", "refund on its way"])
    assert router.route("charged twice", model) == ("billing", "refund on its way")
    assert "billing, technical, general" in model.prompts[0]
    assert "billing specialist" in model.prompts[1]


def test_router_falls_back_on_unknown_label_without_second_model_call():
    model = router.ScriptedModel(["sales"])
    label, reply = router.route("hello", model)
    assert label == "general" and "one working day" in reply
    assert len(model.prompts) == 1


def test_router_demo():
    assert [label for label, _ in router.demo()] == ["billing", "technical", "general"]


# --- planner-executor -------------------------------------------------------------

def test_plan_runs_with_references_between_steps(capsys):
    plan = [
        {"tool": "price_of", "args": {"item": "Museum Pass"}},
        {"tool": "multiply", "args": {"a": "$1", "b": 4}},
        {"tool": "add", "args": {"values": ["$1", "$2", 1.0]}},
    ]
    assert planner_executor.execute(plan) == [18.0, 72.0, 91.0]


@pytest.mark.parametrize("reply, message", [
    ("not json", "not valid JSON"),
    ("[]", "non-empty list"),
    ('[{"tool": "rm_rf", "args": {}}]', "unknown tool"),
    ('[{"tool": "add", "args": [1, 2]}]', "args must be an object"),
])
def test_bad_plans_are_rejected_before_anything_runs(reply, message):
    with pytest.raises(planner_executor.PlanError, match=message):
        planner_executor.make_plan("goal", planner_executor.ScriptedModel([reply]))


def test_forward_reference_is_an_error():
    with pytest.raises(planner_executor.PlanError, match=r"\$2"):
        planner_executor.execute([{"tool": "multiply", "args": {"a": "$2", "b": 1}}])


def test_planner_demo_total(capsys):
    assert "196.50" in planner_executor.demo()
    assert "add(values=[84.0, 94.5, 18.0]) -> 196.5" in capsys.readouterr().out


# --- evaluator-optimizer ----------------------------------------------------------

def test_rule_failures_skip_the_critic_and_feed_back_to_writer():
    writer = evaluator_optimizer.ScriptedModel(["far too long a draft here", "short one"])
    critic = evaluator_optimizer.ScriptedModel(["PASS"])
    final, passed, rounds = evaluator_optimizer.optimize("task", writer, critic, max_words=2)
    assert (final, passed, rounds) == ("short one", True, 2)
    assert len(critic.prompts) == 1
    assert "Too long: 6 words, the limit is 2." in writer.prompts[1]
    assert "far too long a draft here" in writer.prompts[1]


def test_loop_stops_at_round_limit_unapproved():
    writer = evaluator_optimizer.ScriptedModel(["a", "b", "c"])
    critic = evaluator_optimizer.ScriptedModel(["REVISE: no"] * 3)
    assert evaluator_optimizer.optimize("task", writer, critic, max_rounds=3) == ("c", False, 3)


def test_must_mention_is_case_insensitive():
    verdict = evaluator_optimizer.evaluate("t", "Sign in works", evaluator_optimizer.ScriptedModel(["pass"]), 10, ["sign"])
    assert verdict.passed


def test_evaluator_demo():
    final, passed, rounds = evaluator_optimizer.demo()
    assert passed and rounds == 3 and "30 minutes" in final


# --- human in the loop ------------------------------------------------------------

def scripted_refund_model(*extra):
    return human_in_the_loop.ScriptedModel([
        '{"tool": "issue_refund", "args": {"order_id": "A-1", "amount": 5.0}}', *extra,
    ])


def test_risky_tool_runs_only_when_approved():
    human_in_the_loop.LEDGER.clear()
    asked = []

    def approve(tool, args):
        asked.append((tool, args))
        return human_in_the_loop.Decision(True)

    model = scripted_refund_model('{"final": "done"}')
    assert human_in_the_loop.run("t", model, approve) == "done"
    assert asked == [("issue_refund", {"order_id": "A-1", "amount": 5.0})]
    assert human_in_the_loop.LEDGER == ["refund 5.00 on A-1"]


def test_denial_blocks_the_tool_and_reason_reaches_the_model():
    human_in_the_loop.LEDGER.clear()
    model = scripted_refund_model('{"final": "escalated"}')
    deny = lambda tool, args: human_in_the_loop.Decision(False, "over the limit")  # noqa: E731
    assert human_in_the_loop.run("t", model, deny) == "escalated"
    assert human_in_the_loop.LEDGER == []
    assert "denied by a human reviewer: over the limit" in model.prompts[1]


def test_safe_tool_never_asks_for_approval():
    def approve(tool, args):
        raise AssertionError("approver must not be called for safe tools")

    model = human_in_the_loop.ScriptedModel(
        ['{"tool": "look_up_order", "args": {"order_id": "A-9"}}', '{"final": "ok"}'])
    assert human_in_the_loop.run("t", model, approve) == "ok"


def test_step_limit_and_unknown_tool():
    model = human_in_the_loop.ScriptedModel(['{"tool": "nope"}'] * 2)
    with pytest.raises(RuntimeError, match="no final answer after 2 steps"):
        human_in_the_loop.run("t", model, lambda tool, args: human_in_the_loop.Decision(True), max_steps=2)
    assert "no tool called nope" in model.prompts[1]


def test_human_in_the_loop_demo_refunds_only_the_approved_amount():
    assert "refunded 39.00" in human_in_the_loop.demo()
    assert human_in_the_loop.LEDGER == ["refund 39.00 on A-1042"]


# --- memory -----------------------------------------------------------------------

def test_fact_store_persists_dedupes_and_recalls(tmp_path):
    path = tmp_path / "facts.json"
    store = memory.FactStore(path)
    store.save("The user is vegetarian.")
    store.save("The user is vegetarian.")
    store.save("The user lives in Porto.")
    reopened = memory.FactStore(path)
    assert reopened.facts == ["The user is vegetarian.", "The user lives in Porto."]
    assert reopened.recall("Any restaurants in Porto?") == ["The user lives in Porto."]
    assert reopened.recall("What time is it?") == []


def test_window_is_bounded_and_old_turns_are_summarised(tmp_path):
    model = memory.ScriptedModel(["r1", "r2", "summary of turn one", "r3"])
    chat = memory.Conversation(model, memory.FactStore(tmp_path / "f.json"), window=1)
    chat.send("first message")
    chat.send("second message")
    assert chat.turns == [("second message", "r2")] and chat.summary == "summary of turn one"
    chat.model = memory.ScriptedModel(["r3", "summary two"])
    chat.send("third message")
    prompt = chat.model.prompts[0]
    assert "Earlier in this conversation: summary of turn one" in prompt
    assert "second message" in prompt and "first message" not in prompt


def test_remember_marker_is_stripped_and_saved(tmp_path):
    store = memory.FactStore(tmp_path / "f.json")
    chat = memory.Conversation(memory.ScriptedModel(["Got it. REMEMBER: The user owns a kayak."]), store)
    assert chat.send("I own a kayak") == "Got it."
    assert store.facts == ["The user owns a kayak."]


def test_memory_demo_recalls_across_sessions(tmp_path):
    reply, *facts = memory.demo(tmp_path / "facts.json")
    assert facts == ["The user is vegetarian.", "The user lives in Porto."]
    assert "Porto" in reply
    assert json.loads((tmp_path / "facts.json").read_text()) == facts


# --- tool retry -------------------------------------------------------------------

def test_transient_errors_are_retried_with_exponential_backoff():
    service = tool_retry.FlakyWeatherService({"Lisbon": 2})
    waits = []
    assert tool_retry.call_with_retry(service, {"city": "Lisbon"}, sleep=waits.append) == "ok: 18C, clear"
    assert service.calls == 3 and waits == [0.5, 1.0]


def test_gives_up_after_attempts_and_suggests_another_tool():
    service = tool_retry.FlakyWeatherService({"Oslo": 99})
    result = tool_retry.call_with_retry(service, {"city": "Oslo"}, attempts=4, base_delay=1.0, sleep=lambda s: None)
    assert result.startswith("error: still failing after 4 attempts") and service.calls == 4


def test_permanent_and_unexpected_errors_are_not_retried():
    service = tool_retry.FlakyWeatherService({})
    assert "unknown city 'Atlantis'" in tool_retry.call_with_retry(service, {"city": "Atlantis"})
    assert service.calls == 1
    assert tool_retry.call_with_retry(service, {"town": "Oslo"}).startswith("error: TypeError")
    assert tool_retry.call_with_retry(tool_retry.cached_weather, {"city": "Atlantis"}).startswith("error: KeyError")


def test_agent_recovers_from_invalid_json_and_unknown_tool():
    model = tool_retry.ScriptedModel(["oops", '{"tool": "radar", "args": {}}', '{"final": "done"}'])
    assert tool_retry.run("q", model, {"cached_weather": tool_retry.cached_weather}) == "done"
    assert "not valid JSON" in model.prompts[1] and "no tool called 'radar'" in model.prompts[2]


def test_tool_retry_demo_uses_fallback(capsys):
    assert "live data is unavailable" in tool_retry.demo()
    assert "live service calls: 7" in capsys.readouterr().out
