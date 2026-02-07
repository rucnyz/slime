"""
Test suite for SlimeLlm token tracking, message conversion, and response parsing.
Requires a local Qwen tokenizer at /root/Qwen3-4B-Instruct-2507.
Does NOT require a running sglang server.
"""

from transformers import AutoTokenizer
from aigise.rl_integration.slime_llm import SlimeLlm, TokenTracker

MODEL_PATH = "/root/Qwen3-4B-Instruct-2507"


def make_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = SlimeLlm(model="test")
    llm.configure(
        sglang_url="http://localhost:30000/generate",
        tokenizer=tokenizer,
        sampling_params={},
        tools_info=[],
    )
    return llm, tokenizer


def test_token_tracker():
    t = TokenTracker()
    assert t.all_token_ids == []
    assert t.response_length == 0

    t.set_initial_prompt([1, 2, 3])
    assert t.prompt_token_ids == [1, 2, 3]
    assert t.all_token_ids == [1, 2, 3]

    t.add_assistant_tokens([10, 11])
    assert t.response_token_ids == [10, 11]
    assert t.loss_masks == [1, 1]

    t.add_environment_tokens([20, 21, 22])
    assert t.response_token_ids == [10, 11, 20, 21, 22]
    assert t.loss_masks == [1, 1, 0, 0, 0]

    assert t.all_token_ids == [1, 2, 3, 10, 11, 20, 21, 22]
    assert t.response_length == 5
    print("PASS: test_token_tracker")


def test_get_token_delta():
    llm, _ = make_llm()
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    msgs.append({"role": "assistant", "content": "I am an AI."})
    at, am = llm._get_token_delta(msgs, last_role="assistant")
    assert len(at) > 0 and all(m == 1 for m in am)
    print(f"  assistant: {len(at)} tokens, all mask=1")

    msgs.append({"role": "tool", "name": "search", "content": "found 3"})
    tt, tm = llm._get_token_delta(msgs, last_role="tool")
    assert len(tt) > 0 and all(m == 0 for m in tm)
    print(f"  tool: {len(tt)} tokens, all mask=0")

    msgs.append({"role": "user", "content": "Thanks!"})
    ut, um = llm._get_token_delta(msgs, last_role="user")
    assert len(ut) > 0 and all(m == 0 for m in um)
    print(f"  user: {len(ut)} tokens, all mask=0")

    print("PASS: test_get_token_delta")


def test_env_detection():
    llm, tokenizer = make_llm()
    init = [
        {"role": "system", "content": "Agent"},
        {"role": "user", "content": "Go"},
    ]
    pt = tokenizer.apply_chat_template(init, tokenize=False, add_generation_prompt=True)
    llm.tracker.set_initial_prompt(tokenizer(pt, add_special_tokens=False)["input_ids"])
    llm.tracker.messages = list(init)

    llm.tracker.messages.append({"role": "assistant", "content": "OK"})
    at, am = llm._get_token_delta(llm.tracker.messages, last_role="assistant")
    llm.tracker.response_token_ids.extend(at)
    llm.tracker.loss_masks.extend(am)
    prev = len(llm.tracker.response_token_ids)

    cur = list(llm.tracker.messages) + [
        {"role": "tool", "name": "cmd", "content": "done"},
    ]
    llm._track_environment_messages(cur)

    assert len(llm.tracker.response_token_ids) > prev
    new_m = llm.tracker.loss_masks[len(am) :]
    assert all(m == 0 for m in new_m)
    print(f"  env: {len(new_m)} tokens tracked with mask=0")
    print("PASS: test_env_detection")


def test_llm_request_conv():
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    llm, _ = make_llm()
    req = LlmRequest()
    req.config.system_instruction = "Security researcher."
    req.contents = [
        types.Content(role="user", parts=[types.Part(text="Find vuln.")]),
        types.Content(role="model", parts=[types.Part(text="Checking.")]),
        types.Content(
            role="user",
            parts=[types.Part(function_response=types.FunctionResponse(name="cmd", response={"out": "ok"}))],
        ),
    ]
    msgs = llm._llm_request_to_messages(req)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool"], f"Got {roles}"
    assert msgs[3]["name"] == "cmd"
    print(f"  roles: {roles}")
    print("PASS: test_llm_request_conv")


def test_parse_tool():
    llm, _ = make_llm()

    # Qwen-style tool call
    r = '<tool_call>\n{"name": "cmd", "arguments": {"c": "ls"}}\n</tool_call>'
    parts = llm._parse_response_to_parts(r)
    assert len(parts) == 1 and parts[0].function_call is not None, f"Got {len(parts)} parts"
    assert parts[0].function_call.name == "cmd"
    assert parts[0].function_call.args == {"c": "ls"}
    print(f"  qwen tool call: {parts[0].function_call.name}({parts[0].function_call.args})")

    # Plain text
    parts2 = llm._parse_response_to_parts("I found the vulnerability.")
    assert len(parts2) == 1 and parts2[0].text is not None
    print(f"  plain text OK")

    # Text + tool call
    r3 = 'Let me check.\n<tool_call>\n{"name": "scan", "arguments": {"target": "192.168.1.1"}}\n</tool_call>'
    parts3 = llm._parse_response_to_parts(r3)
    assert len(parts3) == 2, f"Expected 2 parts, got {len(parts3)}"
    assert parts3[0].text == "Let me check."
    assert parts3[1].function_call.name == "scan"
    print(f"  text+tool: text='{parts3[0].text}', tool={parts3[1].function_call.name}")

    # Generic JSON tool call
    r4 = '{"name": "run_exploit", "arguments": {"target": "10.0.0.1", "port": 80}}'
    parts4 = llm._parse_response_to_parts(r4)
    assert len(parts4) == 1 and parts4[0].function_call is not None
    assert parts4[0].function_call.name == "run_exploit"
    print(f"  json tool call: {parts4[0].function_call.name}")

    print("PASS: test_parse_tool")


def test_multi_turn_consistency():
    llm, tokenizer = make_llm()

    msgs = [
        {"role": "system", "content": "You are a pen tester."},
        {"role": "user", "content": "Start recon on target."},
    ]
    pt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    llm.tracker.set_initial_prompt(tokenizer(pt, add_special_tokens=False)["input_ids"])
    llm.tracker.messages = list(msgs)

    turns = [
        ("assistant", "Running nmap scan.", None),
        ("tool", "Port 80 open, Port 443 open", "nmap"),
        ("assistant", "Found open ports. Scanning further with nikto.", None),
        ("tool", "Nikto found XSS on /search endpoint", "nikto"),
        ("assistant", "Vulnerability confirmed. Writing report.", None),
    ]

    for role, content, name in turns:
        msg = {"role": role, "content": content}
        if name:
            msg["name"] = name
        llm.tracker.messages.append(msg)
        t, m = llm._get_token_delta(llm.tracker.messages, last_role=role)
        llm.tracker.response_token_ids.extend(t)
        llm.tracker.loss_masks.extend(m)

    # Verify structural properties (exact match not expected due to BPE merge
    # boundary differences in delta tokenization — same as tau-bench, see
    # https://verl.readthedocs.io/en/v0.4.1/sglang_multiturn/multiturn.html)
    full = tokenizer.apply_chat_template(llm.tracker.messages, tokenize=False, add_generation_prompt=False)
    full_t = tokenizer(full, add_special_tokens=False)["input_ids"]
    recon = llm.tracker.all_token_ids

    # Allow small divergence from BPE boundary effects (typically < 10%)
    len_diff = abs(len(recon) - len(full_t))
    diff_pct = len_diff / len(full_t) * 100
    assert diff_pct < 15, f"Token count diverged too much: {len(recon)} vs {len(full_t)} ({diff_pct:.1f}%)"

    trainable = sum(llm.tracker.loss_masks)
    total = len(llm.tracker.loss_masks)
    print(f"  5-turn conversation: {len(full_t)} full tokens, {len(recon)} reconstructed ({diff_pct:.1f}% diff)")
    print(f"  prompt: {len(llm.tracker.prompt_token_ids)} tokens")
    print(f"  response: {total} tokens ({trainable} trainable, {total - trainable} non-trainable)")
    assert trainable > 0 and (total - trainable) > 0
    assert len(llm.tracker.loss_masks) == len(llm.tracker.response_token_ids)

    print("PASS: test_multi_turn_consistency")


if __name__ == "__main__":
    tests = [
        ("TokenTracker", test_token_tracker),
        ("token_delta", test_get_token_delta),
        ("env_detection", test_env_detection),
        ("llm_request", test_llm_request_conv),
        ("parse_tool", test_parse_tool),
        ("multi_turn", test_multi_turn_consistency),
    ]
    for name, fn in tests:
        print(f"=== {name} ===")
        fn()
        print()
    print("ALL TESTS PASSED")
