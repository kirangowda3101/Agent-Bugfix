"""Reward: fraction of a problem's asserts that pass when run in the sandbox."""
import re

_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(completion):
    """Code from the first fenced block, preferring a ```python block. None if there is no block."""
    if not isinstance(completion, str):
        completion = completion[-1]["content"]  # conversational format: list of messages
    python_blocks = re.findall(r"```(?:python|py|Python)[ \t]*\n(.*?)```", completion, re.DOTALL)
    if python_blocks:
        return python_blocks[0]
    m = _FENCE.search(completion)
    return m.group(1) if m else None


def score(sandbox, completions, test_lists, setups):
    """Returns (rewards, details). A completion with no code block scores 0 without running."""
    jobs, details = [], []
    for i, (c, tests, setup) in enumerate(zip(completions, test_lists, setups)):
        code = extract_code(c)
        details.append({"code": code, "passed": [False] * len(tests), "errors": ["no code block"] * len(tests)})
        if code is not None:
            jobs.append({"id": str(i), "code": code, "setup": setup or "", "tests": list(tests)})
    for r in sandbox.run(jobs):
        d = details[int(r["id"])]
        d["passed"], d["errors"] = r["passed"], r["errors"]
    rewards = [sum(d["passed"]) / len(d["passed"]) if d["passed"] else 0.0 for d in details]
    return rewards, details


def make_reward_fn(sandbox, log=None):
    """TRL-compatible reward function. Dataset columns arrive as keyword arguments."""
    def mbpp_pass_fraction(completions, test_list, test_setup_code, **kwargs):
        rewards, details = score(sandbox, completions, test_list, test_setup_code)
        if log is not None:
            log(completions, rewards, details, kwargs)
        return rewards
    return mbpp_pass_fraction
