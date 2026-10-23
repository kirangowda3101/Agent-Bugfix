"""MBPP loading and prompt formatting.

Uses the official splits of the "full" config: train (374), validation (90),
test (500), prompt (10). Training uses train only; validation is for model and
checkpoint selection; test is evaluated once per final model.

The prompt shows only the FIRST assert, so the model knows the function name
and signature. The reward is computed on ALL asserts. Scores are therefore not
comparable to published MBPP numbers, which show all three asserts.
"""
from datasets import load_dataset

DATASET = "google-research-datasets/mbpp"
DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
CONFIG = "full"
MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODEL_REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"

PROMPT = (
    "{text}\n"
    "Your code should pass this test:\n"
    "```python\n{test}\n```\n"
    "Write the complete Python function in a single ```python code block."
)


def load_split(split):
    ds = load_dataset(DATASET, CONFIG, split=split, revision=DATASET_REVISION)
    return ds.map(_to_example, remove_columns=[c for c in ds.column_names if c not in ("task_id",)])


def _to_example(row):
    return {
        "prompt": [{"role": "user", "content": PROMPT.format(text=row["text"], test=row["test_list"][0])}],
        "test_list": row["test_list"],
        "test_setup_code": row["test_setup_code"],
        "reference": row["code"],
    }
