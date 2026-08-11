from datasets import load_dataset

ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

print("total instances:", len(ds))
print()

inst = ds[0]

print("fields available:")
for key in inst:
    print("  -", key)
print()

print("instance_id:", inst["instance_id"])
print("repo:", inst["repo"])
print("base_commit:", inst["base_commit"])
print()
print("FAIL_TO_PASS:", inst["FAIL_TO_PASS"])
print()
print("PASS_TO_PASS count:", len(inst["PASS_TO_PASS"]))
print()
print("problem_statement:")
print(inst["problem_statement"][:1500])

print()
print("smallest PASS_TO_PASS counts:")
sizes = sorted(
    [(len(d["PASS_TO_PASS"]), d["instance_id"], d["difficulty"]) for d in ds]
)
for size, iid, diff in sizes[:15]:
    print(f"  {size:5d}  {iid:40s} {diff}")