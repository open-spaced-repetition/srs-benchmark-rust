"""Seed result/<name>.jsonl with {"user":N} stubs for every enumerated user EXCEPT the
targets, so a resumed run computes ONLY the target users. Usage:
    python _deck_check/seed_stubs.py <name> <maxid> <u1> <u2> ...
"""
import os, sys, json

name = sys.argv[1]
maxid = int(sys.argv[2])
targets = set(int(x) for x in sys.argv[3:])
revdir = r"C:\Users\Andrew\anki-revlogs-10k\revlogs"
allu = [int(d.split("=")[1]) for d in os.listdir(revdir) if d.startswith("user_id=")]
allu = [u for u in allu if u <= maxid]
stubs = [u for u in allu if u not in targets]
os.makedirs("result", exist_ok=True)
with open(f"result/{name}.jsonl", "w") as f:
    for u in stubs:
        f.write(json.dumps({"user": u}) + "\n")
print(f"enumerated<={maxid}: {len(allu)}  stubbed: {len(stubs)}  compute: {sorted(targets)}")
