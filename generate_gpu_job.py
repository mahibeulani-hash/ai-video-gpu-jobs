import json
import os
from generator import generate_clip

def run(job_id):
    job_file = f"/root/jobs/{job_id}.json"
    state_file = f"/root/jobs/{job_id}.state.json"

    job = json.load(open(job_file))
    state = json.load(open(state_file)) if os.path.exists(state_file) else {}

    for i, scene in enumerate(job["scenes"]):
        if state.get(str(i)) == "done":
            continue

        generate_clip(scene, index=i)
        state[str(i)] = "done"
        json.dump(state, open(state_file, "w"))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    args = p.parse_args()
    run(args.job_id)
