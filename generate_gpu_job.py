import json
import os
import torch
from generator import generate_clip

print("===== STARTUP CHECK =====", flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
else:
    raise RuntimeError("CUDA NOT AVAILABLE")


def run(job_id: str):
    job_file = f"/root/job_{job_id}.json"
    state_file = f"/root/state_{job_id}.json"
    output_dir = "/root/outputs"

    os.makedirs(output_dir, exist_ok=True)

    with open(job_file, "r", encoding="utf-8") as f:
        job = json.load(f)

    state = {}
    if os.path.exists(state_file):
        state = json.load(open(state_file))

    script = job["script"]
    num_scenes = job.get("num_scenes", 1)

    scenes = [s.strip() for s in script.split(".") if s.strip()]
    scenes = scenes[:num_scenes]

    for i, s in enumerate(scenes):
        if state.get(str(i)) == "done":
            continue

        scene = {
            "script": s,
            "duration_per_scene": job.get("duration_per_scene", 3),
            "frame_rate": job.get("frame_rate", 12),
            "output_path": f"{output_dir}/{job_id}_scene_{i}.mp4",
        }

        generate_clip(scene, i)

        state[str(i)] = "done"
        json.dump(state, open(state_file, "w"), indent=2)

    print("✅ Job completed", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    run(args.job_id)
