import json
import os
import torch
from generator import generate_clip

# --------------------------------------------------
# STARTUP CHECK
# --------------------------------------------------
print("===== STARTUP CHECK =====", flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Capability:", torch.cuda.get_device_capability(0), flush=True)
else:
    raise RuntimeError("CUDA NOT AVAILABLE — GPU REQUIRED")

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------

def run(job_id: str):
    """
    Multi-user, multi-job safe GPU job runner.

    File layout (Vast RAW mode):
      Job file   : /root/jobs/<job_id>.json
      State file : /root/state_<job_id>.json
      Outputs    : /root/outputs/<job_id>.mp4
    """

    job_file = f"/root/jobs/{job_id}.json"
    state_file = f"/root/state_{job_id}.json"
    output_dir = "/root/outputs"

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # LOAD JOB
    # --------------------------------------------------
    if not os.path.exists(job_file):
        raise FileNotFoundError(f"Job file not found: {job_file}")

    with open(job_file, "r", encoding="utf-8") as f:
        job = json.load(f)

    # --------------------------------------------------
    # LOAD STATE (resume-safe)
    # --------------------------------------------------
    state = {}
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

    # --------------------------------------------------
    # SCENE RESOLUTION
    # --------------------------------------------------
    if "scenes" in job:
        scenes = job["scenes"]

    else:
        script = job.get("script", "").strip()
        if not script:
            raise RuntimeError("Job missing 'script' field")

        num_scenes = job.get("num_scenes", 1)
        duration_per_scene = job.get("duration_per_scene", 3)
        frame_rate = job.get("frame_rate", 12)

        raw_scenes = [s.strip() for s in script.split(".") if s.strip()]
        raw_scenes = raw_scenes[:num_scenes]

        scenes = []
        for s in raw_scenes:
            scenes.append({
                "script": s,
                "duration_per_scene": duration_per_scene,
                "frame_rate": frame_rate,
                "add_music": job.get("add_music", False),
                "music_volume": job.get("music_volume", 0.8),
                "audio_path": job.get("audio_path"),
                "output_path": f"{output_dir}/{job_id}.mp4"
            })

    if not scenes:
        raise RuntimeError("No scenes resolved from job.json")

    # --------------------------------------------------
    # PROCESS SCENES (RESUME SAFE)
    # --------------------------------------------------
    for i, scene in enumerate(scenes):
        if state.get(str(i)) == "done":
            print(f"⏭️ Scene {i} already done, skipping", flush=True)
            continue

        print(f"🎬 Processing scene {i}", flush=True)

        generate_clip(scene, index=i)

        state[str(i)] = "done"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    print("✅ All scenes processed successfully", flush=True)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    run(args.job_id)
