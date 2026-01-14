import json
import os
from generator import generate_clip


def run(job_id: str):
    """
    Runs a GPU video job.
    Supports BOTH:
      1) job.json with explicit "scenes"
      2) job.json with "script" + "num_scenes"
    """

    job_file = f"/root/ai-video-gpu-jobs/job.json"
    state_file = f"/root/ai-video-gpu-jobs/{job_id}.state.json"

    if not os.path.exists(job_file):
        raise FileNotFoundError(f"Job file not found: {job_file}")

    job = json.load(open(job_file))

    state = {}
    if os.path.exists(state_file):
        state = json.load(open(state_file))

    # --------------------------------------------------
    # SCENE RESOLUTION (FIX FOR KeyError: 'scenes')
    # --------------------------------------------------

    if "scenes" in job:
        # Explicit scene list (old / advanced format)
        scenes = job["scenes"]

    else:
        # Script-based job (CURRENT format)
        script = job["script"]
        num_scenes = job.get("num_scenes", 1)

        raw_scenes = [s.strip() for s in script.split(".") if s.strip()]
        raw_scenes = raw_scenes[:num_scenes]

        scenes = []
        for s in raw_scenes:
            scenes.append({
                "script": s,
                "duration_per_scene": job.get("duration_per_scene", 3),
                "frame_rate": job.get("frame_rate", 12),
                "add_music": job.get("add_music", False),
                "music_volume": job.get("music_volume", 0.8),
                "audio_path": job.get("audio_path"),
            })

    if not scenes:
        raise RuntimeError("No scenes resolved from job.json")

    # --------------------------------------------------
    # PROCESS SCENES (RESUME SAFE)
    # --------------------------------------------------

    for i, scene in enumerate(scenes):
        if state.get(str(i)) == "done":
            print(f"⏭️ Scene {i} already done, skipping")
            continue

        print(f"🎬 Processing scene {i}")

        generate_clip(scene, index=i)

        state[str(i)] = "done"
        json.dump(state, open(state_file, "w"))

    print("✅ All scenes processed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    run(args.job_id)
