from pipeline import AutoVideoPipeline
import os

_pipeline = None


def generate_clip(scene, index):
    global _pipeline
    if _pipeline is None:
        _pipeline = AutoVideoPipeline()

    os.makedirs("outputs", exist_ok=True)

    output_path = scene.get("output_path", f"outputs/scene_{index}.mp4")

    _pipeline.generate_video(
        script=scene["script"],
        duration_per_scene=int(scene.get("duration_per_scene", 3)),
        frame_rate=int(scene.get("frame_rate", 12)),
        save_to=output_path,
    )
