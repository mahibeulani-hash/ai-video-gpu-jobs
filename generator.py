from pipeline import AutoVideoPipeline
import os


_pipeline = None


def generate_clip(scene, index):
    """
    Adapter used by generate_gpu_job.py
    Produces REAL video (not animated image).
    """

    global _pipeline
    if _pipeline is None:
        _pipeline = AutoVideoPipeline()

    os.makedirs("outputs", exist_ok=True)

    output_path = scene.get("output_path", f"outputs/scene_{index}.mp4")

    _pipeline.generate_video(
        script=scene["script"],
        duration_per_scene=scene.get("duration_per_scene", 3),
        frame_rate=scene.get("frame_rate", 12),
        save_to=output_path,
    )
