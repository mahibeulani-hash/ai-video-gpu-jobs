from pipeline import AutoVideoPipeline


def generate_clip(scene, index):
    """
    Adapter used by generate_gpu_job.py
    """
    pipeline = AutoVideoPipeline()

    pipeline.generate_video(
        script=scene["script"],
        num_scenes=scene.get("num_scenes", 1),
        duration_per_scene=scene.get("duration_per_scene", 3),
        frame_rate=scene.get("frame_rate", 12),
        add_music=scene.get("add_music", False),
        music_volume=scene.get("music_volume", 0.8),
        audio_path=scene.get("audio_path"),
        save_to=f"outputs/scene_{index}.mp4",
    )
