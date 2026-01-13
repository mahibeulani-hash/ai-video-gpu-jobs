import os
from typing import List
from moviepy.editor import concatenate_videoclips, AudioFileClip

# imports you already use
from sd_generator import SDGenerator
from depth_generator import DepthGenerator
from animator import ImageAnimator


class AutoVideoPipeline:
    """
    End-to-end pipeline:
    Script → Image → Depth → Animation → Video → Audio
    """

    def __init__(self):
        self.sd = SDGenerator()
        self.depth = DepthGenerator()
        self.animator = ImageAnimator()

    def generate_video(
        self,
        script: str,
        num_scenes: int,
        duration_per_scene: int,
        frame_rate: int,
        add_music: bool,
        music_volume: float,
        save_to: str,
        audio_path: str | None = None
    ) -> str:

        scenes = [s.strip() for s in script.split(".") if s.strip()]
        scenes = scenes[:num_scenes]

        if not scenes:
            raise RuntimeError("No valid scenes extracted")

        clips = []

        for idx, scene in enumerate(scenes):
            prompt = (
                scene
                + ", medium shot, sharp facial features, "
                  "symmetrical face, photorealistic skin texture"
            )

            img_path = self.sd.generate_image(
                prompt=prompt,
                steps=6,
                guidance_scale=5.0,
                output_path=f"outputs/scene_{idx}.png"
            )

            depth_path = self.depth.generate_depth(
                img_path,
                output_path=f"outputs/scene_{idx}_depth.png"
            )

            clip = self.animator.animate(
                image_path=img_path,
                depth_path=depth_path,
                duration=duration_per_scene,
                fps=frame_rate
            )

            clips.append(clip)

        video = concatenate_videoclips(clips, method="compose")

        if add_music and audio_path and os.path.exists(audio_path):
            audio = AudioFileClip(audio_path).volumex(music_volume)
            video = video.set_audio(audio.set_duration(video.duration))

        os.makedirs("outputs", exist_ok=True)
        video.write_videofile(save_to, codec="libx264", audio_codec="aac")

        video.close()
        for c in clips:
            c.close()

        return save_to
