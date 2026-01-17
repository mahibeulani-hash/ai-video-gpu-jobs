import torch
import imageio
import numpy as np
from diffusers import CogVideoXPipeline


class AutoVideoPipeline:
    """
    TRUE Text-to-Video using CogVideoX
    """

    def __init__(self):
        print("🚀 Loading CogVideoX T2V model...", flush=True)

        self.pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5b",
            torch_dtype=torch.float16
        ).to("cuda")

        self.pipe.enable_attention_slicing()

        print("✅ CogVideoX loaded", flush=True)

    def generate_video(
        self,
        script: str,
        duration_per_scene: int,
        frame_rate: int,
        save_to: str,
        **kwargs
    ):
        num_frames = int(duration_per_scene * frame_rate)

        print(f"🎥 Generating {num_frames} frames (TEXT → VIDEO)", flush=True)

        with torch.autocast("cuda"):
            video = self.pipe(
                prompt=script,
                num_frames=num_frames,
                height=432,
                width=768,
                num_inference_steps=30,
            ).frames[0]

        frames = [(f * 255).astype(np.uint8) for f in video]
        imageio.mimsave(save_to, frames, fps=frame_rate)

        print(f"✅ Video saved to {save_to}", flush=True)
