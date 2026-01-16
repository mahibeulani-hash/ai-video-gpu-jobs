import torch
import imageio
import numpy as np
from diffusers import StableVideoDiffusionPipeline


class AutoVideoPipeline:
    """
    REAL video generation pipeline using Stable Video Diffusion.
    """

    def __init__(self):
        print("🚀 Loading Stable Video Diffusion model...", flush=True)

        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")

        # OPTIONAL xformers
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ xFormers enabled", flush=True)
        except Exception as e:
            print("⚠️ xFormers not available, continuing without it:", e, flush=True)

        print("✅ Video model loaded", flush=True)

    def generate_video(
        self,
        script: str,
        duration_per_scene: int,
        frame_rate: int,
        save_to: str,
        **kwargs
    ):
        num_frames = duration_per_scene * frame_rate

        print(f"🎥 Generating {num_frames} frames", flush=True)

        with torch.autocast("cuda"):
            result = self.pipe(
                prompt=script,
                num_frames=num_frames,
                height=432,
                width=768,
                num_inference_steps=30,
                guidance_scale=7.5,
                decode_chunk_size=8
            )

        frames = result.frames[0]
        frames = [(frame * 255).astype(np.uint8) for frame in frames]

        imageio.mimsave(save_to, frames, fps=frame_rate)

        print(f"✅ Video saved to {save_to}", flush=True)
        return save_to
