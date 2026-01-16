import torch
import imageio
import numpy as np
from diffusers import (
    StableDiffusionPipeline,
    StableVideoDiffusionPipeline,
)


class AutoVideoPipeline:
    """
    REAL video generation using:
    Text -> Image (SD)
    Image -> Video (SVD)
    """

    def __init__(self):
        print("🚀 Loading text-to-image model...", flush=True)

        self.sd = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        ).to("cuda")

        print("🚀 Loading image-to-video model...", flush=True)

        self.svd = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

        # Optional xformers
        for pipe in (self.sd, self.svd):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        print("✅ Models loaded", flush=True)

    def generate_video(
        self,
        script: str,
        duration_per_scene: int,
        frame_rate: int,
        save_to: str,
        **kwargs
    ):
        # -----------------------------------
        # 1. TEXT -> IMAGE
        # -----------------------------------
        print("🖼️ Generating base image...", flush=True)

        with torch.autocast("cuda"):
            image = self.sd(
                prompt=script,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=432,
                width=768,
            ).images[0]

        # Convert PIL -> torch tensor
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")

        # -----------------------------------
        # 2. IMAGE -> VIDEO
        # -----------------------------------
        num_frames = int(duration_per_scene * frame_rate)
        print(f"🎥 Generating {num_frames} video frames...", flush=True)

        with torch.autocast("cuda"):
            result = self.svd(
                image,
                num_frames=num_frames,
                decode_chunk_size=8,
            )

        frames = result.frames[0]
        frames = [(f * 255).astype(np.uint8) for f in frames]

        imageio.mimsave(save_to, frames, fps=frame_rate)

        print(f"✅ Video saved to {save_to}", flush=True)
        return save_to
