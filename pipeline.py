import torch
import imageio
import numpy as np
from diffusers import (
    StableDiffusionPipeline,
    StableVideoDiffusionPipeline,
)


class AutoVideoPipeline:
    """
    Text -> Image (Stable Diffusion)
    Image -> Video (Stable Video Diffusion)
    """

    def __init__(self):
        print("🚀 Loading Stable Diffusion (text → image)...", flush=True)

        self.sd = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        ).to("cuda")

        print("🚀 Loading Stable Video Diffusion (image → video)...", flush=True)

        self.svd = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

        # xFormers is OPTIONAL
        for pipe in (self.sd, self.svd):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ xFormers enabled", flush=True)
            except Exception:
                print("⚠️ xFormers not available, continuing", flush=True)

        print("✅ Models loaded", flush=True)

    def generate_video(
        self,
        script: str,
        duration_per_scene: int,
        frame_rate: int,
        save_to: str,
        **kwargs
    ):
        # -----------------------------
        # TEXT → IMAGE
        # -----------------------------
        print("🖼️ Generating base image...", flush=True)

        with torch.autocast("cuda"):
            image = self.sd(
                prompt=script,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=432,
                width=768,
            ).images[0]

        # PIL → torch tensor [1,3,H,W]
        image = np.array(image).astype("float32") / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cuda")

        # -----------------------------
        # IMAGE → VIDEO
        # -----------------------------
        num_frames = int(duration_per_scene * frame_rate)
        print(f"🎥 Generating {num_frames} frames...", flush=True)

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
