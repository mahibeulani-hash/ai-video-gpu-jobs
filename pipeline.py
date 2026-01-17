import torch
import numpy as np
import imageio
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from PIL import Image

class AutoVideoPipeline:
    """
    FINAL, WORKING TEXT → VIDEO PIPELINE
    (Image step is internal and hidden)
    """

    def __init__(self):
        print("🚀 Loading models...", flush=True)

        # Text → Image
        self.sd = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        ).to("cuda")

        # Image → Video
        self.svd = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

        self.sd.enable_attention_slicing()
        self.svd.enable_attention_slicing()

        print("✅ Models loaded", flush=True)

    def generate_video(
        self,
        script: str,
        duration_per_scene: int,
        frame_rate: int,
        save_to: str,
    ):
        # ----------------------------------
        # TEXT → IMAGE (hidden step)
        # ----------------------------------
        print("🖼️ Generating base image from text...", flush=True)

        with torch.autocast("cuda"):
            image = self.sd(
                prompt=script
                + ", cinematic lighting, dynamic motion, depth of field, sharp focus",
                num_inference_steps=35,
                guidance_scale=7.5,
                height=576,
                width=1024,
            ).images[0]

        

        # Convert PIL → numpy
        image_np = np.array(image).astype("float32") / 255.0

        # Resize to 224x224 for SVD (MANDATORY)
        image_resized = Image.fromarray((image_np * 255).astype("uint8")).resize(
            (224, 224), Image.BICUBIC
        )

        image = np.array(image_resized).astype("float32") / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cuda")


        # ----------------------------------
        # IMAGE → VIDEO (real temporal diffusion)
        # ----------------------------------
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
