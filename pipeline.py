import os
import torch
import numpy as np
import imageio
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableVideoDiffusionPipeline,
)

# ----------------------------------------------------------
# IMPORTANT: reduce CUDA fragmentation
# ----------------------------------------------------------
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)


class AutoVideoPipeline:
    """
    FINAL, PRODUCTION-SAFE PROMPT → VIDEO PIPELINE

    Internally:
      Text → Image (Stable Diffusion)
      Image → Video (Stable Video Diffusion)

    User experience:
      Prompt → Video
    """

    def __init__(self):
        print("🚀 Loading Stable Diffusion (text → image)...", flush=True)

        self.sd = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        ).to("cuda")

        self.sd.enable_attention_slicing()

        print("🚀 Loading Stable Video Diffusion (image → video)...", flush=True)

        self.svd = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

        # SVD memory safety
        self.svd.enable_attention_slicing()
        self.svd.enable_model_cpu_offload()

        print("✅ Final SVD pipeline ready", flush=True)

    def generate_video(
        self,
        script: str,
        duration_per_scene: int,
        frame_rate: int,
        save_to: str,
    ):
        # --------------------------------------------------
        # 1. TEXT → IMAGE (GPU)
        # --------------------------------------------------
        print("🖼️ Generating base image from prompt...", flush=True)

        with torch.autocast("cuda"):
            image = self.sd(
                prompt=(
                    script
                    + ", cinematic motion, dynamic action, "
                      "camera tracking shot, depth of field"
                ),
                num_inference_steps=35,
                guidance_scale=7.5,
                height=576,
                width=1024,
            ).images[0]

        # --------------------------------------------------
        # 2. FREE SD FROM GPU (CRITICAL)
        # --------------------------------------------------
        self.sd.to("cpu")
        torch.cuda.empty_cache()

        # --------------------------------------------------
        # 3. RESIZE IMAGE FOR SVD (MANDATORY: 224x224)
        # --------------------------------------------------
        image_resized = image.resize((224, 224), Image.BICUBIC)

        image_np = np.array(image_resized).astype("float32") / 255.0
        image_tensor = (
            torch.from_numpy(image_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to("cuda")
        )

        # --------------------------------------------------
        # 4. IMAGE → VIDEO (SVD)
        # --------------------------------------------------
        num_frames = int(duration_per_scene * frame_rate)
        print(f"🎥 Generating {num_frames} video frames...", flush=True)

        with torch.autocast("cuda"):
            result = self.svd(
                image_tensor,
                num_frames=num_frames,
                decode_chunk_size=4,  # ↓ VRAM safe
            )

        frames = result.frames[0]
        frames = [(f * 255).astype(np.uint8) for f in frames]

        # --------------------------------------------------
        # 5. SAVE VIDEO
        # --------------------------------------------------
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        imageio.mimsave(save_to, frames, fps=frame_rate)

        print(f"✅ Video saved: {save_to}", flush=True)
        return save_to
