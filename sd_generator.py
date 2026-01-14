import os
from typing import Optional, List

import torch
from diffusers import StableDiffusionPipeline


class SDGenerator:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)

        # -----------------------------
        # Load pipeline FIRST
        # -----------------------------
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        # -----------------------------
        # GPU optimizations
        # -----------------------------
        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("⚡ xFormers enabled")
            except Exception as e:
                print("⚠️ xFormers not available:", e)

            # Optional – safe
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

            # ❌ DO NOT enable CPU offload on RTX GPUs

        self.pipe.set_progress_bar_config(disable=True)

    # -----------------------------
    # SINGLE IMAGE
    # -----------------------------
    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted",
        height: int = 512,  #512
        width: int = 896, #1024
        steps: int = 20, #28
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
        output_path: str = "outputs/image.png",
        progress_cb=None,   # 👈 ADD THIS
    ) -> str:

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        def callback(step: int, timestep: int, latents):
            if progress_cb:
                progress_cb(step + 1, steps)  # human-friendly

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback=callback,
            callback_steps=1,   # 👈 every step
        )

        image = result.images[0]
        #os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        image.save(output_path)
        print("✅ Image saved to:", os.path.abspath(output_path))   

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return output_path


    # -----------------------------
    # MULTIPLE IMAGES
    # -----------------------------
    def generate_images(
    self,
    prompt: str,
    count: int = 1,
    base_dir: str = "outputs/scenes",
    ):
        os.makedirs(base_dir, exist_ok=True)
        outputs = []

        for i in range(count):
            print(f"🎨 Generating scene {i+1}/{count}")

            def scene_progress(step, total):
                print(f"   ⏳ Scene {i+1}: {step}/{total}")

            path = self.generate_image(
                prompt=prompt,
                seed=i,
                output_path=os.path.join(base_dir, f"scene_{i+1}.png"),
                progress_cb=scene_progress,
            )

            outputs.append(path)

        return outputs

