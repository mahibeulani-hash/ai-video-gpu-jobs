import torch
import cv2
import numpy as np
import os


class DepthGenerator:
    """
    Generates a depth map using MiDaS (CPU-safe).
    """

    def __init__(self):
        print("[DEPTH] Loading MiDaS on CPU", flush=True)
        self.device = "cpu"

        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",  # lighter & faster
            pretrained=True
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms"
        ).small_transform

    @torch.no_grad()
    def generate_depth(
        self,
        image_path: str,
        output_path: str = "outputs/depth.png"
    ) -> str:

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(image).to(self.device)

        prediction = self.model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = (depth * 255).astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, depth)

        return output_path
