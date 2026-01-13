from moviepy.editor import ImageSequenceClip
import numpy as np
import cv2


USE_CINEMATIC_DEPTH = True  # feature flag (rollback-safe)


class ImageAnimator:
    """
    Converts a single image into a short cinematic 2.5D video clip.
    Face regions are automatically protected from depth/parallax warping.
    """

    # --------------------------------------------------
    # FACE DETECTION (FAST, CPU-ONLY)
    # --------------------------------------------------
    def _detect_face_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        mask = np.zeros((h, w), dtype=np.uint8)
        unsafe = False

        for (x, y, fw, fh) in faces:
            # if face touches image edge → unsafe
            if x < 20 or y < 20 or (x + fw) > (w - 20) or (y + fh) > (h - 20):
                unsafe = True

            pad = int(0.55 * fw)  # covers eyes + forehead
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + fw + pad)
            y2 = min(h, y + fh + pad)
            mask[y1:y2, x1:x2] = 255

        return mask, len(faces), unsafe


    # --------------------------------------------------
    # MAIN ANIMATION FUNCTION
    # --------------------------------------------------
    def animate(self, image_path, depth_path, duration, fps):
        PARALLAX_STRENGTH = 0.3  # SAFE for humans

        # ------------------------------------------
        # 1. LOAD IMAGE
        # ------------------------------------------
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError("Image could not be loaded")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # ------------------------------------------
        # 2. LOAD DEPTH MAP
        # ------------------------------------------
        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_map is None:
            raise RuntimeError("Depth map could not be loaded")

        depth_map = cv2.resize(depth_map, (w, h))
        depth_map = depth_map.astype(np.float32) / 255.0

        # ------------------------------------------
        # 3. DEPTH STABILIZATION (ANTI-FACE-MELT)
        # ------------------------------------------
        depth_map = depth_map ** 0.6
        depth_map = depth_map * PARALLAX_STRENGTH

        # ------------------------------------------
        # 4. FACE MASK + DIALOGUE DETECTION
        # ------------------------------------------
        face_mask, face_count, unsafe = self._detect_face_mask(image)

        dialogue_scene = False

        # Two or more faces → dialogue
        if face_count >= 2:
            dialogue_scene = True

        # Single close-up face → dialogue
        if face_count == 1:
            ys, xs = np.where(face_mask > 0)
            if len(ys) > 0:
                face_height = ys.max() - ys.min()
                if face_height / h > 0.18:
                    dialogue_scene = True

        # Unsafe framing → dialogue (reduce motion)
        if unsafe:
            dialogue_scene = True

        # Blur + normalize face mask
        face_mask = cv2.GaussianBlur(face_mask, (51, 51), 0)
        face_mask = face_mask.astype(np.float32) / 255.0
        face_mask = face_mask[..., None]  # (H, W, 1)

        # ------------------------------------------
        # 5. FRAME GENERATION (SAFE, ALWAYS ANIMATED)
        # ------------------------------------------
        frames = []
        num_frames = max(1, int(duration * fps))

        # Dialogue = reduced motion, NOT zero motion
        max_shift = 1 if dialogue_scene else 3

        for i in range(num_frames):
            shift_x = int((i / num_frames) * max_shift)

            M = np.float32([[1, 0, shift_x], [0, 1, 0]])

            warped = cv2.warpAffine(
                image,
                M,
                (w, h),
                borderMode=cv2.BORDER_REFLECT
            )

            # Face protected, background moves
            final = (image * face_mask + warped * (1 - face_mask)).astype(np.uint8)
            frames.append(final)

        # ------------------------------------------
        # 6. MOVIEPY CLIP
        # ------------------------------------------
        clip = ImageSequenceClip(frames, fps=fps)
        return clip
