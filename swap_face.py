import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class swap_face:
    def __init__(self, model_path="models/inswapper_128.onnx"):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(model_path)

    def resize_image(self, image, max_size=1080, min_size=256):
        """Ensures the image is within a reasonable size range."""
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
        elif min(h, w) < min_size:
            scale = min_size / min(h, w)
        else:
            return image  # No resizing needed
        return cv2.resize(image, (int(w * scale), int(h * scale)))

    def swap(self, source_image_path, target_image_path, output_path):
        source_image = cv2.imread(source_image_path)
        target_image = cv2.imread(target_image_path)

        if source_image is None or target_image is None:
            raise ValueError("One of the images could not be loaded. Check the file paths.")

        # Resize images to reasonable dimensions
        source_image = self.resize_image(source_image)
        target_image = self.resize_image(target_image)

        # Detect faces
        source_faces = self.app.get(source_image)
        target_faces = self.app.get(target_image)

        if len(source_faces) == 0:
            raise ValueError("No face detected in source image.")
        if len(target_faces) == 0:
            raise ValueError("No face detected in target image.")

        # Swap all detected faces
        swapped_image = target_image.copy()
        for target_face in target_faces:
            swapped_image = self.swapper.get(swapped_image, target_face, source_faces[0], paste_back=True)

        # Save the output
        cv2.imwrite(output_path, swapped_image)
        print(f"Face swapped image saved at {output_path}")
        return True
