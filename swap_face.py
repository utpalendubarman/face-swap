import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

"""
    face_swapper = FaceSwapper()
    face_swapper.swap_faces("face1.jpg", "face2.jpg", "output2024.jpg")
"""

class FaceSwapper:
    def __init__(self, model_path="models/inswapper_128.onnx"):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(model_path)

    def swap_faces(self, source_image_path, target_image_path, output_path):
        source_image = cv2.imread(source_image_path)
        target_image = cv2.imread(target_image_path)

        # Detect faces
        source_faces = self.app.get(source_image)
        target_faces = self.app.get(target_image)

        if len(source_faces) == 0 or len(target_faces) == 0:
            raise ValueError("No faces detected in one of the images.")

        # Perform face swap (assuming first detected face is the main face)
        swapped_image = self.swapper.get(target_image, target_faces[0], source_faces[0], paste_back=True)

        # Save the output
        cv2.imwrite(output_path, swapped_image)
        print(f"Face swapped image saved at {output_path}")

