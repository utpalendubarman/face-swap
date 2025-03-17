import onnxruntime as ort
import cv2
import numpy as np
import os

"""
    fc=face_cropper()
    res=fc.crop(source="face2.jpg",target_dir="987923-382793")
"""

class face_cropper:
    def __init__(self):
        # Load ONNX model
        self.sess = ort.InferenceSession("models/extractor.onnx")

    def crop(self,source,target_dir):
        # Read the image
        img = cv2.imread(source)
        # Run inference
        scores, bboxes, keypoints, aligned_imgs, landmarks, affine_matrices = self.sess.run(None, {"input": img})
        # Create output directory
        output_dir = target_dir
        os.makedirs(output_dir, exist_ok=True)
        cropped_faces=[]
        # Loop through detected faces
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox  # Get bounding box coordinates
            face = img[y1:y2, x1:x2]  # Crop face from image

            if face.size == 0:  # Skip if empty
                continue

            # Save the cropped face
            face_path = os.path.join(output_dir, f"face_{i}.jpg")
            cv2.imwrite(face_path, face)
            cropped_faces.append(face_path)
            print(f"Saved face {i} at {face_path}")

        print("Face cropping complete!")
        return cropped_faces


