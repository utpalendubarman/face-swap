import onnxruntime as ort
import cv2
import numpy as np
import os

class crop_face:
    def __init__(self, model_path="models/extractor.onnx", margin_ratio=0.4, min_face_size=256):
        """Initialize face cropping with ONNX model, margin, and minimum face size."""
        self.sess = ort.InferenceSession(model_path)
        self.margin_ratio = margin_ratio  # Percentage of face size to add as margin
        self.min_face_size = min_face_size  # Minimum size for cropped face to ensure detection

    def expand_bbox(self, bbox, img_shape):
        """Expands the bounding box while keeping it within image bounds."""
        x1, y1, x2, y2 = map(int, bbox)  # Ensure integer values
        h, w, _ = img_shape

        # Compute face width and height
        face_width = x2 - x1
        face_height = y2 - y1

        # Compute margin
        margin_w = int(face_width * self.margin_ratio)
        margin_h = int(face_height * self.margin_ratio)

        # Expand the bounding box
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)

        return x1, y1, x2, y2

    def resize_face(self, face):
        """Ensures the cropped face is at least min_face_size for better detection."""
        h, w = face.shape[:2]
        if min(h, w) < self.min_face_size:
            scale = self.min_face_size / min(h, w)
            face = cv2.resize(face, (int(w * scale), int(h * scale)))
        return face

    def crop(self, source, target_dir):
        """Detects and crops faces with extra space for better swapping."""
        # Read the image
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"Image not found: {source}")

        # Run face detection
        outputs = self.sess.run(None, {"input": img})
        if len(outputs) < 2:
            raise RuntimeError("Invalid model output. Check the ONNX model.")
        
        scores, bboxes = outputs[0], outputs[1]  # Extract scores and bounding boxes

        # Create output directory
        os.makedirs(target_dir, exist_ok=True)
        cropped_faces = []

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = self.expand_bbox(bbox, img.shape)  # Expand bounding box
            face = img[y1:y2, x1:x2]  # Crop the face

            if face.size == 0:  # Skip if empty
                continue

            face = self.resize_face(face)  # Resize face if too small

            face_path = os.path.join(target_dir, f"face_{i}.jpg")
            cv2.imwrite(face_path, face)

            face_data = {
                "face_path": face_path,
                "scores": scores[i],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
            cropped_faces.append(face_data)
            print(f"Saved expanded face {i} at {face_path}")

        if not cropped_faces:
            raise ValueError("No faces detected in the image.")

        print("Face cropping complete!")
        return cropped_faces
