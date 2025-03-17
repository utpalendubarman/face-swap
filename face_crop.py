import onnxruntime as ort
import cv2
import numpy as np
import os

# Load ONNX model
sess = ort.InferenceSession("models/extractor.onnx")
# Read the image
img = cv2.imread("face2.jpg")
# Run inference
scores, bboxes, keypoints, aligned_imgs, landmarks, affine_matrices = sess.run(None, {"input": img})
# Create output directory
output_dir = "cropped_faces"
os.makedirs(output_dir, exist_ok=True)

# Loop through detected faces
for i, bbox in enumerate(bboxes):
    x1, y1, x2, y2 = bbox  # Get bounding box coordinates
    face = img[y1:y2, x1:x2]  # Crop face from image

    if face.size == 0:  # Skip if empty
        continue

    # Save the cropped face
    face_path = os.path.join(output_dir, f"face_{i}.jpg")
    cv2.imwrite(face_path, face)

    print(f"Saved face {i} at {face_path}")

print("Face cropping complete!")
