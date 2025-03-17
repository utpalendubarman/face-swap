from crop_face import crop_face
from swap_face import swap_face
import cv2
import numpy as np

source = "modi.jpg"
target_dir = "temp/face27"
face_cropper = crop_face()
faces = face_cropper.crop(source=source, target_dir=target_dir)

# Assign source face
x = 1
for face in faces:
    face['face_source'] = "face2.jpg"
    if x % 2 == 0:
        face['face_source'] = "face1.jpg"
    x += 1

# Initialize face swapper
face_swaper = swap_face()
img = cv2.imread(source)

c = 1
# Loop through detected faces
for config in faces:
    x1, y1, x2, y2 = config['x1'], config['y1'], config['x2'], config['y2']  # Bounding box
    print('Swapping:', config['face_path'], "with", config['face_source'])

    swapped_image_path = f"output_{c}.png"
    swapped = face_swaper.swap(config['face_source'], config['face_path'], swapped_image_path)

    # Load the swapped face
    swapped_face = cv2.imread(swapped_image_path)
    if swapped_face is None:
        print(f"Error: Swapped image {swapped_image_path} not found.")
        continue

    # Get the size of the detected face region
    face_width = x2 - x1
    face_height = y2 - y1

    # Resize swapped face to match the face region while keeping the aspect ratio
    swapped_face_resized = cv2.resize(swapped_face, (face_width, face_height))

    # Create a mask (white shape to blend)
    mask = 255 * np.ones(swapped_face_resized.shape, swapped_face_resized.dtype)

    # Compute center of the face region for blending
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Perform seamless cloning
    img = cv2.seamlessClone(swapped_face_resized, img, mask, center, cv2.NORMAL_CLONE)

    c += 1

# Save final image
cv2.imwrite("target3.jpg", img)
print("Face swapping completed. Saved as target1.jpg")
