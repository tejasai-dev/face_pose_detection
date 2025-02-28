import cv2
import os
import numpy as np
import mediapipe as mp
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "classified")

def clear_output_directory():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    for category in ["front", "left", "right"]:
        os.makedirs(os.path.join(OUTPUT_DIR, category), exist_ok=True)

clear_output_directory()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float32)

best_images = {"front": None, "left": None, "right": None}
best_yaws = {"front": float("inf"), "left": float("inf"), "right": float("inf")}

def estimate_pose(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None, None

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print(f"No face detected in {image_path}. Skipping.")
        return image, None, None

    for face_landmarks in results.multi_face_landmarks:
        landmarks = face_landmarks.landmark
        image_points = np.array([
            [landmarks[1].x * w, landmarks[1].y * h],
            [landmarks[152].x * w, landmarks[152].y * h],
            [landmarks[33].x * w, landmarks[33].y * h],
            [landmarks[263].x * w, landmarks[263].y * h],
            [landmarks[61].x * w, landmarks[61].y * h],
            [landmarks[291].x * w, landmarks[291].y * h]
        ], dtype=np.float32)

        # Compute bounding box around the face
        x_min = int(min(landmark.x * w for landmark in landmarks))
        y_min = int(min(landmark.y * h for landmark in landmarks))
        x_max = int(max(landmark.x * w for landmark in landmarks))
        y_max = int(max(landmark.y * h for landmark in landmarks))

        # Draw ellipse around detected face
        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        axes = ((x_max - x_min) // 2, (y_max - y_min) // 2)  # Width and height radii
        cv2.ellipse(image, center, axes, 0, 0, 360, (255, 255, 255), 2)

        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            print(f"Pose estimation failed for {image_path}")
            return image, None, None

        rmat, _ = cv2.Rodrigues(rotation_vector)
        yaw = np.degrees(np.arctan2(rmat[1][0], rmat[0][0]))
        print(f"Yaw angle for {os.path.basename(image_path)}: {yaw:.2f} degrees")

        if -15 <= yaw <= 15:
            pose = "front"
            target_yaw = 0
        elif yaw < -15:
            pose = "right"
            target_yaw = -15
        else:
            pose = "left"
            target_yaw = 15

        print(f"Detected Pose for {os.path.basename(image_path)}: {pose.upper()}")

        deviation = abs(yaw - target_yaw)
        if deviation < best_yaws[pose]:
            best_yaws[pose] = deviation
            best_images[pose] = (image, os.path.basename(image_path))

        # Display yaw value on the image
        cv2.putText(image, f"Yaw: {yaw:.2f} deg", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image, pose, yaw
    
    return image, None, None

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(INPUT_DIR, filename)
        processed_image, pose, yaw = estimate_pose(image_path)

print("Selecting best images per category...")

for pose, best in best_images.items():
    if best:
        best_image, filename = best
        dest_path = os.path.join(OUTPUT_DIR, pose, f"best_{filename}")
        cv2.imwrite(dest_path, best_image)
        print(f"Saved BEST {pose.upper()} image: {filename}")

print("Best image selection completed.")
