import cv2
import numpy as np
from scipy.spatial import Delaunay
import mediapipe as mp
import json

# MediaPipe initialize
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.7)

# human landmark picture loading
target_face_path = "./soseki.png"
landmark_path = "./landmark.json"

# loading target face image
target_face = cv2.imread(target_face_path)
if target_face is None:
    print(f"Error: Could not load target face image from '{target_face_path}'. Please check the file path.")
    exit()

# landmarks are generated by MediaPipe
target_face_rgb = cv2.cvtColor(target_face, cv2.COLOR_BGR2RGB)
target_results = face_mesh.process(target_face_rgb)

if not target_results.multi_face_landmarks:
    print("Warning: Could not detect landmarks on the target face image using MediaPipe. Using manually set landmarks instead.")
    try:
        with open(landmark_path, 'r') as f:
            target_landmarks = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not load landmarks from '{landmark_path}'. Please check the file path.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{landmark_path}'. Please check the file format.")
        exit()
else:
    # Position shape from landmark
    h_t, w_t, _ = target_face.shape
    target_landmarks = [(int(landmark.x * w_t), int(landmark.y * h_t)) for landmark in target_results.multi_face_landmarks[0].landmark]
    # Saving
    with open(landmark_path, 'w') as f:
        json.dump(target_landmarks, f)

# Real-time movie cam
def warp_and_blend_face(src_img, dest_img, src_points, dest_points, alpha=0.5):
    # Fitting texture face parts
    delaunay = Delaunay(dest_points)
    for triangle in delaunay.simplices:
        try:
            # Triangle calculation
            src_tri = np.float32([src_points[i] for i in triangle])
            dest_tri = np.float32([dest_points[i] for i in triangle])

            # Bounding box rectangle
            src_rect = cv2.boundingRect(src_tri)
            dest_rect = cv2.boundingRect(dest_tri)

            # Move each triangle's coordinates within their respective bounding boxes
            src_tri_rect = np.array([[p[0] - src_rect[0], p[1] - src_rect[1]] for p in src_tri], dtype=np.float32)
            dest_tri_rect = np.array([[p[0] - dest_rect[0], p[1] - dest_rect[1]] for p in dest_tri], dtype=np.float32)

            # Get bounding box area
            src_cropped = src_img[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]

            if src_cropped.shape[0] == 0 or src_cropped.shape[1] == 0:
                continue

            # Calculate affine transform matrix and transform src_cropped
            affine_matrix = cv2.getAffineTransform(src_tri_rect, dest_tri_rect)
            warped_cropped = cv2.warpAffine(src_cropped, affine_matrix, (dest_rect[2], dest_rect[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # Generate mask and get triangle area
            mask = np.zeros((dest_rect[3], dest_rect[2], 3), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dest_tri_rect), (1.0, 1.0, 1.0), 16, 0)

            # Apply the transformed result to the corresponding area in the destination image
            dest_cropped = dest_img[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]]
            if dest_cropped.shape[0] == 0 or dest_cropped.shape[1] == 0:
                continue

            blended_cropped = cv2.addWeighted(dest_cropped, 1 - alpha, warped_cropped, alpha, 0)

            # Place the transformed result back into the original frame
            dest_img[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] = blended_cropped
        except IndexError:
            print("Error during warping and blending: Index out of range, skipping this triangle.")
            continue

# Open video cam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB and detect human face
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get human face landmark in real-time
            h, w, _ = frame.shape
            face_coords = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]
            x_min, y_min = np.min(face_coords, axis=0)
            x_max, y_max = np.max(face_coords, axis=0)

            # Checking face area
            if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
                print("Face out of bounds, skipping.")
                continue

            # Calculate face width and height, and resize target image to match real-time face size
            h_t, w_t, _ = target_face.shape  # Redefine here
            face_width = x_max - x_min
            face_height = y_max - y_min
            scale_factor_x = face_width / w_t
            scale_factor_y = face_height / h_t
            scale_factor = min(scale_factor_x, scale_factor_y)
            target_face_resized = cv2.resize(target_face, (int(w_t * scale_factor), int(h_t * scale_factor)))

            # Adjust and fit human face
            resized_target_coords = [(int(point[0] * scale_factor), int(point[1] * scale_factor)) for point in target_landmarks]

            # Fit landmark face
            if len(face_coords) > 0 and len(resized_target_coords) > 0:
                try:
                    warp_and_blend_face(target_face_resized, frame, resized_target_coords, face_coords, alpha=0.5)  # Set transparency to 0.5
                except Exception as e:
                    print(f"Error during warping and blending: {e}")
                    continue

    # Display frame
    cv2.imshow('Rejuvenated Face', frame)

    # 'q' key to quit window
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
