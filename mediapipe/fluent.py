import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh()
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pose_results = pose.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    
    height, width, _ = frame.shape
    posture, eye_status, hand_status, position_status = "", "", "", ""
    
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        head_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * height
        shoulder_left = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height
        shoulder_right = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height
        
        if abs(shoulder_left - shoulder_right) > 20:
            posture = "Miring"
        elif head_y > shoulder_left:
            posture = "Menunduk"
        else:
            posture = "Tegak"
        
        center_x = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * width
        if center_x < width * 0.3:
            position_status = "Terlalu ke kiri"
        elif center_x > width * 0.7:
            position_status = "Terlalu ke kanan"
        else:
            position_status = "Posisi bagus"
    
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for i, landmark in enumerate(face_landmarks.landmark):
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            left_eye_top = face_landmarks.landmark[159].y * height
            left_eye_bottom = face_landmarks.landmark[145].y * height
            right_eye_top = face_landmarks.landmark[386].y * height
            right_eye_bottom = face_landmarks.landmark[374].y * height
            eye_ratio = (left_eye_bottom - left_eye_top + right_eye_bottom - right_eye_top) / 2
            
            eye_status = "Sering berkedip" if eye_ratio < 5 else "Normal"
    
    if hands_results.multi_hand_landmarks:
        hand_status = "Gestur Tangan Aktif"
        for hand_landmarks in hands_results.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        hand_status = "Tidak Menggunakan Gestur"
    
    cv2.putText(frame, f'Postur: {posture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Mata: {eye_status}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Gestur: {hand_status}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Posisi Kamera: {position_status}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('FaceMesh & Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()