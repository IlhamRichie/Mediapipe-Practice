import cv2
import mediapipe as mp
import math

def distance(p1, p2):
    return math.dist([p1.x, p1.y], [p2.x, p2.y])

EYE_DISTANCE_THRESHOLD = 0.05
LIP_THRESHOLD = 0.01             # Untuk deteksi senyum (jarak horizontal bibir)
EYE_BLINK_THRESHOLD = 0.01       # Untuk deteksi kedipan (jarak vertikal kelopak mata)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    label = ""

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Ambil titik-titik penting
            left_eye_outer = landmarks.landmark[33]
            right_eye_outer = landmarks.landmark[263]
            left_lip = landmarks.landmark[61]
            right_lip = landmarks.landmark[291]
            top_lip = landmarks.landmark[13]
            bottom_lip = landmarks.landmark[14]

            # Untuk kedipan mata kiri
            left_eye_top = landmarks.landmark[386]
            left_eye_bottom = landmarks.landmark[374]

            # Hitung jarak
            eye_distance = distance(left_eye_outer, right_eye_outer)
            lip_width = distance(left_lip, right_lip)
            mouth_open = distance(top_lip, bottom_lip)
            left_eye_blink = distance(left_eye_top, left_eye_bottom)

            # Deteksi ekspresi
            label_list = []

            if left_eye_blink < EYE_BLINK_THRESHOLD:
                label_list.append("Mata Tertutup")

            if eye_distance < EYE_DISTANCE_THRESHOLD:
                label_list.append("Mata Terbuka")

            if lip_width > LIP_THRESHOLD:
                label_list.append("Senyum")

            label = " & ".join(label_list) if label_list else "Netral"

            # Gambar landmark wajah
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=1)
            )

    # Tampilkan label prediksi
    cv2.putText(frame, f'Prediksi: {label}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Deteksi Ekspresi Real-time", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()