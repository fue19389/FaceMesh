import cv2
import mediapipe as mp
import time

# Capture photo


cap = cv2.VideoCapture(0)

# Capture webcam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# HIGH RESOLUTION
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 384)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
# lOW RESOLUTION
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 405)

# Variables
pTime = 0
fpoints = []

# Utilization of libraries for mesh and drawing of mediapipe
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

while True:

    # Saving captured image and transforming from BGR TO RGB
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    # Ciclo para dibujar los landmarks, si es que se detecta una cara
    if results.multi_face_landmarks:
        for fno, faceLms in enumerate(results.multi_face_landmarks):
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
            # Ciclo para obtener los landmarks en pixels
            for id, lm in enumerate(faceLms.landmark):
                if id % 5 == 0:
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    fpoints.append([fno, id, x, y])
                    cv2.circle(img, (x, y), 3, (0, 0, 255), cv2.FILLED)

    # Fps and their display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)

    # Show the complete image
    cv2.imshow('Image', img)

    key = cv2.waitKey(30)
    if key == 27: # 27= Esc
        break

print(fpoints)