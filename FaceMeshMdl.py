import cv2
import mediapipe as mp
import time


class FaceMeshDetector:

    def __init__(self, staticMode=False, maxFaces=2, refineLm=False, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLm = refineLm
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        # Utilization of libraries for mesh and drawing of mediapipe
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.refineLm, self.minDetectionCon,
                                                 self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    def findFaceMesh(self, img, draw=True):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        # Ciclo para dibujar los landmarks, si es que se detecta una cara
        nface = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                # Ciclo para obtener los landmarks en pixels
                fpoints = []
                for id, lm in enumerate(faceLms.landmark):
                    if id % 5 == 0: # Here we have the visual identification of multiples of 5 landmarks
                        ih, iw, ic = img.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        fpoints.append([x, y])

                        if draw:
                            # Draw selected landmarks in a differente color (RED)
                            cv2.circle(img, (x, y), 3, (0, 0, 255), cv2.FILLED)
                            # Identify those landmarks
                            # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                            #             0.5, (0, 0, 255), 1)

                nface.append(fpoints)

        return img, nface


def main():
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
    detector = FaceMeshDetector()

    while True:
        # Saving captured image and transforming from BGR TO RGB
        success, img = cap.read()
        img, nface = detector.findFaceMesh(img)
        if len(nface) != 0:
            print(len(nface))

        # Fps and their display
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)

        # Show the complete image
        cv2.imshow('Image', img)

        key = cv2.waitKey(30)
        if key == 27:  # 27= Esc
            break
    print(nface)



if __name__ == '__main__':
    main()


