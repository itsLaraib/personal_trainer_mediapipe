import cv2 as cv
import mediapipe as mp
import time


class poseDetector:
    def __init__(
        self,
        mode=False,
        complexity=1,
        landmarks=True,
        Segmentation=False,
        smoothSegmentation=True,
        detectionConfidence=0.5,
        trackingConfidence=0.5,
    ):
        self.static_image_mode = mode
        self.model_complexity = complexity
        self.smooth_landmarks = landmarks
        self.enable_segmentation = Segmentation
        self.smooth_segmentation = smoothSegmentation
        self.min_detection_confidence = detectionConfidence
        self.min_tracking_confidence = trackingConfidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.static_image_mode,
            self.model_complexity,
            self.smooth_landmarks,
            self.enable_segmentation,
            self.smooth_segmentation,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def poseDraw(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if self.results.pose_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
                    )

            return img

    def getPositions(self, img, draw=True):
        list = []
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            x, y = int(lm.x * w), int(lm.y * h)
            list.append((x, y))
            if draw:
                cv.circle(img,(x,y),5,(255,0,255),-1)
        return list


def main():
    cTime = 0
    pTime = 0
    cap = cv.VideoCapture("Pose-2.mp4")
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.poseDraw(img)
        list=detector.getPositions(img,False)
        print(list)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        if success:
            cv.putText(
                img, str(int(fps)), (10, 60), cv.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2
            )
            cv.imshow("Video", img)
        else:
            break

        if cv.waitKey(20) & 0xFF == ord("d"):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
