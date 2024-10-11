from ultralytics import YOLO
import cv2 

modelPath = "C:/Users/park0/albumentations_examples/notebooks/runs/detect/train3/weights/best.pt"
model = YOLO(modelPath)

# 카메라 켜기
cap = cv2.VideoCapture(0)
# 웹캠 오류 처리
if not cap.isOpened():
    print("camera open failed")
    exit()

while True:
    # 카메라 읽기
    status, img = cap.read()
    if not status:
        print("Can't read camera")
        break
    # 카메라 창 띄우기
    results = model.predict(source = img)
    plots = results[0].plot()
    cv2.imshow("PC_Webcam",plots)
    #cv2.imshow("test", img)
    # esc 누르면 종료하기
    if cv2.waitKey(25) == 27:
        break
cap.release()
cv2.destroyAllWindows()