import cv2, time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frames, t0 = 0, time.time()
while True:
    ok, frame = cap.read()
    frames = frames + 1
    if time.time() - t0 >= 1:
        cv2.setWindowTitle("Preview", f"Preview (FPS ~ {frames})")
        frames, t0 = 0, time.time()
    cv2.imshow('Preview', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'): 
        break

    cap.release()
    cv2.destroyAllWindows()