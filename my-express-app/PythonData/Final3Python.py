import cv2
import numpy as np
import dlib
from imutils import face_utils
import time

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)
    return img_bgr

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\Files\shape_predictor_68_face_landmarks.dat")

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
movement_count = 0
blink_count = 0

# Initialize variables for drowsiness detection
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
prev_blink_state = 0  # Initialize the previous blink state
prev_eye_state = None  # Initialize the previous eye state
prev_movement_state = False  # Initialize the previous movement state

start_time = time.time()  # Initialize the start time
end_time = None
still_timer_start = None  # Initialize stillness timer start
stillness_count = 0  # Initialize stillness counter

movements_per_minute = []  # Array to store number of movements per minute
total_meditation_time = []  # Array to store total time meditated

while True:
    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optical flow code
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray

    magnitude = np.sqrt(flow[..., 0] * 2 + flow[..., 1] * 2)
    threshold = 5.0
    movement_pixels = (magnitude > threshold).sum()

    if movement_pixels > 10000:
        if not prev_movement_state:
            elapsed_time = time.time() - start_time
            #print(f"Movement Detected at {int(round(elapsed_time))} seconds")
            movement_count += 1
            prev_movement_state = True
            still_timer_start = None  # Reset stillness timer
            stillness_count = 0  # Reset stillness counter
    else:
        prev_movement_state = False
        if still_timer_start is None:
            still_timer_start = time.time()  # Start stillness timer

    # Check for stillness
    if still_timer_start is not None:
        elapsed_still_time = time.time() - still_timer_start
        if elapsed_still_time >= 10 and stillness_count == 0:  # Stillness threshold (e.g., 10 seconds)
            # print("Still")
            stillness_count += 1
        elif elapsed_still_time < 10:
            stillness_count = 0  # Reset stillness counter

    # Face detection and blinking code
    faces = detector(gray)
    face_frame = None

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_frame = img[y1:y2, x1:x2]

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Determine blink state
        current_blink_state = max(left_blink, right_blink)

        if current_blink_state == 2 and prev_blink_state == 0:
            blink_count += 1

        prev_blink_state = current_blink_state

        # Determine eye state
        if current_blink_state == 2:
            current_eye_state = "Closed"
        else:
            current_eye_state = "Open"

        # Print eye state changes
        if current_eye_state != prev_eye_state:
            elapsed_time = time.time() - start_time
            #print(f"Eyes {current_eye_state} at {int(round(elapsed_time))} seconds")
            prev_eye_state = current_eye_state

    # Display the frame with movement count and total blinks
    cv2.putText(img, f"Movement Count: {movement_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Total Blinks: {blink_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.namedWindow("Combined",cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(img,700,100)
    # cv2.imshow("Combined", img)

    cv2.namedWindow("Combined", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Combined", 700, 380) 
    # cv2.moveWindow("Combined", 100, 100)
    cv2.imshow("Combined", img) 

 

    # Check if a minute has passed
    elapsed_time = time.time() - start_time
    current_minute = int(elapsed_time / 60)
    if current_minute >= len(movements_per_minute):
        movements_per_minute.append(movement_count)
        movement_count = 0  # Reset the movement count for the new minute
    else:
        movements_per_minute[current_minute] = movement_count

    total_meditation_time.append(elapsed_time)

    key = cv2.waitKey(5)
    if key == ord('q'):
        end_time = time.time()
        break

if end_time is not None:
    total_meditation_time_minutes = (end_time - start_time) / 60
    rounded_meditation_time = round(total_meditation_time_minutes)  # Round off the total meditation time
    print(rounded_meditation_time , movement_count)

# print(movement_count)


cap.release()
cv2.destroyAllWindows()