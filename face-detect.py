import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Disable pyautogui fail-safe (moving mouse to corner won't crash)
pyautogui.FAILSAFE = False

# ─── Thresholds and consecutive frame counts (from eye_control.py) ───
MOUTH_AR_THRESH = 0.6
MOUTH_AR_CONSECUTIVE_FRAMES = 15
EYE_AR_THRESH = 0.19
EYE_AR_CONSECUTIVE_FRAMES = 15
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 10

# ─── Frame counters & state flags ───
MOUTH_COUNTER = 0
EYE_COUNTER = 0
WINK_COUNTER = 0
INPUT_MODE = False
SCROLL_MODE = False
ANCHOR_POINT = (0, 0)

# ─── Colors (BGR) ───
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)

# ─── MediaPipe FaceMesh landmark indices ───
# Right eye (camera-right = your left eye) – used for EAR
RIGHT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                 173, 157, 158, 159, 160, 161, 246]
# Left eye (camera-left = your right eye) – used for EAR
LEFT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                466, 388, 387, 386, 385, 384, 398]

# EAR computation pairs  (top1, bottom1), (top2, bottom2), (left, right)
# Right eye EAR landmarks
R_EYE_TOP1, R_EYE_BOT1 = 159, 145    # vertical pair 1
R_EYE_TOP2, R_EYE_BOT2 = 158, 153    # vertical pair 2
R_EYE_LEFT, R_EYE_RIGHT = 33, 133    # horizontal

# Left eye EAR landmarks
L_EYE_TOP1, L_EYE_BOT1 = 386, 374
L_EYE_TOP2, L_EYE_BOT2 = 385, 380
L_EYE_LEFT, L_EYE_RIGHT = 362, 263

# Mouth MAR landmarks (outer lips)
# Vertical pairs & horizontal pair for mouth aspect ratio
MOUTH_TOP = [13]       # upper lip center
MOUTH_BOTTOM = [14]    # lower lip center
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
# More precise MAR: 3 vertical pairs like the dlib version
MOUTH_V1_TOP, MOUTH_V1_BOT = 82, 87     # vertical pair 1
MOUTH_V2_TOP, MOUTH_V2_BOT = 13, 14     # vertical pair 2
MOUTH_V3_TOP, MOUTH_V3_BOT = 312, 317   # vertical pair 3
MOUTH_H_LEFT, MOUTH_H_RIGHT = 61, 291   # horizontal

# Outer mouth contour for drawing
MOUTH_OUTLINE_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321,
                     375, 291, 409, 270, 269, 267, 0, 37, 39,
                     40, 185]

# Nose tip landmark
NOSE_TIP = 1

# ─── Helper functions (ported from utils.py) ───

def _dist(lm, idx1, idx2, fw, fh):
    """Euclidean distance between two landmarks in pixel space."""
    x1, y1 = lm[idx1].x * fw, lm[idx1].y * fh
    x2, y2 = lm[idx2].x * fw, lm[idx2].y * fh
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def eye_aspect_ratio_mp(landmarks, fw, fh, top1, bot1, top2, bot2, left, right):
    """EAR using MediaPipe landmarks (same formula as dlib version)."""
    A = _dist(landmarks, top1, bot1, fw, fh)
    B = _dist(landmarks, top2, bot2, fw, fh)
    C = _dist(landmarks, left, right, fw, fh)
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio_mp(landmarks, fw, fh):
    """MAR using MediaPipe landmarks (same formula as dlib version)."""
    A = _dist(landmarks, MOUTH_V1_TOP, MOUTH_V1_BOT, fw, fh)
    B = _dist(landmarks, MOUTH_V2_TOP, MOUTH_V2_BOT, fw, fh)
    C = _dist(landmarks, MOUTH_V3_TOP, MOUTH_V3_BOT, fw, fh)
    D = _dist(landmarks, MOUTH_H_LEFT, MOUTH_H_RIGHT, fw, fh)
    if D == 0:
        return 0.0
    return (A + B + C) / (2.0 * D)


def direction(nose_point, anchor_point, w, h, multiple=1):
    """Return direction string given nose and anchor points."""
    nx, ny = nose_point
    x, y = anchor_point
    if nx > x + multiple * w:
        return 'right'
    elif nx < x - multiple * w:
        return 'left'
    if ny > y + multiple * h:
        return 'down'
    elif ny < y - multiple * h:
        return 'up'
    return '-'


def get_landmark_points(landmarks, indices, fw, fh):
    """Convert a list of landmark indices to a numpy array of (x,y) pixel coords."""
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append([int(lm.x * fw), int(lm.y * fh)])
    return np.array(pts, dtype=np.int32)


# ─── Camera setup ───
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        print("Error: Could not read frame from camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # ─── Compute EAR for both eyes ───
        leftEAR = eye_aspect_ratio_mp(
            landmarks, frame_w, frame_h,
            L_EYE_TOP1, L_EYE_BOT1, L_EYE_TOP2, L_EYE_BOT2,
            L_EYE_LEFT, L_EYE_RIGHT
        )
        rightEAR = eye_aspect_ratio_mp(
            landmarks, frame_w, frame_h,
            R_EYE_TOP1, R_EYE_BOT1, R_EYE_TOP2, R_EYE_BOT2,
            R_EYE_LEFT, R_EYE_RIGHT
        )
        ear = (leftEAR + rightEAR) / 2.0
        diff_ear = np.abs(leftEAR - rightEAR)

        # ─── Compute MAR ───
        mar = mouth_aspect_ratio_mp(landmarks, frame_w, frame_h)

        # ─── Nose point ───
        nose_lm = landmarks[NOSE_TIP]
        nose_point = (int(nose_lm.x * frame_w), int(nose_lm.y * frame_h))

        # ─── Draw eye outlines (convex hull) ───
        right_eye_pts = get_landmark_points(landmarks, RIGHT_EYE_IDX, frame_w, frame_h)
        left_eye_pts = get_landmark_points(landmarks, LEFT_EYE_IDX, frame_w, frame_h)
        rightEyeHull = cv2.convexHull(right_eye_pts)
        leftEyeHull = cv2.convexHull(left_eye_pts)
        cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)
        cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)

        # ─── Draw mouth outline (convex hull) ───
        mouth_pts = get_landmark_points(landmarks, MOUTH_OUTLINE_IDX, frame_w, frame_h)
        mouthHull = cv2.convexHull(mouth_pts)
        cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)

        # ─── Draw individual eye & mouth landmark dots ───
        all_face_pts = np.concatenate((right_eye_pts, left_eye_pts, mouth_pts), axis=0)
        for (x, y) in all_face_pts:
            cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)

        # ─── Draw iris landmarks ───
        for idx in [469, 470, 471, 472]:
            lm = landmarks[idx]
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            cv2.circle(frame, (x, y), 3, RED_COLOR, -1)

        for idx in [474, 475, 476, 477]:
            lm = landmarks[idx]
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            cv2.circle(frame, (x, y), 3, BLUE_COLOR, -1)

        # ═══════════════════════════════════════════════════════
        #  WINK DETECTION  →  left wink = left click,
        #                      right wink = right click
        # ═══════════════════════════════════════════════════════
        if diff_ear > WINK_AR_DIFF_THRESH:
            if leftEAR < rightEAR:
                if leftEAR < EYE_AR_THRESH:
                    WINK_COUNTER += 1
                    if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                        pyautogui.click(button='left')
                        print("Left Click! (left wink)")
                        WINK_COUNTER = 0
            elif leftEAR > rightEAR:
                if rightEAR < EYE_AR_THRESH:
                    WINK_COUNTER += 1
                    if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                        pyautogui.click(button='right')
                        print("Right Click! (right wink)")
                        WINK_COUNTER = 0
            else:
                WINK_COUNTER = 0
        else:
            # ═══════════════════════════════════════════════════
            #  BOTH EYES CLOSED  →  toggle scroll mode
            # ═══════════════════════════════════════════════════
            if ear <= EYE_AR_THRESH:
                EYE_COUNTER += 1
                if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                    SCROLL_MODE = not SCROLL_MODE
                    EYE_COUNTER = 0
                    print("Scroll mode:", "ON" if SCROLL_MODE else "OFF")
            else:
                EYE_COUNTER = 0
                WINK_COUNTER = 0

        # ═══════════════════════════════════════════════════════
        #  MOUTH OPEN  →  toggle input (cursor control) mode
        # ═══════════════════════════════════════════════════════
        if mar > MOUTH_AR_THRESH:
            MOUTH_COUNTER += 1
            if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
                INPUT_MODE = not INPUT_MODE
                MOUTH_COUNTER = 0
                ANCHOR_POINT = nose_point
                print("Input mode:", "ON" if INPUT_MODE else "OFF")
        else:
            MOUTH_COUNTER = 0

        # ═══════════════════════════════════════════════════════
        #  INPUT MODE  →  nose movement controls cursor / scroll
        # ═══════════════════════════════════════════════════════
        if INPUT_MODE:
            cv2.putText(frame, "READING INPUT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
            x, y = ANCHOR_POINT
            nx, ny = nose_point
            w, h = 60, 35
            cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
            cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

            dir_str = direction(nose_point, ANCHOR_POINT, w, h)
            cv2.putText(frame, dir_str.upper(), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

            drag = 18
            if dir_str == 'right':
                pyautogui.moveRel(drag, 0)
            elif dir_str == 'left':
                pyautogui.moveRel(-drag, 0)
            elif dir_str == 'up':
                if SCROLL_MODE:
                    pyautogui.scroll(40)
                else:
                    pyautogui.moveRel(0, -drag)
            elif dir_str == 'down':
                if SCROLL_MODE:
                    pyautogui.scroll(-40)
                else:
                    pyautogui.moveRel(0, drag)
        else:
            # ─── Default: move cursor using iris tracking ───
            iris = landmarks[468]
            iris_x = int(iris.x * screen_w)
            iris_y = int(iris.y * screen_h)
            pyautogui.moveTo(iris_x, iris_y)

        # ─── On-screen status indicators ───
        if SCROLL_MODE:
            cv2.putText(frame, 'SCROLL MODE IS ON!', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

        # Debug info (uncomment if needed)
        # cv2.putText(frame, f"MAR: {mar:.2f}", (400, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW_COLOR, 1)
        # cv2.putText(frame, f"L-EAR: {leftEAR:.2f}", (400, 55),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW_COLOR, 1)
        # cv2.putText(frame, f"R-EAR: {rightEAR:.2f}", (400, 80),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW_COLOR, 1)
        # cv2.putText(frame, f"Diff: {diff_ear:.2f}", (400, 105),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW_COLOR, 1)

    cv2.imshow('Eye Controller', frame)

    # Press 'q' or Esc to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cam.release()
cv2.destroyAllWindows()