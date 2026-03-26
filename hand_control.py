import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ── CONFIG ──
SMOOTHING_FACTOR = 5
CLICK_THRESHOLD = 20
CLICK_COOLDOWN = 0.4
SCROLL_THRESHOLD = 40
SCROLL_SPEED = 50

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ── CAMERA & MEDIAPIPE ──
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Changed to detect 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

screen_w, screen_h = pyautogui.size()

# ── VARIABLES ──
prev_x, prev_y = 0, 0
click_time = 0
right_click_time = 0

INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
THUMB_TIP = 4

# Hand connections for drawing skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)             # Palm
]


def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def smooth_move(cx, cy, px, py, f):
    return px + (cx - px) / f, py + (cy - py) / f


# ── MAIN LOOP ──
while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # Use the actual camera frame as background
    canvas = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Draw ALL detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            h_points = []
            for lm in landmarks:
                px = int(lm.x * frame_w)
                py = int(lm.y * frame_h)
                h_points.append((px, py))

            # Draw skeleton lines
            for connection in HAND_CONNECTIONS:
                p1 = h_points[connection[0]]
                p2 = h_points[connection[1]]
                cv2.line(canvas, p1, p2, (0, 200, 0), 2)

            # Draw landmark dots
            for i, pt in enumerate(h_points):
                if i in [INDEX_TIP, MIDDLE_TIP, RING_TIP, THUMB_TIP]:
                    cv2.circle(canvas, pt, 8, (0, 255, 255), -1)
                else:
                    cv2.circle(canvas, pt, 4, (0, 255, 0), -1)

        # Logic for CONTROL (Use the FIRST detected hand)
        control_hand = results.multi_hand_landmarks[0]
        points = []
        for lm in control_hand.landmark:
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            points.append((px, py))

        # ── Key positions ──
        index_tip = points[INDEX_TIP]
        index_pip = points[INDEX_PIP]
        middle_tip = points[MIDDLE_TIP]
        middle_pip = points[MIDDLE_PIP]
        thumb_tip = points[THUMB_TIP]
        
        # Check if fingers are up (Tip above PIP)
        is_index_up = index_tip[1] < index_pip[1]
        is_middle_up = middle_tip[1] < middle_pip[1]

        # ── SCROLL MODE (Index & Middle Up) ──
        if is_index_up and is_middle_up:
            center_y = frame_h // 2
            # Scroll Up
            if index_tip[1] < center_y - 30:
                cv2.putText(canvas, "SCROLL UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pyautogui.scroll(SCROLL_SPEED)
            # Scroll Down
            elif index_tip[1] > center_y + 30:
                cv2.putText(canvas, "SCROLL DOWN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pyautogui.scroll(-SCROLL_SPEED)
            else:
                cv2.putText(canvas, "SCROLL MODE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # ── CURSOR MODE (Only Index Up or Default) ──
        else:
            # Move Cursor
            margin_x = frame_w * 0.15
            margin_y = frame_h * 0.15

            mapped_x = np.interp(index_tip[0],
                                 (int(margin_x), int(frame_w - margin_x)),
                                 (0, screen_w))
            mapped_y = np.interp(index_tip[1],
                                 (int(margin_y), int(frame_h - margin_y)),
                                 (0, screen_h))

            mapped_x = np.clip(mapped_x, 0, screen_w - 1)
            mapped_y = np.clip(mapped_y, 0, screen_h - 1)

            smooth_x, smooth_y = smooth_move(
                mapped_x, mapped_y, prev_x, prev_y, SMOOTHING_FACTOR
            )
            prev_x, prev_y = smooth_x, smooth_y
            pyautogui.moveTo(int(smooth_x), int(smooth_y))

            # ── PINCH CLICKS (Only in Cursor Mode) ──
            index_thumb_dist = get_distance(index_tip, thumb_tip)
            middle_thumb_dist = get_distance(middle_tip, thumb_tip)

            current_time = time.time()

            # Left Click (Index + Thumb)
            if index_thumb_dist < CLICK_THRESHOLD:
                if current_time - click_time > CLICK_COOLDOWN:
                    pyautogui.click()
                    click_time = current_time
                    cv2.circle(canvas, index_tip, 20, (0, 255, 0), 3)

            # Right Click (Middle + Thumb)
            # Note: Harder to do if Middle isn't up, but usually users pinch naturally.
            # If not in scroll mode, we allow right click.
            elif middle_thumb_dist < CLICK_THRESHOLD:
                if current_time - right_click_time > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    right_click_time = current_time
                    cv2.circle(canvas, middle_tip, 20, (0, 0, 255), 3)

    cv2.imshow('Hand Controller', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
