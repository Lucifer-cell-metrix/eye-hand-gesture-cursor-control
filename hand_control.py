import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import sys

# ── CONFIG ──
SMOOTHING_FACTOR = 5
DRAG_SMOOTHING = 2           # Less smoothing during drag
CLICK_THRESHOLD = 25         # Pinch distance for clicks
RIGHT_CLICK_THRESHOLD = 25   # Middle+thumb distance for right click
CLICK_COOLDOWN = 0.5
SCROLL_SPEED = 15
SCROLL_FINGER_DIST = 100
SCROLL_ACCUMULATE = 5
DRAG_HOLD_TIME = 3.0         # Seconds to hold pinch before drag

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ── CAMERA ──
print("Opening camera...")
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    print("Error: Could not open camera.")
    sys.exit(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Warm up camera
print("Warming up camera...")
for _ in range(10):
    cam.read()
    time.sleep(0.05)

# ── MEDIAPIPE ──
print("Initializing hand detection...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,           # Detect both hands
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

screen_w, screen_h = pyautogui.size()

# ── VARIABLES ──
prev_x, prev_y = 0, 0
click_time = 0
right_click_time = 0
scroll_ref_y = None
scroll_accumulated = 0
is_dragging = False
drag_fist_start = None

# Landmark indices
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
THUMB_TIP = 4
PINKY_TIP = 20
PINKY_PIP = 18

# Hand connections for drawing skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def smooth_move(cx, cy, px, py, f):
    return px + (cx - px) / f, py + (cy - py) / f


def count_fingers_up(pts):
    """Count how many fingers (index, middle, ring, pinky) are extended."""
    count = 0
    if pts[INDEX_TIP][1] < pts[INDEX_PIP][1]:
        count += 1
    if pts[MIDDLE_TIP][1] < pts[MIDDLE_PIP][1]:
        count += 1
    if pts[RING_TIP][1] < pts[RING_PIP][1]:
        count += 1
    if pts[PINKY_TIP][1] < pts[PINKY_PIP][1]:
        count += 1
    return count


def get_hand_points(hand_landmarks, frame_w, frame_h):
    """Convert hand landmarks to pixel coordinates."""
    points = []
    for lm in hand_landmarks.landmark:
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        points.append((px, py))
    return points


def draw_hand(canvas, points, color_lines=(0, 200, 0), color_dots=(0, 255, 0), color_tips=(0, 255, 255)):
    """Draw hand skeleton on canvas."""
    for connection in HAND_CONNECTIONS:
        p1 = points[connection[0]]
        p2 = points[connection[1]]
        cv2.line(canvas, p1, p2, color_lines, 2)

    for i, pt in enumerate(points):
        if i in [INDEX_TIP, MIDDLE_TIP, RING_TIP, THUMB_TIP]:
            cv2.circle(canvas, pt, 6, color_tips, -1)
        else:
            cv2.circle(canvas, pt, 4, color_dots, -1)


print("Ready! Show BOTH hands:")
print("  RIGHT hand → Move cursor")
print("  LEFT hand  → Click / Scroll")
print("Press 'q' to quit.")

# ── MAIN LOOP ──
while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    canvas = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        results = hands.process(rgb_frame)
    except Exception as e:
        print(f"MediaPipe error: {e}")
        continue

    # ── Identify which hand is Left and which is Right ──
    right_hand_pts = None
    left_hand_pts = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            pts = get_hand_points(hand_landmarks, frame_w, frame_h)

            if label == "Right":
                right_hand_pts = pts
                draw_hand(canvas, pts,
                          color_lines=(200, 100, 0),    # Blue-ish lines
                          color_dots=(255, 150, 0),     # Blue dots
                          color_tips=(255, 255, 0))     # Cyan tips
            elif label == "Left":
                left_hand_pts = pts
                draw_hand(canvas, pts,
                          color_lines=(0, 100, 200),    # Orange-ish lines
                          color_dots=(0, 150, 255),     # Orange dots
                          color_tips=(0, 255, 255))     # Yellow tips

    # ═══════════════════════════════════════════════════════
    #  RIGHT HAND → CURSOR MOVEMENT
    # ═══════════════════════════════════════════════════════
    if right_hand_pts:
        index_tip = right_hand_pts[INDEX_TIP]

        # Map index finger position to screen
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

        current_smooth = DRAG_SMOOTHING if is_dragging else SMOOTHING_FACTOR
        smooth_x, smooth_y = smooth_move(
            mapped_x, mapped_y, prev_x, prev_y, current_smooth
        )
        prev_x, prev_y = smooth_x, smooth_y
        pyautogui.moveTo(int(smooth_x), int(smooth_y))

        # Label on screen
        cv2.putText(canvas, "R: CURSOR", (right_hand_pts[0][0] - 30, right_hand_pts[0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    # ═══════════════════════════════════════════════════════
    #  LEFT HAND → CLICKS / SCROLL / DRAG
    # ═══════════════════════════════════════════════════════
    if left_hand_pts:
        l_index_tip = left_hand_pts[INDEX_TIP]
        l_index_pip = left_hand_pts[INDEX_PIP]
        l_middle_tip = left_hand_pts[MIDDLE_TIP]
        l_middle_pip = left_hand_pts[MIDDLE_PIP]
        l_thumb_tip = left_hand_pts[THUMB_TIP]

        l_is_index_up = l_index_tip[1] < l_index_pip[1]
        l_is_middle_up = l_middle_tip[1] < l_middle_pip[1]
        l_fingers_up = count_fingers_up(left_hand_pts)

        l_index_thumb_dist = get_distance(l_index_tip, l_thumb_tip)
        l_middle_thumb_dist = get_distance(l_middle_tip, l_thumb_tip)
        l_index_middle_dist = get_distance(l_index_tip, l_middle_tip)

        current_time = time.time()

        # ── SCROLL MODE: Index + Middle up, close together, only 2 fingers ──
        scroll_active = (l_is_index_up and l_is_middle_up
                         and l_index_middle_dist < SCROLL_FINGER_DIST
                         and l_fingers_up == 2)

        if scroll_active:
            avg_y = (l_index_tip[1] + l_middle_tip[1]) // 2

            if scroll_ref_y is not None:
                scroll_accumulated += (avg_y - scroll_ref_y)

                if scroll_accumulated < -SCROLL_ACCUMULATE:
                    cv2.putText(canvas, "SCROLL UP", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    pyautogui.scroll(SCROLL_SPEED)
                    scroll_accumulated = 0
                elif scroll_accumulated > SCROLL_ACCUMULATE:
                    cv2.putText(canvas, "SCROLL DOWN", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    pyautogui.scroll(-SCROLL_SPEED)
                    scroll_accumulated = 0
                else:
                    cv2.putText(canvas, "SCROLL MODE", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(canvas, "SCROLL MODE", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            scroll_ref_y = avg_y

            # Release drag if active
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
                drag_fist_start = None

        # ── CLICK / DRAG MODE ──
        else:
            scroll_ref_y = None
            scroll_accumulated = 0

            # Index + Thumb pinch detection
            pinching = l_index_thumb_dist < CLICK_THRESHOLD

            if is_dragging:
                if not pinching:
                    # Released pinch → stop drag
                    pyautogui.mouseUp()
                    is_dragging = False
                    drag_fist_start = None
                    cv2.putText(canvas, "DROPPED", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(canvas, "DRAGGING", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    cv2.circle(canvas, l_index_tip, 15, (0, 165, 255), 3)

            elif pinching:
                if drag_fist_start is None:
                    drag_fist_start = current_time

                hold_duration = current_time - drag_fist_start
                remaining = DRAG_HOLD_TIME - hold_duration

                if hold_duration >= DRAG_HOLD_TIME:
                    pyautogui.mouseDown()
                    is_dragging = True
                    cv2.putText(canvas, "DRAG STARTED!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                else:
                    cv2.putText(canvas, f"HOLD TO DRAG: {remaining:.1f}s", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    progress = hold_duration / DRAG_HOLD_TIME
                    angle = int(360 * progress)
                    cv2.ellipse(canvas, l_index_tip, (20, 20), 0, 0, angle,
                                (0, 165, 255), 3)

            else:
                # Not pinching
                if drag_fist_start is not None:
                    hold_duration = current_time - drag_fist_start
                    if hold_duration < DRAG_HOLD_TIME:
                        if current_time - click_time > CLICK_COOLDOWN:
                            pyautogui.click()
                            click_time = current_time
                            cv2.putText(canvas, "LEFT CLICK", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    drag_fist_start = None

                # Right click: Middle + Thumb pinch
                if l_middle_thumb_dist < RIGHT_CLICK_THRESHOLD and l_is_middle_up:
                    if current_time - right_click_time > CLICK_COOLDOWN:
                        pyautogui.rightClick()
                        right_click_time = current_time
                        cv2.circle(canvas, l_middle_tip, 20, (0, 0, 255), 3)
                        cv2.putText(canvas, "RIGHT CLICK", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Label on screen
        cv2.putText(canvas, "L: CLICK", (left_hand_pts[0][0] - 30, left_hand_pts[0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

    else:
        # No left hand → release drag
        if is_dragging:
            pyautogui.mouseUp()
            is_dragging = False
            drag_fist_start = None

    # ── Status bar ──
    status_parts = []
    if right_hand_pts:
        status_parts.append("RIGHT: Cursor")
    else:
        status_parts.append("RIGHT: Not detected")
    if left_hand_pts:
        status_parts.append("LEFT: Click/Scroll")
    else:
        status_parts.append("LEFT: Not detected")

    cv2.putText(canvas, " | ".join(status_parts),
                (10, frame_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, "Right Hand=Move | Left: Pinch=Click, Peace=Scroll | 'q' to quit",
                (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    cv2.imshow('Hand Controller', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("Hand controller stopped.")
