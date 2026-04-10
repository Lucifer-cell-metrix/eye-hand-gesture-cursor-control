import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import sys
from collections import deque

# ══════════════════════════════════════════════════════════
#  CONFIG — Tune these for your setup
# ══════════════════════════════════════════════════════════
SMOOTHING_FRAMES = 6         # Number of past positions to average (higher = smoother but laggier)
EMA_ALPHA = 0.35             # Exponential moving average weight (0.1=very smooth, 0.5=responsive)
DRAG_EMA_ALPHA = 0.5         # More responsive during drag
JITTER_THRESHOLD = 3         # Ignore cursor movements smaller than this (pixels) — kills micro-jitter
CLICK_THRESHOLD = 30         # Pinch distance for clicks (pixels)
RIGHT_CLICK_THRESHOLD = 30   # Middle+thumb distance for right click
CLICK_COOLDOWN = 0.45
SCROLL_FINGER_DIST = 110     # Max distance between index & middle tips for scroll
SCROLL_DEAD_ZONE = 25        # Pixels from anchor before scroll starts (prevents accidental scroll)
SCROLL_SPEED_MIN = 2         # Min scroll amount per frame
SCROLL_SPEED_MAX = 30        # Max scroll amount per frame
SCROLL_RAMP_DIST = 120       # Pixels from anchor to reach max scroll speed
SCROLL_COOLDOWN = 0.03       # Seconds between scroll events (lower = smoother)
DRAG_HOLD_TIME = 2.5         # Seconds to hold pinch before drag starts
MARGIN_RATIO = 0.12          # Active hand zone margin (smaller = more screen coverage)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ══════════════════════════════════════════════════════════
#  CAMERA SETUP
# ══════════════════════════════════════════════════════════
print("Opening camera...")
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    print("Error: Could not open camera.")
    sys.exit(1)

# Higher resolution = better hand detection accuracy
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency by minimizing buffer

# Warm up camera
print("Warming up camera...")
for _ in range(15):
    cam.read()
    time.sleep(0.03)

# ══════════════════════════════════════════════════════════
#  MEDIAPIPE HANDS
# ══════════════════════════════════════════════════════════
print("Initializing hand detection...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,            # Higher accuracy model (0=fast, 1=accurate)
    min_detection_confidence=0.7,  # Higher = fewer false detections
    min_tracking_confidence=0.7    # Higher = more stable tracking
)

screen_w, screen_h = pyautogui.size()

# ══════════════════════════════════════════════════════════
#  STATE VARIABLES
# ══════════════════════════════════════════════════════════
# Cursor smoothing with position history (moving average + EMA)
cursor_history_x = deque(maxlen=SMOOTHING_FRAMES)
cursor_history_y = deque(maxlen=SMOOTHING_FRAMES)
smooth_x, smooth_y = screen_w / 2, screen_h / 2  # Start at screen center

click_time = 0
right_click_time = 0
scroll_anchor_y = None       # Y position where scroll mode started (joystick center)
last_scroll_time = 0
is_dragging = False
drag_fist_start = None
frames_without_right_hand = 0
HAND_LOST_FRAMES = 5  # Frames without hand before we stop moving

# Landmark indices
INDEX_TIP = 8
INDEX_PIP = 6
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
THUMB_TIP = 4
THUMB_IP = 3
PINKY_TIP = 20
PINKY_PIP = 18
WRIST = 0

# Hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# ══════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════
def get_distance(p1, p2):
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def ema_smooth(current, previous, alpha):
    """Exponential Moving Average — blends current with previous value."""
    return previous + alpha * (current - previous)


def count_fingers_up(pts):
    """Count how many fingers (index, middle, ring, pinky) are extended.
    A finger is 'up' if its tip is above (lower y) its PIP joint."""
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


def is_thumb_up(pts):
    """Check if thumb is extended (tip is far from index MCP)."""
    return get_distance(pts[THUMB_TIP], pts[INDEX_MCP]) > 60


def get_hand_points(hand_landmarks, frame_w, frame_h):
    """Convert hand landmarks to pixel coordinates."""
    points = []
    for lm in hand_landmarks.landmark:
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        points.append((px, py))
    return points


def draw_hand(canvas, points, color_lines, color_dots, color_tips, label, label_color):
    """Draw hand skeleton with label on canvas."""
    # Draw skeleton lines
    for c in HAND_CONNECTIONS:
        cv2.line(canvas, points[c[0]], points[c[1]], color_lines, 2, cv2.LINE_AA)

    # Draw landmark dots
    for i, pt in enumerate(points):
        if i in [INDEX_TIP, MIDDLE_TIP, RING_TIP, THUMB_TIP, PINKY_TIP]:
            cv2.circle(canvas, pt, 7, color_tips, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)  # White border
        else:
            cv2.circle(canvas, pt, 3, color_dots, -1, cv2.LINE_AA)

    # Draw label near wrist
    wrist = points[WRIST]
    cv2.putText(canvas, label, (wrist[0] - 25, wrist[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2, cv2.LINE_AA)


def map_to_screen(finger_pos, frame_w, frame_h):
    """Map finger position in camera frame to screen coordinates.
    Uses margins to create a comfortable active zone."""
    margin_x = frame_w * MARGIN_RATIO
    margin_y = frame_h * MARGIN_RATIO

    # Map with smooth clamping
    sx = np.interp(finger_pos[0],
                   (int(margin_x), int(frame_w - margin_x)),
                   (0, screen_w))
    sy = np.interp(finger_pos[1],
                   (int(margin_y), int(frame_h - margin_y)),
                   (0, screen_h))

    # Clamp to screen bounds
    sx = np.clip(sx, 0, screen_w - 1)
    sy = np.clip(sy, 0, screen_h - 1)
    return sx, sy


print("=" * 50)
print("  HAND CONTROLLER READY")
print("=" * 50)
print("  RIGHT hand → Move cursor (index finger)")
print("  LEFT hand  → Click / Scroll / Drag")
print("    • Quick pinch     = Left click")
print("    • Hold pinch 2.5s = Drag")
print("    • Middle+thumb    = Right click")
print("    • Peace sign      = Scroll mode")
print("  Press 'q' to quit")
print("=" * 50)

# ══════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════
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

    # ── Identify Left and Right hands ──
    right_hand_pts = None
    left_hand_pts = None
    right_hand_conf = 0
    left_hand_conf = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            confidence = handedness.classification[0].score
            pts = get_hand_points(hand_landmarks, frame_w, frame_h)

            if label == "Right" and confidence > right_hand_conf:
                right_hand_pts = pts
                right_hand_conf = confidence
                draw_hand(canvas, pts,
                          color_lines=(200, 130, 0),
                          color_dots=(255, 180, 50),
                          color_tips=(255, 255, 0),
                          label=f"R {confidence:.0%}",
                          label_color=(255, 200, 50))
            elif label == "Left" and confidence > left_hand_conf:
                left_hand_pts = pts
                left_hand_conf = confidence
                draw_hand(canvas, pts,
                          color_lines=(0, 100, 220),
                          color_dots=(0, 160, 255),
                          color_tips=(0, 255, 255),
                          label=f"L {confidence:.0%}",
                          label_color=(0, 220, 255))

    # ═══════════════════════════════════════════════════════
    #  RIGHT HAND → CURSOR MOVEMENT
    # ═══════════════════════════════════════════════════════
    if right_hand_pts:
        frames_without_right_hand = 0
        index_tip = right_hand_pts[INDEX_TIP]

        # Map finger position to screen
        raw_x, raw_y = map_to_screen(index_tip, frame_w, frame_h)

        # Add to position history for moving average
        cursor_history_x.append(raw_x)
        cursor_history_y.append(raw_y)

        # Step 1: Moving average over recent positions (removes noise)
        avg_x = np.mean(cursor_history_x)
        avg_y = np.mean(cursor_history_y)

        # Step 2: EMA smoothing on top (makes movement feel fluid)
        alpha = DRAG_EMA_ALPHA if is_dragging else EMA_ALPHA
        new_x = ema_smooth(avg_x, smooth_x, alpha)
        new_y = ema_smooth(avg_y, smooth_y, alpha)

        # Step 3: Jitter suppression — ignore tiny movements
        dx = abs(new_x - smooth_x)
        dy = abs(new_y - smooth_y)

        if dx > JITTER_THRESHOLD or dy > JITTER_THRESHOLD:
            smooth_x = new_x
            smooth_y = new_y
            pyautogui.moveTo(int(smooth_x), int(smooth_y))

        # Visual: draw cursor target on camera feed
        cv2.circle(canvas, index_tip, 10, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, index_tip, 3, (255, 255, 0), -1, cv2.LINE_AA)

    else:
        frames_without_right_hand += 1
        if frames_without_right_hand > HAND_LOST_FRAMES:
            cursor_history_x.clear()
            cursor_history_y.clear()

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

        # ── SCROLL MODE: Peace sign (index + middle up, close together, only 2 fingers) ──
        scroll_active = (l_is_index_up and l_is_middle_up
                         and l_index_middle_dist < SCROLL_FINGER_DIST
                         and l_fingers_up == 2)

        if scroll_active:
            avg_y = (l_index_tip[1] + l_middle_tip[1]) // 2

            # Set anchor point when scroll mode first activates
            if scroll_anchor_y is None:
                scroll_anchor_y = avg_y

            # Distance from anchor (positive = hand moved down, negative = up)
            offset = avg_y - scroll_anchor_y

            # Draw anchor line and zone indicator
            anchor_x = (l_index_tip[0] + l_middle_tip[0]) // 2
            cv2.line(canvas, (anchor_x - 40, scroll_anchor_y), (anchor_x + 40, scroll_anchor_y),
                     (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, (anchor_x, avg_y), 8, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.arrowedLine(canvas, (anchor_x, scroll_anchor_y), (anchor_x, avg_y),
                            (0, 255, 0) if offset < 0 else (0, 0, 255), 2, cv2.LINE_AA)

            if abs(offset) > SCROLL_DEAD_ZONE and current_time - last_scroll_time > SCROLL_COOLDOWN:
                # Calculate scroll speed based on distance from anchor (joystick style)
                effective_offset = abs(offset) - SCROLL_DEAD_ZONE
                speed = np.interp(effective_offset, (0, SCROLL_RAMP_DIST), (SCROLL_SPEED_MIN, SCROLL_SPEED_MAX))
                speed = int(speed)

                if offset < 0:
                    # Hand moved UP → scroll UP
                    pyautogui.scroll(speed)
                    cv2.putText(canvas, f"SCROLL UP ({speed})", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    # Hand moved DOWN → scroll DOWN
                    pyautogui.scroll(-speed)
                    cv2.putText(canvas, f"SCROLL DOWN ({speed})", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2, cv2.LINE_AA)

                last_scroll_time = current_time
            else:
                cv2.putText(canvas, "SCROLL READY", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

            # Release drag if active
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
                drag_fist_start = None

        # ── CLICK / DRAG MODE ──
        else:
            scroll_anchor_y = None

            pinching = l_index_thumb_dist < CLICK_THRESHOLD

            if is_dragging:
                if not pinching:
                    pyautogui.mouseUp()
                    is_dragging = False
                    drag_fist_start = None
                    cv2.putText(canvas, "DROPPED", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(canvas, "DRAGGING", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.circle(canvas, l_index_tip, 18, (0, 165, 255), 3, cv2.LINE_AA)

            elif pinching:
                if drag_fist_start is None:
                    drag_fist_start = current_time

                hold_duration = current_time - drag_fist_start
                remaining = DRAG_HOLD_TIME - hold_duration

                if hold_duration >= DRAG_HOLD_TIME:
                    pyautogui.mouseDown()
                    is_dragging = True
                    cv2.putText(canvas, "DRAG STARTED!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(canvas, f"HOLD: {remaining:.1f}s", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                    # Progress ring around pinch point
                    progress = hold_duration / DRAG_HOLD_TIME
                    angle = int(360 * progress)
                    cv2.ellipse(canvas, l_index_tip, (22, 22), 0, 0, angle,
                                (0, 165, 255), 3, cv2.LINE_AA)

            else:
                # Released pinch
                if drag_fist_start is not None:
                    hold_duration = current_time - drag_fist_start
                    if hold_duration < DRAG_HOLD_TIME:
                        if current_time - click_time > CLICK_COOLDOWN:
                            pyautogui.click()
                            click_time = current_time
                            cv2.putText(canvas, "LEFT CLICK", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    drag_fist_start = None

                # Right click: Middle + Thumb pinch
                if l_middle_thumb_dist < RIGHT_CLICK_THRESHOLD and l_is_middle_up:
                    if current_time - right_click_time > CLICK_COOLDOWN:
                        pyautogui.rightClick()
                        right_click_time = current_time
                        cv2.circle(canvas, l_middle_tip, 20, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.putText(canvas, "RIGHT CLICK", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw pinch distance indicator
        cv2.line(canvas, l_index_tip, l_thumb_tip, (100, 100, 100), 1, cv2.LINE_AA)
        mid_pt = ((l_index_tip[0] + l_thumb_tip[0]) // 2, (l_index_tip[1] + l_thumb_tip[1]) // 2)
        cv2.putText(canvas, f"{int(l_index_thumb_dist)}", mid_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    else:
        # No left hand → release drag
        if is_dragging:
            pyautogui.mouseUp()
            is_dragging = False
            drag_fist_start = None
        scroll_anchor_y = None

    # ── Status overlay ──
    r_status = f"R: Cursor ({right_hand_conf:.0%})" if right_hand_pts else "R: ---"
    l_status = f"L: Click ({left_hand_conf:.0%})" if left_hand_pts else "L: ---"

    # Semi-transparent status bar at bottom
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, frame_h - 45), (frame_w, frame_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    cv2.putText(canvas, f"{r_status}  |  {l_status}",
                (10, frame_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "R=Move | L: Pinch=Click, Hold=Drag, Peace=Scroll | 'q' quit",
                (10, frame_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.imshow('Hand Controller', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("Hand controller stopped.")
