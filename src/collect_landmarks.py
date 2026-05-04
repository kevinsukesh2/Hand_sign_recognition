"""Phase 3 script for collecting two-hand landmark samples."""

import argparse
import csv
import math
import time
from pathlib import Path

import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None


CSV_PATH = Path("data") / "gestures_landmarks.csv"
HAND_LANDMARK_COUNT = 21
COORDS_PER_LANDMARK = 3
TOTAL_HANDS = 2


def parse_args():
    """Parse command-line arguments for data collection."""
    parser = argparse.ArgumentParser(
        description="Collect two-hand MediaPipe landmark samples from the webcam."
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Gesture label to save for this collection session, for example Tiger.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=100,
        help="Collection duration in seconds. Default: 100",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Time between automatic samples in seconds. Default: 0.5",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=3,
        help="Countdown time before collection starts in seconds. Default: 3",
    )
    return parser.parse_args()


def create_csv_header():
    """Create the CSV header for label + two hands of landmarks."""
    header = ["label"]

    for hand_index in range(TOTAL_HANDS):
        for landmark_index in range(HAND_LANDMARK_COUNT):
            header.append(f"hand{hand_index}_x{landmark_index}")
            header.append(f"hand{hand_index}_y{landmark_index}")
            header.append(f"hand{hand_index}_z{landmark_index}")

    return header


def ensure_csv_file(csv_path):
    """Create the CSV file and header if the file does not exist yet."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(create_csv_header())


def flatten_hand_landmarks(hand_landmarks):
    """Convert one hand's 21 landmarks into a flat list of x, y, z values."""
    flattened = []

    for landmark in hand_landmarks.landmark:
        flattened.extend([landmark.x, landmark.y, landmark.z])

    return flattened


def zero_hand_features():
    """Return a zero-filled feature list for one missing hand."""
    return [0.0] * (HAND_LANDMARK_COUNT * COORDS_PER_LANDMARK)


def extract_two_hand_features(results):
    """Return a fixed-length feature vector for up to two hands.

    hand0 = Left
    hand1 = Right
    If a hand is missing, its features are filled with zeros.
    If no hands are detected, None is returned.
    """
    if not results.multi_hand_landmarks:
        return None

    hand_features = [None, None]

    handedness_list = results.multi_handedness or []

    for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        target_index = None

        if index < len(handedness_list) and handedness_list[index].classification:
            hand_label = handedness_list[index].classification[0].label.lower()
            if hand_label == "left":
                target_index = 0
            elif hand_label == "right":
                target_index = 1

        if target_index is None or hand_features[target_index] is not None:
            for fallback_index in range(TOTAL_HANDS):
                if hand_features[fallback_index] is None:
                    target_index = fallback_index
                    break

        if target_index is not None and hand_features[target_index] is None:
            hand_features[target_index] = flatten_hand_landmarks(hand_landmarks)

    for hand_index in range(TOTAL_HANDS):
        if hand_features[hand_index] is None:
            hand_features[hand_index] = zero_hand_features()

    features = hand_features[0] + hand_features[1]
    return features


def append_sample_to_csv(csv_path, label, features):
    """Append one labeled sample row to the CSV file."""
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([label] + features)


def draw_status_lines(frame, lines, start_y=30, color=(255, 255, 255)):
    """Draw multiple lines of helpful text on the frame."""
    for index, line in enumerate(lines):
        y = start_y + (index * 30)
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def main():
    """Run the webcam-based landmark collection workflow."""
    args = parse_args()

    if args.duration <= 0:
        print("Error: --duration must be greater than 0.")
        return

    if args.interval <= 0:
        print("Error: --interval must be greater than 0.")
        return

    if args.start_delay < 0:
        print("Error: --start-delay must be 0 or greater.")
        return

    if mp is None:
        print("Error: MediaPipe is not installed in the current environment.")
        return

    if not hasattr(mp, "solutions"):
        print(
            "Error: The installed MediaPipe version does not provide mp.solutions.\n"
            "Please reinstall the project dependencies from requirements.txt."
        )
        return

    ensure_csv_file(CSV_PATH)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the default webcam.")
        return

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    state = "WAITING"
    saved_samples = 0
    countdown_start_time = None
    collection_start_time = None
    next_capture_time = None
    done_start_time = None
    flash_message = ""
    flash_message_until = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            detected_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            now = time.monotonic()

            if state == "WAITING":
                draw_status_lines(
                    frame,
                    [
                        "Phase 3: Collect Landmarks",
                        f"Label: {args.label}",
                        f"Detected hands: {detected_hands}",
                        "Press s to start collection",
                        "Press q to quit",
                    ],
                    color=(255, 255, 255),
                )

            elif state == "COUNTDOWN":
                elapsed = now - countdown_start_time
                remaining = max(0.0, args.start_delay - elapsed)
                countdown_value = max(1, math.ceil(remaining)) if remaining > 0 else 0

                draw_status_lines(
                    frame,
                    [
                        "Get ready",
                        f"Starting in: {countdown_value}",
                    ],
                    color=(0, 255, 255),
                )

                if elapsed >= args.start_delay:
                    state = "COLLECTING"
                    collection_start_time = now
                    next_capture_time = now
                    print(f"Starting collection for label: {args.label}")

            elif state == "COLLECTING":
                elapsed = now - collection_start_time
                time_remaining = max(0.0, args.duration - elapsed)

                if now >= next_capture_time:
                    features = extract_two_hand_features(results)

                    if features is not None:
                        append_sample_to_csv(CSV_PATH, args.label, features)
                        saved_samples += 1
                        print(f"Saved sample {saved_samples} for label: {args.label}")
                        flash_message = "Sample captured"
                        flash_message_until = now + 0.6
                    else:
                        flash_message = "No hand detected - sample will be skipped"
                        flash_message_until = now + 0.6

                    next_capture_time += args.interval

                draw_status_lines(
                    frame,
                    [
                        "Phase 3: Auto Collecting",
                        f"Label: {args.label}",
                        f"Detected hands: {detected_hands}",
                        f"Saved samples this session: {saved_samples}",
                        f"Time remaining: {time_remaining:.1f}s",
                        f"Capture interval: {args.interval}s",
                        "Press q to quit early",
                    ],
                    color=(0, 255, 0),
                )

                if detected_hands == 0:
                    draw_status_lines(
                        frame,
                        ["No hand detected - sample will be skipped"],
                        start_y=250,
                        color=(0, 0, 255),
                    )

                if elapsed >= args.duration:
                    state = "DONE"
                    done_start_time = now
                    print("Collection complete.")
                    print(f"Label: {args.label}")
                    print(f"Saved samples this session: {saved_samples}")
                    print(f"CSV path: {CSV_PATH}")

            elif state == "DONE":
                draw_status_lines(
                    frame,
                    [
                        "Collection complete",
                        f"Label: {args.label}",
                        f"Saved samples this session: {saved_samples}",
                        "Press q to quit",
                    ],
                    color=(0, 255, 0),
                )

                if now - done_start_time >= 2:
                    break

            if flash_message and now <= flash_message_until:
                draw_status_lines(
                    frame,
                    [flash_message],
                    start_y=280,
                    color=(255, 255, 0),
                )

            cv2.imshow("Hand Sign Recognition - Phase 3 Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if state == "COLLECTING":
                    print("Collection complete.")
                    print(f"Label: {args.label}")
                    print(f"Saved samples this session: {saved_samples}")
                    print(f"CSV path: {CSV_PATH}")
                break

            if state == "WAITING" and key == ord("s"):
                state = "COUNTDOWN"
                countdown_start_time = time.monotonic()

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
