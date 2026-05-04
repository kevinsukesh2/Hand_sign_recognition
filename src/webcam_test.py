"""Phase 2 webcam test using OpenCV and MediaPipe Hands."""

import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None


def main():
    """Open the webcam and display detected hand landmarks."""
    if mp is None:
        print("Error: MediaPipe is not installed in the current environment.")
        return

    if not hasattr(mp, "solutions"):
        print(
            "Error: The installed MediaPipe version does not provide mp.solutions.\n"
            "Please install the project dependencies again after pinning MediaPipe "
            "to a compatible version."
        )
        return

    # Create short aliases for the MediaPipe helpers we will use.
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Open the default webcam.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the default webcam.")
        return

    # Set up MediaPipe Hands for up to two hands.
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    try:
        while True:
            # Read one frame from the webcam.
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read a frame from the webcam.")
                break

            # Flip the image so it feels more natural like a mirror.
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB because MediaPipe expects RGB input.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run hand detection on the current frame.
            results = hands.process(rgb_frame)

            detected_hands = 0

            # Draw landmarks and connections if any hands are found.
            if results.multi_hand_landmarks:
                detected_hands = len(results.multi_hand_landmarks)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Display helpful Phase 2 status text on the webcam window.
            cv2.putText(
                frame,
                "Phase 2: Webcam + MediaPipe Hands Test",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Detected hands: {detected_hands}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press q to quit",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Show the live webcam feed with any detected hand skeletons.
            cv2.imshow("Hand Sign Recognition - Phase 2 Test", frame)

            # Quit cleanly when the user presses q.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
