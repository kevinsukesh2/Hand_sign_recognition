"""Phase 7 live webcam inference with an optional shadow clone effect."""

import argparse
import json

import cv2
import joblib
import torch

try:
    import mediapipe as mp
except ImportError:
    mp = None

from effects import apply_shadow_clone_effect
from model import GestureMLP
from utils import extract_two_hand_features, get_project_root


def parse_args():
    """Parse command-line arguments for live inference."""
    parser = argparse.ArgumentParser(
        description="Run live hand sign recognition using the trained gesture classifier."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for showing a predicted gesture. Default: 0.7",
    )
    parser.add_argument(
        "--disable-shadow-clone-effect",
        action="store_true",
        help="Disable the shadow clone visual effect.",
    )
    parser.add_argument(
        "--clone-count",
        type=int,
        default=4,
        help="Number of clone copies to draw when the effect is active. Default: 4",
    )
    parser.add_argument(
        "--clone-alpha",
        type=float,
        default=0.35,
        help="Transparency strength for the clone copies. Default: 0.35",
    )
    return parser.parse_args()


def draw_status_lines(frame, lines, start_y=30, color=(255, 255, 255)):
    """Draw multiple text lines on the webcam frame."""
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


def load_required_file(path):
    """Ensure a required file exists before loading it."""
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return path


def load_artifacts():
    """Load the trained model, label map, scaler, and model config."""
    project_root = get_project_root()
    model_dir = project_root / "models"

    model_path = load_required_file(model_dir / "gesture_classifier.pth")
    label_map_path = load_required_file(model_dir / "label_map.json")
    scaler_path = load_required_file(model_dir / "scaler.pkl")
    config_path = load_required_file(model_dir / "model_config.json")

    with label_map_path.open("r", encoding="utf-8") as file:
        label_map = json.load(file)

    with config_path.open("r", encoding="utf-8") as file:
        model_config = json.load(file)

    scaler = joblib.load(scaler_path)

    input_size = int(model_config["input_size"])
    num_classes = int(model_config["num_classes"])

    model = GestureMLP(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, scaler, label_map


def predict_gesture(model, scaler, label_map, features, threshold):
    """Run one prediction and return the raw label, display label, and confidence."""
    scaled_features = scaler.transform([features])
    feature_tensor = torch.tensor(scaled_features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(feature_tensor)
        probabilities = torch.softmax(logits, dim=1)

    confidence = probabilities.max(dim=1).values.item()
    predicted_index = probabilities.argmax(dim=1).item()
    predicted_label = label_map.get(str(predicted_index), "Unknown")

    if confidence < threshold:
        return predicted_label, "Unknown", confidence

    return predicted_label, predicted_label, confidence


def main():
    """Open the webcam and run live gesture inference."""
    args = parse_args()

    if args.threshold < 0 or args.threshold > 1:
        print("Error: --threshold must be between 0 and 1.")
        return

    if args.clone_count < 1:
        print("Error: --clone-count must be at least 1.")
        return

    if args.clone_alpha < 0 or args.clone_alpha > 1:
        print("Error: --clone-alpha must be between 0 and 1.")
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

    try:
        model, scaler, label_map = load_artifacts()
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return
    except Exception as error:
        print(f"Error: Failed to load model artifacts. {error}")
        return

    print("Loaded model and artifacts successfully.")
    print("Press q to quit.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    selfie_segmentation = None

    if not args.disable_shadow_clone_effect:
        try:
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        except AttributeError:
            print(
                "MediaPipe Selfie Segmentation is not available. "
                "Please check your mediapipe version."
            )
        except Exception as error:
            print(f"Warning: Selfie segmentation could not be started. {error}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the default webcam.")
        return

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read a frame from the webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb_frame)
            segmentation_mask = None

            # Selfie segmentation helps isolate the person so we can place
            # faded shifted copies behind them for the shadow clone effect.
            if selfie_segmentation is not None:
                segmentation_results = selfie_segmentation.process(rgb_frame)
                segmentation_mask = segmentation_results.segmentation_mask

            detected_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            if detected_hands == 0:
                gesture_text = "No hand detected"
                confidence_text = "--"
                raw_predicted_label = None
                confidence = 0.0
            else:
                features = extract_two_hand_features(hand_results)
                raw_predicted_label, display_label, confidence = predict_gesture(
                    model=model,
                    scaler=scaler,
                    label_map=label_map,
                    features=features,
                    threshold=args.threshold,
                )
                gesture_text = display_label
                confidence_text = f"{confidence * 100:.0f}%"

            effect_text = "None"
            effect_enabled = (
                not args.disable_shadow_clone_effect
                and selfie_segmentation is not None
                and raw_predicted_label == "shadow_clone"
                and confidence >= args.threshold
                and segmentation_mask is not None
            )

            if effect_enabled:
                frame = apply_shadow_clone_effect(
                    frame,
                    segmentation_mask,
                    clone_count=args.clone_count,
                    clone_alpha=args.clone_alpha,
                )
                effect_text = "Shadow Clone Jutsu"

            draw_status_lines(
                frame,
                [
                    "Phase 7: Live Gesture Inference + Effects",
                    f"Gesture: {gesture_text}",
                    f"Confidence: {confidence_text}",
                    f"Detected hands: {detected_hands}",
                    f"Effect: {effect_text}",
                    "Press q to quit",
                ],
                color=(0, 255, 0),
            )

            cv2.imshow("Hand Sign Recognition - Phase 7 Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hands.close()
        if selfie_segmentation is not None:
            selfie_segmentation.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
