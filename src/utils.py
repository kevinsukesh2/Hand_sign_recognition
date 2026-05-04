"""Shared helper functions for landmark-based hand sign recognition."""

from pathlib import Path


HAND_LANDMARK_COUNT = 21
COORDS_PER_LANDMARK = 3
TOTAL_HANDS = 2
FEATURE_COUNT = TOTAL_HANDS * HAND_LANDMARK_COUNT * COORDS_PER_LANDMARK


def get_project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def flatten_hand_landmarks(hand_landmarks):
    """Convert one hand's landmarks into a flat x, y, z feature list."""
    flattened = []

    for landmark in hand_landmarks.landmark:
        flattened.extend([landmark.x, landmark.y, landmark.z])

    return flattened


def zero_hand_features():
    """Return a zero-filled feature vector for one missing hand."""
    return [0.0] * (HAND_LANDMARK_COUNT * COORDS_PER_LANDMARK)


def extract_two_hand_features(results):
    """Extract a fixed-length two-hand feature vector from MediaPipe results.

    The feature order matches the training data:
    - hand0 = Left
    - hand1 = Right
    If a hand is missing, that hand is filled with zeros.
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

        # If handedness is missing or a slot is already filled, fall back to detection order.
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

    return hand_features[0] + hand_features[1]
