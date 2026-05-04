"""Visual effects used by the live webcam inference script."""

import cv2
import numpy as np


def shift_image(image, x_offset, y_offset=0):
    """Shift an image by the given x and y offsets."""
    height, width = image.shape[:2]
    transform = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    return cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def apply_shadow_clone_effect(frame, segmentation_mask, clone_count=4, clone_alpha=0.35):
    """Create faded clone copies of the segmented person behind the original.

    Selfie segmentation gives us a mask for the person in the frame. We use
    that mask to isolate the person, shift several copies left and right, and
    blend them back with partial transparency to create a simple shadow clone effect.
    """
    if segmentation_mask is None:
        return frame

    # Turn the soft segmentation output into a smooth foreground mask.
    mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    mask_float = mask.astype(np.float32) / 255.0
    person = (frame.astype(np.float32) * mask_float[..., None]).astype(np.uint8)

    overlay = frame.astype(np.float32).copy()

    base_offsets = [-160, -100, 100, 160]
    offsets = base_offsets[:clone_count]

    if clone_count > len(base_offsets):
        extra_needed = clone_count - len(base_offsets)
        step = 70
        for index in range(extra_needed):
            distance = 220 + (index // 2) * step
            direction = -1 if index % 2 == 0 else 1
            offsets.append(direction * distance)

    for index, x_offset in enumerate(offsets):
        shifted_person = shift_image(person, x_offset)
        shifted_mask = shift_image(mask_float, x_offset)

        # Slightly vary alpha so the clones look layered instead of identical.
        alpha = max(0.15, clone_alpha - (index * 0.03))
        alpha_mask = np.clip(shifted_mask * alpha, 0.0, 1.0)[..., None]

        overlay = (shifted_person.astype(np.float32) * alpha_mask) + (overlay * (1.0 - alpha_mask))

    # Keep the original person crisp on top while the faded clones remain visible
    # mainly around and behind the segmented person.
    foreground_mask = mask_float[..., None]
    final_frame = (frame.astype(np.float32) * foreground_mask) + (overlay * (1.0 - foreground_mask))

    return np.clip(final_frame, 0, 255).astype(np.uint8)
