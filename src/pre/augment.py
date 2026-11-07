import random
import numpy as np
from PIL import Image


def random_scale(img_np: np.ndarray, boxes: np.ndarray, T: int = 640, scale_range: tuple = (0.75, 1.25)):
    """
    Randomly scales the image and bounding boxes to bridge extreme drone altitudes.
    """
    if random.random() > 0.5:
        return img_np, boxes

    scale = random.uniform(*scale_range)
    new_T = int(round(T * scale))
    im = Image.fromarray(img_np).resize((new_T, new_T), Image.Resampling.BILINEAR)
    scaled_img = np.array(im)

    scaled_boxes = boxes.copy()
    if len(scaled_boxes) > 0:
        scaled_boxes[:, :4] *= scale

    if new_T > T:
        max_x = new_T - T
        max_y = new_T - T
        cx = random.randint(0, max_x)
        cy = random.randint(0, max_y)
        cropped_img = scaled_img[cy : cy + T, cx : cx + T]

        if len(scaled_boxes) > 0:
            scaled_boxes[:, [0, 2]] -= cx
            scaled_boxes[:, [1, 3]] -= cy
            scaled_boxes[:, [0, 2]] = np.clip(scaled_boxes[:, [0, 2]], 0, T)
            scaled_boxes[:, [1, 3]] = np.clip(scaled_boxes[:, [1, 3]], 0, T)
            valid = (scaled_boxes[:, 2] - scaled_boxes[:, 0] > 2) & (
                scaled_boxes[:, 3] - scaled_boxes[:, 1] > 2
            )
            scaled_boxes = scaled_boxes[valid]
        return cropped_img, scaled_boxes
    elif new_T < T:
        pad_x = random.randint(0, T - new_T)
        pad_y = random.randint(0, T - new_T)
        padded_img = np.zeros((T, T, 3), dtype=np.uint8)
        padded_img[pad_y : pad_y + new_T, pad_x : pad_x + new_T] = scaled_img

        if len(scaled_boxes) > 0:
            scaled_boxes[:, [0, 2]] += pad_x
            scaled_boxes[:, [1, 3]] += pad_y
        return padded_img, scaled_boxes

    return scaled_img, scaled_boxes


def augment_dataset(image: np.ndarray, boxes: np.ndarray, T: int = 640):
    """
    Data augmentation for aerial drone tiles:
    1. Multi-scale resizing & cropping
    2. Horizontal & vertical flips
    3. 90-degree rotations
    4. Brightness & contrast jitter
    """
    # 1. Multi-scale resizing
    image, boxes = random_scale(image, boxes, T=T, scale_range=(0.75, 1.25))

    # 2. Horizontal flip
    if random.random() < 0.5:
        image = image[:, ::-1]
        if len(boxes) > 0:
            boxes[:, [0, 2]] = T - boxes[:, [2, 0]]

    # 3. Vertical flip
    if random.random() < 0.2:
        image = image[::-1, :]
        if len(boxes) > 0:
            boxes[:, [1, 3]] = T - boxes[:, [3, 1]]

    # 4. 90-degree rotation
    k = random.randint(0, 3)
    for _ in range(k):
        image = np.rot90(image)
        if len(boxes) > 0:
            # (x, y) becomes (y, T - x)
            boxes = np.stack(
                [boxes[:, 1], T - boxes[:, 2], boxes[:, 3], T - boxes[:, 0]], axis=1
            )
    if k > 0:
        image = np.ascontiguousarray(image)

    # 5. Brightness & contrast jitter
    scale = random.uniform(0.8, 1.2)
    shift = random.uniform(-20, 20)
    image = image.astype(np.float32) * scale + shift
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    return image, boxes
