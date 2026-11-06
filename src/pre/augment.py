import random
import numpy as np


# Flips Horizontal and Vertical
# 90* rotation
# Color jitter
def augment_dataset(image: np.ndarray, boxes: np.ndarray, T: int = 640):
    """
    image: NumPy array
    boxes: NumPy array
    T: tile dimension
    """

    # Flips Horizontal
    if random.random() < 0.5:
        image = image[:, ::-1]
        if len(boxes) > 0:
            boxes[:, [0, 2]] = T - boxes[:, [2, 0]]

    # Vertical flipping.
    # Usually less chance, 20% because, Vertical is unsual in drone based visual
    # people could appear less upside down
    if random.random() < 0.2:
        image = image[::-1, :]
        if len(boxes) > 0:
            boxes[:, [1, 3]] = T - boxes[:, [3, 1]]

    # 90-degree rotation
    k = random.randint(0, 3)
    for _ in range(k):
        image = np.rot90(image)
        if len(boxes) > 0:
            # (x, y) becomes (y, T - x)
            boxes = np.stack(
                [boxes[:, 1], T - boxes[:, 2], boxes[:, 3], T - boxes[:, 0]], axis=1
            )

    # 4. Brightness & Contrast Jitter
    scale = random.uniform(0.8, 1.2)
    shift = random.uniform(-20, 20)
    image = image.astype(np.float32) * scale + shift
    image = np.clip(image, 0.0, 255.0)

    return image, boxes
