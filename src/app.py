"""Load, reshape, and label the Dogs vs. Cats training photos."""

from os import listdir
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import img_to_array, load_img

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = Path(__file__).resolve().parent / "train"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGE_SIZE = (200, 200)
PHOTOS_PATH = PROCESSED_DIR / "dogs_vs_cats_photos.npy"
LABELS_PATH = PROCESSED_DIR / "dogs_vs_cats_labels.npy"


def label_from_filename(filename: str) -> float:
    """Return 1.0 for dog photos and 0.0 for cat photos."""
    name = Path(filename).name.lower()
    if name.startswith("dog"):
        return 1.0
    if name.startswith("cat"):
        return 0.0
    raise ValueError(f"Cannot determine label from filename: {filename}")


def list_training_filenames(train_dir: Path = TRAIN_DIR) -> list[str]:
    return sorted(
        filename
        for filename in listdir(train_dir)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def load_training_dataset(
    train_dir: Path = TRAIN_DIR,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load every training photo with Keras, reshape to 200x200, and label it."""
    filenames = list_training_filenames(train_dir)
    if not filenames:
        raise FileNotFoundError(f"No training images found in {train_dir}")

    photos = np.empty((len(filenames), image_size[0], image_size[1], 3), dtype=np.uint8)
    labels = np.empty((len(filenames),), dtype=np.float32)

    for index, filename in enumerate(filenames):
        labels[index] = label_from_filename(filename)
        image = load_img(train_dir / filename, target_size=image_size)
        photos[index] = img_to_array(image, dtype="uint8")
        if (index + 1) % 2500 == 0 or (index + 1) == len(filenames):
            print(f"Loaded {index + 1}/{len(filenames)} photos")

    return photos, labels


def save_dataset(
    photos: np.ndarray,
    labels: np.ndarray,
    photos_path: Path = PHOTOS_PATH,
    labels_path: Path = LABELS_PATH,
) -> tuple[Path, Path]:
    """Persist the (photos, labels) tuple as two NumPy files."""
    photos_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(photos_path, photos)
    np.save(labels_path, labels)
    return photos_path, labels_path


def load_saved_dataset(
    photos_path: Path = PHOTOS_PATH,
    labels_path: Path = LABELS_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Reload the saved (photos, labels) tuple."""
    return np.load(photos_path), np.load(labels_path)


def main() -> None:
    photos, labels = load_training_dataset()
    print(f"photos: {photos.shape} {photos.dtype}")
    print(f"labels: {labels.shape} {labels.dtype}")
    print(f"cats (0.0): {int(np.sum(labels == 0.0))}")
    print(f"dogs (1.0): {int(np.sum(labels == 1.0))}")

    photos_path, labels_path = save_dataset(photos, labels)
    print(f"Saved photos to {photos_path}")
    print(f"Saved labels to {labels_path}")


if __name__ == "__main__":
    main()
