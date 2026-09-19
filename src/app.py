"""Load, reshape, and label the Dogs vs. Cats training photos."""

from os import listdir, link
from pathlib import Path
from random import random, seed
from shutil import copyfile

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "raw" / "train"
PROCESSED_DIR = ROOT / "data" / "processed"
DATASET_HOME = ROOT / "data" / "interim" / "dataset_dogs_vs_cats"
IMAGE_SIZE = (200, 200)
PHOTOS_PATH = PROCESSED_DIR / "dogs_vs_cats_photos.npy"
LABELS_PATH = PROCESSED_DIR / "dogs_vs_cats_labels.npy"
TEST_RATIO = 0.25
SPLIT_SEED = 1


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


def _place_image(src: Path, dst: Path) -> None:
    """Link an image into the class folder without opening its pixels."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        link(src, dst)
    except OSError:
        copyfile(src, dst)


def organize_dataset(
    train_dir: Path = TRAIN_DIR,
    dataset_home: Path = DATASET_HOME,
    test_ratio: float = TEST_RATIO,
    split_seed: int = SPLIT_SEED,
) -> Path:
    """Copy filenames into train/test class folders for flow_from_directory."""
    for split in ("train", "test"):
        for label in ("cats", "dogs"):
            (dataset_home / split / label).mkdir(parents=True, exist_ok=True)

    seed(split_seed)
    for filename in list_training_filenames(train_dir):
        split = "test" if random() < test_ratio else "train"
        if filename.startswith("cat"):
            label = "cats"
        elif filename.startswith("dog"):
            label = "dogs"
        else:
            continue
        _place_image(train_dir / filename, dataset_home / split / label / filename)

    return dataset_home


def create_data_generators(
    dataset_home: Path = DATASET_HOME,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = 64,
) -> tuple[ImageDataGenerator, object, ImageDataGenerator, object]:
    """Create trdata/tsdata and pass them the train and test folders."""
    train_dir = dataset_home / "train"
    test_dir = dataset_home / "test"

    trdata = ImageDataGenerator(rescale=1.0 / 255.0)
    traindata = trdata.flow_from_directory(
        directory=str(train_dir),
        target_size=image_size,
        class_mode="binary",
        batch_size=batch_size,
    )

    tsdata = ImageDataGenerator(rescale=1.0 / 255.0)
    testdata = tsdata.flow_from_directory(
        directory=str(test_dir),
        target_size=image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
    )

    return trdata, traindata, tsdata, testdata


def main() -> None:
    dataset_home = organize_dataset()
    print(f"Organized class folders under {dataset_home}")

    trdata, traindata, tsdata, testdata = create_data_generators()
    print(f"trdata: {type(trdata).__name__}")
    print(f"tsdata: {type(tsdata).__name__}")
    print(f"traindata: {traindata.samples} images, classes={traindata.class_indices}")
    print(f"testdata: {testdata.samples} images, classes={testdata.class_indices}")


if __name__ == "__main__":
    main()
