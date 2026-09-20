"""Load, reshape, and label the Dogs vs. Cats training photos."""

from os import listdir, link
from pathlib import Path
from random import random, seed
from shutil import copyfile

import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "raw" / "train"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DATASET_HOME = ROOT / "data" / "interim" / "dataset_dogs_vs_cats"
IMAGE_SIZE = (200, 200)
MODEL_IMAGE_SIZE = (224, 224)
PHOTOS_PATH = PROCESSED_DIR / "dogs_vs_cats_photos.npy"
LABELS_PATH = PROCESSED_DIR / "dogs_vs_cats_labels.npy"
MODEL_PATH = MODELS_DIR / "efficientnet_b0_224-128-2.keras"
TEST_RATIO = 0.25
SPLIT_SEED = 1
EPOCHS = 5
BATCH_SIZE = 32


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
    class_mode: str = "binary",
    rescale: float | None = 1.0 / 255.0,
) -> tuple[ImageDataGenerator, object, ImageDataGenerator, object]:
    """Create trdata/tsdata and pass them the train and test folders."""
    train_dir = dataset_home / "train"
    test_dir = dataset_home / "test"
    generator_kwargs = {} if rescale is None else {"rescale": rescale}

    trdata = ImageDataGenerator(**generator_kwargs)
    traindata = trdata.flow_from_directory(
        directory=str(train_dir),
        target_size=image_size,
        class_mode=class_mode,
        batch_size=batch_size,
    )

    tsdata = ImageDataGenerator(**generator_kwargs)
    testdata = tsdata.flow_from_directory(
        directory=str(test_dir),
        target_size=image_size,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=False,
    )

    return trdata, traindata, tsdata, testdata


def build_model(input_shape: tuple[int, int, int] = (*MODEL_IMAGE_SIZE, 3)) -> Sequential:
    """EfficientNet-B0 backbone plus dense layers for dog/cat classification."""
    model = Sequential()
    model.add(EfficientNetB0(include_top=False, weights=None, input_shape=input_shape))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    return model


def compile_model(model: Sequential) -> Sequential:
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def make_callbacks(model_path: Path = MODEL_PATH) -> list:
    """ModelCheckpoint + EarlyStopping for fit (replaces deprecated fit_generator)."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        verbose=1,
    )
    early = EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )
    return [checkpoint, early]


def train_model(
    model: Sequential,
    traindata,
    testdata,
    epochs: int = EPOCHS,
) -> object:
    return model.fit(
        traindata,
        validation_data=testdata,
        epochs=epochs,
        callbacks=make_callbacks(),
    )


def evaluate_model(model, testdata) -> tuple[float, float]:
    loss, accuracy = model.evaluate(testdata, verbose=1)
    return float(loss), float(accuracy)


def save_model(model, model_path: Path = MODEL_PATH) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model_path


def load_best_model(model_path: Path = MODEL_PATH):
    """Reload the checkpointed best model from the models folder."""
    return load_model(model_path)


def predict_test_set(model, testdata) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict class probabilities on the test generator (no image display)."""
    probabilities = model.predict(testdata)
    y_pred = probabilities.argmax(axis=1)
    y_true = np.array(testdata.classes)
    accuracy = float((y_pred == y_true).mean())
    return probabilities, y_pred, accuracy


def prepare_generators():
    organize_dataset()
    return create_data_generators(
        image_size=MODEL_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        rescale=None,
    )


def main() -> None:
    trdata, traindata, tsdata, testdata = prepare_generators()
    print(f"trdata: {type(trdata).__name__}")
    print(f"tsdata: {type(tsdata).__name__}")
    print(f"traindata: {traindata.samples} images, classes={traindata.class_indices}")
    print(f"testdata: {testdata.samples} images, classes={testdata.class_indices}")

    if MODEL_PATH.exists():
        print(f"Found saved model at {MODEL_PATH}; skipping retraining")
    else:
        model = compile_model(build_model())
        model.summary()
        history = train_model(model, traindata, testdata)
        print(f"best val_accuracy: {max(history.history['val_accuracy']):.4f}")
        save_model(model)

    best_model = load_best_model()
    loss, accuracy = evaluate_model(best_model, testdata)
    _, y_pred, predict_accuracy = predict_test_set(best_model, testdata)
    print(f"test loss: {loss:.4f}")
    print(f"test accuracy: {accuracy:.4f}")
    print(f"predict accuracy: {predict_accuracy:.4f}")
    print(f"predicted cats: {int((y_pred == 0).sum())}")
    print(f"predicted dogs: {int((y_pred == 1).sum())}")
    print(f"Best model stored at {MODEL_PATH}")


if __name__ == "__main__":
    main()
