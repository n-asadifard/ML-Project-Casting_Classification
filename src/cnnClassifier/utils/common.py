import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

import math
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             fbeta_score, precision_score, recall_score, roc_auc_score)


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())


############################################################
@keras.saving.register_keras_serializable(package="casting")
class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup to peak_lr, then cosine decay to final_lr."""

    def __init__(self, peak_lr: float, total_steps: int, warmup_steps: int, final_lr: float = 1e-6):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.final_lr = float(final_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        warm_lr = self.peak_lr * (step + 1.0) / tf.maximum(1.0, warmup)
        progress = tf.clip_by_value((step - warmup) / tf.maximum(1.0, total - warmup), 0.0, 1.0)
        cos_lr = self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < warmup, warm_lr, cos_lr)

    def get_config(self):
        return dict(peak_lr=self.peak_lr, total_steps=self.total_steps,
                    warmup_steps=self.warmup_steps, final_lr=self.final_lr)



def compute_class_weights(df: pd.DataFrame, label_col: str = "label") -> dict:
    """Inverse-frequency class weights: total / (n_classes * count)."""
    counts = df[label_col].value_counts()
    return {int(c): len(df) / (len(counts) * n) for c, n in counts.items()}


def save_model(path: Path, model: keras.Model):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)    # path must end with .keras


def parse_manifest_line(line):
    # manifest.columns = [filepath, class_name, label]
    filepath, _, label = tf.io.decode_csv(line, record_defaults=["", "", 0])    # 0 defualt value for missing elements
    return filepath, label


def load_image(filepath, img_size):
    """Decode -> grayscale -> resize -> [0, 1]. Shared by training, evaluation and inference."""
    img = tf.io.decode_image(tf.io.read_file(filepath), channels=1, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((*img_size, 1))
    return img


def load_and_preprocess(filepath, label, img_size):
    return load_image(filepath, img_size), tf.cast(label, tf.float32)


def make_dataset(manifest_path, img_size, batch_size: int,
                 training: bool = False, augmenter=None, seed: int = 42) -> tf.data.Dataset:
    """One pipeline for every split. Only training=True shuffles and augments."""
    if augmenter is not None and not training:
        raise ValueError("An augmenter is only allowed for the training split (training=True)")

    manifest_path = Path(manifest_path)
    n = len(pd.read_csv(manifest_path))
    img_size = tuple(int(s) for s in img_size)

    ds = (tf.data.TextLineDataset(str(manifest_path)).skip(1)
          .map(parse_manifest_line, num_parallel_calls=tf.data.AUTOTUNE)
          .map(lambda p, y: load_and_preprocess(p, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
          .cache())
    if training:
        ds = ds.shuffle(n, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and augmenter is not None:
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 1.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def set_trainable(model: keras.Model, freeze_all: bool, n_unfreeze: int | None = None) -> None:
    """Freeze the whole base, or unfreeze only its top n layers (BatchNorm always stays frozen)."""
    if freeze_all:
        if n_unfreeze:
            raise ValueError("freeze_all=True cannot be combined with n_unfreeze")
        model.trainable = False
        return
    if n_unfreeze is None or n_unfreeze <= 0:
        raise ValueError("Either set freeze_all=True or pass n_unfreeze > 0")
    if n_unfreeze > len(model.layers):
        raise ValueError(f"n_unfreeze={n_unfreeze} exceeds the {len(model.layers)} layers of the base model")

    model.trainable = True
    for layer in model.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in model.layers[-n_unfreeze:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def plot_history(history, title: str = "", metrics=("loss", "auc", "recall", "precision")) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.2))
    for ax, m in zip(axes, metrics):
        ax.plot(history.history[m], label="train")
        ax.plot(history.history[f"val_{m}"], label="val")
        ax.set_title(m)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def make_eval_dataset(manifest_path: Path, img_size, batch_size: int) -> tf.data.Dataset:
    """Deterministic pipeline (no shuffle, no augmentation): row i of the output = row i of the manifest."""
    return (tf.data.TextLineDataset(str(manifest_path)).skip(1)
            .map(parse_manifest_line, num_parallel_calls=tf.data.AUTOTUNE)
            .map(lambda p, y: load_and_preprocess(p, y, tuple(img_size)),
                num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))


def choose_threshold(y_true, y_prob, beta: float = 2.0, grid=np.linspace(0.05, 0.95, 91)):
    """Threshold maximizing F-beta; on ties, the middle of the best plateau."""
    scores = np.array([fbeta_score(y_true, y_prob >= t, beta=beta, zero_division=0) for t in grid])
    best_ts = grid[scores >= scores.max() - 1e-9]
    return float(np.median(best_ts)), float(scores.max()), (float(best_ts.min()), float(best_ts.max()))


def binary_metrics(y_true, y_prob, threshold: float, beta: float = 2.0) -> dict:
    """All metrics as plain Python numbers (JSON-serializable)."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "n": int(len(y_true)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fbeta": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }