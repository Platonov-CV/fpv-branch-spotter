import tensorflow as tf
from pathlib import Path
import random


MODULE_DIR = Path(__file__).resolve().parent
FRAME_ROOT = MODULE_DIR / "../data/raw/rugd/frames"
MASK_ROOT = MODULE_DIR  / "../data/processed/rugd/binary_annotations"

IMG_SIZE = (224, 224)

BATCH_SIZE = 8


def _collect_pairs(video_list):
    images, masks = [], []

    for video in video_list:
        frame_dir = FRAME_ROOT / video
        mask_dir  = MASK_ROOT / video

        for img_path in frame_dir.glob("*.png"):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                images.append(str(img_path))
                masks.append(str(mask_path))

    return images, masks


def _load_pair(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
    mask = tf.cast(mask > 0, tf.float32)

    return img, mask


def assemble():
    videos = sorted([v.name for v in FRAME_ROOT.iterdir() if v.is_dir()])
    random.shuffle(videos)

    train_ratio = 0.7
    val_ratio = 0.15

    n = len(videos)
    train_videos = videos[:int(n * train_ratio)]
    val_videos = videos[int(n * train_ratio):int(n * (train_ratio + val_ratio))]
    test_videos = videos[int(n * (train_ratio + val_ratio)):]

    train_imgs, train_masks = _collect_pairs(train_videos)
    val_imgs, val_masks = _collect_pairs(val_videos)
    test_imgs, test_masks = _collect_pairs(test_videos)

    train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_masks))
    val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_masks))
    test_ds = tf.data.Dataset.from_tensor_slices((test_imgs, test_masks))

    train_ds = (
        train_ds
        .cache()
        .shuffle(7500)
        .map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds
        .cache()
        .map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds
        .cache()
        .map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
    )

    return train_ds, val_ds, test_ds
