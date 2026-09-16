"""Download raw MNIST and prepare 12x12 downsampled datasets including 0-9."""

import gzip
import struct
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "juku" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_if_needed(name, filename):
    dest = DATA_DIR / f".raw_{filename}"
    if not dest.exists():
        url = MIRROR + filename
        print(f"Downloading {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        with open(dest, "wb") as f:
            f.write(content)
        print(f"Saved {dest} ({len(content)} bytes)")
    return dest


def parse_images(gz_path):
    with gzip.open(gz_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad magic: {magic}"
        buf = f.read(num * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
    return data


def parse_labels(gz_path):
    with gzip.open(gz_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad magic: {magic}"
        buf = f.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


def downsample_and_save(images, labels, out_path):
    num = len(labels)
    rec_len = 1 + 12 * 12
    out_buf = bytearray(num * rec_len)

    print(f"Downsampling {num} images to 12x12 ...")
    for i in range(num):
        label = int(labels[i])
        img = Image.fromarray(images[i])
        # Use BOX resampling (exact area averaging) matching standard downsampling
        img_12 = img.resize((12, 12), Image.Resampling.BOX)
        pixels = np.array(img_12, dtype=np.uint8).flatten()

        offset = i * rec_len
        out_buf[offset] = label
        out_buf[offset + 1 : offset + rec_len] = pixels.tobytes()

    with gzip.open(out_path, "wb", compresslevel=6) as f:
        f.write(out_buf)
    print(f"Saved {num} records to {out_path} ({len(out_buf)} raw bytes, {out_path.stat().st_size} gz bytes)")


def main():
    train_img_gz = download_if_needed("train_images", FILES["train_images"])
    train_lbl_gz = download_if_needed("train_labels", FILES["train_labels"])
    test_img_gz = download_if_needed("test_images", FILES["test_images"])
    test_lbl_gz = download_if_needed("test_labels", FILES["test_labels"])

    train_images = parse_images(train_img_gz)
    train_labels = parse_labels(train_lbl_gz)
    test_images = parse_images(test_img_gz)
    test_labels = parse_labels(test_lbl_gz)

    print(f"Train: {len(train_labels)} images, digits: {set(train_labels.tolist())}")
    print(f"Test: {len(test_labels)} images, digits: {set(test_labels.tolist())}")

    downsample_and_save(train_images, train_labels, DATA_DIR / "mnist12_0to9_train.bin.gz")
    downsample_and_save(test_images, test_labels, DATA_DIR / "mnist12_0to9_test.bin.gz")
    print("Done!")


if __name__ == "__main__":
    main()
