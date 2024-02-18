# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.models.nn.autobackend import check_class_names
from src.utils import (
    LOGGER,
    emojis,
)
from src.utils.checks import check_font, is_ascii
from src.utils.ops import segments2boxes

HELP_URL = "See https://docs.ultralytics.com/datasets/detect for dataset formatting guidance."
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"  # video suffixes

def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    # sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}detection-object/labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in [6, 8]:  # rotation 270 or 90
                    s = s[1], s[0]
    return s


def verify_image(args):
    """Verify one image."""
    (im_file, cls), prefix = args
    # Number (found, corrupt), message
    nf, nc, msg = 0, 0, ""
    try:
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
    return (im_file, cls), nf, nc, msg


def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                # All labels
                max_cls = lb[:, 0].max()  # max label count
                assert max_cls <= num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int, optional): The color value to fill in the polygons on the mask. Defaults to 1.
        downsample_ratio (int, optional): Factor by which to downsample the mask. Defaults to 1.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Convert a list of polygons to a set of binary masks of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int): The color value to fill in the polygons on the masks.
        downsample_ratio (int, optional): Factor by which to downsample each mask. Defaults to 1.

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def find_dataset_yaml(path: Path) -> Path:
    """
    Find and return the YAML file associated with a Detect, Segment or Pose dataset.

    This function searches for a YAML file at the root level of the provided directory first, and if not found, it
    performs a recursive search. It prefers YAML files that have the same stem as the provided path. An AssertionError
    is raised if no YAML file is found or if multiple YAML files are found.

    Args:
        path (Path): The directory path to search for the YAML file.

    Returns:
        (Path): The path of the found YAML file.
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  # try root level first and then recursive
    assert files, f"No YAML file found in '{path.resolve()}'"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]  # prefer *.yaml files that match
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"
    return files[0]


def check_det_dataset(dataset):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.

    Returns:
        (dict): Parsed dataset information and paths.
    """

    # Read YAML
    data = dataset

    # Checks
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
                )
            LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
            data["val"] = data.pop("validation")  # replace 'validation' key with 'val' key
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"][0])

    data["names"] = check_class_names(data["names"][0])

    # Resolve paths
    path = Path(data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root

    # Set paths
    data["path"] = path  # download scripts
    for k in "train", "val", "test":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse YAML
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            exec(s, {"yaml": data})
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf")  # download fonts

    return data  # dictionary


def check_cls_dataset(dataset, split=""):
    """
    Checks a classification dataset such as Imagenet.

    This function accepts a `dataset` name and attempts to retrieve the corresponding dataset information.
    If the dataset is not found locally, it attempts to download the dataset from the internet and save it locally.

    Args:
        dataset (str | Path): The name of the dataset.
        split (str, optional): The split of the dataset. Either 'val', 'test', or ''. Defaults to ''.

    Returns:
        (dict): A dictionary containing the following keys:
            - 'train' (Path): The directory path containing the training set of the dataset.
            - 'val' (Path): The directory path containing the validation set of the dataset.
            - 'test' (Path): The directory path containing the test set of the dataset.
            - 'nc' (int): The number of classes in the dataset.
            - 'names' (dict): A dictionary of class names in the dataset.
    """

    dataset = Path(dataset)
    data_dir = dataset.resolve()

    train_set = data_dir / "train"
    val_set = (
        data_dir / "val"
        if (data_dir / "val").exists()
        else data_dir / "validation"
        if (data_dir / "validation").exists()
        else None
    )  # data/test or data/val
    test_set = data_dir / "test" if (data_dir / "test").exists() else None  # data/val or data/test
    if split == "val" and not val_set:
        LOGGER.warning("WARNING ⚠️ Dataset 'split=val' not found, using 'split=test' instead.")
    elif split == "test" and not test_set:
        LOGGER.warning("WARNING ⚠️ Dataset 'split=test' not found, using 'split=val' instead.")

    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])  # number of classes
    names = [x.name for x in (data_dir / "train").iterdir() if x.is_dir()]  # class names list
    names = dict(enumerate(sorted(names)))

    # Print to console
    for k, v in {"train": train_set, "val": val_set, "test": test_set}.items():
        prefix = f'{f"{k}:"} {v}...'
        if v is None:
            LOGGER.info(prefix)
        else:
            files = [path for path in v.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
            nf = len(files)  # number of files
            nd = len({file.parent for file in files})  # number of directories
            if nf == 0:
                if k == "train":
                    raise FileNotFoundError(emojis(f"{dataset} '{k}:' no training images found ❌ "))
                else:
                    LOGGER.warning(f"{prefix} found {nf} images in {nd} classes: WARNING ⚠️ no images found")
            elif nd != nc:
                LOGGER.warning(f"{prefix} found {nf} images in {nd} classes: ERROR ❌️ requires {nc} classes, not {nd}")
            else:
                LOGGER.info(f"{prefix} found {nf} images in {nd} classes ✅ ")

    return {"train": train_set, "val": val_set, "test": test_set, "nc": nc, "names": names}

def compress_one_image(f, f_new=None, max_dim=1920, quality=50):
    """
    Compresses a single image file to reduced size while preserving its aspect ratio and quality using either the Python
    Imaging Library (PIL) or OpenCV library. If the input image is smaller than the maximum dimension, it will not be
    resized.

    Args:
        f (str): The path to the input image file.
        f_new (str, optional): The path to the output image file. If not specified, the input file will be overwritten.
        max_dim (int, optional): The maximum dimension (width or height) of the output image. Default is 1920 pixels.
        quality (int, optional): The image compression quality as a percentage. Default is 50%.

    Example:
        ```python
        from pathlib import Path
        from ultralytics.data.utils import compress_one_image

        for f in Path('path/to/dataset').rglob('*.jpg'):
            compress_one_image(f)
        ```
    """

    try:  # use PIL
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, "JPEG", quality=quality, optimize=True)  # save
    except Exception as e:  # use OpenCV
        LOGGER.info(f"WARNING ⚠️ HUB ops PIL failure {f}: {e}")
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)  # ratio
        if r < 1.0:  # image too large
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)
