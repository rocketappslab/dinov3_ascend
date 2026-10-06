import csv
import json
import logging
import os
from typing import Any, Callable, List, Optional, Sequence, Union

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_MANIFEST_EXTENSIONS = {".csv", ".json"}
# Earlier names win when a record contains more than one folder or path field.
_ROOT_KEYS = (
    "image_folder_root",
    "image_folder",
    "image_root",
    "img_root",
    "folder_root",
    "root",
)
_PATH_KEYS = (
    "image_relative_path",
    "relative_path",
    "rel_path",
    "relpath",
    "image_relpath",
    "path",
    "image_path",
    "img_path",
    "filepath",
    "file_path",
    "filename",
    "file_name",
    "image",
    "img",
    "file",
)
_ROOT_KEY_RANK = {name: index for index, name in enumerate(_ROOT_KEYS)}
_PATH_KEY_RANK = {name: index for index, name in enumerate(_PATH_KEYS)}
_ROOT_KEY_SET = set(_ROOT_KEYS)


class ImageDir(ExtendedVisionDataset):
    """Unlabeled images from a directory or CSV/JSON path lists.

    Directory mode walks ``root``, including nested folders. For a large
    collection, pass one or more manifests so paths are read from the files
    instead of scanned:

        ImageDir:root=/data/images:extra=/data/a.csv
        ImageDir:root=/data/images:extra=/data/a.csv,/data/b.json
        ImageDir:root=/data/images:extra=/data/a.csv:extra=/data/b.json
        ImageDir:root=/data/a.csv,/data/b.json
        ImageDir:root=/data/images:extra=/data/lists

    ``extra`` may be a ``.csv`` or ``.json`` file, a comma-separated list of
    those files, or a directory of them. Paths are concatenated in the order
    given. A JSON object may contain several lists; every list is loaded.
    Each record can store an image folder and a path relative to that folder.
    CSV columns (or JSON fields) are ``root`` / ``image_root`` /
    ``image_folder`` and ``path`` / ``relative_path`` / ``image``. A JSON
    object may also set one folder for a list of relative paths:

        {"root": "/data/images", "images": ["a.jpg", "nested/b.jpg"]}
        [{"image_root": "/data/images", "relative_path": "a.jpg"}]

    A relative path with no folder of its own is resolved against ``root``
    when ``root`` is a directory, otherwise against a sibling folder of the
    same name (``demo.csv`` next to ``demo/``). An absolute path is used as
    written. A folder stored on the record wins over the dataset directory.

    ``image_root`` is the image folder joined with every relative path. It
    wins over a folder stored on a record. One folder applies to every list.
    Several lists can each use their own folder: pass the folders in the same
    order as the lists.

        ImageDir:root=/data/list.csv:image_root=/data/images
        ImageDir:root=/data/a.csv,/data/b.csv:image_root=/data/images_a,/data/images_b

    Write that CSV from a folder with ``write_image_list`` or:

        python -m dinov3.data.datasets.image_dir /root/workspace/rocket/Dataset/demo
    """

    def __init__(
        self,
        *,
        root: str,
        extra: Optional[Union[str, Sequence[str]]] = None,
        image_root: Optional[str] = None,
        split: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        del split
        paths, source = _collect_paths(root, extra, image_root)
        if not paths:
            raise FileNotFoundError(f"No images found from {source}")
        logger.info("ImageDir loaded %s paths from %s", f"{len(paths):,d}", source)
        self._paths = paths

    def get_image_data(self, index: int) -> bytes:
        with open(self._paths[index], "rb") as handle:
            return handle.read()

    def get_target(self, index: int) -> int:
        return 0

    def __len__(self) -> int:
        return len(self._paths)


def _image_roots(image_root: Optional[str]) -> List[str]:
    if image_root is None or not str(image_root).strip():
        return []
    roots = _split_spec(str(image_root))
    for folder in roots:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Image folder root not found: {folder}")
    return roots


def _roots_for_manifests(image_root: Optional[str], count: int) -> List[Optional[str]]:
    """Pair image folders with lists.

    One folder is used for every list. Several folders must line up with the
    lists in the same order.
    """
    roots = _image_roots(image_root)
    if not roots:
        return [None] * count
    if len(roots) == 1:
        return roots * count
    if len(roots) != count:
        raise ValueError(
            f"Got {len(roots)} image folder roots for {count} lists; "
            "pass one folder per list, in the same order, or a single folder for all lists"
        )
    return list(roots)


def _collect_paths(
    root: str,
    extra: Optional[Union[str, Sequence[str]]],
    image_root: Optional[str] = None,
) -> tuple:
    specs = _manifest_specs(extra)
    if specs:
        return _load_manifests(root, specs, image_root), ",".join(specs)
    root_specs = _split_spec(root)
    if len(root_specs) > 1 or (root_specs and not os.path.isdir(root_specs[0]) and _looks_like_manifest(root_specs[0])):
        return _load_manifests(None, root_specs, image_root), root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Image directory not found: {root}")
    if len(_image_roots(image_root)) > 1:
        raise ValueError("Multiple image folder roots need one list per folder")
    return _walk_images(root), root


def _manifest_specs(extra: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if extra is None:
        return []
    if isinstance(extra, str):
        return _split_spec(extra)
    specs = []
    for item in extra:
        specs.extend(_split_spec(item))
    return specs


def _split_spec(spec: str) -> List[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def _looks_like_manifest(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _MANIFEST_EXTENSIONS or os.path.isdir(path)


def _load_manifests(root: Optional[str], specs: Sequence[str], image_root: Optional[str] = None) -> List[str]:
    dataset_root = root if root and os.path.isdir(root) else None
    manifests: List[str] = []
    for spec in specs:
        manifests.extend(_expand_manifest_input(spec))
    folders = _roots_for_manifests(image_root, len(manifests))
    paths: List[str] = []
    for manifest, folder in zip(manifests, folders):
        paths.extend(_load_manifest(manifest, dataset_root, folder))
    return paths


def _expand_manifest_input(spec: str) -> List[str]:
    if os.path.isdir(spec):
        found = []
        for dirpath, _, filenames in os.walk(spec):
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                if _is_manifest(path):
                    found.append(path)
        if not found:
            raise FileNotFoundError(f"No .csv or .json lists found under {spec}")
        found.sort()
        return found
    if not os.path.isfile(spec):
        raise FileNotFoundError(f"Image manifest not found: {spec}")
    if not _is_manifest(spec):
        raise ValueError(f"Image manifest must be .csv or .json, got {spec}")
    return [spec]


def _is_manifest(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in _MANIFEST_EXTENSIONS


def _fallback_base_dir(manifest: str) -> str:
    directory = os.path.dirname(os.path.abspath(manifest))
    sibling = os.path.join(directory, os.path.splitext(os.path.basename(manifest))[0])
    if os.path.isdir(sibling):
        return sibling
    return directory


def _best_column(header: Sequence[str], rank: dict) -> Optional[int]:
    matches = [(index, name) for index, name in enumerate(header) if name in rank]
    if not matches:
        return None
    return min(matches, key=lambda item: rank[item[1]])[0]


def _best_field(item: dict, rank: dict) -> Optional[str]:
    lowered = {}
    for key, value in item.items():
        if isinstance(key, str):
            lowered[key.strip().lower()] = value
    for name in rank:
        value = lowered.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _folder_base(folder: Optional[str], manifest_dir: str) -> Optional[str]:
    if not folder or not folder.strip():
        return None
    folder = folder.strip()
    if os.path.isabs(folder):
        return folder
    return os.path.normpath(os.path.join(manifest_dir, folder))


def _join_image(
    image: str,
    folder: Optional[str],
    dataset_root: Optional[str],
    manifest: str,
    image_root: Optional[str] = None,
) -> str:
    """Join an image folder root with a relative image path."""
    image = image.strip()
    if not image or os.path.isabs(image):
        return image
    if image_root:
        return os.path.join(image_root, image)
    manifest_dir = os.path.dirname(os.path.abspath(manifest))
    base = _folder_base(folder, manifest_dir) or dataset_root or _fallback_base_dir(manifest)
    return os.path.join(base, image)


def _load_manifest(path: str, base_dir: Optional[str], image_root: Optional[str] = None) -> List[str]:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".csv":
        return _paths_from_csv(path, base_dir, image_root)
    if extension == ".json":
        return _paths_from_json(path, base_dir, image_root)
    raise ValueError(f"Image manifest must be .csv or .json, got {path}")


def _csv_rows(path: str) -> List[List[str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        first_line = handle.readline()
        if not first_line:
            return []
        if "\t" in first_line and "," not in first_line:
            delimiter = "\t"
        elif ";" in first_line and "," not in first_line:
            delimiter = ";"
        else:
            delimiter = ","
        handle.seek(0)
        return [row for row in csv.reader(handle, delimiter=delimiter) if any(cell.strip() for cell in row)]


def _paths_from_csv(path: str, base_dir: Optional[str], image_root: Optional[str] = None) -> List[str]:
    rows = _csv_rows(path)
    if not rows:
        return []
    header = [cell.strip().lower() for cell in rows[0]]
    column = _best_column(header, _PATH_KEY_RANK)
    root_column = _best_column(header, _ROOT_KEY_RANK)
    if column is None:
        column = 0
        data_rows = rows
        root_column = None
    else:
        data_rows = rows[1:]
    paths = []
    for row in data_rows:
        if column >= len(row):
            continue
        folder = row[root_column].strip() if root_column is not None and root_column < len(row) else None
        resolved = _join_image(row[column], folder, base_dir, path, image_root)
        if resolved:
            paths.append(resolved)
    return paths


def _extract_path(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return _best_field(item, _PATH_KEY_RANK)
    return None


def _extract_root(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        return _best_field(item, _ROOT_KEY_RANK)
    return None


def _paths_from_json(path: str, base_dir: Optional[str], image_root: Optional[str] = None) -> List[str]:
    with open(path, encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    file_root = _extract_root(payload)
    items = _json_items(payload, path)
    paths = []
    for item in items:
        extracted = _extract_path(item)
        if extracted is None:
            continue
        folder = _extract_root(item) or file_root
        resolved = _join_image(extracted, folder, base_dir, path, image_root)
        if resolved:
            paths.append(resolved)
    return paths


def _json_items(payload: Any, path: str) -> List[Any]:
    if isinstance(payload, list):
        return _flatten_items(payload)
    if isinstance(payload, dict):
        items: List[Any] = []
        for value in payload.values():
            if isinstance(value, list):
                items.extend(_flatten_items(value))
        if items:
            return items
        if _extract_path(payload):
            return [payload]
        if payload and all(isinstance(value, str) for value in payload.values()):
            values = [
                value
                for key, value in payload.items()
                if str(key).strip().lower() not in _ROOT_KEY_SET
            ]
            if values:
                return values
    raise ValueError(
        f"Unsupported JSON manifest {path}: expected a list of paths, "
        "lists of objects with a path field, or an object containing one or more lists"
    )


def _flatten_items(items: Sequence[Any]) -> List[Any]:
    flat: List[Any] = []
    for item in items:
        if isinstance(item, list):
            flat.extend(_flatten_items(item))
        else:
            flat.append(item)
    return flat


def _walk_images(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS:
                paths.append(os.path.join(dirpath, name))
    paths.sort()
    return paths


def write_image_list(root: str, output: str, relative: bool = True) -> int:
    """Write images under ``root`` to a CSV list with ``root`` and ``path`` columns.

    ``path`` is relative to the image folder unless ``relative`` is false.
    ``root`` is the absolute image folder, so the list can be loaded by itself.
    Returns the number of paths written.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = _walk_images(root)
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    image_root = os.path.abspath(root)
    output_dir = os.path.dirname(os.path.abspath(output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path"])
        for path in paths:
            if relative:
                path = os.path.relpath(path, image_root)
            writer.writerow([path])
    logger.info("Wrote %s image paths to %s", f"{len(paths):,d}", output)
    return len(paths)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert an image folder to an ImageDir CSV list.")
    parser.add_argument("root", help="Folder of images, for example /root/workspace/rocket/Dataset/demo")
    parser.add_argument("-o", "--output", help="CSV to write. Default: <root>.csv next to the folder")
    parser.add_argument("--absolute", action="store_true", help="Store absolute paths instead of paths relative to the folder")
    args = parser.parse_args(argv)
    output = args.output
    if not output:
        output = os.path.abspath(args.root).rstrip(os.sep) + ".csv"
    count = write_image_list(args.root, output, relative=not args.absolute)
    print(f"Wrote {count} paths to {output}")


if __name__ == "__main__":
    main()
