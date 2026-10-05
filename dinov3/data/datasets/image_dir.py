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
# Earlier names win when a header contains more than one path column.
_PATH_KEYS = (
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
_PATH_KEY_RANK = {name: index for index, name in enumerate(_PATH_KEYS)}


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
    CSV rows are paths, or a table whose header names the path column
    (``path``, ``image``, ``filename``, ``file_name``, and similar).
    Relative paths are resolved against ``root`` when ``root`` is a directory.
    When the dataset root is the list file itself, they are resolved against
    the CSV ``root`` column, or against a sibling folder of the same name
    (``demo.csv`` next to ``demo/``).

    Write that CSV from a folder with ``write_image_list`` or:

        python -m dinov3.data.datasets.image_dir /root/workspace/rocket/Dataset/demo
    """

    def __init__(
        self,
        *,
        root: str,
        extra: Optional[Union[str, Sequence[str]]] = None,
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
        paths, source = _collect_paths(root, extra)
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


def _collect_paths(root: str, extra: Optional[Union[str, Sequence[str]]]) -> tuple:
    specs = _manifest_specs(extra)
    if specs:
        return _load_manifests(root, specs), ",".join(specs)
    root_specs = _split_spec(root)
    if len(root_specs) > 1 or (root_specs and not os.path.isdir(root_specs[0]) and _looks_like_manifest(root_specs[0])):
        return _load_manifests(None, root_specs), root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Image directory not found: {root}")
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


def _load_manifests(root: Optional[str], specs: Sequence[str]) -> List[str]:
    image_root = root if root and os.path.isdir(root) else None
    paths: List[str] = []
    for spec in specs:
        for manifest in _expand_manifest_input(spec):
            paths.extend(_load_manifest(manifest, image_root))
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


def _resolve_path(entry: str, base_dir: str) -> str:
    entry = entry.strip()
    if not entry or os.path.isabs(entry):
        return entry
    return os.path.join(base_dir, entry)


def _load_manifest(path: str, base_dir: Optional[str]) -> List[str]:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".csv":
        return _paths_from_csv(path, base_dir)
    if extension == ".json":
        return _paths_from_json(path, base_dir)
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


def _paths_from_csv(path: str, base_dir: Optional[str]) -> List[str]:
    rows = _csv_rows(path)
    if not rows:
        return []
    header = [cell.strip().lower() for cell in rows[0]]
    matches = [(index, name) for index, name in enumerate(header) if name in _PATH_KEY_RANK]
    root_column = None
    if matches:
        column = min(matches, key=lambda item: _PATH_KEY_RANK[item[1]])[0]
        data_rows = rows[1:]
        if "root" in header:
            root_column = header.index("root")
    else:
        column = 0
        data_rows = rows
    fallback = base_dir or _fallback_base_dir(path)
    paths = []
    for row in data_rows:
        if column >= len(row):
            continue
        row_base = fallback
        if base_dir is None and root_column is not None and root_column < len(row) and row[root_column].strip():
            row_base = row[root_column].strip()
        resolved = _resolve_path(row[column], row_base)
        if resolved:
            paths.append(resolved)
    return paths


def _extract_path(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        matches = [(key, item[key]) for key in _PATH_KEYS if isinstance(item.get(key), str) and item[key].strip()]
        if matches:
            return min(matches, key=lambda item: _PATH_KEY_RANK[item[0]])[1]
    return None


def _paths_from_json(path: str, base_dir: Optional[str]) -> List[str]:
    with open(path, encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if base_dir is None:
        file_root = payload.get("root") if isinstance(payload, dict) else None
        base_dir = file_root.strip() if isinstance(file_root, str) and file_root.strip() else _fallback_base_dir(path)
    items = _json_items(payload, path)
    paths = []
    for item in items:
        extracted = _extract_path(item)
        if extracted is None:
            continue
        resolved = _resolve_path(extracted, base_dir)
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
            return list(payload.values())
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
        writer.writerow(["root", "path"])
        for path in paths:
            if relative:
                path = os.path.relpath(path, image_root)
            writer.writerow([image_root, path])
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
