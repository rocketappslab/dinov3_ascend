#!/usr/bin/env python3
"""Keep images Pillow can open, and write those paths to a CSV."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def check_one(path: str) -> tuple[str, str | None]:
    try:
        with Image.open(path) as im:
            im.convert("RGB")
            im.load()
        return path, None
    except (OSError, UnidentifiedImageError, ValueError) as e:
        return path, f"{type(e).__name__}: {e}"


def iter_images(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                yield str(Path(dirpath) / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/mnt/data/workspace/rocket/Dataset/gastronet_unzip",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--output", default="valid_images.csv")
    parser.add_argument("--bad-list", default="bad_images.txt")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    paths = list(iter_images(root))
    print(f"Found {len(paths)} images under {root}  workers={args.workers}")

    ok = 0
    bad: list[tuple[str, str]] = []
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path"])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, (path, err) in enumerate(
                pool.map(check_one, paths, chunksize=64), 1
            ):
                if err is None:
                    writer.writerow([path])
                    ok += 1
                else:
                    bad.append((path, err))
                    print(f"BAD  {path}  ({err})")
                if i % 1000 == 0 or i == len(paths):
                    handle.flush()
                    print(f"progress {i}/{len(paths)}  ok={ok} bad={len(bad)}")

    with open(args.bad_list, "w", encoding="utf-8") as handle:
        for path, err in bad:
            handle.write(f"{path}\t{err}\n")

    print(f"Wrote {ok} paths to {args.output}")
    print(f"Wrote {len(bad)} failures to {args.bad_list}")


if __name__ == "__main__":
    main()