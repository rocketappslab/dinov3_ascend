


import argparse
import sys
import time
from multiprocessing import Pool, cpu_count
import os
from PIL import Image
import multiprocessing
from pathlib import Path
import sys
from functools import partial
import re
from PIL import Image, ImageDraw
import pytesseract
from tqdm import tqdm
import pandas as pd
import os
import requests
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Extract text
def sanitize_file_path(filepath):
    path = Path(filepath)
    # Add flags=re.ASCII so \w strictly matches ASCII characters
    clean_stem = re.sub(r'[^\w\.-]', '', path.stem, flags=re.ASCII)
    
    return path.with_name(f"{clean_stem}{path.suffix}")
img1 = "/home/ei/workspace/rocket/Project/nanfang_capsule/src/data/鏉庡瓙鍜宊000202.jpg"
img2 = "/home/ei/workspace/rocket/Project/nanfang_capsule/src/data/183_112932.jpg"
img3 = "/home/ei/workspace/rocket/Project/nanfang_capsule/src/data/鑻忚壓鏂嘷000088.jpg"
img4 = "/home/ei/workspace/rocket/Project/nanfang_capsule/src/data/闄堝叏鑻盻000069.jpg"
img5 = "/home/ei/workspace/rocket/Project/nanfang_capsule/src/data/闄堝浗閿媉000024.jpg" #omom black 520*520

def get_ocr(image):
  url = "http://127.0.0.1:4443/ocr"
  payload = {"image_path": image}

  response = requests.post(url, json=payload)
  response = response.json()
  bbox = response.get('results')
  return bbox

def get_shapes_by_brand(brand):
    if brand == "side":
        shapes = []
    elif brand == "omom-hd":
        shapes = [{
      "points": [
        [
          512.0,
          0.0
        ],
        [
          403.2884141331142,
          20.9531635168447
        ]
      ],
      "shape_type": "rectangle",
    },
    {
      "points": [
        [
          109.12243221035334,
          21.36400986031224
        ],
        [
          0.0,
          -8.881784197001252e-16
        ]
      ],
      "shape_type": "rectangle",
    },
    {
      "points": [
        [
          407.23418241577656,
          491.0468364831553
        ],
        [
          512.0,
          512.0
        ]
      ],
      "shape_type": "rectangle",
    }]
    elif brand == "omom":
        shapes = [{
      "points": [
        [
          202.95,
          221.79999999999998
        ],
        [
          240.0,
          234.3
        ]
      ],
      "shape_type": "rectangle",
    },
    {
      "points": [
        [
          191.75,
          10.400000000000002
        ],
        [
          240.0,
          0.0
        ]
      ],
      "shape_type": "rectangle",
    },
    {
      "points": [
        [
          37.30493058879847,
          12.2929631402585
        ],
        [
          0.0,
          0.24423926631791382
        ]
      ],
      "shape_type": "rectangle",
    }]
    elif brand == "richen-force":
        shapes = [{
      "points": [
        [
          257.9712378017463,
          15.921931176168465
        ],
        [
          319.86132511556247,
          0.5136106831022086
        ]
      ],
      "shape_type": "rectangle",
    },
    {
      "points": [
        [
          3.9473684210526336,
          303.6842105263158
        ],
        [
          132.5438596491228,
          317.1929824561403
        ]
      ],
      "shape_type": "rectangle",
    }]
    elif brand == "ankon":
        shapes = [{
      "points": [
        [
          73.78274268104776,
          25.613698236425908
        ],
        [
          0.0,
          0.0
        ]
      ],
      "shape_type": "rectangle",
    },
    {
      "points": [
        [
          392.7349768875192,
          24.603354533915272
        ],
        [
          480.0,
          0.0
        ]
      ],
      "shape_type": "rectangle",
    },
    {
      "points": [
        [
          363.28197226502306,
          456.50231124807397
        ],
        [
          480.0,
          480.0
        ]
      ],
      "shape_type": "rectangle",
    },
    {
      "points": [
        [
          76.864406779661,
          458.1971153846154
        ],
        [
          0.0,
          480.0
        ]
      ],
      "shape_type": "rectangle",
    }]
    else:
        raise
    return shapes

def get_config_by_img(image, input_path):
    brand = ""
    _, h = image.size
    if h == 480:
        chars = get_ocr(str(input_path))
        if chars is None or len(chars) <= 2:
            brand = "side"
        else:
            brand = "ankon"
    elif h == 512:
        brand = "omom-hd"
    elif h == 240:
        brand = "omom"
    elif h == 320:
        brand = "richen-force"
    else:
        raise
    return brand, get_shapes_by_brand(brand)

def preprocess(input_path, output_path, save_img=True):
  meta = {
    'brand': '',
    'disease_type': '',
    'input_path': '',
    'output_path': '',
  }
  meta['input_path'] = input_path
  folder_path = os.path.dirname(output_path)
  if folder_path: # Check to ensure the path isn't empty
    os.makedirs(folder_path, exist_ok=True)
  with Image.open(input_path) as img:
      # Convert to RGB mode if necessary (handles PNG transparency/palettes safely)
      if img.mode in ("RGBA", "P"):
          img = img.convert("RGB")
      brand, shapes = get_config_by_img(img, input_path)
      meta['brand'] = brand

      draw = ImageDraw.Draw(img)
      # Draw solid black rectangle
      for shape in shapes:
          if shape['shape_type'] == "rectangle":
              xy = shape['points']
              x0, y0, x1, y1 = xy[0][0],xy[0][1],xy[1][0],xy[1][1]

              # Ensure x0 <= x1 and y0 <= y1
              xy = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
              draw.rectangle(xy, fill="black")
      if save_img:
        img.save(output_path)
        meta['output_path'] = output_path
  parts = list(input_path.parts)
  meta['disease_type'] = parts[10]
  return meta

  

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
OUTPUT_FORMAT_MAP = {
    "jpeg": "JPEG",
    "jpg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
}


def process_file(input, output, file_path):
    """Wrapper function for multiprocessing"""
    try:
        # output_img_path = sanitize_file_path(Path(str(file_path).replace(input, output, 1)))
        output_img_path = Path(str(file_path).replace(input, output, 1)) # keep original filename
        result = preprocess(file_path, output_img_path, save_img=True)
        result['error'] = ''
        return True, result
    except Exception as e:
        return False, {"input_path": file_path,
                        "error": str(e)}


if __name__ == "__main__":
  input = "/home/ei/workspace/rocket/Project/nanfang_capsule/data/unzipped/v2_plus_v1sb"
  output = "/home/ei/workspace/rocket/Project/nanfang_capsule/data/preprocessed_oriname/v2_plus_v1sb"

  input_dir = Path(input)
  if not input_dir.is_dir():
      sys.exit(f"Error: input directory not found: {input_dir}")

  files = sorted(
      f for f in input_dir.rglob("*")
      if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
  )
  if not files:
      sys.exit(f"Error: no supported images found in {input_dir} "
                f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})")
  
  print(f"Found {len(files)} files to process")
  
  # files = files[:100]

  # Create partial function with input and output paths
  process_func = partial(process_file, input, output)


  # Create a pool of worker processes
  results = []
  with multiprocessing.Pool(processes=os.cpu_count()) as pool:
      # Use imap_unordered for better performance with tqdm
      with tqdm(total=len(files), desc="Processing images", unit="file") as pbar:
          for result in pool.imap_unordered(process_func, files):
              results.append(result)
              pbar.update(1)

  # Report results
  successful = [r for r in results if r[0]]
  failed = [r for r in results if not r[0]]

  print(f"\nProcessing complete:")
  print(f"  Successful: {len(successful)} files")
  print(f"  Failed: {len(failed)} files")

  if failed:
      print("\nFailed files:")
      for _, error in failed:
          print(f"  Error: {error}")

  # Convert to DataFrame
  rows = []
  for success, info in results:
      row = {
          'success': success,
          'brand': info.get('brand', ''),
          'disease_type': info.get('disease_type', ''),
          'input_path': str(info.get('input_path', '')),  # Convert PosixPath to string
          'output_path': str(info.get('output_path', '')),
          'error': str(info.get('error', '')
          )
      }
      rows.append(row)

  df = pd.DataFrame(rows)

  # Save to CSV
  df.to_csv('results.csv', index=False)

  print(f"Results saved to results.csv")
  print(df)
