# %%
import pandas as pd
import re
import numpy as np


df = pd.read_csv("results.csv")
# df['error'] = df.apply(lambda x: 'ok' if x.success else 'corrupted', axis=1)

# print(df[df['error']=='corrupted'])
# # %%
# print(list(df[df['error']=='ok'].input_path))
# %%
def naming(path):
    if 'lesion_frames' in path:
        return 'lesion_frames'
    elif 'lesion_context' in path:
        return 'lesion_context'
    else:
        return 'wrong naming'
df['naming'] = df.apply(lambda x: naming(x.input_path), axis=1)
# %%
import re

def find_chinese_pattern(text):
    # Matches sequences of Chinese (Han) characters
    pattern = r'[\u4e00-\u9fff]+'
    matches = re.findall(pattern, text)
    if matches:
        return ','.join(matches)
    else:
        return ''
df['chinese'] = df.apply(lambda x: find_chinese_pattern(x.input_path), axis=1)
print(df[df['chinese'] != ''])
# %%
import re
from pathlib import Path
# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True)
def get_match_path(context_path):
    # Reference path
    context_path = Path(context_path)

    # Get base directory and target file name
    for i in range(1,3):
        base_dir = context_path.parents[i]  # 
        target_filename = context_path.name  # 
        # Search all subdirectories inside 'lesion_frames'
        # frames_dir = base_dir / "lesion_frames"
        matches = list(base_dir.glob(f"**/{target_filename}"))
        if len(matches) == 2:
            return matches
    return []
    
def dive(path):
    if 'lesion_frames' in path or 'lesion_frame' in path:
        candidates = get_match_path(path)
        if len(candidates) == 0:
            return 'not in context'
    return ''

df['frameNotInContext'] = df.apply(lambda x: dive(x.input_path), axis=1)
print(df[df['frameNotInContext']==''])
print(df[df['frameNotInContext']=='not in context'])
# %%
df['path'] = df.apply(lambda x: x.input_path.replace('/home/ei/workspace/rocket/Project/nanfang_capsule/data/unzipped/v1/', ''), axis=1)


# Save to CSV
df[['brand', 'disease_type', 'path', 'naming', 'chinese', 'frameNotInContext']].to_csv('inspect.csv', index=False)
print(df)
# %%
list(df[df['frameNotInContext']=='not in context'].input_path)
# %%


import csv
import os


def save_files_to_csv(folder_path, output_csv_path):
    # Header for the CSV file
    headers = ["File Name", "File Path", "Extension"]

    with open(
        output_csv_path, mode="w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        # Walk through all directories and subdirectories
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                # Get the full absolute file path
                full_path = os.path.abspath(os.path.join(root, file_name))

                # Split the filename and extension
                _, extension = os.path.splitext(file_name)

                # Write row to CSV
                writer.writerow([file_name, full_path, extension])


# --- Example Usage ---
folder_to_scan = "/home/ei/workspace/rocket/Project/nanfang_capsule/data/unzipped/v2_plus_v2smallbowel_v2esophagus"
output_csv = "files_list.csv"

save_files_to_csv(folder_to_scan, output_csv)
print(f"File list saved successfully to {output_csv}")
# %%
