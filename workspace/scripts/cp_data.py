import os
import glob
import shutil


input_dirs = [
    "",
    "",
    "",
    "",
]
target_dir = ""

if not os.path.exists(target_dir):
    os.makedirs(target_dir)


files_lists = []
for folder in input_dirs:
    files = sorted(glob.glob(os.path.join(folder, "data_*.pt")))
    files_lists.append(files)

mixed_files = []
max_len = max(len(files) for files in files_lists)
for i in range(max_len):
    for file_list in files_lists:
        if i < len(file_list):
            mixed_files.append(file_list[i])

for idx, file_path in enumerate(mixed_files):
    new_name = f"data_{idx}.pt"
    target_path = os.path.join(target_dir, new_name)
    shutil.copy(file_path, target_path)
    print(f"Copied {file_path} to {target_path}")
