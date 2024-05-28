import os
from tqdm import tqdm
from datasets import load_dataset
import json

# load online
# dataset = load_dataset("polinaeterna/pokemon-blip-captions")
# caption_column="text"

# load local
DATASET_DIR="/home/ao/workspace/IL-SIM-DATA/metaworld_data/folder/train"
data_files = {}
data_files["train"] = os.path.join(DATASET_DIR, "**")
dataset = load_dataset(
    "imagefolder",
    data_files=data_files,
)
caption_column="additional_feature"


root_dir = "/home/ao/workspace/IL-SIM-DATA/metaworld_data/sam"  # out_dir
images_dir = "images"
captions_dir = "captions"

images_dir_absolute = os.path.join(root_dir, images_dir)
captions_dir_absolute = os.path.join(root_dir, captions_dir)

if not os.path.exists(root_dir):
    os.makedirs(os.path.join(root_dir, images_dir))

if not os.path.exists(os.path.join(root_dir, images_dir)):
    os.makedirs(os.path.join(root_dir, images_dir))
if not os.path.exists(os.path.join(root_dir, captions_dir)):
    os.makedirs(os.path.join(root_dir, captions_dir))

image_format = "png"
json_name = "partition/data_info.json"
if not os.path.exists(os.path.join(root_dir, "partition")):
    os.makedirs(os.path.join(root_dir, "partition"))

absolute_json_name = os.path.join(root_dir, json_name)
data_info = []

order = 0
for item in tqdm(dataset["train"]): 
    image = item["image"]
    image.save(f"{images_dir_absolute}/{order}.{image_format}")
    with open(f"{captions_dir_absolute}/{order}.txt", "w") as text_file:
        text_file.write(item[caption_column])
    
    width, height = 512, 512
    ratio = 1
    data_info.append({
        "height": height,
        "width": width,
        "ratio": ratio,
        "path": f"images/{order}.{image_format}",
        "prompt": item[caption_column],
    })
        
    order += 1

with open(absolute_json_name, "w") as json_file:
    json.dump(data_info, json_file)