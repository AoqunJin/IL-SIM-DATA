import os
import cv2
from tqdm import tqdm

# 设置输入和输出路径
input_dir = "metaworld_data/corner/video"
output_dir = "metaworld_data/folder/train"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 初始化 metadata.csv 文件
metadata_file = os.path.join(output_dir, "metadata.csv")
with open(metadata_file, "w") as f:
   f.write("image,text\n")

# 遍历视频文件
counter = 1

with tqdm(os.listdir(input_dir)) as td:
    for filename in td:
        if filename.endswith(".avi"):
            video_path = os.path.join(input_dir, filename)
            cap = cv2.VideoCapture(video_path)

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 提取关键帧
            for i in range(frame_count):
                ret, frame = cap.read()
                if i % (fps // 2) == 0:  # 每 0.5 秒提取一帧
                    output_filename = f"{counter:04d}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, frame)

                    # 更新 metadata.csv
                    feature = " ".join(filename.split("-")[:-2])
                    with open(metadata_file, "a") as f:
                        f.write(f"{output_filename},{feature}\n")

                    counter += 1

            cap.release()

print("Finish!")


# from datasets import load_dataset

# dataset = load_dataset(path=r"C:\Users\aojia\Desktop\train")
# dataset.push_to_hub("Jaq203/MetaWorld-MT50-Expand", token="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")