import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import clip

class ViD(Dataset):
    def __init__(self, file_list, preprocess):
        super().__init__()
        self.file_list = file_list
        self.preprocess = preprocess

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        obss = np.load(self.file_list[index])["arr_0"].astype(np.uint8)
        obss = torch.cat([self.preprocess(Image.fromarray(obs)).unsqueeze(0) for obs in obss], dim=0)
        return obss, self.file_list[index]

def process(file_list):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(rank)
    model, preprocess = clip.load('ViT-B/32', "cuda")
    
    dataset = ViD(file_list, preprocess)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=8, sampler=sampler)
    sampler.set_epoch(0)

    if rank == 0: pbar = tqdm(total=len(dataset))
    for data in dataloader:
        for idx, batch in enumerate(data[0]):
            with torch.no_grad():
                image_features = model.encode_image(batch.cuda())
                file_path = data[1][idx]

                np.savez_compressed(
                    file_path[:-3] + "clip",
                    image_features.cpu().numpy()
                )

            if rank == 0: pbar.update(world_size)

def main():
    path = '/data/jaq/metaworld_data/corner/state/'
    file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('obss.npz')]

    process(file_list)

if __name__ == "__main__":
    main()
    # torchrun --nproc-per-node=4 encoding.py
