import os 
import torch 
import random 
import torchvision.transforms as T
import configparser
import albumentations as A
import numpy as np 
import json 
import open_clip
from torch.utils.data import Dataset
from typing import Any 
from glob import glob
from PIL import Image 
from einops import rearrange, repeat
from albumentations.pytorch import ToTensorV2
from data.obj_track_dataset import OTMetaDataset

GOT_IGNORE_VIDEO=["001898", "009302", "008628", "001858", "001849", "009058", "009065",
                        "001665", "004419", "009059", "009211", "009210", "009032", "009102",
                          "002446", "008925", "001867", "009274", "009323", "002449", "009031",
                            "009094", "005912", "007643", "007788", "008917", "009214", "007610",
                              "009320", "007645", "009027", "008721", "008931", "008630", "txt"]



def generate_prompt_bbox_info(k_indices, groundth_txt, meta_info_ini):
    with open(groundth_txt, "r") as f:
        bbox_lines = f.readlines()
    selected_bboxes = [bbox_lines[i].strip() for i in k_indices]

    ori_bbox = []

    config = configparser.ConfigParser()
    config.read(meta_info_ini)
    img_w, img_h = tuple(map(int, config["METAINFO"]["resolution"][1:-1].split(", ")))

    for i, bbox  in enumerate(selected_bboxes):
        x, y, w, h = map(float, bbox.split(","))
        x1, y1, x2, y2 = x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h
        x1, y1, x2, y2 = map(lambda v: min(max(v, 0.0), 1.0), [x1, y1, x2, y2])
        ori_bbox.append([x1, y1, x2, y2, 0])
    
    def get_grid_index(x, y, total_columns=192):
        grid_size = 2  # Since the image is divided into 192x192 grids on a 384x384 image
        grid_x = int(x / grid_size)
        grid_y = int(y / grid_size)
        return grid_y * total_columns + grid_x
     
    # Parse meta_info.ini
    meta_info = {}
    with open(meta_info_ini, "r") as f:
        lines = f.readlines()
        for line in lines:
            if ":" in line and "class" in line:
                key, value = line.split(":")
                key = key.strip()
                value = value.strip()
                meta_info[key] = value 
    
    # Generate the prompt
    object_class = meta_info["object_class"]
    motion_class = meta_info["motion_class"]

    prompt = f"A {object_class} that is {motion_class}. "
    seg_phrase = f"{object_class}"
    return prompt, ori_bbox, seg_phrase


def draw_bboxes_on_frames(frames, bboxes):
    bboxes = np.array(bboxes)
    from PIL import Image, ImageDraw
    _, _, height, width = frames.shape

    frames_np = (frames).numpy().transpose(0, 2, 3, 1)
    frames_np = frames_np.astype(np.uint8)
    
    cnt = 0
    for frame, bbox in zip(frames_np, bboxes):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin *= 480
        xmax *= 480
        ymin *= 320
        ymax *= 320
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0), width=2)
        img.save(f'test{cnt}.jpg')
        cnt += 1


class GOT10KDataset(OTMetaDataset):
    def __init__(
        self,
        path: str,
        n_sample_frames: int = 8,
        width: int = 256,
        height: int = 256,
        split: str = "train",
        fallback_prompt: str = "",
        num_max_obj: int = 8,
        preprocessor = None 
    ):
        
        self.preprocessor = preprocessor
        self.fallback_prompt = fallback_prompt
        # List all foders (each folder corresponds to a video)
        data_path = os.path.join(path,split)
        video_folders = glob(f"{data_path}/*")
        video_folders.sort()
        self.video_folders =[
            ok_name for ok_name in video_folders if not any(drop in ok_name for drop in GOT_IGNORE_VIDEO)
        ]
        
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        embedding_path = os.path.join(path,"obj_embedding.json")
        with open(embedding_path,"r") as f:
            self.obj_embedding = json.load(f)

    def get_prompt_and_bbox(self, folder_path, k_index):
        groundth_txt_path = os.path.join(folder_path, "groundtruth.txt")
        meta_info_ini_path = os.path.join(folder_path, "meta_info.ini")

        if os.path.exists(groundth_txt_path) and os.path.exists(meta_info_ini_path):
            return generate_prompt_bbox_info(k_index, groundth_txt_path, meta_info_ini_path)
        else:
            return self.fallback_prompt 

    def __len__(self):
        return len(self.video_folders) 
    

    def __getitem__(self, index: int) -> Any:
        sampled_frame_files, resize_transform, w_scale, h_scale, k_index = self.get_frame_batch(self.video_folders[index])
        prompt, ori_bboxes, obj_class = self.get_prompt_and_bbox(self.video_folders[index], k_index)
        # Define the transform using albumentations
        crop_transform = A.Compose(
           [A.Resize(height=self.height, width=self.width),
            ToTensorV2(),],
            bbox_params=A.BboxParams(format="albumentations"),
        )

        # Apply crop transform to each fram and update bboxes
        frames = []
        resize_bboxes = []
        for idx, frame_file in enumerate(sampled_frame_files):
            image = np.array(Image.open(frame_file).convert("RGB")).astype(np.uint8)
            transformed = crop_transform(image=image, bboxes=[ori_bboxes[idx],])
            frames.append(transformed["image"])
            resize_bboxes.append(transformed["bboxes"][0]) 
        
        frames = torch.stack(frames)
        
        def dummy_normalize_bboxes(bboxes):
            return [[xmin, ymin, xmax, yman] for xmin, ymin, xmax, yman, dummy_class in bboxes]
        
        out_dict = {}
        if self.preprocessor is not None:
            first_frame_img = Image.open(sampled_frame_files[0]).convert("RGB")
            first_frame_tensor = self.preprocessor(first_frame_img).unsqueeze(0)
            out_dict["first_frame_tensor"] = first_frame_tensor

        bboxes = torch.Tensor(dummy_normalize_bboxes(resize_bboxes))
        bboxes = torch.clamp(bboxes, min=0.0, max=1.0).unsqueeze(1)
        
        bboxes_mask = torch.ones(self.n_sample_frames,1)  # num_frame, 1, 4

        obj_embedding = torch.tensor(self.obj_embedding[obj_class])
        obj_embedding = obj_embedding.repeat(self.n_sample_frames,1).unsqueeze(1) # shape n_sample, 1, 1024
        prompt_ids = open_clip.tokenize(prompt)

        out_dict["pixel_values"] = self.normalize_input(frames)
        out_dict["prompt_ids"] = prompt_ids
        out_dict["boxes"] = bboxes
        out_dict["text_embeddings"] = obj_embedding
        out_dict["masks"] = bboxes_mask 
        out_dict["data_name"] = "got10k"
        out_dict["text"] = prompt 
        out_dict["video_folder"] = self.video_folders[index].split("/")[-1]
        out_dict["obj_class"] = obj_class
        return out_dict 
        

if __name__ == "__main__":
    dataset = GOT10KDataset(
        path="/data/leuven/333/vsc33366/projects/Diffusion-Video/dataset/got-10k",
        split="train",
        width=480,
        height=320,
        n_sample_frames=8,
        fallback_prompt="A person that is walking"
    )

    idx = 0 
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8
    )

    for step, batch in enumerate(train_dataloader):
        print(batch["text_prompt"])
        #import ipdb;ipdb.set_trace()