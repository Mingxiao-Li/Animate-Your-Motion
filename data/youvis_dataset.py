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


# special_id = './dataset/youvis/train/JPEGImages/823e7a86e8'
class YOUVISDataset(OTMetaDataset):
   ## in all dataset  num_max_obj=25
    def __init__(
        self,
        path: str,
        n_sample_frames: int = 8,
        width: int = 256,
        height: int = 256,
        split: str = "train",
        num_max_obj: int = 8,
        fallback_prompt: str = "",
        preprocessor=None,
    ):

        data_path = os.path.join(path, split)
        video_folders = glob(f"{data_path}/JPEGImages/*") 
        video_folders.sort()
        # print(f"In total {len(video_folders)} videos")  # 2985
        invalid_count = 0  # 93
        invalid_video = []  # 95
        self.preprocessor = preprocessor
        for folder_path in video_folders:
            # if folder_path == special_id:
            #     import ipdb; ipdb.set_trace()
            all_img = sorted(glob(f"{folder_path}/*.jpg"))
            if len(all_img) < n_sample_frames:
                invalid_count += 1
                # video_folders.remove(folder_path)
                invalid_video.append(folder_path)
                # print(f"{folder_path} contains less than {n_sample_frames} images")
        # print(f"In total {invalid_count} invalid videos")
        valid_video_folders = []
        for folder_path in video_folders:
            if folder_path not in invalid_video:
                valid_video_folders.append(folder_path)
        # print(f"In total {len(video_folders)} valid videos")

        self.video_folders = valid_video_folders  # should be 2890
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.num_max_obj = num_max_obj
        self.load_prompt_and_data(path, split)
    
    def load_prompt_and_data(self, path, split):
        prompt_path = os.path.join(path, "caption.json")
        instance_path = os.path.join(path, split, "instances.json")
        annotations_path = os.path.join(path, split, "annotations_info_with_cat_emb.json")
        with open(prompt_path, "r") as f:
            prompts = json.load(f)
        
        with open(instance_path, "r") as f:
            instances = json.load(f)
        
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        
        self.categories_embedding = annotations["category_embedding"]
        self.prompts = prompts
        self.annotations = annotations
        self.videos_info = instances["videos"]
        self.categories_dict = instances['categories']

      
    def __len__(self):
        return len(self.video_folders)
    

    def get_prompt_and_bbox(self, folder_path, k_indices):
        video_name = folder_path.split("/")[-1] 
        annotation_info = self.annotations[video_name]
        all_boxes = []
        all_objects = []
        all_objects_embeddings = []
        all_obj_mask = []
        all_objects = []
        img_w, img_h = annotation_info[0]["width"], annotation_info[0]["height"]
        for _, bbox_info in enumerate(annotation_info):
            # bbox_info: one object's box in all other frames
            selected_bboxs = [bbox_info['bboxes'][i] for i in k_indices]

            ori_bbox = []
            ori_obj = []
            ori_obj_embedding = []
            ori_mask = []   # has object 1 , non object 0
            for i, bbox in enumerate(selected_bboxs):
                if bbox is not None:
                    x, y, w, h = map(float, bbox)
                    x1, y1, x2, y2 = x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h
                    x1, y1, x2, y2 = map(lambda v: min(max(v, 0.0), 1.0), [x1, y1, x2, y2])
                    ori_bbox.append([x1, y1, x2, y2, 0])
                    object_class = self.categories_dict[bbox_info["category_id"]-1]["name"]
                    object_embedding = torch.tensor(self.categories_embedding[object_class])
                    ori_obj_embedding.append(object_embedding)
                    ori_obj.append(object_class)
                    ori_mask.append(1)
                else:
                    ori_bbox.append([0.,0.,1.,1.,0]) # 1 for none object , all image
                    ori_obj.append('image') # image for none object
                    ori_mask.append(0)
                    ori_obj_embedding.append(torch.zeros(1,1024))
         
            all_boxes.append(ori_bbox)
            all_obj_mask.append(ori_mask)
            all_objects.append(ori_obj)
            all_objects_embeddings.append(torch.cat(ori_obj_embedding,dim=0))
            
        # while len(all_objects) < self.num_max_obj:
        #     all_objects.append(["image"]*8)
        prompt = self.prompts[video_name]
        all_objects_embeddings = torch.cat([x.unsqueeze(0) for x in all_objects_embeddings], dim=0)
        
        return prompt, all_boxes, all_objects, all_obj_mask, all_objects_embeddings
    

    def __getitem__(self, index: int) -> Any:
        sampled_frame_files, resize_transform, w_scale, h_scale, k_index = self.get_frame_batch(self.video_folders[index], random_sample=True, annotations=self.annotations)
        prompt, ori_bboxes, obj_class, obj_mask, obj_embeddings = self.get_prompt_and_bbox(self.video_folders[index], k_index)
        # ori_bboxes list of list [[box_1, box_2, ...], [box_1, box_2, ...]]

        # Define the transform using albumentations
        crop_transform = A.Compose(
           [A.Resize(height=self.height, width=self.width),
            ToTensorV2(),],
            bbox_params=A.BboxParams(format="albumentations"),
        )
        
        # Apply crop transform to each frame and update bboxes
        frames = []
        resize_bboxes = [] 
        
        # reshape box list to  [[1,1,1],[2,2,2],[3,3,3],...]
        reshape_bboxes = [list(group) for group in zip(*ori_bboxes)]
        for idx, frame_file in enumerate(sampled_frame_files):
            image = np.array(Image.open(frame_file).convert("RGB")).astype(np.uint8)
            try:
                transformed = crop_transform(image=image,bboxes=reshape_bboxes[idx],)
            except:
                import ipdb;ipdb.set_trace()
            frames.append(transformed["image"])
            resize_bboxes.append(transformed["bboxes"])
        
        out_dict = {}
        if self.preprocessor is not None:
            first_frame_img = Image.open(sampled_frame_files[0]).convert("RGB")
            first_frame_tensor = self.preprocessor(first_frame_img).unsqueeze(0)
            out_dict["first_frame_tensor"] = first_frame_tensor

        # reshape back to box [[1,2,3],[1,2,3],...] 1->[x1,y1,x2,y2]
        resize_bboxes = [list(group) for group in zip(*resize_bboxes)]
        frames = torch.stack(frames)

        bboxes = torch.Tensor(resize_bboxes)       # shape num,num_frame,4
        bboxes = torch.clamp(bboxes, min=0.0, max=1.0)
        
        # the bboxes shape num_obj, num_frame, 4, --> num_frame, num_obj, 4
        bboxes = bboxes.transpose(0,1)

        # obj_embedding -> num_obj num_frame  convert to num_frame, num_obj
        obj_embeddings = obj_embeddings.transpose(0,1)

        # obj_mask -> num_obj num_frame convert to num_frame, num_obj
        obj_mask = torch.Tensor(obj_mask).transpose(0,1)
       

        ## pad to batch 
        all_obj_bbox = torch.zeros(self.n_sample_frames, self.num_max_obj, 4)
        _,num_obj ,_  =bboxes.shape 
        if num_obj > self.num_max_obj:
            num_obj = self.num_max_obj
        all_obj_bbox[:,:num_obj,:] = bboxes [:,:num_obj,:4] 

        all_obj_embedding = torch.zeros(self.n_sample_frames, self.num_max_obj,obj_embeddings.shape[-1])
        all_obj_embedding[:, :num_obj,:]=obj_embeddings[:,:num_obj,:]

        all_obj_mask = torch.zeros(self.n_sample_frames, self.num_max_obj)
        all_obj_mask[:,:num_obj] = obj_mask[:,:num_obj]
        prompt_ids = open_clip.tokenize(prompt)

        out_dict["pixel_values"] = self.normalize_input(frames)
        out_dict["prompt_ids"] = prompt_ids 
        out_dict["boxes"] = all_obj_bbox
        out_dict["text_embeddings"] = all_obj_embedding 
        out_dict["masks"] = all_obj_mask 
        out_dict["data_name"] = "youvis"
        out_dict["text"] = prompt 
        out_dict["video_folder"] = self.video_folders[index].split("/")[-1]
        return out_dict 
    



if __name__ == "__main__":
    dataset = YOUVISDataset(
        path="./dataset/youvis",
        split="train",
        width=256,
        height=256,
        n_sample_frames=16,
    )
    print(len(dataset))

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    for step, batch in enumerate(train_dataloader):
        # print(batch["text_prompt"])
        if step % 10 == 0:
            print(step)

  