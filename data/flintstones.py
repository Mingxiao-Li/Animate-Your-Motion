from torch.utils.data import Dataset
from einops import rearrange
from PIL import Image
from util import b2f 
from transformers import CLIPTokenizer

import pickle
import numpy as np 
import torch 

class FlinstStonesDataset(Dataset):

    def __init__(
            self,
            pretrained_model_path: str, 
            data_path: str,
            split: str = "train",  # train or val
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 2,
    ):  
        
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.sample_start_idx = sample_start_idx
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.data, self.key_lisst = self._load_data(data_path, split)
        self.text_max_length = self.tokenizer.model_max_length

    def _load_data(self, path, split):
        content = open(path, "rb")
        data = pickle.load(content)[split]
        key_list = list(data.keys())

        return data, key_list
        
    
    def __len__(self):
        return len(self.key_lisst)
    

    def __getitem__(self, index):
        key = self.key_lisst[index]
        cur_data = self.data[key]

        prompt = cur_data['text']
        video = [b2f(b) for b in cur_data['video']]

        sample_index = list(range(self.sample_start_idx, len(video), self.sample_frame_rate))[:self.n_sample_frames]
        video_tensor_list = [torch.tensor(np.array(video[index])).unsqueeze(0) for index in sample_index] 
        video_tensor = torch.cat(video_tensor_list, dim=0)  # f, w, h ,c
        video_tensor = rearrange(video_tensor, "f h w c -> f c h w")
        tokens = self.tokenizer(
            prompt, max_length=self.text_max_length,padding="max_length", truncation=True, return_tensors="pt"
        )
        prompt_ids = tokens.input_ids[0]
        prompt_padding_mask = tokens.attention_mask[0]

        batch = {
            "pixel_values": (video_tensor / 127.5 - 1.0),
            "prompt_ids": prompt_ids,
            "prompt_padding_mask": prompt_padding_mask,
            "prompt_text": prompt
        }

        return batch 

if __name__ == "__main__":
    data_path = "./dataset/Flintstones/flintstones.pkl"
    pretrained_model_path = "./stable-diffusion-v1-4"
    #tokenizer = CLIPTokenizer.from_pretrained(pretraiend_model_path, subfolder="tokenizer")
    train_dataset = FlinstStonesDataset(
        pretrained_model_path=pretrained_model_path,
        data_path=data_path
    )
    count = 0
    for data in train_dataset:
        if data["pixel_values"].shape[0] != 8:
            import ipdb 
            ipdb.set_trace()
        print(data["pixel_values"].shape)
        print(data["prompt_ids"].shape)
        count += 1
        print(count)
