import os
import os.path as osp
import csv
import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
from transformers import CLIPTokenizer

class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            pretrained_model_path: str,
            split: str = "2M_train",  # train or val
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.data_path = osp.join(data_path, split)
        # self.data_path = '/user/leuven/347/vsc34708/stage/dataset/webvid/2M_val/'
        video_ids = os.listdir(osp.join(self.data_path, 'videos'))
        self.video_ids = [idx.replace('.mp4', '') for idx in video_ids]
        self.gt = self.get_gt()

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_max_length = self.tokenizer.model_max_length

    def get_gt(self):
        gt = dict()
        text_path = osp.join(self.data_path, 'results.csv')
        with open(text_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                vid, text = row[0], row[1]
                if vid in self.video_ids:
                    gt[vid] = text
        return gt

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):
        vid = self.video_ids[index]
        # load and sample video frames
        vr = decord.VideoReader(osp.join(self.data_path, 'videos', vid+'.mp4'), width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        prompt = self.gt.get(vid)

        tokens = self.tokenizer(
            prompt, max_length=self.text_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        prompt_ids = tokens.input_ids[0]
        prompt_padding_mask = tokens.attention_mask[0]
        
        # for testing no text prompt exp
        #prompt = ""
        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids,
            "prompt_padding_mask": prompt_padding_mask,
            "prompt_text": prompt,
        }

        return example


if __name__ == "__main__":
    data_path = "/user/leuven/347/vsc34708/stage/dataset/webvid/2M_val/"
    pretrained_model_path = "/user/leuven/347/vsc34708/stage/stable-diffusion-v1-4"
    train_dataset = TuneAVideoDataset(data_path=data_path, pretrained_model_path=pretrained_model_path)
    count = 0
    for data in train_dataset:
        print(data["pixel_values"].shape)
        print(data["prompt_ids"].shape)
        count += 1
        if count >= 5:
            break
