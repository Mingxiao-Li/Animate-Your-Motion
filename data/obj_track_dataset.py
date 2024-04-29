import os 
import os 
import json 
import numpy as np 
import torch 
import random 
import torchvision.transforms as T

from torch.utils.data import Dataset
from typing import List
from PIL import Image 
from collections import defaultdict
from tqdm import tqdm 
from torchvision import transforms
from glob import glob 
from einops import rearrange, repeat

class OTMetaDataset(Dataset):


    def check_objcets_in_sampled_vide(self, annotation, video_name, indeices):
        annotation_info = annotation[video_name]
        zero_obj = True 
        
        for bbox_info in enumerate(annotation_info):
            bboxs = [bbox_info['bboxes'][i] for i in indeices]
            pass 
        return zero_obj

    def get_frame_batch(self, folder_path, interval=1, random_sample=False, annotations=None):
        # Get all jpg images in the folder

        fram_files = sorted(glob(f"{folder_path}/*.jpg"))

        # Sample frame at the given interval
        if not random_sample:
            sampled_frame_files = fram_files[:: interval]
            indices = list(range(0, len(fram_files), interval))
            # Ensure we don't sample more that the required number of frames
            start_idx = 0
            if len(sampled_frame_files) >= self.n_sample_frames:
                start_idx = random.randint(0, len(sampled_frame_files)-self.n_sample_frames)
                sampled_frame_files = sampled_frame_files[start_idx: start_idx + self.n_sample_frames]
            indices=indices[start_idx: start_idx + self.n_sample_frames]

        elif random_sample:
            assert len(fram_files) >= self.n_sample_frames*interval
            start_idx = random.randint(0, len(fram_files)-interval*self.n_sample_frames)
            sampled_frame_files = fram_files[start_idx::interval]
            if len(sampled_frame_files) > self.n_sample_frames:
                sampled_frame_files = sampled_frame_files[:self.n_sample_frames]
            indices = list(range(start_idx,start_idx+self.n_sample_frames*interval,interval))
 

        # Read the first frame to determine the original size and get the resie transform
        f_sample = Image.open(sampled_frame_files[0])
        resize_transform, w_scale, h_scale = self.get_frame_buckets(f_sample)

        return sampled_frame_files, resize_transform, w_scale, h_scale, indices
    

    def get_frame_buckets(self, frame):
        w, h = frame.size 
        width, height = self.width, self.height 

        w_scale = width / w 
        h_scale = height / h 
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize, w_scale, h_scale
    

    def normalize_input(
        self,
        item,
        mean=[0.5, 0.5, 0.5],
        std =[0.5, 0.5, 0.5],
        use_simple_norm=False 
       ):
        if item.dtype == torch.uint8 and not use_simple_norm:
            item = rearrange(item, "f c h w -> f h w c")
            item = item.float() / 255.0
            mean = torch.tensor(mean)
            std = torch.tensor(std)

            out = rearrange((item-mean) / std, "f h w c -> f c h w")
            return out 
        else:
            item = rearrange(item, "f c h w -> f h w c")
            return rearrange(item / 127.5 - 1.0, "f h w c -> f c h w")
        





class ObjectTrackingDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_name: List[str],
                 root_path: str,
                 split: str = "train",
                 width: int = 256,
                 height: int = 256,
                 n_sample_frames: int = 8,
                 got_data_sample_ratio: int = 5
                 ):
        
        dataset_path = []
        for name in data_name:
            dataset_path.append(os.path.join(root_path, name))
        self.tokenizer = tokenizer
        self.n_sample_frames = n_sample_frames
        self.split = split 
        self.image_size = (height, width)

        self.data = []

        assert len(data_name) !=0, "data_name is empty"

        if "got" in data_name:
            got_data_path =os.path.join(root_path,"got-10k")
            self.data.extend(self._load_got_dataset(got_data_path, split, got_data_sample_ratio))
        
        if "youvis" in data_name:
            youvis_data_path = os.path.join(root_path, "youvis")
            self.data.extend(self._load_youvis_dataset(youvis_data_path, split))
        
        import ipdb;ipdb.set_trace()

    def resize_img_and_adjust_box(self, pil_image_list, image_size, box_coordinates):
        """
        box coordinate = [x, y, width, height]
        --> change to [x1,y1, x2, y2] and represented by ratio (0~1)
        x -> horizontal left to right  0 to max
        y -> vertical top to down 0 to max
        """
        width, height = image_size
        WW, HH = pil_image_list[0].size 

        w_scale = width / WW 
        h_scale = height / HH 
        
        resized_img_list = []
        for pil_image in pil_image_list:
            pil_image = pil_image.resize(
                tuple(round(WW*w_scale),round(HH*h_scale)), resample=Image.BICUBIC
            )
            tensor_img = transforms.ToTensor()(pil_image)
            resized_img_list.append(tensor_img)
            
        for obj_boxes in box_coordinates:
            for box in obj_boxes:
                if box is None:
                    continue 
                else:
                    box[0] = box[0] / WW 
                    box[1] = box[1] / HH 
                    box[2] = box[2] / WW + box[0]
                    box[3] = box[3] / HH + box[1]
        
        tensor_box_coordinates = torch.tensor(box_coordinates)
        tensor_image_list = torch.cat(resized_img_list,dim=0)
        return tensor_image_list, tensor_box_coordinates
        

    def center_crop_arr(self, pile_image, image_size):
        pass 


    def _load_got_dataset(self, data_path, split, got_data_sample_ratio):
        got_data_list = load_got_dataset(data_path, split, got_data_sample_ratio)
        return got_data_list

    def _load_youvis_dataset(self,data_path, split):
        youvis_data_list = load_youvis_dataset(data_path, split) 
        return youvis_data_list
    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        current_data = self.data[index]
        data_dict = {}

        video_image_list = current_data["images"]
        box_coordinates = current_data["box_coordinates"]
        objects = current_data["objects"]

        if "caption" in current_data:
            caption = current_data["caption"]
        else:
            caption = ""
        
        # get image list here 

        img_tensor, box_coordinates = self.resize_img_and_adjust_box(
            video_image_list, self.image_size, box_coordinates,
        )

        prompt_ids = self.tokenizer(
            caption, max_length=77, padding='max_length', truncation=True, return_tensors='pt'
        )
        
        ## process image tensor 
        
        data_dict["prompt_ids"] = prompt_ids
        data_dict["boxes"] = box_coordinates
        data_dict["prompt_padding_mask"]
        data_dict['text_masks'] # # indicating how many objects still there 
        data_dict['mask']  # obj mask
        return data_dict







#--------------------------------------------------------------------
    
def load_youvis_dataset(data_path, split):
    """
    data_info-> structure dict
    data keys = info, videos, categories, annotations
    data[video] (list)[0] = dict{ file_names, height, width, length, id}
    data['categories'] (list)[0] = { supercategory , id, name }
    data['annotations] (list)[0] = {video_id, height, width, length, bboxes, category_id, id,}
    """
    print("Loading YouVis Dataset ...")
    data_list = []
    img_folder = os.path.join(data_path, f'{split}/JEPGImages')
    info_file = os.path.join(data_path, f'{split}/instances.json')
    caption_file = os.path.join(data_path, 'caption.json')

    with open(info_file, 'r') as file:
        data_info = json.load(file)
    
    categories = data_info['categories']
    videos_info = data_info['videos']  
    annotations = data_info['annotations']
    with open(caption_file, 'r') as file:
        captions = json.load(file)
    data_dict = defaultdict(dict)
    for anno in tqdm(annotations, total=len(annotations)):
        video_id = anno['video_id']
        video_info = videos_info[video_id]
        if len(data_dict[video_id]) == 0:
            data_dict[video_id]['video_name'] = video_info['file_names'][0].split('/')[0]
            data_dict[video_id]['caption'] = captions[data_dict[video_id]['video_name']]
            height, width = anno['height'], anno['width']
            data_dict[video_id]['resolution'] = (height, width)
            data_dict[video_id]['objects'] = [categories[anno['category_id']]]
            data_dict[video_id]['box_coordinates'] = [anno['bboxes']]
            
            data_dict[video_id]['images'] = []
            for img_name in video_info['file_names']:
                img_path = os.path.join(img_folder, img_name)
                try:
                    img = Image.open(img_path)
                    data_dict[video_id]['images'].append(img)
                except IOError:
                    print(f'Error loading images: {img_path}')
            
        else:
            assert anno['height'] == data_dict[video_id]['resolution'][0]
            assert anno['width'] == data_dict[video_id]['resolution'][1]
            data_dict[video_id]['objects'].append(categories[anno['category_id']]) 
            data_dict[video_id]['box_coordinates'].append(anno['bboxes'])
    
    for k,v in data_dict.items():
        data_list.append(
            v.update(
                {
                    "video_id": k
                }
            )
        )
    return data_list



def load_got_dataset(data_path, split, got_data_sample_ratio):
    """
    Return data_list[{}]
    {
    image_nmae: name
    images: [img1,img2...]
    objects: [bear] 
    resolution: (heihgt, width)
    box_coordinates: [[x1,y1,x2,y2]]
    }
    """
    print("Loading GOT-10K dataset ...")
    data_list = []
    data_path = os.path.join(data_path,split)
    all_images = os.listdir(data_path)
    for img_name in tqdm(all_images):
        data_dict = {}
        data_dict["video_name"] = img_name
        folder_path = os.path.join(data_path, img_name)
        data_dict["images"] = []
        for img_file in sorted(os.listdir(folder_path))[::got_data_sample_ratio]:
            if img_file.endswith(".jpg") or img_file.endswith(".jpeg"):
                img_path = os.path.join(folder_path, img_file)
                try:
                    img = Image.open(img_path)
                    data_dict["images"].append(img)
                except IOError:
                    print(f"Error loading image: {img_path}")
        meta_info_path = os.path.join(folder_path, "meta_info.ini")
        major_class, resolution = read_got_major_class(meta_info_path)
        data_dict["objects"] = [major_class]
        data_dict["resolution"] = resolution

        box_info_path = os.path.join(folder_path, "groundtruth.txt")
        data_dict["box_coordinates"] = read_box_coordinate(box_info_path)
        data_list.append(data_dict)
    return data_list
         

def read_got_major_class(file_path):
    major_class = None
    resolution = None
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("major_class"):
                major_class = line.split(":")[1].strip()
            elif line.startswith("resolution"):
                resolution_str = line.split(":")[1].strip().strip("()")
                resolution = tuple(map(int, resolution_str.split(",")))
            
            if major_class is not None and resolution is not None:
                break 
    return major_class, resolution



def read_box_coordinate(file_path):
    coordinates = []
    with open(file_path, "r") as file:
        for line in file:
            coordinate = [float(x) for x in line.strip().split(",")]
            coordinates.append(coordinate)
    return [coordinates]


if __name__ == "__main__":
    from torch.utils.data import DataLoader 
    import transformers 
    pretrained_model_path = "/data/leuven/333/vsc33366/projects/Diffusion-Video/stable-diffusion-v1-4"
    tokenizer = transformers.CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    
    obj_dataset = ObjectTrackingDataset(
        tokenizer=tokenizer,
        data_name=["youvis"],
        root_path="/data/leuven/333/vsc33366/projects/Diffusion-Video/dataset",
    )

    dataloader = DataLoader(
        obj_dataset,
        batch_size=2,
        num_workders=0
    )
    
    for i, batch in enumerate(dataloader):
        x = batch
        import ipdb;ipdb.set_trace()
