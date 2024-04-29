from torch.utils.data import Dataset
from data.got10k_dataset import GOT10KDataset
from data.youvis_dataset import YOUVISDataset
from typing import List
import torch 
import importlib
import os 

class CatObjTrackVideoDataset(Dataset):

    def __init__(
            self,
            path: str,
            name: str="object_tracking",
            datasets: List[str]=["got10k", "youvis"],
            n_sample_frames: int=8,
            width: int = 256,
            height: int = 256,
            split: str = "train",
            num_max_obj: int = 8,
            fallback_prompt: str = "",
            repeats: int=None,
            preprocessor = None,
            ):
        
        dataset_params={
            "n_sample_frames": n_sample_frames,
            "width": width,
            "height": height,
            "split" : split,
            "num_max_obj": num_max_obj,
            "fallback_prompt": fallback_prompt,
        }
        
        if repeats is None:
            repeats = [1] * len(datasets)
        else:
            assert len(repeats) == len(datasets)

        #offset_map = []
        which_dataset = []
        cul_previous_dataset_length = 0 
        # building dataset
        self.datasets = []
        for idx, dataset in enumerate(datasets):
            repeat = repeats[idx]
            if dataset == "got10k":
                print("Loading GOT10K Dataset !!!")
                dataset_params["path"] = os.path.join(path,"got-10k") 
                dataset = GOT10KDataset(
                       **dataset_params,
                       preprocessor=preprocessor)
            elif dataset == "youvis":
                print("Loading YOUVIS Dataset !!!")
                dataset_params["path"] = os.path.join(path, "youvis")
                dataset = YOUVISDataset(
                    **dataset_params,
                    preprocessor=preprocessor)
                
            self.datasets.append(dataset)
            for _ in range(repeat):
                #offset_map.append(torch.ones(len(dataset) * cul_previous_dataset_length))
                which_dataset.append(torch.ones(len(dataset)) * idx)
                cul_previous_dataset_length += len(dataset)
        
       # offset_map = torch.cat(offset_map, dim=0).long()
        self.total_length = cul_previous_dataset_length
        
       # self.mapping = torch.arange(self.total_length) - offset_map
        self.which_dataset = torch.cat(which_dataset, dim=0).long()
        shuffle_index = torch.randperm(self.which_dataset.shape[0])
        self.which_dataset = self.which_dataset[shuffle_index]

    def __getitem__(self, index: int):
        dataset = self.datasets[self.which_dataset[index]]
       # print(dataset)
        data_idx = index % len(dataset)
        return dataset[data_idx]
    
    def __len__(self):
        return self.total_length
    

def collate_fn(data_list,max_num_obj, num_sample_frames):
    """
    data: dict{pixel_value ; prompt_ids; bboxes ; bboxes_class ; masks}

    masks -> bbox masks
    text_embeddings -> bbox_embeddings
    """
    data_batch = {
        "pixel_values" : [],
        "prompt_ids"  : [],
        "boxes": [],
        "text_embeddings" :[],
        "masks": [],
        "name": [],
        "first_frame": []
        }
    
    for data in data_list:
        if data["data_name"] == "got10k":
            # got10k dataset shape=[num_frame,box_coord]
            # pad to multi objs shape=[num_obj, num_frame, box_coord]
            try:
                objs = torch.zeros(num_sample_frames, max_num_obj, 4)
                objs[:,:1,:] = data["boxes"]
                data["boxes"] = objs

                # pad obj embedding, got10k single obj bboxes_class=str 
                # pad it to [[[obj]* frame ] [0,0] obj]
                obj_emb = torch.zeros(num_sample_frames, max_num_obj, data["text_embeddings"].shape[-1])
                obj_emb[:,:1,:]=data["text_embeddings"]
                data["text_embeddings"] = obj_emb

                # pad obj mask, got10k single obj masks [[1]*num_frame]
                # pad it to [[1]* num_frame, [,0,0,] obj]
                bbox_mask = torch.zeros(num_sample_frames,max_num_obj)
                bbox_mask[:,:1] = data["masks"]
                data["masks"] = bbox_mask
            except:
                import ipdb;ipdb.set_trace()
        
        if "first_frame_tensor" in data:
            data_batch["first_frame"].append(data["first_frame_tensor"])

        data_batch["name"].append(data["data_name"])
        data_batch["pixel_values"].append(data["pixel_values"].unsqueeze(0))
        data_batch["boxes"].append(data["boxes"].unsqueeze(0))
        data_batch["text_embeddings"].append(data["text_embeddings"].unsqueeze(0))
        data_batch["masks"].append(data["masks"].unsqueeze(0))
        data_batch["prompt_ids"].append(data["prompt_ids"])
    data_batch["pixel_values"] = torch.cat(data_batch["pixel_values"],dim=0)
    data_batch["boxes"] = torch.cat(data_batch["boxes"],dim=0)
    data_batch["text_embeddings"] = torch.cat(data_batch["text_embeddings"],dim=0)
    data_batch["masks"] =torch.cat(data_batch["masks"], dim=0)
    data_batch["prompt_ids"] = torch.cat(data_batch["prompt_ids"],dim=0)
    if len(data_batch["first_frame"]) > 0:
        try:
            data_batch["first_frame"] = torch.cat(data_batch["first_frame"],dim=0)
        except:
            import ipdb;ipdb.set_trace()
    return data_batch
    


if __name__ == "__main__":
    from functools import partial
    dataset = CatObjTrackVideoDataset(
        path="/data/leuven/333/vsc33366/projects/Diffusion-Video/dataset"
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(collate_fn,max_num_obj=8, num_sample_frames=8)
    )

    for step, batch in enumerate(dataloader):
        for k,v in batch.items():
            if isinstance(v,torch.Tensor):
                print(k, v.shape)
            else:
                print(k, v)
