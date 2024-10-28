import json 
import os 
import torch
import open_clip 
from modelscope_t2v.text_to_video_synthesis_model import get_model_scope_t2v_models
from tqdm import tqdm 

def preprocess_got10k(pretrained_ms_pat,path):
    """
    Generate embedding for category and save
    """
    all_objects = []
    splits = ["train", "val"]
    for split in splits:
        data_path = os.path.join(path,split)
        all_foldes = os.listdir(data_path)
        for folder in tqdm(all_foldes,total=len(all_foldes)):
            if "txt" in folder:
                continue
            meta_info_path = os.path.join(data_path,folder,"meta_info.ini")
            meta_info = {}
            with open(meta_info_path,"r") as f:
                lines = f.readlines()
                for line in lines:
                    if ":" in line and "class" in line:
                        key, value = line.split(":")
                        key = key.strip()
                        value = value.strip()
                        meta_info[key] = value 
            obj = meta_info["object_class"]
            if obj not in all_objects:
                all_objects.append(obj)
    
    _, _, text_encoder, _ = get_model_scope_t2v_models(
        pretrained_ms_pat, 
        )
    text_encoder = text_encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    embedding_dict = {}
    for obj in tqdm(all_objects,total=len(all_objects)):
        obj_embedding = get_token_embedding(obj, text_encoder)
        embedding_dict[obj] = obj_embedding.detach().cpu().numpy().tolist()

    with open(os.path.join(path,"obj_embedding.json"),"w") as f:
        json.dump(embedding_dict,f)



def preprocess_youvis(pretrained_ms_path,path):
    """
    Generate new annotations {video_name: annotation}
    and category embeddings
    """
    _, _, text_encoder, _, _, _ = get_model_scope_t2v_models(
        pretrained_ms_path, 
        )
    
    text_encoder = text_encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    instance_path = os.path.join(path,"instances.json")
    with open(instance_path,"r") as f:
        data = json.load(f)
    

     # process categories embedding
    embedding_dict = {}
    for category in data["categories"]:
        obj = category["name"]
        cat_embedding = get_token_embedding(obj,text_encoder)
        embedding_dict[obj] = cat_embedding.detach().cpu().numpy().tolist()

    videos = data['videos']
    annotations = data["annotations"]
    new_annotations = {}
    new_annotations["category_embedding"] = embedding_dict
    for anno in annotations:
        video_id = anno['video_id']
        
        try:
            video_name = videos[video_id-1]['file_names'][0].split('/')[0]
        except:
            import ipdb;ipdb.set_trace()
        if video_name in new_annotations:
            new_annotations[video_name].append(anno)
        else:
            new_annotations[video_name] = [anno]
    

    with open(f"{path}/annotations_info_with_cat_emb.json", "w") as f:
        json.dump(new_annotations,f)
    print("Done !!")


def get_token_embedding(text, text_encoder):
    token = open_clip.tokenize(text)
    index = torch.where(token[0] == 49407)[0].item()
    token_embedding = text_encoder.encode(text)
    return token_embedding[:,index,:]


if __name__ == "__main__":
    pretrained_ms_path = "" # path to the pretrained ms model
    path = ""# path to the dataset

    preprocess_youvis(pretrained_ms_path,path)


