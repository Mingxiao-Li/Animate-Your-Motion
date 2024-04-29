import argparse
import datetime
import logging
import inspect
import itertools
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage
import torch
import torch.utils.checkpoint
import diffusers
import transformers
import random 
import numpy as np 
import inference_utils
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from data.gligen_dataset.concat_dataset import ConCatDataset
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from trackers import MyWandbTracker
from torchvision import transforms
from PIL import Image, ImageDraw
from collections import OrderedDict
# from transformers import CLIPTokenizer
from text_grounding_input import GroundingNetInput
from modelscope_t2v.ms_pipeline import MSVideoGenerationPipeline
from train import Trainer
from modelscope_t2v.text_to_video_synthesis_model import get_model_scope_t2v_models
from util import save_videos_grid, draw_boxes
from data.concate_dataset import CatObjTrackVideoDataset, collate_fn
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")
import albumentations as A
import open_clip 
from albumentations.pytorch import ToTensorV2


def get_token_embedding(text, text_encoder):
    token = open_clip.tokenize(text)
    index = torch.where(token[0] == 49407)[0].item()
    token_embedding = text_encoder.encode(text)
    return token_embedding[:,index,:]

def data_preprocess(input_data_path):
    bbox_path = f"{input_data_path}/bbox.txt"
    img_path = f"{input_data_path}/img.png"
    image = np.array(Image.open(img_path).convert("RGB")).astype(np.uint8)
    
    crop_transform = A.Compose(
           [A.Resize(height=256, width=256),
            ToTensorV2(),],
            bbox_params=A.BboxParams(format="albumentations"),
        )
    height, width = image.shape[0],image.shape[1]

    bboxes = []
    with open(bbox_path, "r") as f:
        all_box = f.readlines()
        for box in all_box:
            box = box.strip().split(",")
            box = [float(x) for x in box]
            box[0] = box[0] / width
            box[1] = box[1] / height 
            box[2] = box[2] / width 
            box[3] = box[3] / height
            box.append(0)
            bboxes.append(box)
    transformed = crop_transform(image=image, bboxes=bboxes)
    resized_bboxes=transformed["bboxes"]
    frame = transformed["image"]

    resized_bboxes=[
        [bb[0],bb[1],bb[2],bb[3]] for bb in resized_bboxes
    ]
    return frame, resized_bboxes
    


def main(
        root_path: str,
        pretrained_model_path: str,
        pretrained_weight_path: None,
        train_grounding: bool = False,
        prompt: bool=str,
        gradient_accumulation_steps: int = 1,
        mixed_precision: Optional[str] = "fp16",
        enable_xformers_memory_efficient_attention: bool = True,
        cat_init: bool = False,
        seed: Optional[int] = None,
        trained_ckpt_path: str = None,
        video_length: int = 8,
        cross_vision: bool = False,
        output_dir: str= "I", 
        control_net: bool = False,
        input_data_path: str ="",
        guidance_scale: float = 7.5
    ):
    
    accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    #################################################
    # Load scheduler and models
    unet, vae, text_encoder, diffusion, position_net, visual_encoder, controlnet_encoder = get_model_scope_t2v_models(
        root_path, 
        train_grounding=train_grounding,
        pretrained_weight_path=pretrained_weight_path,
        cat_init=cat_init,
        cross_vision=cross_vision,
        control_net=control_net,
    )
    
    if trained_ckpt_path is not None:
        unet_weights = torch.load(os.path.join(trained_ckpt_path, 'pytorch_model.bin'))
        position_path = os.path.join(trained_ckpt_path, 'pytorch_model_1.bin')
        control_net_path = os.path.join(trained_ckpt_path, 'pytorch_model_2.bin')
        visual_encoder_path = os.path.join(trained_ckpt_path, 'pytorch_model_2.bin')
      
        missing_keys, unexpected_keys = unet.load_state_dict(unet_weights, strict=False)
        print("********************************")
        print("Loading pretrained weights from ", trained_ckpt_path)
        print(" Missing Keys : ", missing_keys)
        print(" Unexpected Keys :", unexpected_keys)
        print("********************************")
        assert len(unexpected_keys) == 0, "get unexpected_keys when loading ckpt"

        if position_net is not None and os.path.exists(position_path):
            position_net_weights = torch.load(position_path)
            missing_keys, unexpected_keys = position_net.load_state_dict(position_net_weights, strict=True)
            assert len(unexpected_keys) == 0, "get unexpected_keys when loading position net"

        if controlnet_encoder is not None:
            control_net_path = os.path.join(trained_ckpt_path, 'pytorch_model_2.bin')
            control_net_weights = torch.load(control_net_path)
            missing_keys, unexpected_keys = controlnet_encoder.load_state_dict(control_net_weights, strict=True)
            assert len(unexpected_keys) == 0, "get unexpected_keys when loading control net"
        
        if visual_encoder is not None:
            visual_encoder_path = os.path.join(trained_ckpt_path, 'pytorch_model_2.bin')
            visual_encoder_weights = torch.load(visual_encoder_path)
            missing_keys, unexpected_keys = visual_encoder.load_state_dict(visual_encoder_weights, strict=False)
            assert len(unexpected_keys) == 0, "get unexpected_keys when loading visual encoder"
    #################################################
    
    #################################################
    # Set trainable and non-trainable parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    if position_net is not None:
        position_net.requires_grad_(False)
    if visual_encoder is not None:
        visual_encoder.requires_grad_(False)
    if controlnet_encoder is not None:
        controlnet_encoder.requires_grad_(False)

    ################################################

    ################################################
    # Set xofrmer or gradient checkpoint for saving memory
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    #################################################

   
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)
    if position_net is not None:
        position_net.to(accelerator.device)
    if visual_encoder is not None:
        visual_encoder.to(accelerator.device)

    if controlnet_encoder is not None:
        controlnet_encoder.to(accelerator.device)


    pipeline = MSVideoGenerationPipeline(
        cross_vision=cross_vision,
        vae=vae, 
        text_encoder=text_encoder, 
        unet=unet, 
        position_net=position_net,
        diffusion=diffusion, 
        visual_encoder=visual_encoder,
        controlnet_encoder=controlnet_encoder)
    
    prompt = [prompt]
    resized_rgb_img, resized_bboxes = data_preprocess(
        input_data_path=input_data_path
    )
    
    bboxes_tensor = torch.tensor(resized_bboxes).unsqueeze(0).unsqueeze(2)
    pixel_img = resized_rgb_img.unsqueeze(0) / 127.5 -1.0
    #zeros_embd = torch.zeros(1, 8, 8, 1024)
    text_embedding = get_token_embedding("dog", text_encoder)
    text_embedding = text_embedding.unsqueeze(0).unsqueeze(1).repeat(1,8,1,1)
    #zeros_embd[:,:,:1,:] = text_embedding
    #text_embedding = zeros_embd.to(accelerator.device)
    # cond_pixel=  shape [n,3,256,256] (-1,1)
    input_batch = {
        "pixel_values": pixel_img.to(accelerator.device),
        "boxes": bboxes_tensor.to(accelerator.device).to(torch.float32),
        "text_embeddings": text_embedding.to(torch.float32), 
        "masks": torch.ones(1,8,1,device=accelerator.device),
    }

    
    output = pipeline(
            prompt = prompt,
            video_length = video_length,
            height = 32, 
            width = 32, 
            num_inference_steps = 50,
            guidance_scale = guidance_scale,
            first_frame_pixel = pixel_img,
            batch = input_batch,
            obj_oriented_attn_ratio = 0.0,
            cat_init=cat_init
        )
    
    save_videos_grid(output,f"{output_dir}/cus_pred.gif",n_rows=1,fps=8)
    with open(f"{output_dir}/prompt.txt","w") as f:
        f.write(prompt[0])
    
    boxes_pred_imgs = []
    img_list = torch.transpose(output[0],1,0)
    boxes = bboxes_tensor[0]
    for i, img in enumerate(img_list):
        pred_box_img_path = f"{output_dir}/pred_box_img"
        os.makedirs(pred_box_img_path, exist_ok=True)

        pil_box_pred_img = draw_boxes(img * 256 / 127.5 -1, boxes[i], return_img=True)
        pil_box_pred_img.save(f"{pred_box_img_path}/pred_box_img_{i}.png")
        boxes_pred_imgs.append(pil_box_pred_img)
    boxes_pred_imgs[0].save(f"{output_dir}/boxes_pred.gif",save_all=True,append_images=boxes_pred_imgs[1:],optimize=False,duration=8)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/flintstones.yaml")
    parser.add_argument("--train_grounding", type=str, default=None)
    parser.add_argument("--cat_init", action="store_true")
    parser.add_argument("--pretrained_weight_path", type=str, default=None)
    parser.add_argument("--video_length", type=int, default=8)
    parser.add_argument("--trained_ckpt_path", type=str, default=None)
    parser.add_argument("--output_dir", default=".",type=str)
    parser.add_argument("--cross_vision", action="store_true")
    parser.add_argument("--control_net", action="store_true", help="add image control same as control net")
    parser.add_argument("--input_data_path",type=str,default="")
    parser.add_argument("--guidance_scale",type=float,default=7.5)
    parser.add_argument("--prompt", type=str, default="")
    
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    OmegaConf.update(config, "train_grounding", args.train_grounding)
    OmegaConf.update(config, "cat_init", args.cat_init)
    OmegaConf.update(config, "trained_ckpt_path", args.trained_ckpt_path)
    OmegaConf.update(config, "video_length", args.video_length)
    OmegaConf.update(config, "pretrained_weight_path", args.pretrained_weight_path)
    OmegaConf.update(config, "cross_vision", args.cross_vision)
    OmegaConf.update(config, "output_dir", args.output_dir)
    OmegaConf.update(config, "control_net", args.control_net)
    #OmegaConf.update(config, "data_path", args.data_path)
    OmegaConf.update(config, "guidance_scale", args.guidance_scale)
    OmegaConf.update(config, "prompt", args.prompt)
    OmegaConf.update(config, "input_data_path", args.input_data_path)
    main(**config)
