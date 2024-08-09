import argparse
import logging
import os
from typing import Dict, Optional
from omegaconf import OmegaConf
import torch
import torch.utils.checkpoint
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from modelscope_t2v.ms_pipeline import MSVideoGenerationPipeline
from modelscope_t2v.text_to_video_synthesis_model import get_model_scope_t2v_models
from util import save_videos_grid,draw_boxes,visual_img
from data.concate_dataset import CatObjTrackVideoDataset
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")

def main(
        root_path: str,
        pretrained_model_path: str,
        eval_data: Dict,
        pretrained_weight_path: None,
        train_grounding: bool = False,
        gradient_accumulation_steps: int = 1,
        mixed_precision: Optional[str] = "fp16",
        enable_xformers_memory_efficient_attention: bool = True,
        cat_init: bool = False,
        seed: Optional[int] = None,
        trained_ckpt_path: str = None,
        video_length: int = 8,
        use_image_dataset: bool = False,
        use_tmp_window_attention: bool = False,
        tmp_window_size: int = 3,
        cross_vision: bool = False,
        use_clip_visual_encoder: bool = False, 
        dataset: str= "got10k",
        num_workers: int = 1,
        batch_size: int =2 ,
        guidance_scale: float = 7.5,
        output_dir: str=".",
        control_net: bool = False,
        img_free_guidance: float=0.0,
        ti_free_guidance: float = 0.0,
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
        use_tmp_window_attention=use_tmp_window_attention,
        tmp_window_size=tmp_window_size,
        use_image_dataset=use_image_dataset,
        cross_vision=cross_vision,
        use_clip_visual_encoder=use_clip_visual_encoder,
        control_net=control_net
    )
    
    if trained_ckpt_path is not None:
        import ipdb;ipdb.set_trace()
        weights = torch.load(os.path.join(trained_ckpt_path,"pytorch_model.bin"))
        missing_keys, unexpected_keys = unet.load_state_dict(weights, strict=False)
        print("********************************")
        print("Loading pretrained weights from ", trained_ckpt_path)
        print(" Missing Keys : ", missing_keys)
        print(" Unexpected Keys :", unexpected_keys)
        print("********************************")
        assert len(unexpected_keys) == 0, "get unexpected_keys when loading ckpt"

        position_net_weights = torch.load(os.path.join(trained_ckpt_path,'pytorch_model_1.bin'))
        missing_keys, unexpected_keys = position_net.load_state_dict(position_net_weights, strict=False)
        print("***********Position-Net*********************")
        print(" Missing Keys : ", missing_keys)
        print(" Unexpected Keys :", unexpected_keys)
        print("********************************")
       
        if visual_encoder is not None:
            visual_encoder_weights = torch.load(os.path.join(trained_ckpt_path,"pytorch_model_2.bin"))
            missing_keys, unexpected_keys = visual_encoder.load_state_dict(visual_encoder_weights, strict=False)
            print("***********Visual Encoder*********************")
            print(" Missing Keys : ", missing_keys)
            print(" Unexpected Keys :", unexpected_keys)
            print("********************************")
        
        if controlnet_encoder is not None:
            control_net_weights = torch.load(os.path.join(trained_ckpt_path, "pytorch_model_2.bin"))
            missing_keys, unexpected_keys = controlnet_encoder.load_state_dict(control_net_weights, strict=True)
            assert len(unexpected_keys) == 0, "get unexpected_keys when loading control net"

    #################################################
    # Set trainable and non-trainable parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    if controlnet_encoder is not None:
        controlnet_encoder.requires_grad_(False)
    if visual_encoder is not None:
        visual_encoder.requires_grad_(False)
    ################################################

    ################################################
    # Set xofrmer or gradient checkpoint for saving memory
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    #################################################
    
    eval_data.datasets=[dataset]
    if dataset == "youvis":
        eval_data.split="train"
    eval_dataset = CatObjTrackVideoDataset(**eval_data)
    
    if dataset == "youvis":
        max_num = 180
    else:
        max_num = len(eval_dataset)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=False,
        num_workers = num_workers,
        batch_size = batch_size,
    )
    
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
        diffusion=diffusion, 
        position_net=position_net,
        visual_encoder=visual_encoder,
        controlnet_encoder=controlnet_encoder)
    
    num_obj = 0
    for index, batch in enumerate(eval_dataloader):
        print(f"Genreating {index}/{max_num//batch_size} samples")
        gt_video = batch["pixel_values"].to(accelerator.device)
        prompt = batch["text"]
        cond_pixel = gt_video[:,0]
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(accelerator.device)
        if num_obj >= max_num:
            break 
        num_obj += batch_size
      
        #import ipdb;ipdb.set_trace()
        #cond_pixel = torch.zeros_like(cond_pixel)
        #batch["pixel_values"] = torch.zeros_like(batch["pixel_values"])
        
        output = pipeline(
            prompt = prompt,
            video_length = video_length,
            height = 32, 
            width = 32, 
            num_inference_steps = 50,
            guidance_scale = guidance_scale,
            first_frame_pixel = cond_pixel,
            batch = batch,
            obj_oriented_attn_ratio = 0.0,
            img_free_guidance=img_free_guidance,
            ti_free_guidance=ti_free_guidance,
            cat_init=cat_init
        )
        if dataset == "got10k":
            batch["boxes"] = batch["boxes"].reshape(batch_size, 8, -1) #ideo_length,-1)
            outputs = output[:,:,:8,:,:]
            for i in range(batch_size):
                print("Saving -> ", batch['text'][i])
                folder = batch["video_folder"][i]
                save_path = f"{output_dir}/{folder}"
                os.makedirs(save_path, exist_ok=True)

                output = outputs[i]
                save_videos_grid(gt_video[i].transpose(0,1).unsqueeze(0).cpu(),f"{save_path}/gt_video.gif",n_rows=1, fps=video_length,rescale=True)
                save_videos_grid(output.unsqueeze(0),f"{save_path}/pred_video.gif",n_rows=1,fps=video_length)
                with open(f"{save_path}/prompt.txt", "w") as file:
                    file.write(batch["text"][i])
                img_list = torch.transpose(output, 1, 0)
                boxes = batch["boxes"][i]
                boxes_pred_imgs = []
                boxes_gt_imgs = []
         
                for j, img in enumerate(img_list):
                    pred_box_img_path = f"{save_path}/pred_box_img"
                    os.makedirs(pred_box_img_path, exist_ok=True)
                    
                    pil_box_pred_img = draw_boxes(img * 255/ 127.5 -1, boxes[j].unsqueeze(0), return_img=True)
                    pil_box_pred_img.save(f"{pred_box_img_path}/pred_box_img_{j}.png")
                    boxes_pred_imgs.append(pil_box_pred_img)

                    gt_box_img_path = f"{save_path}/gt_box_img"
                    os.makedirs(gt_box_img_path, exist_ok=True)
                    pil_box_gt_img = draw_boxes(gt_video[i][j],boxes[j].unsqueeze(0), return_img=True)
                    pil_box_gt_img.save(f"{gt_box_img_path}/gt_box_img_{j}.png")
                    boxes_gt_imgs.append(pil_box_gt_img)

                    pred_img_path = f"{save_path}/pred_img"
                    os.makedirs(pred_img_path, exist_ok=True)
                    pil_pred_img = visual_img(img * 255/ 127.5 -1)
                    pil_pred_img.save(f"{pred_img_path}/pred_img_{j}.png")

                    gt_img_path = f"{save_path}/gt_img"
                    os.makedirs(gt_img_path,exist_ok=True) 
                    pil_gt_img = visual_img(gt_video[i][j])
                    pil_gt_img.save(f"{gt_img_path}/gt_img_{j}.png")
                    
                    cur_box = boxes[j].cpu().tolist()
                    with open(f"{save_path}/box_{j}.txt", "w") as f:
                        string = ",".join([str(x) for x  in cur_box]) 
                        f.write(string + "\n")
                
                boxes_pred_imgs[0].save(f"{save_path}/boxes_pred.gif",save_all=True,append_images=boxes_pred_imgs[1:],optimize=False,duration=8)
                boxes_gt_imgs[0].save(f"{save_path}/boxes_gt.gif",save_all=True,append_images=boxes_gt_imgs[1:],optimize=False,duration=8)
        elif dataset == "youvis":
            for i in range(batch_size):
                print("Saving -> ", batch['text'][i])
                folder = batch["video_folder"][i]
                save_path = f"{output_dir}/{folder}"
                os.makedirs(save_path, exist_ok=True)
                save_videos_grid(gt_video[i].transpose(0,1).unsqueeze(0).cpu(),f"{save_path}/gt_video.gif",n_rows=1, fps=video_length,rescale=True)
                save_videos_grid(output[i].unsqueeze(0),f"{save_path}/pred_video.gif",n_rows=1,fps=video_length)
                with open(f"{save_path}/prompt.txt", "w") as file:
                    file.write(batch["text"][i])
                img_list = torch.transpose(output[i], 1, 0)
                batch["boxes"] = batch["boxes"].view(batch_size, video_length, 8, -1)
                boxes = batch["boxes"][i]
                boxes_pred_imgs = []
                boxes_gt_imgs = []
                for j, img in enumerate(img_list):
                    pred_box_img_path = f"{save_path}/pred_box_img"
                    os.makedirs(pred_box_img_path, exist_ok=True)
                    pil_box_pred_img = draw_boxes(img * 255/ 127.5 -1, boxes[j], return_img=True)
                    pil_box_pred_img.save(f"{pred_box_img_path}/pred_box_img_{j}.png")
                    boxes_pred_imgs.append(pil_box_pred_img)

                    gt_box_img_path = f"{save_path}/gt_box_img"
                    os.makedirs(gt_box_img_path, exist_ok=True)
                    pil_box_gt_img = draw_boxes(gt_video[i][j],boxes[j], return_img=True)
                    pil_box_gt_img.save(f"{gt_box_img_path}/gt_box_img_{j}.png")
                    boxes_gt_imgs.append(pil_box_gt_img)

                    pred_img_path = f"{save_path}/pred_img"
                    os.makedirs(pred_img_path, exist_ok=True)
                    pil_pred_img = visual_img(img * 255/ 127.5 -1)
                    pil_pred_img.save(f"{pred_img_path}/pred_img_{j}.png")

                    gt_img_path = f"{save_path}/gt_img"
                    os.makedirs(gt_img_path,exist_ok=True) 
                    pil_gt_img = visual_img(gt_video[i][j])
                    pil_gt_img.save(f"{gt_img_path}/gt_img_{j}.png")
                    
                    cur_box = boxes[j].cpu().tolist()
                    with open(f"{save_path}/box_{j}.txt", "w") as f:
                        for bb in cur_box:
                            string = ",".join([str(x) for x  in bb]) 
                            f.write(string + "\n")
                
                boxes_pred_imgs[0].save(f"{save_path}/boxes_pred.gif",save_all=True,append_images=boxes_pred_imgs[1:],optimize=False,duration=8)
                boxes_gt_imgs[0].save(f"{save_path}/boxes_gt.gif",save_all=True,append_images=boxes_gt_imgs[1:],optimize=False,duration=8)

    print("Mission Completed !!!")
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/flintstones.yaml")
    parser.add_argument("--train_grounding", default=None, type=str)
    parser.add_argument("--cat_init", action="store_true")
    parser.add_argument("--pretrained_weight_path", type=str, default=None)
    parser.add_argument("--video_length", type=int, default=8)
    parser.add_argument("--trained_ckpt_path", type=str, default=None)
    parser.add_argument("--use_tmp_window_attention", action="store_true")
    parser.add_argument("--tmp_window_size", type=int, default=3)
    parser.add_argument("--use_image_dataset",action="store_true")
    parser.add_argument("--cross_vision", action="store_true")
    parser.add_argument("--use_clip_visual_encoder", action="store_true")
    parser.add_argument("--dataset",type=str,default="got10k")
    parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--batch_size",type=int,default=2)
    parser.add_argument("--guidance_scale", type=float,default=7.5)
    parser.add_argument("--output_dir",type=str, default="")
    parser.add_argument("--control_net",action="store_true")
    parser.add_argument("--img_free_guidance",type=float,default=0.0)
    parser.add_argument("--ti_free_guidance",type=float,default=0.0, help="text and img free guidance scale")


    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    OmegaConf.update(config, "train_grounding", args.train_grounding)
    OmegaConf.update(config, "cat_init", args.cat_init)
    OmegaConf.update(config, "trained_ckpt_path", args.trained_ckpt_path)
    OmegaConf.update(config, "video_length", args.video_length)
    OmegaConf.update(config, "use_tmp_window_attention", args.use_tmp_window_attention)
    OmegaConf.update(config, "tmp_window_size", args.tmp_window_size)
    OmegaConf.update(config, "pretrained_weight_path", args.pretrained_weight_path)
    OmegaConf.update(config, "use_image_dataset", args.use_image_dataset)
    OmegaConf.update(config, "cross_vision", args.cross_vision)
    OmegaConf.update(config, "use_clip_visual_encoder", args.use_clip_visual_encoder)
    OmegaConf.update(config, "dataset", args.dataset)
    OmegaConf.update(config, "num_workers", args.num_workers)
    OmegaConf.update(config, "batch_size", args.batch_size)
    OmegaConf.update(config, "guidance_scale", args.guidance_scale)
    OmegaConf.update(config, "output_dir", args.output_dir)
    OmegaConf.update(config, "control_net", args.control_net)
    OmegaConf.update(config, "img_free_guidance", args.img_free_guidance)
    OmegaConf.update(config, "ti_free_guidance", args.ti_free_guidance)
    main(**config)
