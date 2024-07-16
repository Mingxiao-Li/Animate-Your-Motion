import argparse
import datetime
import logging
import inspect
import itertools
import math
import os
from typing import Dict, Optional, Tuple, List
from omegaconf import OmegaConf
import open_clip 
import torch
import torch.utils.checkpoint
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from data.gligen_dataset.concat_dataset import ConCatDataset
from data.concate_dataset import CatObjTrackVideoDataset, collate_fn

from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from trackers import MyWandbTracker
# from transformers import CLIPTokenizer
from functools import partial
from train import Trainer
from modelscope_t2v.text_to_video_synthesis_model import get_model_scope_t2v_models

def add_additional_channels(state_dict, num_additional_channels):
    "state_dict should be just from unet model, not the entire SD or GLIGEN"

    if num_additional_channels != 0:
    
        new_conv_weight = torch.zeros(320, 4+num_additional_channels, 3, 3 )

        for key,value in state_dict.items():
            if key == "input_blocks.0.0.weight":
                old_conv_weight = value
                new_conv_weight[:,0:4,:,:] = old_conv_weight
                state_dict[key] = new_conv_weight
                
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")

def main(
        root_path: str,
        pretrained_model_path: str,
        output_dir: str,
        train_data: Dict,
        pretrained_weight_path: None,
        validation_data: Dict = None,
        train_grounding: bool = False,
        validation_steps: int = 100,
        trainable_modules: List[str] = (
            "attn1.to_q",
            "attn2.to_q",
            "attn_temp",
        ),
        use_wandb: bool = False,
        tracker: Dict = None,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        max_train_steps: int = 500,
        learning_rate: float = 3e-5,
        scale_lr: bool = False,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = True,
        checkpointing_steps: int = 500,
        resume_from_checkpoint: Optional[str] = None,
        mixed_precision: Optional[str] = "fp16",
        use_8bit_adam: bool = False,
        enable_xformers_memory_efficient_attention: bool = True,
        cat_init: bool = False,
        seed: Optional[int] = None,
        use_img_free_guidance: bool = False,
        tune_from_ckpt_path: str= None,
        use_tmp_window_attention: bool =False,
        only_cat_first_frame: bool = False,
        use_image_dataset: bool = False,
        tmp_window_size: int = 3,
        cross_vision: bool = False,
        use_clip_visual_encoder: bool = False,
        control_net: bool = False,
):
    
    *_, config = inspect.getargvalues(inspect.currentframe())
    tracker.group = "model_scope"

    if use_wandb:
        my_tracker = MyWandbTracker(tracker)
        accelerator = Accelerator(
            log_with=my_tracker,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
    

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(f"video_model_scope")
        if use_wandb:
            my_tracker.store_init_configuration(config)


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
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
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
        control_net=control_net,
    )
    
    tune_pos_net = True
    if tune_from_ckpt_path is not None:
        unet_weights = torch.load(os.path.join(tune_from_ckpt_path, 'pytorch_model.bin'))
        for key, value in unet_weights.items():
            model_weight = unet.state_dict().get(key)
            try:
                value.shape != model_weight.shape
            except:
                import ipdb;ipdb.set_trace()
            if value.shape != model_weight.shape:
                print(key, value.shape, model_weight.shape)
                unet_weights[key] = model_weight
                trainable_modules.append(key.replace('.weight', '').replace('.bias', ''))
        if cat_init and unet_weights["input_blocks.0.0.weight"].shape[1] == 4:
            unet_weights["input_blocks.0.0.weight"] = unet.state_dict().get("input_blocks.0.0.weight")
        missing_keys, unexpected_keys = unet.load_state_dict(unet_weights, strict=False)
        print("********************************")
        print("Loading pretrained weights from ", tune_from_ckpt_path)
        print(" Missing Keys : ", missing_keys)
        print(" Unexpected Keys :", unexpected_keys)
        print("********************************")
        assert len(unexpected_keys) == 0, "get unexpected_keys when loading ckpt"
        
        position_net_path = os.path.join(tune_from_ckpt_path, 'pytorch_model_1.bin')
        if os.path.exists(position_net_path):
            position_net_weights = torch.load(position_net_path)
        else:
            position_net_weights = None
        if position_net is not None and position_net_weights is not None:
            missing_keys, unexpected_keys = position_net.load_state_dict(position_net_weights, strict=True)
            assert len(unexpected_keys) == 0, "get unexpected_keys when loading position net"
            tune_pos_net = False
            print("Successfully load position net weights from ",position_net_path)
        
        print("Freeze alpha_attn and dense !!")
        for name,module in unet.named_modules():
            if "gate_attn" in name:
                if hasattr(module,"alpha_attn"):
                    module.alpha_attn.requires_grad = False
                if hasattr(module,"alpha_dense"):
                    module.alpha_dense.requires_grad = False
        
        visual_encoder_path = os.path.join(tune_from_ckpt_path,"pytorch_model_2.bin")
        if os.path.exists(visual_encoder_path):
            visual_encoder_weights = torch.load(visual_encoder_path)
        else:
            visual_encoder_weights = None 
        if visual_encoder is not None and visual_encoder_weights is not None:
            missing_keys, unexpected_keys = visual_encoder.load_state_dict(visual_encoder_weights)
            assert len(unexpected_keys) == 0, "get unexpected_keys when loading visual_encoder"
            print("Successfully load visual encoder weights form ", visual_encoder_path)
    else:   
        unet_weights = unet.state_dict() 
        
    # Load control_encoder weights
    if controlnet_encoder is not None:
        controlnet_trainable_keys = list()
        if tune_from_ckpt_path is not None:
            control_path = os.path.join(tune_from_ckpt_path, 'pytorch_model_2.bin')
        if tune_from_ckpt_path is not None and os.path.exists(control_path):
            control_weights = torch.load(control_path)
            for k, v in controlnet_encoder.state_dict().items():
                if k not in unet_weights:
                    controlnet_trainable_keys.append(k.split('.')[0])
            print("Successfully load control net weights from ",control_path)
        else:
            control_weights = dict()
            if unet_weights["input_blocks.0.0.weight"].shape[1] == 8:
                unet_weights["input_blocks.0.0.weight"] = unet_weights["input_blocks.0.0.weight"][:, :4]
            for k, v in controlnet_encoder.state_dict().items():
                if k in unet_weights:
                    control_weights[k] = unet_weights[k]
                else:
                    control_weights[k] = v
                    controlnet_trainable_keys.append(k.split('.')[0])
        _ = controlnet_encoder.load_state_dict(control_weights, strict=True)
       
        
    
    ##################################s###############
    if use_wandb:
        my_tracker.watch(unet)

    #################################################
    # Set trainable and non-trainable parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    training_modules = []

  # temporal_conv --> for tmp convolution
  # 2.transformer_blocks.
    if trainable_modules is not None:
        trainable_modules = list(set(trainable_modules))
        for name, module in unet.named_modules():
            for train_module_name in trainable_modules:
                if train_module_name in name:
                    if module not in training_modules:
                        training_modules.append(name)
                    for params in module.parameters():
                        params.requires_grad = True
        for m in training_modules:
            print(f"Tune UNet module --> {m}")
    else:
        for name, module in unet.named_modules():
            for params in module.parameters():
                params.requires_grad = True
        print("Tune all parameters in UNet")
    
    for name, module in unet.named_modules():
        if "gate_attn" in name:
            if hasattr(module, "cross_alpha_attn"):
                module.cross_alpha_attn.requires_grad = True
                print("Train --> gate_attn.cross_alpha_attn")
            if hasattr(module, "cross_alpha_dense"):
                module.cross_alpha_dense.requires_grad = True 
                print("Train --> gate_attn.cross_alpha_dense")
            if hasattr(module, "beta_g"):
                module.beta_g.requires_grad = True 
                print("Train --> gate_attn.beta_g")
            if hasattr(module, "beta_v"):
                module.beta_v.requires_grad = True 
                print("Train --> gate_attn.beta_v")
            

    if visual_encoder is not None:
        print("Tune Vision Encoder")
        visual_encoder.requires_grad_(True)
    if position_net is not None:
        if tune_pos_net:
            print("Tune Position Net")
            position_net.requires_grad_(True)
        else:
            print("NOT Tune Position Net")
            position_net.requires_grad_(False)
    if controlnet_encoder is not None:
        print("Tune ControlNet Encoder Zero Convs")
        controlnet_encoder.requires_grad_(False)
        for name, module in controlnet_encoder.named_modules():
            if name in controlnet_trainable_keys:
                print(f"Tune ControlNet modules --> {name}")
                for params in module.parameters():
                    params.requires_grad = True
        
    ################################################ 
   
    ################################################
    # Set xofrmer or gradient checkpoint for saving memory
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    #################################################
    
    #################################################
    # Set optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb 
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if scale_lr:
        # WARNING!!! Be very careful to use scale_lr, especially train_batch_size.
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
    def combine_para(gens):
        for gen in gens:
            if gen is not None:
                for ele in gen.parameters():
                    yield ele 
    
    optimizer = optimizer_cls(
        combine_para([unet, visual_encoder, position_net, controlnet_encoder]),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon
    )
    #################################################
    if train_data.name == "flintstones":
        from data.flintstones import FlinstStonesDataset
        train_data.pop("name")
        train_dataset = FlinstStonesDataset(**train_data)
        train_data.split = "val"
        val_dataset = FlinstStonesDataset(**train_data)
        tokenizer = train_dataset.tokenizer

        train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        num_workers=0,
        )

    elif train_data.name == "webvid":
        from data.webvid import TuneAVideoDataset
        train_data.pop("name")
        train_dataset = TuneAVideoDataset(**train_data)
        train_data.split = "2M_val"
        val_dataset = TuneAVideoDataset(**train_data)
        tokenizer = train_dataset.tokenizer
        train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        num_workers=0,shuffle=True 
        )

    elif train_data.name == "gligen":
        train_dataset = ConCatDataset(train_data.dataset_names, train_data.data_path, train=True)
        val_dataset = None
        tokenizer = transformers.CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        num_workers=0,shuffle=True 
        )

    elif train_data.name == "object_tracking":
        if visual_encoder is not None:
            preprocessor = visual_encoder.preprocessor
        else:
            preprocessor = None
        train_dataset = CatObjTrackVideoDataset(
            **train_data,
            preprocessor = preprocessor 
        )
        val_dataset = None
        tokenizer = None  # use open_clip tokenize
        train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True,
        batch_size=train_batch_size,
        num_workers=0,
        collate_fn=partial(collate_fn,max_num_obj=train_data.num_max_obj, num_sample_frames=train_data.n_sample_frames)
        )
    else:
        raise NotImplementedError

    trainer = Trainer(
        accelerator=accelerator,
        tokenizer=tokenizer,
        logger=logger,
        train_dataset=train_dataset,
        num_process=accelerator.num_processes,
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_clip_visual_encoder=use_clip_visual_encoder
    )
    
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=val_batch_size
        ) 

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )
    

    unet, optimizer, train_dataloader, lr_scheduler, diffusion = accelerator.prepare(
                unet,  optimizer, train_dataloader, lr_scheduler, diffusion
            )
    
    position_net = accelerator.prepare(position_net) if position_net is not None else None 
    visual_encoder = accelerator.prepare(visual_encoder) if visual_encoder is not None else None
    if controlnet_encoder is not None:
        controlnet_encoder = accelerator.prepare(controlnet_encoder) if controlnet_encoder is not None else None



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
    

    global_step = 0
    first_epoch = 0
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        accelerator.print(f"Resuming from checkpoint {resume_from_checkpoint}")
        accelerator.load_state(resume_from_checkpoint)
        global_step = int(resume_from_checkpoint.split("-")[-1])
        
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    trainer.train_model_scope(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        position_net=position_net,
        visual_encoder=visual_encoder,
        controlnet_encoder=controlnet_encoder,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        weight_dtype=weight_dtype,
        train_dataloader=train_dataloader,
        accelerator=accelerator,
        first_epoch=first_epoch,
        global_step=global_step,
        max_grad_norm=max_grad_norm,
        output_dir=output_dir,
        checkpointing_steps=checkpointing_steps,
        scheduler=scheduler,
        validation_steps=validation_steps,
        cat_init=cat_init,
        seed=seed,
        device=accelerator.device,
        diffusion=diffusion,
        validation_dataloader=val_dataloader,
        use_img_free_guidance=use_img_free_guidance,
        only_cat_first_frame=only_cat_first_frame,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/flintstones.yaml")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int,default=1)
    parser.add_argument("--train_grounding", type=str, default=None, 
                        help="specify which class is used to conduct grounding, \
                            'vanilla' means gligen grounding; \
                            'double' means two branches grounding; \
                            'merge' means merging grounding")
    parser.add_argument("--cat_init", action="store_true")
    parser.add_argument("--pretrained_weight_path", type=str, default=None)
    parser.add_argument("--use_img_free_guidance", action="store_true")
    parser.add_argument("--tune_from_ckpt_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--trainable_modules", nargs='+', help="input_string", default=[])
    parser.add_argument("--use_tmp_window_attention", action="store_true")
    parser.add_argument("--tmp_window_size", type=int, default=3)
    parser.add_argument("--only_cat_first_frame", action="store_true")
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--use_image_dataset", action="store_true")
    parser.add_argument("--use_clip_visual_encoder", action="store_true",help="select if use clip visual encoder, the cross vision must be True")
    parser.add_argument("--cross_vision", action="store_true")
    parser.add_argument("--control_net", action="store_true", help="add image control same as control net")

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if len(args.trainable_modules) != 0:
        OmegaConf.update(config, "trainable_modules", args.trainable_modules)

    OmegaConf.update(config, "train_batch_size", args.train_batch_size)
    OmegaConf.update(config, "val_batch_size", args.val_batch_size)
    OmegaConf.update(config, "output_dir", args.output_dir)
    OmegaConf.update(config, "use_wandb", args.use_wandb)
    OmegaConf.update(config, "gradient_accumulation_steps", args.gradient_accumulation_steps)
    OmegaConf.update(config, "train_grounding", args.train_grounding)
    OmegaConf.update(config, "cat_init", args.cat_init)
    OmegaConf.update(config, "pretrained_weight_path", args.pretrained_weight_path)
    OmegaConf.update(config, "use_img_free_guidance", args.use_img_free_guidance)
    OmegaConf.update(config, "tune_from_ckpt_path", args.tune_from_ckpt_path)
    OmegaConf.update(config, "resume_from_checkpoint", args.resume_from_checkpoint)
    OmegaConf.update(config, "use_tmp_window_attention", args.use_tmp_window_attention)
    OmegaConf.update(config, "tmp_window_size", args.tmp_window_size)
    OmegaConf.update(config, "only_cat_first_frame", args.only_cat_first_frame)
    OmegaConf.update(config, "scale_lr", args.scale_lr)
    OmegaConf.update(config, "use_image_dataset", args.use_image_dataset)
    OmegaConf.update(config, "cross_vision", args.cross_vision)
    OmegaConf.update(config, "use_clip_visual_encoder", args.use_clip_visual_encoder)
    OmegaConf.update(config, "control_net", args.control_net)
    print("-----CONFIG-----")
    print(config)
    print("----------------")
    main(**config)
