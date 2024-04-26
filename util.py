import os
import imageio
import importlib
import numpy as np
import io 
import base64

from typing import Union
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import numpy as np 
from PIL import Image, ImageDraw
from tqdm import tqdm
from einops import rearrange


def visual_img(img_tensor,name=None):
    img = (img_tensor.squeeze(0).detach().cpu()+1)*127.5
    img = img.numpy().astype(np.uint8)
    img = img.transpose(1,2,0)
    img = Image.fromarray(img)
    if name is not None:
        img.save(name)
    else:
        return img



def draw_boxes_of_predicted_image(output, boxes_tensor, name):
    output = output.squeeze(2)
    output = (255.*output).detach().cpu().squeeze(0)
    output = output.numpy().astype(np.uint8)
    output = output.transpose(1,2,0)
    img = Image.fromarray(output)
    draw = ImageDraw.Draw(img)
    boxes_tensor = boxes_tensor.cpu()
    for box in boxes_tensor:
        box = (128*box).int()
        if box.sum() == 0:
            continue 
        draw.rectangle(box.tolist(),outline="red",width=2)
    img.save(name)


def draw_boxes(img_tensor, boxes_tensor, name="test",return_img=False):
    pil_img = visual_img(img_tensor)
    draw = ImageDraw.Draw(pil_img)
    boxes_tensor = boxes_tensor.cpu()
    colors = [
    'red', 'green', 'blue', 'yellow',
    'purple', 'orange', 'pink', 'cyan'
]
    for i,box in enumerate(boxes_tensor):
        box= (img_tensor.shape[-1]*box).int()

        if box.sum() == 0:
            continue
        draw.rectangle(box.tolist(),outline=colors[i],width=2)

    if not return_img:
        pil_img.save(name)
    else:
        return pil_img


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def b2f(b):
    f = Image.open(io.BytesIO(base64.b64decode(b)))
    w, h = f.size
    w, h = int(128*w/h), 128
    f = f.resize([w, h]).crop([(w-128)//2, 0, (w-128)//2+128, 128])
    return f


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, duration=fps)


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def task_ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, task, prompt=""):
    first_frame_gt_latent = video_latent[:,:, 0, :,: ].unsqueeze(2)
    last_frame_gt_latent = video_latent[:,:, -1, :,: ].unsqueeze(2)
    if task == "infilling":
        first_inv = ddim_inversion(pipeline=pipeline,
                                  ddim_scheduler=ddim_scheduler,
                                  video_latent=first_frame_gt_latent,
                                  num_inv_steps=num_inv_steps,
                                  prompt=prompt)[0]
        last_inv = ddim_inversion(pipeline=pipeline,
                                  ddim_scheduler=ddim_scheduler,
                                  video_latent=last_frame_gt_latent,
                                  num_inv_steps=num_inv_steps,
                                  prompt=prompt)[0]
        latents = torch.cat([first_inv, last_inv], dim=2)
    elif task == "prediction":
        latents = ddim_inversion(pipeline=pipeline,
                                  ddim_scheduler=ddim_scheduler,
                                  video_latent=first_frame_gt_latent,
                                  num_inv_steps=num_inv_steps,
                                  prompt=prompt)[0]
    elif task == "rewinding":
        latents = ddim_inversion(pipeline=pipeline,
                                  ddim_scheduler=ddim_scheduler,
                                  video_latent=last_frame_gt_latent,
                                  num_inv_steps=num_inv_steps,
                                  prompt=prompt)[0]
    else:
        return None 
    
    return latents


def generate_frame_mask(num_frames, task):
    mask = torch.ones(num_frames, num_frames)
    if task == "infilling":
        mask[0,1:]=0
        mask[-1,:-1]=0
    elif task == "prediction":
        mask[0,1:] = 0 
    elif task == "rewind":
        mask[-1:-1] = 0 
    return mask 

def generate_pos_coords(w, h, f):
    coord_x, coord_y = np.meshgrid(np.arange(w), np.arange(h))
    coord_x = np.tile(coord_x[None], (f, 1, 1)) / (w - 1)
    coord_y = np.tile(coord_y[None], (f, 1, 1)) / (h - 1)
    coord_z = np.tile(np.arange(f)[:, None, None], (1, h, w)) / (f - 1)
    coords = np.stack((coord_x, coord_y, coord_z), axis=1).astype(np.float32)  # f*3*h*w
    return coords


class SpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialConv, self).__init__()
        # Cannot initialize multiple Conv layers as zero, as it would lead to zero gradiant.
        self.zero_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=13, padding=6, stride=1)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=9, padding=4, stride=1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2, stride=1),
        )

        self.zero_init()

    def zero_init(self):
        for p in self.zero_conv.parameters():
            nn.init.zeros_(p)

    def forward(self, x):
        out = self.conv_layers(self.zero_conv(x))
        return out


