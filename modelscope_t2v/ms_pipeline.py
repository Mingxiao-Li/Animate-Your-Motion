from diffusers.utils import deprecate, logging, BaseOutput
from dataclasses import dataclass
from typing import Union, List, Optional
from diffusers.pipeline_utils import DiffusionPipeline
from einops import rearrange
import torch 
import numpy as np 
import torch.cuda.amp as amp
from text_grounding_input import GroundingNetInput
from util import save_videos_grid
logger = logging.get_logger(__name__)


@dataclass
class MSVideoGenerationOutput(BaseOutput):
    video: Union[torch.Tensor, np.ndarray]

def preprocess_noise(noise):
    noise_shared = torch.randn(1, 4, 1, 16, 16)
    noise_shared = noise_shared.expand_as(noise).to(noise.device)
    noise = torch.sqrt(torch.tensor(1/2)) * noise + torch.sqrt(torch.tensor(1/2))*noise_shared
    return noise 

class MSVideoGenerationPipeline(DiffusionPipeline):

    def __init__(
            self,
            vae,
            text_encoder,
            unet,
            position_net,
            diffusion,
            visual_encoder,
            cross_vision,
            controlnet_encoder,
        ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            unet=unet,
            diffusion=diffusion,
        )
        self.cross_vision = cross_vision
        if position_net is not None:
            self.register_modules(position_net=position_net)
        else:
            self.position_net = None
        
        if visual_encoder is not None:
            self.register_modules(visual_encoder=visual_encoder)
        else:
            self.visual_encoder = None

        if controlnet_encoder is not None:
            self.register_modules(controlnet_encoder=controlnet_encoder)
        else:
            self.controlnet_encoder = None
        # if self.unet.position_net is not None:
        self.grounding_input = GroundingNetInput()
        
    

    def get_first_frame_cond(self, img, noise, weight_dtype):
        # img shape --> 1 x 3 x 128 x 128
        pixel_values = img.to(weight_dtype)
        latent = self.vae.encode(pixel_values.to(self.vae.device)).sample()
        latent = latent * 0.18215
        latent = latent.unsqueeze(2).expand_as(noise)
        return latent


    def get_grounding_cond(self,):
        pass

    def __call__(
            self,
            prompt: Union[str, List[str]],
            video_length: Optional[int],
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            img_free_guidance: float = 0.0,
            ti_free_guidance: float = 0.0,
            first_frame_pixel=None,
            batch=None,
            obj_oriented_attn_ratio=0.0,
            cat_init=False,
            # negative_prompt: Optional[Union[str, List[str]]] = None,
            # eta: float = 0.0,
            # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            **kwargs,
        ):

        # Default height and width to unet
        height = height or self.unet.config.dim 
        width = width or self.unet.config.dim 

        #definde call parameters
        num_samples = 1 if isinstance(prompt, str) else len(prompt)
        device = self.unet.device 

        #do_classifier_free_guidance = guidance_scale > 1.0

        text_emb = self.text_encoder.encode(prompt).to(device)
        
        null_text_emb = self.text_encoder.encode(" ").to(device)
    
        context = [null_text_emb, text_emb]                    
        latent_h, latent_w = height, width
        noise = torch.randn(num_samples, 4, video_length, latent_h, latent_w).to(device)
        #noise = preprocess_noise(noise)
        if batch is not None and self.position_net is not None:
            if len(batch["boxes"].shape) == 4:
                batch_size, num_frame, num_obj, _ = batch["boxes"].shape
                batch["boxes"] = batch["boxes"].view(batch_size*num_frame,num_obj,-1)
                batch["text_embeddings"] = batch["text_embeddings"].view(batch_size*num_frame,num_obj,-1)
                batch["masks"] = batch["masks"].view(batch_size*num_frame,-1)
           
            grounding_input = self.grounding_input.prepare(batch)
            grounding_context = self.position_net(
                boxes = grounding_input["boxes"],
                masks = grounding_input["masks"],
                positive_embeddings = grounding_input["positive_embeddings"]
            )
   
            grounding_input = {
                "context": grounding_context,
                "masks": batch["masks"]
            }
            #grounding_input['context'] = grounding_context
        else:
            grounding_input = {}
           
        
        first_frame_cond_latent = None
        if self.unet.config.cat_init or self.cross_vision:
        
            first_frame_cond_latent = self.get_first_frame_cond(first_frame_pixel, noise=noise, weight_dtype=self.vae.dtype)
        if self.unet.config.cat_init or self.controlnet_encoder is not None:
            assert first_frame_pixel is not None, "First frame should be provided !!!"
            first_frame_cond_latent = self.get_first_frame_cond(first_frame_pixel, noise=noise, weight_dtype=self.vae.dtype)
        
        with torch.no_grad():
            with amp.autocast(enabled=True):
                # if grounding_context is None:
                #     model_kwargs = [ {
                #         'y': context[1].unsqueeze(0).repeat(num_samples, 1,1)
                #
                #     }, {
                #         'y': context[0].unsqueeze(0).repeat(num_samples, 1,1)
                #     }]
                # else:
                
                if not self.cross_vision:
                    model_kwargs = [{
                        'y': context[1], #.unsqueeze(0).repeat(num_samples, 1,1),
                        'grounding_input': grounding_input
                    }, {
                        'y': context[0].repeat(num_samples, 1,1),
                        'grounding_input': grounding_input
                    }]
                   
                    x0 = self.diffusion.ddim_sample_loop(
                        noise=noise,
                        model=self.unet,
                        controlnet_encoder=self.controlnet_encoder,
                        model_kwargs=model_kwargs,
                        guide_scale=guidance_scale,
                        ddim_timesteps=num_inference_steps,
                        eta=0.0,
                        first_frame_cond=first_frame_cond_latent,
                        obj_oriented_attn_ratio=obj_oriented_attn_ratio,
                        img_free_guidance=img_free_guidance,
                        ti_free_guidance=ti_free_guidance,
                        cat_init=cat_init
                    )
                else:
                    first_latent = rearrange(first_frame_cond_latent, "b c f h w ->  b f c h w")[:,0,:,:]
                    visual_feat = self.visual_encoder(first_latent)
                    visual_feat = rearrange(visual_feat,"b c w h -> b (w h) c")
                    visual_feat = visual_feat.unsqueeze(1).repeat(1, video_length,1,1)
                    visual_feat = rearrange(visual_feat, "b  f l d -> (b f) l d")
                    model_kwargs = [{
                        "y": context[1],# .unsqueeze(0).repeat(num_samples, 1,1),
                        "grounding_input": grounding_input,
                        "visual_feat": visual_feat
                    }, {
                        "y": context[0].repeat(num_samples,1,1),
                        "grounding_input": grounding_input,
                        "visual_feat": visual_feat
                    }]
                  
                    x0 = self.diffusion.ddim_sample_loop(
                        noise = noise,
                        model = self.unet,
                        controlnet_encoder=self.controlnet_encoder,
                        model_kwargs = model_kwargs,
                        guide_scale = guidance_scale,
                        ddim_timesteps = num_inference_steps,
                        eta = 0.0,
                        first_frame_cond=first_frame_cond_latent,
                        img_free_guidance=img_free_guidance,
                        ti_free_guidance=ti_free_guidance,
                        cat_init=cat_init
                    )

            scale_factor = 0.18215
            video_data = 1. / scale_factor * x0
            bs_vd = video_data.shape[0]
            video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
            self.vae.to(device)
            video_data = self.vae.decode(video_data.to(self.vae.dtype))
            video_data = rearrange(
                video_data, '(b f) c h w -> b c f h w', b=bs_vd
            )
        video_data = video_data.type(torch.float32).cpu()
        video_data = (video_data / 2 + 0.5).clamp(0, 1)
        return video_data