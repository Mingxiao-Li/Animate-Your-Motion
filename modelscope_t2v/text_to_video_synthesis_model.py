import os
from os import path as osp
from typing import Any, Dict

import open_clip
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from einops import rearrange

from modelscope_t2v.autoencoder import AutoencoderKL
from modelscope_t2v.diffusion import GaussianDiffusion, beta_schedule
from modelscope_t2v.unet_sd import UNetSD
from modelscope_t2v.config import Config
from models.position_net import PositionNet
from modelscope_t2v.controlnet import ControlNet


class TextToVideoSynthesis(nn.Module):
    
    def __init__(self, ckpt_root_path, is_inference=True):
        super().__init__()
       
        self.device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        
        self.config = Config.from_file(
            os.path.join(ckpt_root_path, "configuration.json")
        )
        cfg = self.config.model.model_cfg
        self.sd_model = UNetSD(
            in_dim=cfg['unet_in_dim'],
            dim=cfg['unet_dim'],
            y_dim=cfg['unet_y_dim'],
            context_dim=cfg['unet_context_dim'],
            out_dim=cfg['unet_out_dim'],
            dim_mult=cfg['unet_dim_mult'],
            num_heads=cfg['unet_num_heads'],
            head_dim=cfg['unet_head_dim'],
            num_res_blocks=cfg['unet_res_blocks'],
            attn_scales=cfg['unet_attn_scales'],
            dropout=cfg['unet_dropout'],
            temporal_attention=cfg['temporal_attention'],
            use_control=False, #cfg['control_net'],
        )

        self.sd_model.load_state_dict(
            torch.load(
                os.path.join(ckpt_root_path, "text2video_pytorch_model.pth")
            )
        )
        
        if is_inference:
            self.sd_model.eval()
        self.sd_model.to(self.device)
        
        # Initialize diffusion
        betas = beta_schedule(
            'linear_sd',
            cfg['num_timesteps'],
            init_beta=0.00085,
            last_beta=0.0120
        )

        self.diffusion = GaussianDiffusion(
            betas=betas,
            mean_type=cfg['mean_type'],
            var_type=cfg['var_type'],
            loss_type=cfg['loss_type'],
            rescale_timesteps=False
        )

        # Initialize autoencoder 
        ddconfig = {
            'double_z': True,
            'z_channels': 4,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0
        }

        self.autoencoder = AutoencoderKL(
            ddconfig, 4,
            os.path.join(ckpt_root_path, self.config.model.model_args.ckpt_autoencoder)
        )

        self.autoencoder.to(self.device)

        # Initialize open clip
        self.config.model.model_args.ckpt_clip

      

        self.clip_encoder = FrozenOpenCLIPEmbedder(
            version = os.path.join(ckpt_root_path,self.config.model.model_args.ckpt_clip),
            layer='penultimate'
        )

        self.clip_encoder.to(self.device)


    def forward(self, input:Dict[str, Any]):
        r"""
        The entry function of text to image synthesis task.
        1. Using diffusion model to generate the video's latent representation.
        2. Using vqgan model (autoencoder) to decode the video's latent representation to visual space.

        Args:
            input (`Dict[Str, Any]`):
                The input of the task
        Returns:
            A generated video (as pytorch tensor).
        """
        y = input['text_emb']
        zero_y = input['text_emb_zero']
        context = torch.cat([zero_y, y], dim=0).to(self.device)
        # synthesis
        with torch.no_grad():
            num_sample = 1  # here let b = 1
            max_frames = 8 #self.config.model.model_args.max_frames
            latent_h, latent_w = input['out_height'] // 8, input[
                'out_width'] // 8
            with amp.autocast(enabled=True):
                x0 = self.diffusion.ddim_sample_loop(
                    noise=torch.randn(num_sample, 4, max_frames, latent_h,
                                      latent_w).to(
                                          self.device),  # shape: b c f h w
                    model=self.sd_model,
                    model_kwargs=[{
                        'y':
                        context[1].unsqueeze(0).repeat(num_sample, 1, 1)
                    }, {
                        'y':
                        context[0].unsqueeze(0).repeat(num_sample, 1, 1)
                    }],
                    guide_scale=9.0,
                    ddim_timesteps=50,
                    eta=0.0)

                scale_factor = 0.18215
                video_data = 1. / scale_factor * x0
                bs_vd = video_data.shape[0]
                video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
                self.autoencoder.to(self.device)
                video_data = self.autoencoder.decode(video_data)
                if self.config.model.model_args.tiny_gpu == 1:
                    self.autoencoder.to('cpu')
                video_data = rearrange(
                    video_data, '(b f) c h w -> b c f h w', b=bs_vd)
                
        return video_data.type(torch.float32).cpu()



class FrozenOpenCLIPEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = ['last', 'penultimate']

    def __init__(self,
                 arch='ViT-H-14',
                 version='open_clip_pytorch_model.bin',
                 device='cuda',
                 max_length=77,
                 freeze=True,
                 layer='last'):
        super().__init__()
        assert layer in self.LAYERS
        model, _, preprocessor = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=version
        )

        #del model.visual 
        self.model = model 
        self.visual = clip_visual_encoder(model, preprocessor)

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        
        self.layer = layer 
        if self.layer == 'last':
            self.layer_idx = 0 
        elif self.layer == 'penultimate':
            self.layer_idx = 1
        else:
            raise NotImplementedError()
    
    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False 
    
    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z 
    
    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text) # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x)
        return x 
    
    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break 
            x = r(x, attn_mask=attn_mask)
        return x 
    
    def encode(self, text):
        return self(text)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skip = None
        else:
            self.skip = nn.Conv2d(in_c, out_c, ksize, 1, ps)

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)
    
    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skip is not None:
            try:
                h + self.skip(x)
            except:
                import ipdb; ipdb.set_trace()
            return h + self.skip(x)
        else:
            return h + x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

class clip_visual_encoder(nn.Module):
    def __init__(self, clip_model, preprocessor):
        super().__init__()
        visual_encoder = clip_model.visual
        self.preprocessor = preprocessor
        self.conv1 = visual_encoder.conv1
        self.class_embedding = visual_encoder.class_embedding
        self.positional_embedding = visual_encoder.positional_embedding
        self.patch_dropout = visual_encoder.patch_dropout
        self.ln_pre = visual_encoder.ln_pre
        self.transformer = visual_encoder.transformer
        self.ln_post = visual_encoder.ln_post
        self._global_pool = visual_encoder._global_pool
         
    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        pooled, tokens = self._global_pool(x)

        return tokens 




class VisualEncoder(nn.Module):
    def __init__(self, channels=[256, 512, 1024, 1024], nums_rb=3, cin=4, ksize=3, sk=False, use_conv=True):
        super(VisualEncoder, self).__init__()
        self.channels = channels 
        self.nums_rb = nums_rb 
        self.body = []
        self.preprocessor = None

        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i == 2) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i-1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv)
                    )
                elif (i == 1) and ( j == 0):
                    self.body.append(
                        ResnetBlock(channels[i-1], channels[i], down=True, ksize=ksize, sk=sk, use_conv = use_conv)
                    )
                elif (i == 3) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i-1], channels[i], down=True, ksize=ksize, sk=sk, use_conv = use_conv)
                    )
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                    )
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x):

        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
        return x


    

def add_additional_channels(state_dict, num_additional_channels):
    "state_dict should be just from unet model, not the entire SD or GLIGEN"

    if num_additional_channels != 0:
    
        new_conv_weight = torch.zeros(320, 4+num_additional_channels, 3, 3 )

        for key, value in state_dict.items():
            if key == "input_blocks.0.0.weight":
                old_conv_weight = value
                new_conv_weight[:,0:4,:,:] = old_conv_weight
                state_dict[key] = new_conv_weight


def get_model_scope_t2v_models(
        cfg_root_path, 
        pretrained_weight_path=None, 
        train_grounding=False, 
        cat_init=False, 
        use_tmp_window_attention=False,
        tmp_window_size=3,
        use_image_dataset=False,
        cross_vision=False,
        control_net=False,
        use_clip_visual_encoder=False,
):
    
    config = Config.from_file(
        os.path.join(cfg_root_path, "configuration.json")
    )

    if pretrained_weight_path is None:
        pretrained_weight_path = os.path.join(cfg_root_path, "text2video_pytorch_model.pth")
        pretrained_weights_dict = torch.load(
               pretrained_weight_path
            )
    else:
        pretrained_weights_dict = torch.load(pretrained_weight_path)
    
    cfg = config.model.model_cfg 
    cfg["cat_init"] = cat_init
    cfg["use_gate_self_attn"] = train_grounding
    print("grounding type --> ", train_grounding)
    if cat_init:
       add_additional_channels(pretrained_weights_dict, num_additional_channels=4)
    
    if train_grounding:
        position_net = PositionNet(in_dim=1024, out_dim=1024)
    else:
        position_net = None

    sd_model = UNetSD(
        in_dim=cfg['unet_in_dim'],
        dim=cfg['unet_dim'],
        y_dim=cfg['unet_y_dim'],
        context_dim=cfg['unet_context_dim'],
        out_dim=cfg['unet_out_dim'],
        dim_mult=cfg['unet_dim_mult'],
        num_heads=cfg['unet_num_heads'],
        head_dim=cfg['unet_head_dim'],
        num_res_blocks=cfg['unet_res_blocks'],
        attn_scales=cfg['unet_attn_scales'],
        dropout=cfg['unet_dropout'],
        temporal_attention=cfg['temporal_attention'],
        cat_init=cfg["cat_init"],
        use_gate_self_attn=cfg["use_gate_self_attn"],
        use_image_dataset=use_image_dataset,
        use_tmp_window_attention=use_tmp_window_attention,
        tmp_window_size=tmp_window_size,
        use_control=control_net,
    )

    if control_net:
        model_weights = sd_model.state_dict()
        for key, value in pretrained_weights_dict.items():
            m_w = model_weights[key]
            if value.shape != m_w.shape:
                assert (len(value.shape) == 4 or len(value.shape) == 1)
                new_value = torch.zeros_like(m_w)
                if len(value.shape) == 4:
                    v_shape = value.shape[1]
                    w_shape = new_value.shape[1]
                    new_value[:, :v_shape] = value
                    new_value[:, v_shape:] = value[:, (v_shape-w_shape):]
                else:
                    v_shape = value.shape[0]
                    w_shape = new_value.shape[0]
                    new_value[:v_shape] = value
                    new_value[v_shape:] = value[(v_shape-w_shape):]
                pretrained_weights_dict[key] = new_value
    missing_keys, unexpected_keys = sd_model.load_state_dict(pretrained_weights_dict, strict=False)
    print("********************************")
    print("Loading pretrained weights from ", pretrained_weight_path)
    print(" Missing Keys : ", missing_keys)
    print(" Unexpected Keys :", unexpected_keys)
    print("********************************")
    assert len(unexpected_keys) == 0, "get unexpected_keys when loading ckpt"
    ddconfig = {
            'double_z': True,
            'z_channels': 4,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0
        }

    autoencoder = AutoencoderKL(
        ddconfig, 4,
        os.path.join(cfg_root_path, config.model.model_args.ckpt_autoencoder)
    )

    clip_encoder = FrozenOpenCLIPEmbedder(
        version=os.path.join(cfg_root_path, config.model.model_args.ckpt_clip),
        layer="penultimate"
    )

    betas = beta_schedule(
            'linear_sd',
            cfg['num_timesteps'],
            init_beta=0.00085,
            last_beta=0.0120
        )

    diffusion = GaussianDiffusion(
            betas=betas,
            mean_type=cfg['mean_type'],
            var_type=cfg['var_type'],
            loss_type=cfg['loss_type'],
            rescale_timesteps=False
        )
    
    ### if use control net --> define the control net as visual encoder here
    visual_encoder = None 
    if cross_vision:
        if not use_clip_visual_encoder:
            visual_encoder = VisualEncoder() 
        else:
            visual_encoder = clip_encoder.visual 
    
    controlnet_encoder = None
    if control_net:
        controlnet_encoder = ControlNet()
   
    return sd_model, autoencoder, clip_encoder, diffusion, position_net, visual_encoder, controlnet_encoder


if __name__ == "__main__":
    root_path = "/staging/leuven/stg_00116/modelscope/text-to-video"
    model = TextToVideoSynthesis(root_path)
    import ipdb; ipdb.set_trace()