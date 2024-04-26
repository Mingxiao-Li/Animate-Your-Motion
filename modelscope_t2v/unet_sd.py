# Part of the implementation is borrowed and modified from stable-diffusion,
# publicly avaialbe at https://github.com/Stability-AI/stablediffusion.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
import pdb

__all__ = ['UNetSD']


def has_inf_or_nan(x):
    return (torch.isinf(x) | (x != x)).any().item()

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def generate_weight_sequence(n):
    if n % 2 == 0:
        max_weight = n // 2
        weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
    else:
        max_weight = (n + 1) // 2
        weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
    return weight_sequence


def create_masks(tensor_shape, boxes, box_masks, context, fg_text_index, token_length):
    device = boxes.device
    fs, hw = tensor_shape
    h = w = int(hw**0.5)
    vis_masks = torch.ones((fs, hw, hw)).to(device)

    v2t_mask = None
    if context is not None:
        v2t_mask = torch.zeros((fs, hw, context.shape[1])).to(device)
        all_text_index = torch.arange(token_length).to(device)
        fg_text_mask = torch.zeros(token_length).to(device)
        for (sid, eid) in fg_text_index:
            if eid > 0:
                fg_text_mask[sid:eid] = 1
        all_fg_text_index = fg_text_mask.nonzero().reshape(-1)
        bg_text_index = all_text_index[~(all_text_index[:, None] == all_fg_text_index).any(dim=1)]

    # Scale the box coordinates to x dimensions
    scaled_boxes = boxes * torch.tensor([w - 1, h - 1, w - 1, h - 1]).to(device)

    # for temporal mask generation
    all_mask_v = torch.zeros((fs, h, w))
    # Iterate through each frame and each box
    for f in range(fs):
        boxes_f = torch.round(scaled_boxes[f][box_masks[f]]).int()
        fg_index = torch.tensor([])
        mask_v = torch.zeros((h, w))
        for box_id, (x1, y1, x2, y2) in enumerate(boxes_f):
            # get fg region index
            mask_b = torch.zeros((h, w))
            mask_b[y1:y2 + 1, x1:x2 + 1] = box_id + 1
            mask_v[y1:y2 + 1, x1:x2 + 1] = box_id + 1
            ind_y, ind_x = torch.nonzero(mask_b).T
            index = ind_y * h + ind_x
            fg_index = torch.cat((fg_index, index))

            # fg region self-attn
            vis_masks[f, index, :] = 0
            vis_masks[f, :, index] = 0
            vis_masks[f, index[:, None], index] = 1

            # fg text and fg region cross-attn
            if v2t_mask is not None:
                sid, eid = fg_text_index[box_id]
                v2t_mask[f, index, sid:eid] = 1
        all_mask_v[f] = mask_v
        # bg regions self-attn
        fg_index = torch.unique(fg_index)
        all_index = torch.arange(hw)
        bg_obj_index = all_index[~(all_index[:, None] == fg_index).any(dim=1)]
        vis_masks[f, bg_obj_index[:, None], bg_obj_index] = 1
        # bg text and bg region cross-attn
        if v2t_mask is not None:
            v2t_mask[f, bg_obj_index[:, None], bg_text_index] = 1
    # temporal mask self-attn
    all_mask_v = all_mask_v.reshape(fs, -1).transpose(0, 1)  # hw*fs
    temp_mask = (all_mask_v.unsqueeze(1) == all_mask_v.unsqueeze(2)).to(device)  # hw*fs*fs

    if v2t_mask is not None:
        v2t_mask = v2t_mask.bool()
    return vis_masks.bool(), v2t_mask, temp_mask


class UNetSD(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 in_dim=7,
                 dim=512,
                 y_dim=512,
                 context_dim=512,
                 out_dim=6,
                 dim_mult=[1, 2, 3, 4],
                 num_heads=None,
                 head_dim=64,
                 num_res_blocks=3,
                 attn_scales=[1 / 2, 1 / 4, 1 / 8],
                 use_scale_shift_norm=True,
                 dropout=0.1,
                 temporal_attn_times=2,
                 temporal_attention=True,
                 use_checkpoint=False,
                 use_image_dataset=False,
                 use_fps_condition=False,
                 use_gate_self_attn=False,
                 use_sim_mask=False,
                 cat_init=False,
                 use_tmp_window_attention=False,
                 tmp_window_size=3,
                 use_control=False,
                 ):
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super(UNetSD, self).__init__()
        self.in_channels = 4
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        self.num_heads = num_heads
        self.cat_init = cat_init
        # parameters for spatial/temporal attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))

        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(
                nn.Linear(dim, embed_dim), nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)

        # encoder
        self.input_blocks = nn.ModuleList()
        if not cat_init:
            init_block = nn.ModuleList([nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        else:
            init_block = nn.ModuleList([nn.Conv2d(2 * self.in_dim, dim, 3, padding=1)])

        if temporal_attention:
            init_block.append(
                TemporalTransformer(
                    dim,
                    num_heads,
                    head_dim,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    disable_self_attn=disabled_sa,
                    use_linear=use_linear_in_temporal,
                    multiply_zero=use_image_dataset,
                    use_tmp_window_attention=use_tmp_window_attention,
                    tmp_window_size=tmp_window_size))

        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim,
                out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                # residual (+attention) blocks
                block = nn.ModuleList([
                    ResBlock(
                        in_dim,
                        embed_dim,
                        dropout,
                        out_channels=out_dim,
                        use_scale_shift_norm=False,
                        use_image_dataset=use_image_dataset,
                    )
                ])
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=self.context_dim,
                            use_gate_self_attn=use_gate_self_attn,
                            disable_self_attn=False,
                            use_linear=True,
                            ))
                    if self.temporal_attention:
                        block.append(
                            TemporalTransformer(
                                out_dim,
                                out_dim // head_dim,
                                head_dim,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_temporal,
                                multiply_zero=use_image_dataset,
                                use_tmp_window_attention=use_tmp_window_attention,
                                tmp_window_size=tmp_window_size))

                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)

        if use_control:
            shortcut_dims = [2*i for i in shortcut_dims]
        # middle
        self.middle_block = nn.ModuleList([
            ResBlock(
                out_dim,
                embed_dim,
                dropout,
                use_scale_shift_norm=False,
                use_image_dataset=use_image_dataset,
            ),
            SpatialTransformer(
                out_dim,
                out_dim // head_dim,
                head_dim,
                depth=1,
                context_dim=self.context_dim,
                disable_self_attn=False,
                use_gate_self_attn=use_gate_self_attn,
                use_linear=True,
                )
        ])

        if self.temporal_attention:
            self.middle_block.append(
                TemporalTransformer(
                    out_dim,
                    out_dim // head_dim,
                    head_dim,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    disable_self_attn=disabled_sa,
                    use_linear=use_linear_in_temporal,
                    multiply_zero=use_image_dataset,
                    use_tmp_window_attention=use_tmp_window_attention,
                    tmp_window_size=tmp_window_size
                ))

        self.middle_block.append(
            ResBlock(
                out_dim,
                embed_dim,
                dropout,
                use_scale_shift_norm=False,
                use_image_dataset=use_image_dataset,
            ))

        # decoder
        self.output_blocks = nn.ModuleList()
        for i, (in_dim,
                out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                # residual (+attention) blocks
                block = nn.ModuleList([
                    ResBlock(
                        in_dim + shortcut_dims.pop(),
                        embed_dim,
                        dropout,
                        out_dim,
                        use_scale_shift_norm=False,
                        use_image_dataset=use_image_dataset,
                    )
                ])
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=1024,
                            disable_self_attn=False,
                            use_gate_self_attn=use_gate_self_attn,
                            use_linear=True,
                            ))

                    if self.temporal_attention:
                        block.append(
                            TemporalTransformer(
                                out_dim,
                                out_dim // head_dim,
                                head_dim,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_temporal,
                                multiply_zero=use_image_dataset,
                                use_tmp_window_attention=use_tmp_window_attention,
                                tmp_window_size=tmp_window_size))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(
                        out_dim, True, dims=2.0, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)

        # head
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))

        # zero out the last layer params
        nn.init.zeros_(self.out[-1].weight)
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (TemporalTransformer, SpatialTransformer, ResBlock)):
            module.gradient_checkpointing = value

    def forward(
            self,
            x,
            t,
            y,
            fps=None,
            video_mask=None,
            grounding_input=None,
            focus_present_mask=None,
            prob_focus_present=0.,
            obj_oriented_attn_ratio=0.0,
            mask_last_frame_num=0,  # mask last frame num
            visual_feat=None,
            control_feat=None,
    ):
        """
        prob_focus_present: probability at which a given batch sample will focus on the present
                            (0. is all off, 1. is completely arrested attention across time)
        """
        if isinstance(grounding_input, dict):
            if t[0] / 1000 > (1.0 - obj_oriented_attn_ratio):
                grounding_input["obj_oriented_attn"] = True
            else:
                grounding_input["obj_oriented_attn"] = False

        batch, device = x.shape[0], x.device
        self.batch = batch

        # image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(
                focus_present_mask, lambda: prob_mask_like(
                    (batch, ), prob_focus_present, device=device))

        time_rel_pos_bias = None
        # embeddings
        if self.use_fps_condition and fps is not None:
            e = self.time_embed(sinusoidal_embedding(
                t, self.dim)) + self.fps_embedding(
                    sinusoidal_embedding(fps, self.dim))
        else:
            e = self.time_embed(sinusoidal_embedding(t, self.dim))
        context = y
   
        # repeat f times for spatial e and context
        f = x.shape[2]
        e = e.repeat_interleave(repeats=f, dim=0)
        context = context.repeat_interleave(repeats=f, dim=0)

        # always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        # encoder
        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,
                                     focus_present_mask, video_mask, 
                                     grounding_input=grounding_input,
                                     visual_feat=visual_feat)
            xs.append(x)

        # middle
        # if isinstance(control_feat, list):
        #     x = x + control_feat.pop()
        for block in self.middle_block:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,
                                     focus_present_mask, video_mask, 
                                     grounding_input=grounding_input,
                                     visual_feat=visual_feat)

        # decoder
        for block in self.output_blocks:
            if isinstance(control_feat, list):
                x_in = torch.cat([xs.pop(), control_feat.pop()], dim=1)
            else:
                x_in = xs.pop()
            x = torch.cat([x, x_in], dim=1)
            x = self._forward_single(
                block,
                x,
                e,
                context,
                time_rel_pos_bias,
                focus_present_mask,
                video_mask,
                reference=xs[-1] if len(xs) > 0 else None,
                grounding_input=grounding_input,
                visual_feat=visual_feat)

        # head
        x = self.out(x)
        # reshape back to (b c f h w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=batch)
        return x

    def _forward_single(self,
                        module,
                        x,
                        e,
                        context,
                        time_rel_pos_bias,
                        focus_present_mask,
                        video_mask,
                        reference=None,
                        grounding_input=None,
                        visual_feat=None):
        if isinstance(module, ResidualBlock):
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            x = module(x, context, grounding_input, visual_feat)
        elif isinstance(module, TemporalTransformer):
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, context, grounding_input)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, CrossAttention):
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            x = module(x, context, grounding_input, visual_feat)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, Resample):
            x = module(x, reference)
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, e, context,
                                         time_rel_pos_bias, focus_present_mask,
                                         video_mask, reference, grounding_input=grounding_input,
                                         visual_feat=visual_feat)
        else:
            x = module(x)
        return x


def sinusoidal_embedding(timesteps, dim):
    # check input
    half = dim // 2
    timesteps = timesteps.float()
    # compute sinusoidal embedding
    sinusoid = torch.outer(
        timesteps, torch.pow(10000,
                             -torch.arange(half).to(timesteps).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    if dim % 2 != 0:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    return x


class CrossAttention(nn.Module):

    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            if sim.shape != mask.shape:
                mask = rearrange(mask, 'b () (i j) -> b i j', i=sim.shape[1])
            sim.masked_fill_(~mask.bool(), max_neg_value)
        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)
        # if exists(mask):
        #     print(sim.shape)
        #     print(sim[:,:sim.shape[1]-24,:sim.shape[1]-16-8].sum(2).mean(1))
        #     print(sim[:,:sim.shape[1]-24,sim.shape[1]+24:].sum(2).mean(1))
        #     import ipdb;ipdb.set_trace()
        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if has_inf_or_nan(out):
            import ipdb; ipdb.set_trace()
        return self.to_out(out)


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 disable_self_attn=False,
                 use_linear=False,
                 use_checkpoint=True,
                 use_gate_self_attn=False,
               ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d],
                disable_self_attn=disable_self_attn,
                use_gate_self_attn=use_gate_self_attn,
                checkpoint=use_checkpoint,
                ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(
                    inner_dim, in_channels, kernel_size=1, stride=1,
                    padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, grounding_input=None, visual_feat=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        # import ipdb; ipdb.set_trace()
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], grounding_input=grounding_input, visual_feat=visual_feat)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 disable_self_attn=False,
                 use_linear=False,
                 use_checkpoint=True,
                 only_self_att=True,
                 multiply_zero=False,
                 use_tmp_window_attention=False,
                 tmp_window_size=3,
                 ):
        super().__init__()
        self.use_tmp_window_attention = use_tmp_window_attention
        self.window_size = tmp_window_size
        self.multiply_zero = multiply_zero
        self.only_self_att = only_self_att
        if self.only_self_att:
            context_dim = None
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        
        if not use_linear:
            self.proj_in = nn.Conv1d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d],
                checkpoint=use_checkpoint) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(
                    inner_dim, in_channels, kernel_size=1, stride=1,
                    padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, grounding_input=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if self.only_self_att:
            context = None
        if not isinstance(context, list):
            context = [context]
        b, c, f, h, w = x.shape
        x_in = x
        x = self.norm(x)
        
       
        if not self.use_linear:
            x = rearrange(x, 'b c f h w -> (b h w) c f').contiguous()
            x = self.proj_in(x)
        if self.use_linear:
            x = rearrange(
                x, '(b f) c h w -> b (h w) f c', f=self.frames).contiguous()
            x = self.proj_in(x)

        
        if self.use_tmp_window_attention:
            x = rearrange(x, 'bhw c f -> bhw f c').contiguous()
            ## generating sliding windows
            self.window_size=3
            windows_inds = [[i,i+self.window_size] for i in range(x.shape[-2]-self.window_size+1)]
            slice_all_x = []
            for win_start, win_end in windows_inds:
                slice_all_x.append(x[:,win_start:win_end,:])
            all_x = torch.cat(slice_all_x,dim=0)
            for i, block in enumerate(self.transformer_blocks):
                all_x = block(all_x)
            num_blocks = len(windows_inds)
            all_x = rearrange(all_x, "(b w) c d -> b w c d", w=num_blocks)
            value = torch.zeros_like(x)
            weights = torch.zeros_like(x)
        
            for i, (win_start, win_end) in enumerate(windows_inds):
                value[:,win_start: win_end] = all_x[:,i]
                weights[:,win_start:win_end] += torch.ones_like(weights[:,win_start:win_end])
            x = value/weights   
            x = rearrange(x, '(b hw) f c -> b hw f c', b=b).contiguous()         
        elif self.only_self_att:
            x = rearrange(x, 'bhw c f -> bhw f c').contiguous()
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, grounding_input=grounding_input, is_temp=True)
            x = rearrange(x, '(b hw) f c -> b hw f c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) c f -> b hw f c', b=b).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                context[i] = rearrange(
                    context[i], '(b f) l con -> b f l con',
                    f=self.frames).contiguous()
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_i_j = repeat(
                        context[i][j],
                        'f l con -> (f r) l con',
                        r=(h * w) // self.frames,
                        f=self.frames).contiguous()
                    x[j] = block(x[j], context=context_i_j)

        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) f c -> b f c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw f c -> (b hw) c f').contiguous()
            x = self.proj_out(x)
            x = rearrange(
                x, '(b h w) c f -> b c f h w', b=b, h=h, w=w).contiguous()

        if self.multiply_zero:
            x = 0.0 * x + x_in
        else:
            x = x + x_in
        return x


class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True,
                 checkpoint=True,
                 use_gate_self_attn=False,
                 disable_self_attn=False,
                 ):
        super().__init__()
        attn_cls = CrossAttention
        # By default self.attn1 is self attention on image features
        assert disable_self_attn == False
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else
            None)  # is a self-attention if not self.disable_self_attn

        self.use_gate_self_attn = True if use_gate_self_attn else False
        if self.use_gate_self_attn:
            if use_gate_self_attn == "vanilla":
                self.gate_attn = GATEDSelfAttentionDense(
                    query_dim=dim,
                    context_dim=1024,
                    n_heads=n_heads,
                    d_head=d_head,
                )
            elif use_gate_self_attn == "double":
                self.gate_attn = DoubleGATEDSelfAttention(
                    query_dim = dim,
                    context_dim = 1024,
                    n_heads = n_heads,
                    d_head = d_head,
                )
            elif use_gate_self_attn == "merge":
                self.gate_attn = MergeGatedSelfAttention(
                    query_dim = dim,
                    context_dim = 1024,
                    n_heads = n_heads,
                    d_head = d_head 
                )
            elif use_gate_self_attn == "vanilla_cross_vision":
                self.gate_attn = GATEDSelfAttentionDense(
                    query_dim=dim,
                    context_dim=1024,
                    n_heads=n_heads,
                    d_head=d_head,
                    v_dim=1024,
                    cross_vision=True
                )
            elif use_gate_self_attn == "vanilla_cross_vision_parallel":
                self.gate_attn = GATEDSelfAttentionDense(
                    query_dim=dim,
                    context_dim=1024,
                    n_heads=n_heads,
                    d_head=d_head,
                    v_dim=1024,
                    parallel=True
                )
            elif use_gate_self_attn == "vanilla_cross_vision_clip_vision":
                self.gate_attn = GATEDSelfAttentionDense(
                    query_dim=dim,
                    context_dim=1024,
                    n_heads=n_heads,
                    d_head=d_head,
                    v_dim=1280,
                    parallel=True
                )
            elif use_gate_self_attn == "vanilla_cross_vision_parallel_clip_vision":
                self.gate_attn = GATEDSelfAttentionDense(
                    query_dim=dim,
                    context_dim=1024,
                    n_heads=n_heads,
                    d_head=d_head,
                    v_dim=1280,
                    parallel=True
                )
            else:
                raise ValueError(
                    f"Do not support {use_gate_self_attn}"
                )

        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # By default self.attn2 is cross attention on image and text
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint


    def forward(self, x, context=None, grounding_input=None,  visual_feat=None, is_temp=False,):
        visual_self_mask = None
        text_cross_mask = None
        temp_mask = None
        self_mask = None

        if isinstance(grounding_input, dict):
            grounding_context = grounding_input.get('context')
            grounding_mask = grounding_input.get('masks')

            if 'fg_index' in grounding_input.keys():
                if grounding_input is not None:
                    grounding_context = grounding_input.get('context')
                    box_masks = grounding_input.get('masks').bool()
                    boxes = grounding_input.get('boxes')
                    fg_text_index = grounding_input.get("fg_index")
                    token_length = grounding_input.get("text_length")
                    obj_oriented_attn = grounding_input.get("obj_oriented_attn")
                else:
                    boxes = grounding_context = box_masks = fg_text_index = token_length = None
                    obj_oriented_attn = False
                if boxes is not None and obj_oriented_attn:
                    shape = (x.shape[1], x.shape[0]) if is_temp else x.shape[:2]
                    visual_self_mask, text_cross_mask, temp_mask = create_masks(
                        shape, boxes, box_masks, context, fg_text_index, token_length)
                self_mask = temp_mask if is_temp else visual_self_mask
        else:
            grounding_context = grounding_mask = None
     

        x = self.attn1(
            self.norm1(x),
            context=context if self.disable_self_attn else None, mask=self_mask) + x
        # add self gate attention 
        if self.use_gate_self_attn and grounding_context is not None:
            x = self.gate_attn(x=x, context=grounding_context, mask=grounding_mask, visual_tokens=visual_feat)
        cross_mask = temp_mask if is_temp else text_cross_mask
        x = self.attn2(self.norm2(x), context=context, mask=cross_mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


# feedforward
class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(
            dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout),
                                 nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(
                self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    :param use_temporal_conv: if True, use the temporal convolution.
    :param use_image_dataset: if True, the temporal parameters will not be optimized.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        use_temporal_conv=True,
        use_image_dataset=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock_v2(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                use_image_dataset=use_image_dataset)

    def forward(self, x, emb, batch_size):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb, batch_size)

    def _forward(self, x, emb, batch_size):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv:
            h = rearrange(h, '(b f) c h w -> b c f h w', b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, 'b c f h w -> (b f) c h w')
        return h


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if self.use_conv:
            self.op = nn.Conv2d(
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Resample(nn.Module):

    def __init__(self, in_dim, out_dim, mode):
        assert mode in ['none', 'upsample', 'downsample']
        super(Resample, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mode = mode

    def forward(self, x, reference=None):
        if self.mode == 'upsample':
            assert reference is not None
            x = F.interpolate(x, size=reference.shape[-2:], mode='nearest')
        elif self.mode == 'downsample':
            x = F.adaptive_avg_pool2d(
                x, output_size=tuple(u // 2 for u in x.shape[-2:]))
        return x


class ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 embed_dim,
                 out_dim,
                 use_scale_shift_norm=True,
                 mode='none',
                 dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_scale_shift_norm = use_scale_shift_norm
        self.mode = mode

        # layers
        self.layer1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1))
        self.resample = Resample(in_dim, in_dim, mode)
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim,
                      out_dim * 2 if use_scale_shift_norm else out_dim))
        self.layer2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1))
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Conv2d(
            in_dim, out_dim, 1)
        # zero out the last layer params
        nn.init.zeros_(self.layer2[-1].weight)

    def forward(self, x, e, reference=None):
        identity = self.resample(x, reference)
        x = self.layer1[-1](self.resample(self.layer1[:-1](x), reference))
        e = self.embedding(e).unsqueeze(-1).unsqueeze(-1).type(x.dtype)
        if self.use_scale_shift_norm:
            scale, shift = e.chunk(2, dim=1)
            x = self.layer2[0](x) * (1 + scale) + shift
            x = self.layer2[1:](x)
        else:
            x = x + e
            x = self.layer2(x)
        x = x + self.shortcut(identity)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, context_dim=None, num_heads=None, head_dim=None):
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)

        # layers
        self.norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        if context_dim is not None:
            self.context_kv = nn.Linear(context_dim, dim * 2)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x, context=None):
        r"""x:       [B, C, H, W].
            context: [B, L, C] or None.
        """
        identity = x
        b, c, h, w, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        x = self.norm(x)
        q, k, v = self.to_qkv(x).view(b, n * 3, d, h * w).chunk(3, dim=1)
        if context is not None:
            ck, cv = self.context_kv(context).reshape(b, -1, n * 2,
                                                      d).permute(0, 2, 3,
                                                                 1).chunk(
                                                                     2, dim=1)
            k = torch.cat([ck, k], dim=-1)
            v = torch.cat([cv, v], dim=-1)

        # compute attention
        attn = torch.matmul(q.transpose(-1, -2) * self.scale, k * self.scale)
        attn = F.softmax(attn, dim=-1)

        # gather context
        x = torch.matmul(v, attn.transpose(-1, -2))
        x = x.reshape(b, c, h, w)
        # output
        x = self.proj(x)
        return x + identity


class TemporalConvBlock_v2(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim=None,
                 dropout=0.0,
                 use_image_dataset=False):
        super(TemporalConvBlock_v2, self).__init__()
        if out_dim is None:
            out_dim = in_dim  # int(1.5*in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)))

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if self.use_image_dataset:
            x = identity + 0.0 * x
        else:
            x = identity + x
        return x


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        mask = torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
        # aviod mask all, which will cause find_unused_parameters error
        if mask.all():
            mask[0] = False
        return mask


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
    raise ValueError(f'unsupported dimensions: {dims}')


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
    raise ValueError(f'unsupported dimensions: {dims}')



# GATEDSelfAttention
class GATEDSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head, v_dim=None, cross_vision=False, parallel=False):
        super().__init__()

        # no problem with linear weights
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = CrossAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        self.scale = 1
        self.parallel = parallel 
        self.cross_vision = cross_vision 
        if self.cross_vision or self.parallel:
            assert v_dim is not None
            self.cross_attn = CrossAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
            self.cross_ff = FeedForward(query_dim, glu=True)
            self.v_linear = nn.Linear(v_dim, query_dim)
            self.cross_norm1 = nn.LayerNorm(query_dim)
            self.cross_norm2 = nn.LayerNorm(query_dim)

            self.register_parameter('cross_alpha_attn', nn.Parameter(torch.tensor(0.)))
            self.register_parameter('cross_alpha_dense', nn.Parameter(torch.tensor(0.)))
        
        if self.parallel:
            self.register_parameter('beta_g', nn.Parameter(torch.tensor(0.)))
            self.register_parameter('beta_v', nn.Parameter(torch.tensor(0.)))



    def forward(self, x, context, mask, visual_tokens=None):
        # batch max_num_obj
        context = self.linear(context)
        mask = None
        if mask is not None:
            one_mask_for_visual = torch.ones(mask.shape[0],x.shape[1]).to(mask.device)
            mask = torch.cat([one_mask_for_visual , mask],dim=1)
            mask = mask.unsqueeze(1).repeat(1,mask.shape[-1],1)
        
        x_g = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(self.norm1(torch.cat([x, context], dim=1)),mask=mask)[:, :x.shape[1], :]
        x_g = x_g + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x_g))
        
        # print("alpha_attn",torch.tanh(self.alpha_attn))
        # print("alpha_dense",torch.tanh(self.alpha_dense))
        if self.cross_vision:
            visual_context = self.v_linear(visual_tokens)
            x_v = x_g + self.scale * torch.tanh(self.cross_alpha_attn) * self.cross_attn(self.cross_norm1(x_g),context=visual_context,mask=mask)
            x_out = x_v + self.scale * torch.tanh(self.cross_alpha_dense) * self.cross_ff(self.cross_norm2(x_v))
            # print("cross_alpha_attn", torch.tanh(self.cross_alpha_attn))
            # print("cross_alpha_dense", torch.tanh(self.cross_alpha_dense))
            #import ipdb;ipdb.set_trace()

        elif self.parallel:
            visual_context = self.v_linear(visual_tokens)
            x_v = x + self.scale * torch.tanh(self.cross_alpha_attn) * self.cross_attn(self.cross_norm1(x),context=visual_context,mask=mask)
            x_v = x_v + self.scale * torch.tanh(self.cross_alpha_dense) * self.cross_ff(self.cross_norm2(x_v))
            x_out = x + torch.tanh(self.beta_g)*x_g + torch.tanh(self.beta_v)*x_v
            
            # print("cross_alpha_attn", torch.tanh(self.cross_alpha_attn))
            # print("cross_alpha_dense", torch.tanh(self.cross_alpha_dense))
            # print("beta_g", torch.tanh(self.beta_g))
            # print("beta_v", torch.tanh(self.beta_v))
            #import ipdb;ipdb.set_trace()
        else:
            x_out = x_g

        if torch.isnan(x).any(): 
            import ipdb;ipdb.set_trace()
        
        return x_out




class DoubleGATEDSelfAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        self.linear = nn.Linear(context_dim, query_dim)
        self.attn = CrossAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        self.scale = 1

        self.visual_linear = nn.Linear(context_dim, query_dim)
        self.visual_attn = CrossAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.visual_ff = FeedForward(query_dim, glu=True)
        self.visual_norm1 = nn.LayerNorm(query_dim)
        self.visual_norm2 = nn.LayerNorm(query_dim)
        self.register_parameter('visual_alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('visual_alpha_dense', nn.Parameter(torch.tensor(0.)))

        self.register_parameter('beta_g', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('beta_v', nn.Parameter(torch.tensor(0.)))
    
    def forward(self, x, context, mask, visual_tokens):
        # batch max_num_obj 
        context = self.linear(context)
        if mask is not None:
            one_mask_for_visual = torch.ones(mask.shape[0],x.shape[1]).to(mask.device)
            mask = torch.cat([one_mask_for_visual , mask],dim=1)
            mask = mask.unsqueeze(1).repeat(1,mask.shape[-1],1)
        x_g = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(
            self.norm1(torch.cat([x, context], dim=1)),mask=mask)[:, :x.shape[1], :]
        x_g = x_g + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x_g))
        
        visual_context = self.visual_linear(visual_tokens)
        x_v = x + self.scale * torch.tanh(self.visual_alpha_attn) * self.visual_attn(
            self.visual_norm1(torch.cat([x,visual_context], dim=1)))[:, :x.shape[1],:]
        x_v = x_v + self.scale * torch.tanh(self.visual_alpha_dense) * self.visual_ff(self.visual_norm2(x_v))
        x = x+ torch.tanh(self.beta_g) * x_g + torch.tanh(self.beta_v) * x_v 
        
        
        return x


class MergeGatedSelfAttention(nn.Module):

    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        self.linear = nn.Linear(context_dim, query_dim)
        self.v_linear = nn.Linear(context_dim, query_dim)

        self.attn = CrossAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))
        self.scale = 1


    def forward(self, x, context, mask, visual_tokens):
        context = self.linear(context)
        v_context = self.v_linear(visual_tokens)
        
        if mask is not None:
            one_mask_for_visual = torch.ones(mask.shape[0],x.shape[1]+visual_tokens.shape[1]).to(mask.device)
            mask = torch.cat([one_mask_for_visual , mask],dim=1)
            mask = mask.unsqueeze(1).repeat(1,mask.shape[-1],1)
            #mask = None
            #if mask is not None:
            #    import ipdb;ipdb.set_trace()
        # check mask, check if type embedding is needed
        x_1 = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(self.norm1(torch.cat([x,v_context, context], dim=1)),mask=mask)[:, :x.shape[1], :]
        x_2 = x_1 + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x_1))
        
        # x_1 = x + 0.45 * self.attn(self.norm1(torch.cat([x,v_context, context], dim=1)),mask=mask)[:, :x.shape[1], :]
        # x_2 = x_1 + 0.45 * self.ff(self.norm2(x_1))
        
        # show_norm_info({
        #     "x":x,
        #     "x_1":x_1,
        #     "x_2":x_2,
        # }, dim=2)
        
        
        # # distance mean, max, min
        # print("Infor of  x x_1")
        # show_distance_info(x,x_1, dim=2)
 
        # print("Infor of x x_2 ") 
        # show_distance_info(x, x_2, dim=2)

        # print("Infor of x_1 x_2")
        # show_distance_info(x_1,x_2, dim=2)
        
        #import ipdb;ipdb.set_trace()
        return x_2


def show_distance_info(x1, x2, dim):
    dis = (x1-x2).pow(2).sum(dim).sqrt()
    mean = dis.mean()
    max = dis.max()
    min = dis.min()
    var = dis.var()
    
    print("--------------------")
    print("Mean Distance --> ", mean)
    print("Max Distance --> ", max)
    print("Min Distance --> ", min)
    print("Var Distance --> ",var)
    print("--------------------")


def show_norm_info(x_dict,dim):
    """
    x_dict = {x:, x_1: , x_2: }
    """
 
    all_info = {}
    for name, x in x_dict.items():
        norm_info = torch.norm(x,dim=dim)
        mean = norm_info.mean(1)
        var = norm_info.var(1)
        max = norm_info.max(1)
        min = norm_info.min(1)
        all_info[name]={"mean":mean,
                        "var": var,
                        "max": max,
                        "min": min
                        }
    
    print("----------Norm Info----------")
    
    for name in all_info.keys():
        print(f"Mean of {name}", all_info[name]["mean"][0])
    
    for name in all_info.keys():
        print(f"Var of {name}", all_info[name]["var"])

    for name in all_info.keys():
        print(f"Max of {name}", all_info[name]["max"][0])
    
    for name in all_info.keys():
        print(f"Min of {name}", all_info[name]["min"][0])
    print("-----------------------------")

