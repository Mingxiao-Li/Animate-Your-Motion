import torch
import torch.nn as nn
from einops import rearrange, repeat
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from modelscope_t2v.unet_sd import TemporalTransformer, ResBlock, SpatialTransformer, Downsample

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


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

class ControlNet(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 in_dim=4,
                 dim=320,
                 context_dim=1024,
                 num_heads=8,
                 head_dim=64,
                 num_res_blocks=2,
                 dropout=0.1,
                 dim_mult=[1, 2, 4, 4],
                 attn_scales=[1, 1 / 2, 1 / 4],
                 temporal_attention=True,
                 use_fps_condition=False,
                 use_image_dataset=False,
                 use_tmp_window_attention=False,
                 tmp_window_size=3,
                 ):
        
        super(ControlNet, self).__init__()
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        transformer_depth = 1
        disabled_sa = False
        use_linear_in_temporal = False
        self.embed_dim = embed_dim
        self.dim_mult = dim_mult
        self.in_dim = in_dim 
        self.temporal_attention = temporal_attention
        self.num_res_blocks = num_res_blocks
        self.context_dim = context_dim
        self.use_fps_condition = use_fps_condition
        self.use_image_dataset = use_image_dataset
        self.attn_scales = attn_scales


        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(
                nn.Linear(dim, embed_dim), nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)
        

        self.input_blocks = nn.ModuleList()
        init_block = nn.ModuleList([
            nn.Conv2d(self.in_dim, dim, 3, padding=1)
            ])

        self.zero_convs = nn.ModuleList(
            [self.make_zero_conv(dim)]
        )

        self.input_hint_block = zero_module(conv_nd(2, 4, dim, 3, padding=1))
        #     nn.Sequential(
        #     # conv_nd(2, 4, 16, 3, padding=1),
        #     # nn.SiLU(),
        #     # conv_nd(2, 16, 16, 3, padding=1),
        #     # nn.SiLU(),
        #     # conv_nd(2, 16, 32, 3, padding=1),
        #     # nn.SiLU(),
        #     # conv_nd(2, 32, 32, 3, padding=1),
        #     # nn.SiLU(),
        #     # conv_nd(2, 32, 96, 3, padding=1),
        #     # nn.SiLU(),
        #     # conv_nd(2, 96, 96, 3, padding=1),
        #     # nn.SiLU(),
        #     # conv_nd(2, 96, 256, 3, padding=1, stride=2),
        #     # nn.SiLU(),
        #     zero_module(conv_nd(2, 4, dim, 3, padding=1)),
        # )

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
                    tmp_window_size=tmp_window_size)
            )
        
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
                            use_gate_self_attn=False,
                            disable_self_attn=False,
                            use_linear=True)
                    )
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
                self.zero_convs.append(
                    self.make_zero_conv(in_dim)
                )

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)
                    self.zero_convs.append(
                        self.make_zero_conv(out_dim)
                    )

         # middle
        # self.middle_block = nn.ModuleList([
        #     ResBlock(
        #         out_dim,
        #         embed_dim,
        #         dropout,
        #         use_scale_shift_norm=False,
        #         use_image_dataset=use_image_dataset,
        #     ),
        #     SpatialTransformer(
        #         out_dim,
        #         out_dim // head_dim,
        #         head_dim,
        #         depth=1,
        #         context_dim=self.context_dim,
        #         disable_self_attn=False,
        #         use_gate_self_attn=False,
        #         use_linear=True)
        #      ])
        #
        # if self.temporal_attention:
        #     self.middle_block.append(
        #         TemporalTransformer(
        #             out_dim,
        #             out_dim // head_dim,
        #             head_dim,
        #             depth=transformer_depth,
        #             context_dim=context_dim,
        #             disable_self_attn=disabled_sa,
        #             use_linear=use_linear_in_temporal,
        #             multiply_zero=use_image_dataset,
        #             use_tmp_window_attention=use_tmp_window_attention,
        #             tmp_window_size=tmp_window_size
        #         ))
        #
        # self.middle_block.append(
        #     ResBlock(
        #         out_dim,
        #         embed_dim,
        #         dropout,
        #         use_scale_shift_norm=False,
        #         use_image_dataset=use_image_dataset,
        #     ))
        #
        # self.middle_block_out = self.make_zero_conv(out_dim)
    

    def make_zero_conv(self, channels):
        return zero_module(conv_nd(2, channels, channels, 1, padding=0))
    
    def forward(self, x, hint, t, y, **kwargs):
        emb = self.time_embed(sinusoidal_embedding(t, self.dim))
        guided_hint = self.input_hint_block(hint)

        # repeat f times for spatial e and context
        f = x.shape[2]
        self.batch = x.shape[0]
        emb = emb.repeat_interleave(repeats=f, dim=0)
        y = y.repeat_interleave(repeats=f, dim=0)
        guided_hint = guided_hint.repeat_interleave(repeats=f, dim=0)

        outs = []
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                x = self._forward_single(module, x, emb, y)
                x += guided_hint
                guided_hint = None
            else:
                x = self._forward_single(module, x, emb, y)
            outs.append(zero_conv(x))

        # x = self._forward_single(self.middle_block, x, emb, y)
        # outs.append(self.middle_block_out(x))

        return outs

    def _forward_single(self, module, x, emb, y):
        if isinstance(module, ResBlock):
            x = x.contiguous()
            x = module(x, emb, self.batch)
        elif isinstance(module, SpatialTransformer):
            x = module(x, y)
        elif isinstance(module, TemporalTransformer):
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, y)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, emb, y)
        else:
            x = module(x)
        return x
