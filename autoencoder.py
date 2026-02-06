from diffusion_networks import *

class UNetBlock(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, up=False, down=False, attention=False,
                 num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
                 resample_filter=[1, 1], resample_proj=False, adaptive_scale=True,
                 init=dict(), init_zero=dict(init_weight=0), init_attn=None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
       
        self.conv1 = Conv2d(in_channels=out_channels,
                            out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3,
                              kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels,
                               out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        x = self.conv1(torch.nn.functional.dropout(
            x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(
                x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

class AutoEncoder(torch.nn.Module):
    def __init__(self,
                 # Image resolution at input/output.
                 img_resolution,
                 # Number of color channels at input.
                 in_channels,
                 # Number of color channels at output.
                 out_channels,
                 # Base multiplier for the number of channels.
                 model_channels=128,
                 latent_channels=1,
                 # Per-resolution multipliers for the number of channels.
                 channel_mult=[1, 1, 1, 1],
                 # Number of residual blocks per resolution.
                 num_blocks=4,
                 # List of resolutions with self-attention.
                 attn_resolutions=[16],
                 # Dropout probability of intermediate activations.
                 dropout=0.10,
                 # Dropout probability of class labels for classifier-free guidance.
                 label_dropout=0,
                 # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
                 encoder_type='standard',
                 # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                 decoder_type='standard',
                 # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
                 resample_filter=[1, 1],
                 ):
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
       
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(
                        in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)

        self.to_latent = Conv2d(
            in_channels=cout,
            out_channels=latent_channels,
            kernel=1,
        )

        self.from_latent = Conv2d(
            in_channels=latent_channels,
            out_channels=cout,
            kernel=1,
        )

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(
                        in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(
                    num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def encoder(self, x):
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        x=self.to_latent(x)
        return x

    def decoder(self, x):
        x=self.from_latent(x)
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                x = block(x)
        x=aux if aux is not None else x
        return x

    def forward(self, x, class_labels=None):

        # Conditioning.
        if class_labels is not None:
            tmp = class_labels
            x = torch.cat((x, tmp), dim=1)

        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec

