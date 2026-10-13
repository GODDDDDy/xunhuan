import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel

from .networks import get_norm_layer, ResBlock, DenseBlock, SEBlock


class DefaultModel(BaseModel):
    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False):
        super(DefaultModel, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # for S1 and S2
        self.feature_extraction_S1 = nn.Sequential(
            nn.Conv2d(3, init_dim // 2, kernel_size=7, stride=1, padding=3, bias=use_bias),
            ResBlock(init_dim // 2, norm_layer, use_dropout, use_bias)
        )
        self.feature_extraction_S2 = nn.Sequential(
            nn.Conv2d(3, init_dim // 2, kernel_size=7, stride=1, padding=3, bias=use_bias),
            ResBlock(init_dim // 2, norm_layer, use_dropout, use_bias)
        )
        self.S12_feature_fuse = nn.Sequential(
            nn.Conv2d(init_dim, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.ReLU(True),
            DenseBlock(in_channel=init_dim, growth_rate=32, n_blocks=3)
        )
        self.S12_feature_downsampling1 = nn.Sequential(
            nn.Conv2d(init_dim, init_dim * 2, kernel_size=1, stride=1),
            ResBlock(init_dim * 2, norm_layer, use_dropout, use_bias),
            nn.Conv2d(init_dim * 2, init_dim * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        )
        self.S12_feature_downsampling2 = nn.Sequential(
            nn.Conv2d(init_dim * 2, init_dim * 4, kernel_size=1, stride=1),
            ResBlock(init_dim * 4, norm_layer, use_dropout, use_bias),
            nn.Conv2d(init_dim * 4, init_dim * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
        )

        # using Stokes feature guided skip connection instead of vanilla skip connection
        self.fgsc_s = nn.Sequential(
            nn.Conv2d(4 * init_dim * 2, 4 * init_dim, kernel_size=1, stride=1, bias=use_bias),
            ResBlock(4 * init_dim, norm_layer, use_dropout, use_bias),
            SEBlock(4 * init_dim, 8),
        )
        self.fgsc_m = nn.Sequential(
            nn.Conv2d(2 * init_dim * 2, 2 * init_dim, kernel_size=1, stride=1, bias=use_bias),
            ResBlock(2 * init_dim, norm_layer, use_dropout, use_bias),
            SEBlock(2 * init_dim, 8),
        )

        # for S0
        self.feature_extraction_S0 = nn.Sequential(
            nn.Conv2d(3, init_dim, kernel_size=7, stride=1, padding=3, bias=use_bias),
            ResBlock(init_dim, norm_layer, use_dropout, use_bias)
        )
        self.S0_feature_downsampling1 = nn.Sequential(
            nn.Conv2d(init_dim, init_dim * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(init_dim * 2),
            nn.ReLU(inplace=True),
            ResBlock(init_dim * 2, norm_layer, use_dropout, use_bias),
        )
        self.S0_feature_downsampling2 = nn.Sequential(
            nn.Conv2d(init_dim * 2, init_dim * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(init_dim * 4),
            nn.ReLU(inplace=True),
            ResBlock(init_dim * 4, norm_layer, use_dropout, use_bias),
        )
        self.embed_block = nn.Sequential(
            DenseBlock(in_channel=init_dim * 4, growth_rate=16, n_blocks=3)
        )
        self.S0_feature_upsampling1 = nn.Sequential(
            nn.ConvTranspose2d(init_dim * 4 * 2, init_dim * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(init_dim * 2),
            nn.ReLU(inplace=True),
            ResBlock(init_dim * 2, norm_layer, use_dropout, use_bias),
        )
        self.S0_feature_upsampling2 = nn.Sequential(
            nn.ConvTranspose2d(init_dim * 2 * 2, init_dim, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(init_dim),
            nn.ReLU(inplace=True),
            ResBlock(init_dim, norm_layer, use_dropout, use_bias),
        )
        self.out = nn.Sequential(
            nn.Conv2d(init_dim, 3, kernel_size=1, stride=1, bias=use_bias),
            nn.Tanh()
        )

    def forward(self, S0_hat, S1_hat, S2_hat):
        feature_S1 = self.feature_extraction_S1(S1_hat)
        feature_S2 = self.feature_extraction_S2(S2_hat)
        feature_S12 = self.S12_feature_fuse(torch.cat([feature_S1, feature_S2], dim=1))
        feature_S12_m = self.S12_feature_downsampling1(feature_S12)
        feature_S12_s = self.S12_feature_downsampling2(feature_S12_m)

        feature_S0 = self.feature_extraction_S0(S0_hat)
        feature_S0_m = self.S0_feature_downsampling1(feature_S0)
        feature_S0_s = self.S0_feature_downsampling2(feature_S0_m)

        skip_s = self.fgsc_s(torch.cat([feature_S0_s, feature_S12_s], dim=1))
        skip_m = self.fgsc_m(torch.cat([feature_S0_m, feature_S12_m], dim=1))

        feature_S0_s_out = self.embed_block(feature_S0_s)
        feature_S0_m_out = self.S0_feature_upsampling1(torch.cat([feature_S0_s_out, skip_s], dim=1))
        feature_S0_out = self.S0_feature_upsampling2(torch.cat([feature_S0_m_out, skip_m], dim=1))
        res = self.out(feature_S0_out)
        S0 = torch.clamp(res + S0_hat, min=0, max=1)
        return S0
