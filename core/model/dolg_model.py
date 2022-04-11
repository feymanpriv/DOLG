#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Written by feymanpriv

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import core.net as net
from core.config import cfg
from core.model.resnet import ResNet, ResHead
from core.model.resnet import GeneralizedMeanPoolingP

""" Dolg models """

class DOLG(nn.Module):
    """ DOLG model """
    def __init__(self):
        super(DOLG, self).__init__()
        self.pool_l= nn.AdaptiveAvgPool2d((1, 1)) 
        self.pool_g = GeneralizedMeanPoolingP(norm=3.0) 
        self.fc_t = nn.Linear(cfg.MODEL.S4_DIM, cfg.MODEL.S3_DIM, bias=True)
        self.fc = nn.Linear(cfg.MODEL.S4_DIM, cfg.MODEL.HEADS.REDUCTION_DIM, bias=True)
        self.globalmodel = ResNet()
        self.localmodel = SpatialAttention2d(cfg.MODEL.S3_DIM)
        self.desc_cls = Arcface(cfg.MODEL.HEADS.REDUCTION_DIM, cfg.MODEL.NUM_CLASSES)

    def forward(self, x, targets):
        """ Global and local orthogonal fusion """
        f3, f4 = self.globalmodel(x)
        fl, _ = self.localmodel(f3)
        
        fg_o = self.pool_g(f4)
        fg_o = fg_o.view(fg_o.size(0), cfg.MODEL.S4_DIM)
        
        fg = self.fc_t(fg_o)
        fg_norm = torch.norm(fg, p=2, dim=1)
        
        proj = torch.bmm(fg.unsqueeze(1), torch.flatten(fl, start_dim=2))
        proj = torch.bmm(fg.unsqueeze(2), proj).view(fl.size())
        proj = proj / (fg_norm * fg_norm).view(-1, 1, 1, 1)
        orth_comp = fl - proj

        fo = self.pool_l(orth_comp)
        fo = fo.view(fo.size(0), cfg.MODEL.S3_DIM)

        final_feat=torch.cat((fg, fo), 1)
        global_feature = self.fc(final_feat)

        global_logits = self.desc_cls(global_feature, targets)
        return global_feature, global_logits

    '''
    def forward(self, x, targets):
        """ Global and local orthogonal fusion """
        feamap3, feamap4 = self.globalmodel(x)

        g_f = self.pool_g(feamap4)
        b, c, h, w = g_f.size(0), g_f.size(1), g_f.size(2), g_f.size(3)
        x = g_f.view(b, -1)
        x = self.fc_t(x)
        g_f = x.view(b, c // 2, h, w)
        e_f = g_f.expand_as(feamap3)

        local_feamap3, _ = self.localmodel(feamap3)
        proj = torch.sum(e_f * local_feamap3, dim=1) / 
                    torch.sum(e_f * e_f, dim=1).unsqueeze(1) * e_f
        
        orth_comp = local_feamap3 - proj
        p_f = self.pool_l(orth_feamap3)
        p_f = p_f.view(p_f.size(0), -1)
        g_f = g_f.view(g_f.size(0), -1)

        global_feature=torch.cat((g_f, p_f), 1)
        global_feature = self.fc(global_feature)

        global_logits = self.desc_cls(global_feature, targets)
        return global_feature, global_logits
    '''


class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    '''
    def __init__(self, in_c, act_fn='relu', with_aspp=cfg.MODEL.WITH_MA):
        super(SpatialAttention2d, self).__init__()
        
        self.with_aspp = with_aspp
        if self.with_aspp:
            self.aspp = ASPP(cfg.MODEL.S3_DIM)
        self.conv1 = nn.Conv2d(in_c, cfg.MODEL.S3_DIM, 1, 1)
        self.bn = nn.BatchNorm2d(cfg.MODEL.S3_DIM, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(cfg.MODEL.S3_DIM, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

        for conv in [self.conv1, self.conv2]: 
            conv.apply(net.init_weights)

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        if self.with_aspp:
            x = self.aspp(x)
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)

        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        x = att * feature_map_norm
        return x, att_score
    
    def __repr__(self):
        return self.__class__.__name__


class ASPP(nn.Module):
    '''
    Atrous Spatial Pyramid Pooling Module 
    '''
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp = []
        self.aspp.append(nn.Conv2d(in_c, 512, 1, 1))

        for dilation in [6, 12, 18]:
            _padding = (dilation * 3 - dilation) // 2
            self.aspp.append(nn.Conv2d(in_c, 512, 3, 1, padding=_padding, dilation=dilation))
        self.aspp = nn.ModuleList(self.aspp)

        self.im_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(in_c, 512, 1, 1),
                                     nn.ReLU())
        conv_after_dim = 512 * (len(self.aspp)+1)
        self.conv_after = nn.Sequential(nn.Conv2d(conv_after_dim, 1024, 1, 1), nn.ReLU())
        
        for dilation_conv in self.aspp:
            dilation_conv.apply(net.init_weights)
        for model in self.im_pool:
            if isinstance(model, nn.Conv2d):
                model.apply(net.init_weights)
        for model in self.conv_after:
            if isinstance(model, nn.Conv2d):
                model.apply(net.init_weights)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        aspp_out = [F.interpolate(self.im_pool(x), scale_factor=(h,w), mode="bilinear", align_corners=False)]
        for i in range(len(self.aspp)):
            aspp_out.append(self.aspp[i](x))
        aspp_out = torch.cat(aspp_out, 1)
        x = self.conv_after(aspp_out)
        return x


class Arcface(nn.Module):
    """ Additive Angular Margin Loss """
    def __init__(self, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN

        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets):
        # get cos(theta)
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self._s
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )

