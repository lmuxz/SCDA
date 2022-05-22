# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_funcs import entropy_func
import utils.loss as loss
from models import *
from utils.utils import get_coeff

from trainer.base_trainer import BaseTrainer


__all__ = ['ToAlign']


class ToAlign(BaseTrainer):
    def __init__(self, cfg):
        super(ToAlign, self).__init__(cfg)

    def build_base_models(self):
        basenet_name = self.cfg.MODEL.BASENET
        kwargs = {
            'embedding_input': self.cfg.MODEL.EMBED_INPUT,
            'embedding_dim': self.cfg.MODEL.EMBED_DIM,
            'num_dim': self.cfg.MODEL.NUM_DIM,
            'num_classes': self.cfg.DATASET.NUM_CLASSES,
            'hda': True,
            'toalign': True,
        }

        basenet = eval(basenet_name)(**kwargs).cuda()

        return basenet

    def build_models(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        # backbone
        self.base_net = self.build_base_models()
        # discriminator
        self.d_net = eval(self.cfg.MODEL.DNET)(
            in_feature=self.cfg.DATASET.NUM_CLASSES,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            out_feature=self.cfg.MODEL.D_OUTDIM
        ).cuda()

        self.registed_models = {'base_net': self.base_net, 'd_net': self.d_net}
        parameter_list = self.base_net.get_parameters() + self.d_net.get_parameters()
        self.build_optim(parameter_list)

    def one_step(self, data_src, data_tar):
        num_inputs_src, cate_inputs_src, labels_src = data_src['num_input'].cuda(), data_src['cate_input'].cuda(), data_src['output'].cuda()

        num_inputs_tar, cate_inputs_tar = data_tar['num_input'].cuda(), data_tar['cate_input'].cuda()

        # --------- classification --------------
        outputs_all_src = self.base_net(num_inputs_src, cate_inputs_src)  # [f, y, z]
        assert len(outputs_all_src) == 3, \
            f'Expected return with size 3, but got {len(outputs_all_src)}'
        loss_cls_src = F.cross_entropy(outputs_all_src[1], labels_src)
        focals_src = outputs_all_src[-1]

        # --------- alignment --------------
        outputs_all_src = self.base_net(num_inputs_src, cate_inputs_src, toalign=True, labels=labels_src)  # [f_p, y_p, z_p]
        outputs_all_tar = self.base_net(num_inputs_tar, cate_inputs_tar)  # [f, y, z]
        assert len(outputs_all_src) == 3 and len(outputs_all_tar) == 3, \
            f'Expected return with size 3, but got {len(outputs_all_src)}'
        focals_tar = outputs_all_tar[-1]

        logits_all = torch.cat((outputs_all_src[1], outputs_all_tar[1]), dim=0)
        softmax_all = nn.Softmax(dim=1)(logits_all)
        focals_all = torch.cat((focals_src, focals_tar), dim=0)

        ent_tar = entropy_func(nn.Softmax(dim=1)(outputs_all_tar[1].data)).mean()

        # domain alignment
        loss_alg = loss.d_align_uda(
            softmax_output=softmax_all, d_net=self.d_net,
            coeff=get_coeff(self.ite, max_iter=self.cfg.TRAIN.TTL_ITE), ent=self.cfg.METHOD.ENT
        )

        # hda
        loss_hda = focals_all.abs().mean()

        loss_ttl = loss_cls_src + loss_alg * self.cfg.METHOD.W_ALG + loss_hda

        # update
        self.step(loss_ttl)

        # display
        if self.ite % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_alg: {loss_alg.item():.3f}',
                f'l_hda: {loss_hda.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'ent_tar: {ent_tar.item():.3f}',
                f'best_neg_loss: {self.best_neg_loss:.3f}',
            ])