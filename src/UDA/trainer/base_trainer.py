# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import sys
import logging
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.utils import save_model
from utils.lr_scheduler import inv_lr_scheduler
from sklearn.model_selection import train_test_split

from datasets import *
from models import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.setup()
        self.build_datasets()
        self.build_models()
        # self.resume_from_ckpt()

    def setup(self):
        self.start_ite = 0
        self.ite = 0
        self.best_neg_loss = -float('inf')

    def build_datasets(self):
        logging.info(f'building dataset')
        self.dataset_loaders = {}

        source_train_input, source_test_input, source_train_output, source_test_output = train_test_split(
            self.cfg.TRAIN.SOURCE_INPUT, 
            self.cfg.TRAIN.SOURCE_OUTPUT, 
            test_size=self.cfg.TRAIN.TEST_SIZE
        )

        target_train_input, target_test_input = train_test_split(
            self.cfg.TRAIN.TARGET_INPUT, 
            test_size=self.cfg.TRAIN.TEST_SIZE)


        self.dataset_loaders['source_train'] = DataLoader(
            TabularDataset(source_train_input, self.cfg.TRAIN.CATE_INDEX, 0, source_train_output),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.WORKERS,
            drop_last=True
        )
        self.dataset_loaders['source_test'] = DataLoader(
            TabularDataset(source_test_input, self.cfg.TRAIN.CATE_INDEX, 0, source_test_output),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )
        self.dataset_loaders['target_train'] = DataLoader(
            TabularDataset(target_train_input, self.cfg.TRAIN.CATE_INDEX, 1),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.WORKERS,
            drop_last=True
        )
        self.dataset_loaders['target_test'] = DataLoader(
            TabularDataset(target_test_input, self.cfg.TRAIN.CATE_INDEX, 1),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )

        self.len_src = len(self.dataset_loaders['source_train'])
        self.len_tar = len(self.dataset_loaders['target_train'])

        logging.info(f'source train example length: {self.len_src}'
                     f'/{len(self.dataset_loaders["source_test"])}')
        logging.info(f'target train example length: {self.len_tar}'
                     f'/{len(self.dataset_loaders["target_test"])}')

    def build_models(self):
        logging.info(f'building models')
        self.base_net = self.build_base_models()
        self.registed_models = {'base_net': self.base_net}
        parameter_list = self.base_net.get_parameters()
        self.build_optim(parameter_list)


    def build_base_models(self):
        basenet_name = self.cfg.MODEL.BASENET
        kwargs = {
            'embedding_input': self.cfg.MODEL.EMBED_INPUT,
            'embedding_dim': self.cfg.MODEL.EMBED_DIM,
            'num_dim': self.cfg.MODEL.NUM_DIM,
            'num_classes': self.cfg.DATASET.NUM_CLASSES,
        }

        basenet = eval(basenet_name)(**kwargs).cuda()

        return basenet

    def build_optim(self, parameter_list: list):
        self.optimizer = optim.SGD(
            parameter_list,
            lr=self.cfg.TRAIN.LR,
            momentum=self.cfg.OPTIM.MOMENTUM,
            weight_decay=self.cfg.OPTIM.WEIGHT_DECAY,
            nesterov=True
        )
        self.lr_scheduler = inv_lr_scheduler

    def resume_from_ckpt(self):
        last_ckpt = os.path.join(self.cfg.TRAIN.OUTPUT_CKPT, 'models-last.pt')
        if os.path.exists(last_ckpt):
            ckpt = torch.load(last_ckpt)
            for k, v in self.registed_models.items():
                v.load_state_dict(ckpt[k])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_ite = ckpt['ite']
            self.best_neg_loss = ckpt['best_neg_loss']
            logging.info(f'> loading ckpt from {last_ckpt} | ite: {self.start_ite} | best_neg_loss: {self.best_neg_loss:.3f}')
        else:
            logging.info('--> training from scratch')

    def train(self):
        # start training
        for _, v in self.registed_models.items():
            v.train()
        for self.ite in range(self.start_ite, self.cfg.TRAIN.TTL_ITE):
            # test
            if self.ite % self.cfg.TRAIN.TEST_FREQ == self.cfg.TRAIN.TEST_FREQ - 1 and self.ite != self.start_ite:
                self.base_net.eval()
                self.test()
                self.base_net.train()

            self.current_lr = self.lr_scheduler(
                self.optimizer,
                ite_rate=self.ite / self.cfg.TRAIN.TTL_ITE * self.cfg.METHOD.HDA.LR_MULT,
                lr=self.cfg.TRAIN.LR,
            )

            # dataloader
            if self.ite % self.len_src == 0 or self.ite == self.start_ite:
                iter_src = iter(self.dataset_loaders['source_train'])
            if self.ite % self.len_tar == 0 or self.ite == self.start_ite:
                iter_tar = iter(self.dataset_loaders['target_train'])

            # forward one iteration
            data_src = iter_src.__next__()
            data_tar = iter_tar.__next__()
            self.one_step(data_src, data_tar)
            # if self.ite % self.cfg.TRAIN.SAVE_FREQ == 0 and self.ite != 0:
            #     self.save_model(is_best=False, snap=True)

    def one_step(self, data_src, data_tar):
        num_inputs_src, cate_inputs_src, labels_src = data_src['num_input'].cuda(), data_src['cate_input'].cuda(), data_src['output'].cuda()

        outputs_all_src = self.base_net(num_inputs_src, cate_inputs_src)  # [f, y]

        loss_cls_src = F.cross_entropy(outputs_all_src[1], labels_src)

        loss_ttl = loss_cls_src

        # update
        self.step(loss_ttl)

        # display
        if self.ite % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'best_neg_loss: {self.best_neg_loss:.3f}',
            ])

    def display(self, data: list):
        log_str = f'I:  {self.ite}/{self.cfg.TRAIN.TTL_ITE} | lr: {self.current_lr:.5f} '
        # update
        for _str in data:
            log_str += '| {} '.format(_str)
        logging.info(log_str)

    def step(self, loss_ttl):
        self.optimizer.zero_grad()
        loss_ttl.backward()
        self.optimizer.step()

    def test(self):
        logging.info('--> testing on source_test')
        src_loss = self.test_func(self.dataset_loaders['source_test'], self.base_net)
        # logging.info('--> testing on target_test')
        # tar_acc = self.test_func(self.dataset_loaders['target_test'], self.base_net)
        is_best = False
        if src_loss > self.best_neg_loss:
            self.best_neg_loss = src_loss
            is_best = True

        # display
        log_str = f'I:  {self.ite}/{self.cfg.TRAIN.TTL_ITE} | src_acc: {src_loss:.3f} |' \
                  f'best_neg_loss: {self.best_neg_loss:.3f}'
        logging.info(log_str)

        self.save_model(is_best=is_best)

    def test_func(self, loader, model):
        with torch.no_grad():
            iter_test = iter(loader)
            print_freq = max(len(loader) // 5, self.cfg.TRAIN.PRINT_FREQ)

            cum_loss = 0
            for i in range(len(loader)):
                data = iter_test.__next__()
                num_inputs, cate_inputs, labels = data['num_input'].cuda(), data['cate_input'].cuda(), data['output'].cuda()
                outputs_all = model(num_inputs, cate_inputs)  # [f, y, ...]

                loss = float(F.cross_entropy(outputs_all[1], labels).detach().cpu())
                cum_loss += loss

        return - cum_loss / (i+1)


    def predict(self, target_test:np.ndarray):
        loader = DataLoader(
            TabularDataset(target_test, self.cfg.TRAIN.CATE_INDEX, 1),
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )

        with torch.no_grad():
            iter_test = iter(loader)

            res = []
            for i in range(len(loader)):
                data = iter_test.__next__()
                num_inputs, cate_inputs = data['num_input'].cuda(), data['cate_input'].cuda()
                outputs_all = self.base_net(num_inputs, cate_inputs)  # [f, y, ...]

                res.append(F.softmax(outputs_all[1]).detach())

        return torch.cat(res, dim=0).cpu().numpy()[:,1]

    def save_model(self, is_best=False, snap=False):
        data_dict = {
            'optimizer': self.optimizer.state_dict(),
            'ite': self.ite,
            'best_neg_loss': self.best_neg_loss
        }
        for k, v in self.registed_models.items():
            data_dict.update({k: v.state_dict()})
        save_model(self.cfg.TRAIN.OUTPUT_CKPT, data_dict=data_dict, ite=self.ite, is_best=is_best, snap=snap)
