from typing import Any, Type, Union, List, Optional

import torch
import torch.nn as nn

import numpy as np

from models.base_model import BaseModel

class NumNN(BaseModel):
    def __init__(self,
                num_dim,
                num_classes: int = 2,
                **kwargs
                ):
        super().__init__(num_classes=num_classes, **kwargs)

        self.input_layer = nn.Sequential(
            nn.Linear(num_dim, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.1),
        )

        self.feature_layers = [self.input_layer]

        self._fdim = 32

        # head
        self.build_head()

    def get_backbone_parameters(self):
        feature_layers_params = []
        for m in self.feature_layers:
            feature_layers_params += list(m.parameters())
        parameter_list = [{'params': feature_layers_params, 'lr_mult': 1}]

        return parameter_list

    def forward_backbone(self, num_inputs, cate_inputs=None):
        return self.input_layer(num_inputs)