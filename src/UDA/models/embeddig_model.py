from typing import Any, Type, Union, List, Optional

import torch
import torch.nn as nn

import numpy as np

from models.base_model import BaseModel

class EmbedNN(BaseModel):
    def __init__(self,
                embedding_input, embedding_dim, num_dim,
                num_classes: int = 2,
                **kwargs
                ):
        super().__init__(num_classes=num_classes, **kwargs)

        self.embed = nn.ModuleList()
        for i in range(len(embedding_input)):
            self.embed.append(nn.Embedding(embedding_input[i], embedding_dim[i]))

        input_dim = np.sum(embedding_dim) + num_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.feature_layers = [self.embed, self.input_layer]

        self._fdim = 32

        # head
        self.build_head()

    def get_backbone_parameters(self):
        feature_layers_params = []
        for m in self.feature_layers:
            feature_layers_params += list(m.parameters())
        parameter_list = [{'params': feature_layers_params, 'lr_mult': 1}]

        return parameter_list

    def forward_backbone(self, num_inputs, cate_inputs):
        embeddings = []
        for i in range(len(self.embed)):
            embeddings.append(self.embed[i](cate_inputs[:, i]))
        embedding = torch.cat(embeddings, 1)
        inputs = torch.cat((embedding, num_inputs), 1)
        return self.input_layer(inputs)