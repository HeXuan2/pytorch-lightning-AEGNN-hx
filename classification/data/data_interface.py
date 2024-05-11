# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch
class DInterface(pl.LightningDataModule):

    def __init__(self, num_workers=0,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(dataType="train")
            self.valset = self.instancialize(dataType="val")

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(dataType="test")

        # # If you need to balance your data using Pytorch Sampler,
        # # please uncomment the following lines.
    
        # with open('./data/ref/samples_weight.pkl', 'rb') as f:
        #     self.sample_weight = pkl.load(f)

    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
    #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

    def my_collate_fn(self, batch):
        # Unzip the batch data
        automs_index, feats_batch, edges_batch, coors_batch, adj_batch, mask_batch, batch_size, dataset_type, mask, target, weight = zip(*batch)
        #这里会返回所有批次的张量,用元组存储下来
        return automs_index, feats_batch, edges_batch, coors_batch, adj_batch, mask_batch, batch_size, dataset_type, mask, target, weight

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.my_collate_fn)
        # return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.my_collate_fn)
        # return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.my_collate_fn)
        # return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        #class_args =['data_dir', 'class_num', 'train', 'no_augment', 'aug_prob', 'img_mean', 'img_std']
        #存的是standard_data.py中__init__里面的参数
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]

        #main.py里面所有的参数
        inkeys = self.kwargs.keys()

        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        #arg1存储了从main.py拿到的参数，并传递给standard_data.py
        args1.update(other_args)

        return self.data_module(**args1)
