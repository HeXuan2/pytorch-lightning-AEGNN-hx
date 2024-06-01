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

import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data

from torchvision import transforms
from sklearn.model_selection import train_test_split

from classification.data.MoleData import MoleDataSet
from classification.utils.tool import split_data, load_data, process_data
from rdkit import Chem,RDLogger
import random
from torch.utils.data.dataset import Dataset

RDLogger.DisableLog('rdApp.*')

class StandardData(data.Dataset):
    def __init__(self, data_dir=None,
                 split_type=None,
                 split_ratio=None,
                 seed=None,
                 dataset_type=None,
                 batch_size=None,
                 dataType=None,
            ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value.

        data_path = op.join(self.data_dir, 'bace.csv')
        dataAll = load_data(data_path)
        train_data, val_data, test_data = split_data(dataAll, self.split_type, self.split_ratio, self.seed, None)  # scaffold or random


        if(self.dataType=="train"):
            self.data=train_data

        elif(self.dataType=="test"):
            self.data =test_data
        elif(self.dataType=="val"):
            self.data =val_data

        print("self.dataType=",self.dataType)

        self.data_dict = process_data(self.data, self.batch_size)  # 传入的参数data是经过有效性检验的MoleData对象，batch_size=50

        self.automs_index_all = self.data_dict.get('automs_index_all')
        self.feats_batch_all = self.data_dict.get('feats_batch_all')
        self.edges_batch_all = self.data_dict.get('edges_batch_all')
        self.coors_batch_all = self.data_dict.get('coors_batch_all')
        self.adj_batch_all = self.data_dict.get('adj_batch_all')
        self.mask_batch_all = self.data_dict.get('mask_batch_all')

        self.lenBatch=len(self.automs_index_all)


    def __len__(self):
        return self.lenBatch

    def __getitem__(self, count):

        # 100
        # 80 10 10
        # 50
        # 30

        batch = len(self.automs_index_all[count])
        beginData=count*self.batch_size
        label=MoleDataSet(self.data[beginData:beginData+batch]).label()

        mask = torch.Tensor([[x is not None for x in tb] for tb in label])
        target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])
        mask, target = mask.cuda(), target.cuda()
        weight = torch.ones(target.shape).cuda()

        return self.automs_index_all[count], self.feats_batch_all[count], self.edges_batch_all[count],self.coors_batch_all[count],self.adj_batch_all[count], self.mask_batch_all[count], batch, self.dataset_type,mask,target,weight


