# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
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

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from model import MInterface
from data import DInterface

from utilsMain import load_model_path_by_args


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_acc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)

    load_path = load_model_path_by_args(args)

    data_module = DInterface(**vars(args))

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.ckpt_path = load_path

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    # args.callbacks = load_callbacks()
    # args.logger = logger

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=30, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    #加载数据集
    parser.add_argument('--dataset', default='standard_data', type=str)
    parser.add_argument('--data_dir', default='data/MoleculeNet', type=str)
    parser.add_argument('--model_name', default='network', type=str)
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--weight_decay', default=1e-7, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    parser.add_argument('--task_num', type=int, default=1,
                   help='The number of task in multi-task training.')
    parser.add_argument('--dataset_type', type=str, choices=['classification', 'regression'], default='classification',
                   help='The type of dataset.')
    parser.add_argument('--split_type', type=str, choices=['random', 'scaffold'], default='random',
                   help='The type of data splitting.')
    parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   help='The ratio of data splitting.[train,valid,test]')

    # Model Hyperparameters
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--layer_num', default=5, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=200)

    args = parser.parse_args()

    main(args)