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
import torch
import importlib
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from classification.utils.tool import compute_mae_mse_rmse, compute_rsquared
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

# 创建 TensorBoardLogger 实例，并在训练器中使用
logger = TensorBoardLogger('tb_logs', name='my_experiment')

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()


    def forward(self, automs_index, feats_batch, edges_batch, coors_batch, adj_batch, mask_batch, batch, dataset_type):
        return self.model(automs_index, feats_batch, edges_batch, coors_batch, adj_batch, mask_batch, batch, dataset_type)

    def training_step(self, batch, batch_idx):

        #由于这个框架会自动增加维度，比如在standard_data.py里面__getitem__返回的是维度大小为（32，2）的张量，
        # 到这里batch会自动给它增加一个维度，因此输入模型的时候我们再给它降维就好了

        automs_index, feats_batch, edges_batch, coors_batch, adj_batch, mask_batch, batch_size, dataset_type, mask, target, weight =batch



        automs_index = automs_index[batch_idx]
        feats_batch = feats_batch[batch_idx]
        edges_batch = edges_batch[batch_idx]
        coors_batch = coors_batch[batch_idx]
        adj_batch = adj_batch[batch_idx]
        mask_batch = mask_batch[batch_idx]
        batch_size = batch_size[batch_idx]
        dataset_type = dataset_type[batch_idx]
        mask = mask[batch_idx]
        target = target[batch_idx]
        weight = weight[batch_idx]

        print("----training_step----batch_idx", batch_idx,"len(automs_index)=",len(automs_index))

        pred = self(automs_index, feats_batch, edges_batch, coors_batch, adj_batch, mask_batch, batch_size, dataset_type)
        loss = self.loss_function(pred, target) * weight * mask
        loss = loss.sum() / mask.sum()

        print("loss=", loss)
        # 使用TensorBoard记录损失值
        logger.experiment.add_scalar("Loss/train", loss, global_step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        # batch_idx 表示第几轮epoch
        print("----validation_step----batch_idx", batch_idx)
        automs_index, feats_batch, edges_batch, coors_batch, adj_batch, mask_batch, batch_size, dataset_type, mask, target, weight = batch

        automs_index=automs_index[batch_idx]
        feats_batch=feats_batch[batch_idx]
        edges_batch=edges_batch[batch_idx]
        coors_batch=coors_batch[batch_idx]
        adj_batch=adj_batch[batch_idx]
        mask_batch=mask_batch[batch_idx]
        batch_size=batch_size[batch_idx]
        dataset_type=dataset_type[batch_idx]

        mask=mask[batch_idx]
        target=target[batch_idx]
        weight=weight[batch_idx]

        pred = self(automs_index, feats_batch, edges_batch, coors_batch, adj_batch, mask_batch, batch_size, dataset_type)

        target = target.squeeze(0)
        pred = pred.data.cpu().numpy()
        S = np.array(pred).flatten()

        T = np.array(target).flatten()

        mae, mse, rmse = compute_mae_mse_rmse(T, S)
        print(mae, mse, rmse)
        r2 = compute_rsquared(S, T)

        R2 = r2_score(T, S)
        MSE = mean_squared_error(T, S)

        if dataset_type == 'classification':
            auc = roc_auc_score(T, S)
        else:
            auc = -1 * MSE

        AUCs = [abs(auc), R2, mse, mae, rmse, r2]
        print('AUC: ', AUCs)

        # 使用 logger 记录指标
        logger.experiment.add_scalar("Validation/AUC", auc, self.global_step)
        logger.experiment.add_scalar("Validation/R2", R2, self.global_step)
        logger.experiment.add_scalar("Validation/MSE", MSE, self.global_step)
        logger.experiment.add_scalar("Validation/MAE", mae, self.global_step)
        logger.experiment.add_scalar("Validation/RMSE", rmse, self.global_step)
        logger.experiment.add_scalar("Validation/RSquared", r2, self.global_step)


        return [batch_idx, abs(auc), R2, mse, mae, rmse, r2]

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            # self.loss_function = F.mse_loss
            self.loss_function = nn.MSELoss()
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            # self.loss_function = F.binary_cross_entropy
            self.loss_function = nn.BCELoss()
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]

        inkeys = self.hparams.keys()

        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
