import os
import csv
import logging
import math
import numpy as np
import torch
import torch.nn as nn

from .creat_data_DC import smile_to_graph
from .scaffold import scaffold_split
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve, roc_auc_score

from ..data.MoleData import MoleData, MoleDataSet


def compute_mae_mse_rmse(target, prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))
    mae = sum(absError) / len(absError)
    mse = sum(squaredError) / len(squaredError)
    RMSE = mse ** 0.5
    return mae, mse, RMSE


def compute_rsquared(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    r2 = round((SSR / SST) ** 2, 3)
    return r2

def process_data(data, iter_step):
    # 传入的参数data是经过有效性检验的MoleData对象，iter_step=batch_size=50
    smile_used = 0
    automs_index_all = []
    feats_batch_all = []
    edges_batch_all = []
    coors_batch_all = []
    adj_batch_all = []
    mask_batch_all = []

    for i in range(0, len(data), iter_step):

        if smile_used + iter_step > len(data): #第一次0+50 ，第二次 50+50=100
            data_now = MoleDataSet(data[i:len(data)])
        else:
            data_now = MoleDataSet(data[i:i + iter_step]) #第一次拿0-49的data，第二次拿1-50的data

        #data_now就是每次处理50个，最后再处理不足50个的。
        smile = data_now.smile()

        automs_index, feats_batch, edges_batch, coors_batch, adj_batch, mask_batch = smile_to_graph(smile)

        automs_index_all.append(automs_index)
        feats_batch_all.append(feats_batch)
        edges_batch_all.append(edges_batch)
        coors_batch_all.append(coors_batch)
        adj_batch_all.append(adj_batch)
        mask_batch_all.append(mask_batch)

        smile_used += iter_step #第一次为50，第二次100；

    return {'automs_index_all': automs_index_all, 'feats_batch_all': feats_batch_all,
            'edges_batch_all': edges_batch_all,
            'coors_batch_all': coors_batch_all, 'adj_batch_all': adj_batch_all, 'mask_batch_all': mask_batch_all}


def mkdir(path,isdir = True):
    if isdir == False:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok = True)


def set_log(name,save_path):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    #处理器
    log_stream = logging.StreamHandler()
    log_stream.setLevel(logging.DEBUG)
    log.addHandler(log_stream)  #记录器
    
    mkdir(save_path)
    #处理器
    log_file_d = logging.FileHandler(os.path.join(save_path, 'debug.log'))
    log_file_d.setLevel(logging.DEBUG)
    log.addHandler(log_file_d)
    
    return log


def get_header(path):
    with open(path) as file:
        header = next(csv.reader(file))
    
    return header


def get_task_name(path):
    task_name = get_header(path)[1:]
    
    return task_name



def load_data(path):
    with open(path) as file:
        reader = csv.reader(file)
        next(reader)
        lines = []


        for line in reader:
            lines.append(line)
        data = []
        #lines=[['[Cl].CC(C)NCC(O)COc1cccc2ccccc12', '1'], ['C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CCCl', '1']]

        for line in lines:
            one = MoleData(line)
            data.append(one)

        # data里面存了很多moleData对象
        data = MoleDataSet(data)
        fir_data_len = len(data)

        data_val = []
        smi_exist = []

        for i in range(fir_data_len):
            if data[i].mol is not None:
                smi_exist.append(i)

        data_val = MoleDataSet([data[i] for i in smi_exist])

        now_data_len = len(data_val)

        print('There are ',now_data_len,' smiles in total.')
        if fir_data_len - now_data_len > 0:
            print('There are ',fir_data_len , ' smiles first, but ',fir_data_len - now_data_len, ' smiles is invalid.  ')
    #返回MoleDataSet对象
    return data_val



def split_data(data,type,size,seed,log):
    # 切分数据集，传入的参数data是经过有效性检验的MoleData对象，type=split_type='random'，size=split_ratio=[0.8, 0.1, 0.1]，seed=30

    assert len(size) == 3 #因为是切分三个数据集
    assert sum(size) == 1 #判断和是否为1

    if type == 'random':
        #调用moleData对象中的random_data，来打乱data中列表的顺序
        data.random_data(seed)
        #计算，训练集有多少个smile序列
        train_size = int(size[0] * len(data))
        # 计算，验证集有多少个smile序列
        val_size = int(size[1] * len(data))

        train_val_size = train_size + val_size
        #取0-train_size-1个data中的simle
        train_data = data[:train_size]
        #取train_size-train_val_size-1
        val_data = data[train_size:train_val_size]
        #剩下的就是测试集
        test_data = data[train_val_size:]
        #返回训练集、测试集、验证集的每一个MoleDataSet对象
        return MoleDataSet(train_data),MoleDataSet(val_data),MoleDataSet(test_data)

    elif type == 'scaffold':
        return scaffold_split(data, size, seed, log)
    else:
        return None
        #raise ValueError('Split_type is Error.')

    '''
    elif type == 'scaffold':  #根据这个scaffold进行划分，可以看看这个scaffod
        return scaffold_split(data,size,seed,log)
    '''


def get_label_scaler(data):
    smile = data.smile()
    label = data.label()
    
    label = np.array(label).astype(float)
    ave = np.nanmean(label,axis=0)
    ave = np.where(np.isnan(ave),np.zeros(ave.shape),ave)
    std = np.nanstd(label,axis=0)
    std = np.where(np.isnan(std),np.ones(std.shape),std)
    std = np.where(std==0,np.ones(std.shape),std)
    
    change_1 = (label-ave)/std
    label_changed = np.where(np.isnan(change_1),None,change_1)
    label_changed.tolist()
    data.change_label(label_changed)
    
    return [ave,std]

def get_loss(type):
    if type == 'classification':
        return nn.BCELoss()
    elif type == 'regression':
        return nn.MSELoss()
    else:
        raise ValueError('data type Error.')

def prc_auc(label,pred):
    prec, recall, _ = precision_recall_curve(label,pred)
    result = auc(recall,prec)
    return result

def rmse(label,pred):
    result = mean_squared_error(label,pred)
    return math.sqrt(result)

def get_metric(metric):
    if metric == 'auc':
        return roc_auc_score
    elif metric == 'prc-auc':
        return prc_auc
    elif metric == 'rmse':
        return rmse
    else:
        raise ValueError('Metric Error.')


def get_scaler(path):
    state = torch.load(path, map_location=lambda storage, loc: storage)
    if state['data_scaler'] is not None:
        ave = state['data_scaler']['means']
        std = state['data_scaler']['stds']
        return [ave,std]
    else:
        return None



def rmse(label,pred):
    result = mean_squared_error(label,pred)
    result = math.sqrt(result)
    return result


"""

Noam learning rate scheduler with piecewise linear increase and exponential decay.

The learning rate increases linearly from init_lr to max_lr over the course of
the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
Then the learning rate decreases exponentially from max_lr to final_lr over the
course of the remaining total_steps - warmup_steps (where total_steps =
total_epochs * steps_per_epoch). This is roughly based on the learning rate
schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).

"""

class NoamLR(_LRScheduler):
    def __init__(self,optimizer,warmup_epochs,total_epochs,steps_per_epoch,
                 init_lr,max_lr,final_lr):
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        return list(self.lr)

    def step(self,current_step=None):
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]
