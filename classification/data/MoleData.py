from rdkit import Chem,RDLogger
import random
from torch.utils.data.dataset import Dataset

RDLogger.DisableLog('rdApp.*')

class MoleData:
    def __init__(self, line):
        self.smile = line[0]
        self.mol = Chem.MolFromSmiles(self.smile)
        # 返回label标签的数组形如[1.0,0,1.0]
        self.label = [float(x) if x != '' else None for x in line[1:]]

    def task_num(self):
        # 几变量预测
        return len(self.label)

    def change_label(self, label):
        # 修改label
        self.label = label


class MoleDataSet(Dataset):
    def __init__(self, data):
        # 很多个moleData对象
        self.data = data
        self.scaler = None

    def smile(self):
        smile_list = []
        for one in self.data:
            smile_list.append(one.smile)
        return smile_list

    def mol(self):
        mol_list = []
        for one in self.data:
            mol_list.append(one.mol)
        return mol_list

    def label(self):
        label_list = []
        for one in self.data:
            label_list.append(one.label)
        return label_list

    def task_num(self):
        if len(self.data) > 0:
            return self.data[0].task_num()
        else:
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def random_data(self, seed):
        random.seed(seed)
        random.shuffle(self.data)

    def change_label(self, label):
        assert len(self.data) == len(label)
        for i in range(len(label)):
            self.data[i].change_label(label[i])

