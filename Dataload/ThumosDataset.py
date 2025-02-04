import os.path as osp
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class THUMOSDataSet(Dataset):
    def __init__(self, args, flag='train'):
        assert flag in ['train', 'test', 'valid']
        self.pickle_root = args.data_root
        self.sessions = getattr(args, flag + '_session_set')
        self.enc_steps = args.enc_layers
        self.training = flag == 'train'
        self.feature_pretrain = args.feature_type
        self.inputs = []#存储test数据集的真实分布（train的时候就存储val数据集的）

        self.subnet = 'val' if self.training else 'test'

        target_all = pickle.load(
            open(osp.join(self.pickle_root, 'thumos_' + self.subnet + '_anno.pickle'), 'rb'))

    

        for session in self.sessions:
            # print("session的值: "+session)
            target = target_all[session]['anno']
            # print("target的值: ",target.shape[0],"  ",target.shape[1])
            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(
                    range(seed, target.shape[0], 1),
                    range(seed + self.enc_steps, target.shape[0], 1)):  # target.shape[0]
                enc_target = target[start:end]
                class_h_target = enc_target[self.enc_steps - 1]
                if class_h_target.argmax() != 21:
                    #这一步将test数据集的真实分布（每帧22个类的0/1分布）加载
                    #加载的格式['video_test_0000292', 1, 65, array([[1., 0., 0., ..., 0., 0., 0.],。。。。
                    self.inputs.append([session, start, end, enc_target])


        # print("inputs的值")
        # print(self.inputs[0])
        # print(self.inputs[1])
        #这一步加载事先用图像特征和光流法特征
        # 特征和帧的总数是1:6的关系，图像特征为2048维度，光流特征为2048维度，加在一起变成4096维度
        self.feature_All = pickle.load(open(osp.join(
            self.pickle_root, 'thumos_all_feature_{}_Kinetics.pickle'.format(self.subnet)), 'rb'))
        print(osp.join(self.pickle_root, 'thumos_all_feature_{}_Kinetics.pickle'.format(self.subnet)))


    def __getitem__(self, index):
        session, start, end, enc_target = self.inputs[index]
        camera_inputs = self.feature_All[session]['rgb'][start:end]
        camera_inputs = torch.tensor(camera_inputs)

        motion_inputs = self.feature_All[session]['flow'][start:end]
        motion_inputs = torch.tensor(motion_inputs)

        camera_inputs = torch.cat((camera_inputs, motion_inputs), dim=-1)
        enc_target = torch.tensor(enc_target)

        return camera_inputs, enc_target

    def __len__(self):
        return len(self.inputs)


if __name__ == '__main__':
    target_all = pickle.load(
            open('../../../../../data/ssd1/zhanghaoran/zhr/thumos14/thumos_test_anno.pickle', 'rb'))
    target = target_all['video_test_0001163']['anno']
    
    print(target[90:110])
    # from misc.config import parse_args
    # import json

    # args = parse_args()
    # with open(args.dataset_file, 'r') as f:
    #     data_info = json.load(f)['THUMOS']

    # args.train_session_set = data_info['train_session_set']
    # args.test_session_set = data_info['test_session_set']
    # dataset = THUMOSDataSet(args)

    # sampler_val = torch.utils.data.SequentialSampler(dataset)
    # data_loader_val = DataLoader(dataset, args.batch, sampler=sampler_val,
    #                              drop_last=False, pin_memory=True, num_workers=8)
    # for feature, target in data_loader_val:
    #     print(feature.shape, target.shape)
