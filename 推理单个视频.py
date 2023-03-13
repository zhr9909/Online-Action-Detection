import json
from loss.evaluate import Colar_evaluate

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from Dataload.ThumosDataset import THUMOSDataSet
from model.ColarModel import Colar_dynamic, Colar_static
from misc import init

import pickle
import os.path as osp


def evaluate(args):
    print(type(args.cuda_id))
    device = torch.device(2)
    model_static = Colar_static(args.input_size, args.numclass, device, args.kmean)
    model_dynamic = Colar_dynamic(args.input_size, args.numclass)
    model_dict = torch.load(args.checkpoint, map_location=torch.device(2))
    # print(model_dict)
                    # model_static.load_state_dict(model_dict['model_static'])
                    # print('aaaa3')
                    # model_dynamic.load_state_dict(model_dict['model_dynamic'])
    kkk = torch.load('./checkpoint/zhr1.pth', map_location=torch.device(2))
    model_static.load_state_dict(kkk['model_static'])
    model_dynamic.load_state_dict(kkk['model_dynamic'])

    model_dynamic.to(2)
    model_static.to(2)

    #将thumos数据加载，然后对里面顺序加载
    dataset_val = THUMOSDataSet(flag='test', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    #将数据分成一个个大小为batch_size的tensor
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, pin_memory=True, num_workers=args.num_workers)

    #将模型设置成评估模式
    model_dynamic.eval()
    model_static.eval()

    score_val_x = []
    target_val_x = []
    i = 0

    feature_All = pickle.load(open(osp.join(
            '../../../../data/ssd1/zhanghaoran/zhr/thumos14', 'thumos_all_feature_test_Kinetics.pickle'), 'rb'))
    print("session_26:")
    print(len(feature_All['video_test_0000026']['rgb'][0]))
    print(len(feature_All['video_test_0000026']['flow'][0]))
    camera_inputs = feature_All['video_test_0000026']['rgb']
    camera_inputs = torch.tensor(camera_inputs)

    motion_inputs = feature_All['video_test_0000026']['flow']
    motion_inputs = torch.tensor(motion_inputs)

    #[1039, 4096]
    from ipdb import set_trace
    # set_trace()
    #print((camera_inputs[0] * 32).shape)

    camera_inputs = torch.cat((camera_inputs, motion_inputs), dim=-1)
    camera_inputs = torch.cat((camera_inputs[0].repeat(32,1), camera_inputs), dim=0)
    camera_inputs = torch.cat((camera_inputs, camera_inputs[-1].repeat(32,1)), dim=0)


    for i in range(32, camera_inputs.shape[0] - 32):
        start = i - 32
        end = i + 32
        print(camera_inputs[start:end].unsqueeze(0).shape)
        # 这一段把动态分支判断出的最大类别提取出来了
        dynamic_sorce = model_dynamic(camera_inputs[start:end].unsqueeze(0).to(device))
        dynamic_sorce = dynamic_sorce.permute(0, 2, 1)
        dynamic_sorce = dynamic_sorce[:, -1:, :21]
        dynamic_sorce = F.softmax(dynamic_sorce, dim=-1)
        dynamic_sorce_class = torch.max(dynamic_sorce,-1).indices[0].item()
        # print('图像最大:',dynamic_sorce)

        # 这一段把静态分支判断出的最大类别提取出来了
        static_sorce = model_static(camera_inputs[start:end].unsqueeze(0).to(device)[:, -1:, :],device)
        static_sorce = static_sorce.permute(0, 2, 1)
        static_sorce = F.softmax(static_sorce, dim=-1)
        static_sorce_class = torch.max(static_sorce,-1).indices[0].item()
        # print('静态最大:',static_sorce)

        final_score = dynamic_sorce*0.7+static_sorce*0.3
        final_score = torch.max(final_score,-1).indices[0].item()
        print('最终类别:',final_score)


    #camera_inputs是rgb特征和光流特征，enc_target是test或者val数据集的真实类别分布标签
    for camera_inputs, enc_target in data_loader_val:
        inputs = camera_inputs.to(device)
        target_val = enc_target[:, -1:, :21].to(device)

        with torch.no_grad():
            # dynamic的输入:torch.Size([128, 64, 4096]),128代表batch的大小，64指的是64个特征段取得的特征向量，每个4096维度
            # static的输入: torch.Size([128, 1, 4096])
            # 经过底下模型之后，dynamic输出为torch.Size([128, 22, 64])，static的输出为torch.Size([128, 21, 1])
            enc_score_dynamic = model_dynamic(inputs)
            enc_score_static = model_static(inputs[:, -1:, :], device)

        #计算出静态分支得到的类别数值分布
        enc_score_static = enc_score_static.permute(0, 2, 1)
        enc_score_static = enc_score_static[:, :, :21]

        #计算出动态分支得到的类别数值分布
        enc_score_dynamic = enc_score_dynamic.permute(0, 2, 1)
        enc_score_dynamic = enc_score_dynamic[:, -1:, :21]

        # ======修改过后=====
        # enc_score_dynamic  torch.Size([128, 1, 21])
        # enc_score_static  torch.Size([128, 1, 21])

        #将动态和静态分支的分支用个比例加起来
        score_val = enc_score_static * 0.3 + enc_score_dynamic * 0.7
        score_val = F.softmax(score_val, dim=-1)

        # 这一步之后变成(128, 21)的shape，注意这已经不是张量了,而是'numpy.ndarray'
        # 注意这里128，就是指代128帧，每帧的21个类别分数，模型是每次input 128个帧进去，
        # 然后静态分支对每帧输出21类别，动态分支对每帧的前64个帧输出类别然后取当前这帧
        score_val = score_val.contiguous().view(-1, 21).cpu().numpy()
        target_val = target_val.contiguous().view(-1, 21).cpu().numpy()
        #将本次的模型计算的值和 本次真实值 都存到总列表里
        score_val_x += list(score_val)
        target_val_x += list(target_val)
        print('\r train-------------------{:.4f}%'.format((i / 1600) * 100), end='')
        i += 1

    score_val_x = np.asarray(score_val_x)
    target_val_x = np.asarray(target_val_x)
    print(score_val_x.shape)
    print(type(score_val_x))
    all_probs = np.asarray(score_val_x).T
    all_classes = np.asarray(target_val_x).T
    print(all_probs.shape, all_classes.shape)
    results = {'probs': all_probs, 'labels': all_classes}
    Colar_evaluate(results, 1, 'test', '')


if '__main__' == __name__:
    args = init.parse_args()
    print(type(args.cuda_id))
    with open(args.dataset_file, 'r') as f:
        data_info = json.load(f)['THUMOS']

    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']

    evaluate(args)
