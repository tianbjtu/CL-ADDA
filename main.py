import argparse
import sys
import os

import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import scipy.io as scio

from dataSet import *
from myModel import *
from testModel import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
cuda = torch.cuda.is_available()
if cuda:
    sys.stdout.write("Let's use %d GPUS\n" % (torch.cuda.device_count()))

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

if not os.path.exists("./image"):
    os.makedirs("./image")

if not os.path.exists("./result"):
    os.makedirs("./result")

if not os.path.exists("./data"):
    os.makedirs("./data")

if not os.path.exists("./checkpoint"):
    os.makedirs("./checkpoint")

if not os.path.exists("./logs"):
    os.makedirs("./logs")

if not os.path.exists("./saved_model/"):
    os.makedirs("./saved_model/")

# Age 最佳参数，学习率0.01  学习率不变 对比比例1  恢复比例1 预测比例0.1  衰减0.0001 Adam 预测损失函数MSE  800
# IQ 最佳参数，学习率0.01  学习率不变 对比比例1  恢复比例10 预测比例1  衰减0.0001 Adam 预测损失函数MAE

parser = argparse.ArgumentParser()
parser.add_argument('--n_roi', type=int, default=200, help='number of ROIs')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of CLM encoder training')
parser.add_argument('--n_epochs_dsm', type=int, default=100, help='number of epochs of DownStreamModel training')
parser.add_argument('--batchSize', type=int, default=128, help='size of the batches')  #32


parser.add_argument('--lr', type=float, default=0.001, help='learning rate for CLM encoder')
parser.add_argument('--lr_dsm', type=float, default=0.01, help='learning rate for DownStreamModel')
parser.add_argument('--cl_rate', type=float, default=1)
parser.add_argument('--pred_rate',type=float,default=1)
parser.add_argument('--pred_type', type=str, default='Age', help='prediction type for Model')
parser.add_argument('--data_type',type=str,default='CamCan')
parser.add_argument('--load_encoder', type=bool, default=False)
parser.add_argument('--load_dsm', type=bool, default=False)

parser.add_argument('--weightdecay', type=float, default=0.0001, help='regularization')
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./saved_model/', help='path to save model')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=str, help='path to checkpoint directory')
parser.add_argument('--log_path', type=str, default='./logs/', help='path to record log')
# 打印的频率，默认为5
parser.add_argument('-p', '--save-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
opt = parser.parse_args()


#################### Parameter Initialization #######################

name = str(opt.pred_type)
opt.save_path = os.path.join(opt.save_path, name)
opt.checkpoint_dir = os.path.join(opt.checkpoint_dir, name)
opt.log_path = os.path.join(opt.log_path, name)

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)
if not os.path.exists(opt.checkpoint_dir):
    os.makedirs(opt.checkpoint_dir)
if not os.path.exists(opt.log_path):
    os.makedirs(opt.log_path)
else:
    deldir(opt.log_path)
    os.makedirs(opt.log_path)


if opt.pred_type=='IQ':
    data_filepath="****.npy"
    data_info="****.npy"
elif opt.pred_type=='Age':
    data_filepath = "****.npy"
    data_info = "****.npy"

num_fold=10

data=np.load(data_filepath)
SubjInfo = np.load(data_info)
data_FC_train=get_generate_fc(data,"train")
data_FC_test=get_generate_fc(data,"test")

train_index_list,test_index_list,_=generate_order(SubjInfo,num_fold)

if __name__ == '__main__':


    true_label=SubjInfo
    true_label = true_label[:, np.newaxis]
    pd_label=np.zeros(true_label.shape)

    for foldNo in range(num_fold):

        print("*===========fold%d============**"%(foldNo))
        foldNoName=str(foldNo)
        foldlagpath = os.path.join(opt.log_path, foldNoName)
        if not os.path.exists(foldlagpath):
            os.makedirs(foldlagpath)

        train_id=train_index_list[foldNo]
        valid_id = train_id[int(len(train_id) * 0.9):]  #
        train_id = train_id[0:int(len(train_id) * 0.9)]  #
        test_id=test_index_list[foldNo]


        train_fc_list = data_FC_train[train_id, :]
        valid_fc_list = data_FC_test[valid_id, :]  #
        test_fc_list = data_FC_test[test_id, :]


        train_SubjInfo = SubjInfo[train_id]
        valid_SubjInfo = SubjInfo[valid_id]  #
        test_SubjInfo = SubjInfo[test_id]

        dataset_train = MyDataset(train_fc_list, train_SubjInfo)
        dataset_valid = MyDataset(valid_fc_list, valid_SubjInfo)  #
        dataset_test = MyDataset(test_fc_list, test_SubjInfo)

        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=opt.batchSize, num_workers=0, shuffle=True)

        valid_loader = torch.utils.data.DataLoader(
            dataset_valid, batch_size=opt.batchSize, num_workers=0, shuffle=True)  #

        test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=opt.batchSize, num_workers=0, shuffle=True)

        model = Model(opt, cuda,foldNo,foldlagpath)

        best_model_path= model.train_cl_encoder(train_loader=train_loader, test_loader=valid_loader, max_epoch=opt.n_epochs,
                                            save_model=opt.save_model, load_model=opt.load_encoder)

        best_pred_result=testModel(best_model_path,test_fc_list,opt.n_roi,cuda)
        pd_label[test_id,:]=best_pred_result

    result_acc, result_MAE = get_Acc(true_label, pd_label)
    print('finally_acc: {:.7f} finally_MAE: {:.7f}\n'.format(result_acc, result_MAE))

    scio.savemat("result/result%s.mat"%(opt.pred_type), {'true_label': true_label, 'pd_label': pd_label,"acc": result_acc,"MAE": result_MAE})

