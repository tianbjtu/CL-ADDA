import time
import copy

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from losses import *
import os
import scipy.stats as stats
import math


def adjust_learning_rate(optimizer, init_lr, epoch, n_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / n_epochs))
    for param_group in optimizer.param_groups:
        # if 'fix_lr' in param_group and param_group['fix_lr']:
        #     param_group['lr'] = init_lr
        # else:
        #     param_group['lr'] = cur_lr
        param_group['lr'] = cur_lr


def get_Acc(predata,data_label):
    predata=np.squeeze(predata)
    data_label=np.squeeze(data_label)
    if (len(np.unique(np.squeeze(data_label).astype(int))) == 2):
        # 性别预测
        predata = np.where(predata > 0.5, 1, 0)
        accuracy_num = (predata == data_label).sum()

        return accuracy_num
    else:
        # 预测结果与真实结果的相关系数
        a = np.squeeze(predata)
        b = np.squeeze(data_label)
        r = stats.pearsonr(a, b)
        MAE=np.mean(abs(a - b))
        return r[0],MAE


# 初始化权重
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    # elif classname.find("BatchNorm1d") != -1:
    #     torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     torch.nn.init.constant_(m.bias.data, 0.0)


class BrainNetCNN_encoder(torch.nn.Module):

    def __init__(self, in_planes,node_num,out_feature_num):
        super(BrainNetCNN_encoder, self).__init__()
        self.in_planes = in_planes
        self.d = node_num
        self.out_feature_num=out_feature_num

        self.E2N = torch.nn.Conv2d(1, 64, (1, self.d))
        self.bn2_3 = torch.nn.BatchNorm2d(64)
        self.N2G = torch.nn.Conv2d(64, self.out_feature_num, (self.d, 1))
        self.bn2_4 = torch.nn.BatchNorm2d(self.out_feature_num)

    def forward(self, x):

        out = F.leaky_relu(self.bn2_3(self.E2N(x)), negative_slope=0.33)
        out = F.leaky_relu(self.bn2_4(self.N2G(out)), negative_slope=0.33)
        out=out.view(out.size(0), -1)
        return out



class ContrastiveLearningModel(torch.nn.Module):
    def __init__(self, node_num):
        super(ContrastiveLearningModel, self).__init__()

        self.node_num = node_num
        self.feature_dim = 256

        self.model_encoder = BrainNetCNN_encoder(1,self.node_num,self.feature_dim)

        self.proj_latent_dim = 100
        self.pred_latent_dim = 100

        self.projector = nn.Sequential(nn.Linear(self.feature_dim, self.proj_latent_dim, bias=False),
                                       nn.BatchNorm1d(self.proj_latent_dim),
                                       nn.LeakyReLU(negative_slope=0.33),
                                       nn.Linear(self.proj_latent_dim, self.feature_dim, bias=False),
                                       nn.BatchNorm1d(self.feature_dim, affine=False))

        self.predictor = nn.Sequential(nn.Linear(self.feature_dim, self.pred_latent_dim, bias=False),
                                       nn.BatchNorm1d(self.pred_latent_dim),
                                       nn.LeakyReLU(negative_slope=0.33),
                                       nn.Linear(self.pred_latent_dim, self.feature_dim))  # 2


        self.model_predict = DownStreamModel()

    def forward(self, view1,view2=None,istest=False):
        if istest:
            endcode_feature1 = self.model_encoder(view1)
            pred_label = self.model_predict(endcode_feature1)
            return pred_label,endcode_feature1
        else:
            endcode_feature1 = self.model_encoder(view1)
            endcode_feature2 = self.model_encoder(view2)

            z1 = self.projector(endcode_feature1)
            z2 = self.projector(endcode_feature2)

            p1 = self.predictor(z1)  # NxC
            p2 = self.predictor(z2)  # NxC

            pred_label=self.model_predict(endcode_feature1,endcode_feature2)
            return p1, p2, z1, z2, pred_label



class DownStreamModel(torch.nn.Module):
    def __init__(self):
        super(DownStreamModel, self).__init__()

        self.feature_dim = 256
        self.latent_dim1 = 128
        self.latent_dim2 = 30

        self.latent_layer = nn.Sequential(nn.Linear(self.feature_dim, self.latent_dim1),
                                          nn.BatchNorm1d(self.latent_dim1),
                                          nn.LeakyReLU(negative_slope=0.33),
                                          #nn.Dropout(p=0.5),
                                          nn.Linear(self.latent_dim1, self.latent_dim2),
                                          nn.BatchNorm1d(self.latent_dim2),
                                          nn.LeakyReLU(negative_slope=0.33),
        )

        self.label_lay = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(self.latent_dim2, 1),
            #nn.LeakyReLU(negative_slope=0.33)
            nn.LeakyReLU(negative_slope=0.33)
        )

    def forward(self, feature1,feature2=None):
        if feature2 is None:
            latent_feature = self.latent_layer(feature1)
            pred_label=self.label_lay(latent_feature)
            return pred_label
        else:
            latent_feature1 = self.latent_layer(feature1)
            latent_feature2 = self.latent_layer(feature2)
            pred_label1 = self.label_lay(latent_feature1)
            pred_label2 = self.label_lay(latent_feature2)
            pred_label = 0.8*pred_label1 + 0.2*pred_label2
            return pred_label



# 模型v1
class Model:
    def __init__(self, arg,cuda,foldNo,foldlagpath):

        self.args=arg
        self.foldNo=foldNo
        self.foldlagpath=foldlagpath
        self.clm = ContrastiveLearningModel(arg.n_roi)
        self.writer = SummaryWriter(log_dir=foldlagpath)

        self.clm_loss = nn.CosineSimilarity(dim=-1)
        self.ds_loss=nn.L1Loss(reduction="mean")
        self.cuda=cuda
        self.pred_type = arg.pred_type
        self.save_freq=arg.save_freq

        self.cl_rate=arg.cl_rate
        self.pred_rate=arg.pred_rate


        if self.cuda:
            self.clm = nn.DataParallel(self.clm)
            self.clm = self.clm.cuda()
            self.clm_loss=self.clm_loss.cuda()
            self.ds_loss=self.ds_loss.cuda()


        self.init_optimizer(arg.optim, arg.lr, arg.weightdecay)
        self.init_save_path(arg.save_path,arg.checkpoint_dir)

    def init_save_path(self, save_path,checkpoint_dir, encoder_save_path=None,dsm_save_path=None):
        self.save_path = save_path
        self.checkpoint_dir =checkpoint_dir
        if encoder_save_path is None:
            self.encoder_save_path = os.path.join(self.save_path, 'CLM_fold%d.pth'%(self.foldNo))
            # self.dsm_save_path = os.path.join(self.save_path, 'DSM_%d.pth' % (self.pred_type))
        else:
            self.encoder_save_path = os.path.join(self.save_path, encoder_save_path)
            # self.dsm_save_path = os.path.join(self.save_path, dsm_save_path)


    def init_optimizer(self, opt_name, lr_encoder,weightdecay):
        if opt_name == 'Adam':
            self.opt_encoder = torch.optim.Adam(self.clm.parameters(), lr=lr_encoder, weight_decay=weightdecay)
            # self.opt_dsm = torch.optim.Adam(self.dsm.parameters(), lr=lr_dsm, weight_decay=weightdecay)

        elif opt_name == 'SGD':
            self.opt_encoder = torch.optim.SGD(self.clm.parameters(), lr=lr_encoder, momentum=0.9,
                                               weight_decay=weightdecay, nesterov=True)
            # self.opt_dsm = torch.optim.SGD(self.dsm.parameters(), lr=lr_dsm, momentum=0.9,
            #                                    weight_decay=weightdecay, nesterov=True)



    def load_clm(self):
        self.clm.load_state_dict(
            torch.load(self.encoder_save_path, map_location='cpu'))

    # def load_dsm(self):
    #     self.dsm.load_state_dict(
    #         torch.load(self.dsm_save_path, map_location='cpu'))

    # 训练对比组件
    def train_cl_encoder(self, train_loader,test_loader, max_epoch=200, save_model=False, load_model=False):
        if load_model:
            self.load_clm()
            print("Succesfully load CLM encoder!")
            return
        else:
            self.clm.apply(weights_init_normal)
        best_loss = 1e10
        best_acc_label = 1e10
        best_epoch = 0
        for epoch in range(0, max_epoch):
            since = time.time()
            # tr_cl_loss, tr_sim_loss, tr_pd_loss,tr_acc,tr_MAE,tr_loss = self.train_encoder_one_epoch(train_loader)
            tr_dict = self.train_encoder_one_epoch(train_loader)
            # te_pd_loss_ori,te_pd_loss_new,te_acc_ori,te_MAE_ori,te_acc_new,te_MAE_new,te_loss_ori,te_loss_new = self.test_encoder(test_loader)
            te_dict = self.test_encoder(test_loader)

            #adjust_learning_rate(self.opt_encoder, self.args.lr, epoch,self.args.n_epochs)

            time_elapsed = time.time() - since
            print('*====**')
            print('Fold: {:d} {:.0f}m {:.0f}s , best epoch {:d}   best loss: {:.7f} best acc_label: {:.7f}'.format(self.foldNo,time_elapsed // 60, time_elapsed % 60,
                                                                                  best_epoch, best_loss,best_acc_label))

            tr_cl_loss, tr_pd_loss, tr_loss, tr_acc_label, tr_MAE = tr_dict["loss_cl_all"],\
                                                                                               tr_dict["loss_pd_all"],\
                                                                                               tr_dict["loss_all"],\
                                                                                               tr_dict["acc"],\
                                                                                               tr_dict["MAE"]

            te_pd_loss, te_loss, te_acc_label, te_MAE =te_dict["loss_pd_all"],\
                                                       te_dict["loss_all"],\
                                                       te_dict["acc"],\
                                                       te_dict["MAE"]


            print('Fold: {:d} Epoch: {:03d}\n'
                  'Train_loss: CL_Loss: {:.7f} pd_loss: {:.7f} Loss: {:.7f}\n'
                  'Test_loss: pd_loss: {:.7f} Loss: {:.7f}\n'
                  'Train_acc: acc_label: {:.7f} MAE: {:.7f}\n'
                  'Test_acc: acc_label: {:.7f} MAE: {:.7f}\n'.format(self.foldNo,epoch, tr_cl_loss,tr_pd_loss,tr_loss,
                                                                                       te_pd_loss,te_loss,
                                                                                       tr_acc_label,tr_MAE,
                                                                                       te_acc_label, te_MAE))

            self.writer.add_scalars('contrastive_loss', {'train_cl_loss': tr_cl_loss}, epoch)
            self.writer.add_scalars('prediction_loss', {'train_pd_loss': tr_pd_loss, 'test_pd_loss': te_pd_loss}, epoch)
            self.writer.add_scalars('loss', {'train_loss': tr_loss, 'test_loss': te_loss}, epoch)
            self.writer.add_scalars('acc_label', {'train_acc_label': tr_acc_label, 'test_acc_label': te_acc_label}, epoch)
            self.writer.add_scalars('MAE', {'train_MAE': tr_MAE, 'test_MAE': te_MAE}, epoch)


            if epoch % self.save_freq ==0 :
                print("saving breakpoint encoder %d epoch"%(epoch))
                model_wts = copy.deepcopy(self.clm.state_dict())
                if save_model:
                    torch.save(model_wts, os.path.join(self.checkpoint_dir, 'CLM_fold%d_cheakpoint.pth'%(self.foldNo)))

            if te_loss < best_loss and epoch > 50:
                print("saving best encoder")
                best_epoch = epoch
                best_acc_label=te_acc_label
                best_loss = te_loss
                best_model_wts = copy.deepcopy(self.clm.state_dict())
                print("--------------train-------------")
                print("label: ",tr_dict['label'][0:10])
                print("label_pred: ", tr_dict['label_pred'][0:10])
                print("--------------test-------------")
                print("label: ", te_dict['label'][0:10])
                print("label_pred: ", te_dict['label_pred'][0:10])

                if save_model:
                    torch.save(best_model_wts, self.encoder_save_path)

            if epoch == max_epoch-1 :
                print("fold%d saving finall encoder"%(self.foldNo))

                model_wts = copy.deepcopy(self.clm.state_dict())
                if save_model:
                    torch.save(model_wts, os.path.join(self.save_path, 'CLM_fold%d_finally.pth'%(self.foldNo)))

        return self.encoder_save_path

    # 训练对比组件一轮
    def train_encoder_one_epoch(self, train_loader):
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        self.clm.train()
        loss_all=0
        loss_cl_all=0
        loss_pd_all=0
        step = 0

        all_label = []
        all_pred = []
        for batch_views,labels in train_loader:
            self.opt_encoder.zero_grad()
            batch_views = batch_views.type(Tensor)

            out1,out2,z1,z2,pred_out=self.clm(batch_views[:,0],batch_views[:,1])
            labels = labels.type(Tensor)
            labels = labels[:, np.newaxis]

            loss_cl = 1-(self.clm_loss(out1, z2.detach()).mean() + self.clm_loss(out2, z1.detach()).mean()) * 0.5
            loss_pd = self.ds_loss(pred_out, labels)
            all_pred += pred_out.detach().cpu().numpy().tolist()
            all_label += labels.detach().cpu().numpy().tolist()


            step = step + 1
            loss = self.cl_rate*loss_cl +self.pred_rate*loss_pd
            loss.backward()
            loss_cl_all += loss_cl.item() * batch_views.shape[0]
            loss_pd_all += loss_pd.item() * batch_views.shape[0]
            loss_all += loss.item() * batch_views.shape[0]
            self.opt_encoder.step()

        all_pred=np.array(all_pred)
        all_label=np.array(all_label)
        acc, MAE = get_Acc(all_pred, all_label)

        print('train...........')
        for param_group in self.opt_encoder.param_groups:
            print("LR encoder", param_group['lr'])
        #self.scheduler_encoder.step()

        dict={}
        dict["loss_cl_all"] = loss_cl_all / len(train_loader.dataset)
        dict["loss_pd_all"] = loss_pd_all / len(train_loader.dataset)
        dict["label"]=np.squeeze(all_label)
        dict["label_pred"]=np.squeeze(all_pred)
        dict["acc"]=acc
        dict["MAE"]=MAE
        dict["loss_all"]= loss_all/len(train_loader.dataset)

        return dict

    # 测试对比组件
    def test_encoder(self, loader):
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        self.clm.eval()
        loss_all = 0
        loss_pd_all = 0

        all_label = []
        all_pred = []

        with torch.no_grad():
            for batch_views,labels in loader:
                batch_views = batch_views.type(Tensor)

                pred_out,_ = self.clm(batch_views[:, 0],istest=True)
                labels = labels.type(Tensor)
                labels = labels[:, np.newaxis]


                loss_pd = self.ds_loss(pred_out, labels)
                all_pred += pred_out.detach().cpu().numpy().tolist()
                all_label += labels.detach().cpu().numpy().tolist()


                loss = self.pred_rate*loss_pd

                loss_pd_all += loss_pd.item() * batch_views.shape[0]
                loss_all += loss.item() * batch_views.shape[0]

            all_pred = np.array(all_pred)
            all_label = np.array(all_label)
            acc, MAE = get_Acc(all_pred, all_label)

        dict = {}
        dict["loss_pd_all"] = loss_pd_all / len(loader.dataset)
        dict["label"] = np.squeeze(all_label)
        dict["label_pred"] = np.squeeze(all_pred)
        dict["acc"] = acc
        dict["MAE"] = MAE
        dict["loss_all"] = loss_all / len(loader.dataset)

        return dict