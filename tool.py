import numpy as np
from nilearn import connectome
import os
from sklearn.preprocessing import MinMaxScaler

def silde_type(data, movie_NO):
    if movie_NO ==1 :
        result_data = data[ :,:, 29:268]
        result_data = np.concatenate((result_data, data[ :,:, 294:509]), 2)
        result_data = np.concatenate((result_data, data[ :,:, 535:718]), 2)
        result_data = np.concatenate((result_data, data[ :,:, 743:802]), 2)

    if movie_NO == 2:
        result_data = data[ :,:, 29:251]
        result_data = np.concatenate((result_data, data[ :,:, 276:530]), 2)
        result_data = np.concatenate((result_data, data[ :,:, 554:799]), 2)


    if movie_NO ==3 :
        result_data = data[ :,:, 29:204]
        result_data = np.concatenate((result_data, data[ :,:, 230:409]), 2)
        result_data = np.concatenate((result_data, data[ :,:, 434:633]), 2)
        result_data = np.concatenate((result_data, data[ :,:, 659:796]), 2)

    if movie_NO == 4:
        result_data = data[:,:, 29:257]
        result_data = np.concatenate((result_data, data[:,:, 281:506]), 2)
        result_data = np.concatenate((result_data, data[:,:, 531:782]), 2)

    return result_data

# def generate_high_amplitude_connectivity(timeseries,kind='correlation',loc=0):
#
#     timeseries_after_zscore=MinMaxScaler().fit_transform(timeseries)
#     ntime, nnodes = timeseries_after_zscore.shape
#
#     # indices of unique edges(upper triangle)
#     u, v = np.where(np.triu(np.ones(nnodes), 1))
#
#     #generate edge time series
#     ets = timeseries_after_zscore[:, u] * timeseries_after_zscore[:, v]
#
#     # calculate co - fluctuation amplitude at each frame
#     temp1=np.square(ets)
#     sum_temp1=np.sum(temp1,1)
#     rms = np.sqrt(sum_temp1)
#
#     idxsort = np.argsort(rms)[::-1]
#     if loc==0:
#         # fraction of high - and low - amplitude frames to retain
#         frackeep = 0.1
#         nkeep = int(ntime * frackeep)
#         # sort co - fluctuation amplitude
#         # estimate fc using just high - amplitude frames
#         idxsort=idxsort[0:nkeep]
#         idxsort=np.sort(idxsort)
#         top_timeseries = timeseries[idxsort,:]
#     elif loc==1:
#         # fraction of high - and low - amplitude frames to retain
#         frackeep = 0.1
#         nkeep = int(ntime * frackeep)
#         # sort co - fluctuation amplitude
#         # estimate fc using just high - amplitude frames
#         idxsort = idxsort[nkeep:2*nkeep]
#         idxsort = np.sort(idxsort)
#         top_timeseries = timeseries[idxsort, :]
#
#     fc = np.array(np.corrcoef(top_timeseries, rowvar=0))
#     fc=fc[np.newaxis,:]
#
#     return fc

def generate_high_amplitude_connectivity(timeseries):

    timeseries_after_zscore=MinMaxScaler().fit_transform(timeseries)
    ntime, nnodes = timeseries_after_zscore.shape

    # indices of unique edges(upper triangle)
    u, v = np.where(np.triu(np.ones(nnodes), 1))

    #generate edge time series
    ets = timeseries_after_zscore[:, u] * timeseries_after_zscore[:, v]

    # calculate co - fluctuation amplitude at each frame
    temp1=np.square(ets)
    sum_temp1=np.sum(temp1,1)
    rms = np.sqrt(sum_temp1)

    idxsort = np.argsort(rms)[::-1]

    idxsort1 = idxsort[0:len(idxsort)//2]
    #idxsort1 = np.sort(idxsort1)

    idxsort2 = idxsort[len(idxsort)//2:]
    #idxsort2 = np.sort(idxsort2)

    top_timeseries1 = timeseries[idxsort1, :]
    top_timeseries2 = timeseries[idxsort2, :]

    fc1 = np.array(np.corrcoef(top_timeseries1, rowvar=0))
    fc1 = fc1[np.newaxis,:]
    fc2 = np.array(np.corrcoef(top_timeseries2, rowvar=0))
    fc2 = fc2[np.newaxis, :]

    return fc1,fc2


def mean2(x):
    y = np.sum(x) / (x.shape[0]*x.shape[0])
    return y


def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
    return r



# def get_generate_fc(data,generate_type="train"):
#
#     fc_list=[]
#     if generate_type=="train":
#         for index in range(data.shape[0]):
#             result = []
#             temp_data = data[index, :, :]
#             movie_view1 = temp_data
#             temp = np.array(np.corrcoef(movie_view1, rowvar=0))
#             temp = temp[np.newaxis, :]
#             result.append(temp)
#             movie_view2 = temp_data
#             temp = generate_high_amplitude_connectivity(movie_view2,loc=0)
#             result.append(temp)
#             result = np.array(result)
#             print("corr: %.6f"%(corr2(result[0,0,:,:],result[1,0,:,:])))
#             fc_list.append(result)
#     else:
#         for index in range(data.shape[0]):
#             result = []
#             temp_data = data[index, :, :]
#             movie_view1 = temp_data
#             fc = np.array(np.corrcoef(movie_view1, rowvar=0))
#             fc = fc[np.newaxis, :]
#             result.append(fc)
#             movie_view2 = temp_data
#             temp = generate_high_amplitude_connectivity(movie_view2, loc=0)
#             result.append(temp)
#             result = np.array(result)
#             fc_list.append(result)
#
#     return fc_list
def get_generate_fc(data,generate_type="train"):

    fc_list=[]

    if generate_type=="train":

        for index in range(data.shape[0]):
            result = []
            temp_data = data[index, :, :]
            movie_view = temp_data
            fc1,fc2=generate_high_amplitude_connectivity(movie_view)
            result.append(fc1)
            result.append(fc2)
            result = np.array(result)
            print("corr: %.6f"%(corr2(result[0,0,:,:],result[1,0,:,:])))
            fc_list.append(result)

    elif generate_type=="test":
        for index in range(data.shape[0]):
            result = []
            temp_data = data[index, :, :]
            movie_view1 = temp_data
            fc = np.array(np.corrcoef(movie_view1, rowvar=0))
            fc = fc[np.newaxis, :]
            result.append(fc)
            result = np.array(result)
            fc_list.append(result)

    fc_list=np.array(fc_list)
    return fc_list


def generate_order(SubjInfo,num_fold):

    np.random.seed(10)

    SubjNoUse = [i for i in range(len(SubjInfo))]
    SubjNoUse = np.array(SubjNoUse)
    np.random.shuffle(SubjNoUse)
    np.random.seed(None)

    index_order=SubjNoUse
    train_index_list,test_index_list,Permed=NFold(index_order,num_fold)

    return train_index_list,test_index_list,Permed

def NFold(index_order, FoldNum):

    SampleSize=index_order.shape[0]
    MinFeatNumPerFold = int(np.floor(SampleSize / FoldNum))
    Permed=index_order
    SampleNo={}
    for Tmp in range(FoldNum):
        SampleNo[Tmp] = Permed[Tmp * MinFeatNumPerFold: (Tmp+1) * MinFeatNumPerFold]

    if np.mod(SampleSize, FoldNum): # np.mod(a, m): 返回 a 除以 m 后的余数
        for Tmp in range(np.mod(SampleSize, FoldNum)):
            SampleNo[Tmp]=SampleNo[Tmp].tolist()
            SampleNo[Tmp].append(Permed[FoldNum * MinFeatNumPerFold + Tmp])
            SampleNo[Tmp]=np.array(SampleNo[Tmp])
    TestNo={}
    TrainNo={}
    for Tmp in range(FoldNum):
        TestNo[Tmp]=SampleNo[Tmp]
        for i in range(FoldNum):
            if i!=Tmp:
                if Tmp in TrainNo.keys():
                    A = SampleNo[i].tolist()
                    TrainNo[Tmp]=TrainNo[Tmp]+A
                else:
                    A=SampleNo[i].tolist()
                    TrainNo[Tmp]=A
        TrainNo[Tmp]=np.array(TrainNo[Tmp])
    return TrainNo,TestNo,Permed


# 删除文件夹
def deldir(dir):
    if not os.path.exists(dir):
        return False
    if os.path.isfile(dir):
        os.remove(dir)
        return
    for i in os.listdir(dir):
        t = os.path.join(dir, i)
        if os.path.isdir(t):
            deldir(t)  # 重新调用次方法
        else:
            os.unlink(t)
    if os.path.exists(dir):
        os.removedirs(dir)  # 递归删除目录下面的空文件夹


