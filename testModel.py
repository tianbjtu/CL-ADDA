import numpy as np
import torch
import torch.nn as nn
from myModel import ContrastiveLearningModel

def testModel(bestModel_path,test_fc,n_roi,cuda):
    data_fc=np.array(test_fc)
    data_fc=torch.tensor(data_fc)
    model=ContrastiveLearningModel(n_roi)
    if cuda:
        model = nn.DataParallel(model)
        model =model.cuda()

    model.load_state_dict(torch.load(bestModel_path, map_location='cpu'))
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    with torch.no_grad():

        data_fc = data_fc.type(Tensor)
        pd_label, _ = model(data_fc[:, 0],istest=True)

    pd_label=pd_label.detach().cpu().numpy()

    return pd_label

