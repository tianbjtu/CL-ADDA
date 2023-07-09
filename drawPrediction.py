import numpy as np
import pandas as pd
import scipy.io as scio
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams['axes.unicode_minus']=False
sns.set_theme(style="ticks",font='Times New Roman')
fig,sub_ax=plt.subplots(nrows=1,ncols=2,figsize = (20,9))
fig.subplots_adjust(hspace=0.3,wspace=0.3)

def draw1():
    data = scio.loadmat("result/resultAge.mat")
    Predicted_data=data['pd_label'][:,0]
    true_label=data['true_label'][:,0]
    R=data['acc'][:,0]
    MAE=data['MAE'][:,0]


    with sns.axes_style("whitegrid"):
        ax1 = sns.scatterplot(x=true_label, y=Predicted_data, color="#1663A9", marker='o', ax=sub_ax[0])
        ax1.set_xlabel("Actual Age", fontsize=25,weight='bold')
        ax1.set_ylabel("Predicted Age", fontsize=25, weight='bold')
        ax1.set_xlim(0,100)
        ax1.set_ylim(0,100)
        vals = ax1.get_xticks()
        ax1.set_xticklabels([int(x) for x in vals], fontsize=20,rotation=45, weight='bold')
        vals = ax1.get_yticks()
        ax1.set_yticklabels([int(x) for x in vals], fontsize=20, weight='bold')
        ax1.text(70, 10, 'R = %.3f' % R, fontsize=15)
        ax1.text(70, 5, 'MAE = %.3f years' % MAE, fontsize=15)

        parameter = np.polyfit(true_label, Predicted_data, 1)
        y = parameter[0] * true_label + parameter[1]
        # y1 = parameter[0] * true_label + parameter[1]+MAE
        # y2 = parameter[0] * true_label + parameter[1] - MAE
        sns.lineplot(true_label, y,ci=None, color='r', linewidth=3,ax=sub_ax[0])
        # sns.lineplot(true_label, y1, ci=None, color='r', linewidth=3, linestyle="--",ax=sub_ax[0])
        # sns.lineplot(true_label, y2, ci=None, color='r', linewidth=3, linestyle="--",ax=sub_ax[0])
        ax1.set_title('(a)', loc='left', fontsize=35, y=1.05, x=-0.15, weight='bold')

def draw2():
    data = scio.loadmat("result/resultIQ.mat")
    Predicted_data = data['pd_label'][:, 0]
    true_label = data['true_label'][:, 0]
    R = data['acc'][:, 0]
    MAE = data['MAE'][:, 0]

    with sns.axes_style("whitegrid"):
        ax1 = sns.scatterplot(x=true_label, y=Predicted_data, color="#1663A9", marker='o', ax=sub_ax[1])
        ax1.set_xlabel("Actual IQ", fontsize=25, weight='bold')
        ax1.set_ylabel("Predicted IQ", fontsize=25, weight='bold')
        ax1.set_xlim(0, 50)
        ax1.set_ylim(0, 50)
        vals = ax1.get_xticks()
        ax1.set_xticklabels([int(x) for x in vals], fontsize=20, rotation=45, weight='bold')
        vals = ax1.get_yticks()
        ax1.set_yticklabels([int(x) for x in vals], fontsize=20, weight='bold')
        ax1.text(40, 5, 'R = %.3f' % R, fontsize=15)
        ax1.text(40, 2.5, 'MAE = %.3f' % MAE, fontsize=15)

        parameter = np.polyfit(true_label, Predicted_data, 1)
        y = parameter[0] * true_label + parameter[1]
        # y1 = parameter[0] * true_label + parameter[1]+MAE
        # y2 = parameter[0] * true_label + parameter[1] - MAE
        sns.lineplot(true_label, y, ci=None, color='r', linewidth=3, ax=sub_ax[1])
        # sns.lineplot(true_label, y1, ci=None, color='r', linewidth=3, linestyle="--",ax=sub_ax[0])
        # sns.lineplot(true_label, y2, ci=None, color='r', linewidth=3, linestyle="--",ax=sub_ax[0])
        ax1.set_title('(b)', loc='left', fontsize=35, y=1.05, x=-0.15, weight='bold')


draw1()
draw2()
fig.savefig("./image/fig5.tiff",dpi=300)
plt.show()
