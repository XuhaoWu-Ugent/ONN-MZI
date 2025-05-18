#this code is used to draw Fig. 5 in our paper.
#please remove the plotting.py.forScat's file suffix before use this code.
import numpy as np
import matplotlib.pyplot as plt
import plotting
import pickle
import os
#if you want to get the data,please modify the commented out code in attention/model/Binary_Classification.py/train_ours
fig, axes = plotting.init_gridspec(1, 4, 4)
num=0
X_vals = my_list = [ i for _ in range(4)  for i in range(1, 41) ]
X_vals=np.asarray(X_vals)
sorted_array = np.repeat(np.arange(1, 5), 40)
for type_data in ["tr","te"]:
    Y_vals=[]
    for dataset in ["emotion","sst","hate","rotten_tomatoes"]:
        with open('./scat/loss_'+type_data+'_'+dataset+'.pkl', 'rb') as file:
            Y_vals_row = pickle.load(file) 
            Y_vals.append(Y_vals_row['sim_topk_loss_'+type_data])
    Y_vals=np.asarray(Y_vals)
    Y_vals=Y_vals.flatten()
    lines=plotting.plot_scatter_by_class(axes[num], X_vals, Y_vals,yhat=sorted_array)
    plotting.annotate(axes[num], xlabel="Epoch", ylabel="Top-K Surrogate Loss on "+ ("Train" if type_data=='tr' else "Test")+" Set", title="")
    num+=1
    Y_vals=[]
    for dataset in ["emotion","sst","hate","rotten_tomatoes"]:
        with open('./scat/percent_'+type_data+'_'+dataset+'.pkl', 'rb') as file:
            Y_vals_row = pickle.load(file) 
            Y_vals.append(Y_vals_row['topk_overlap_'+type_data])
    Y_vals=np.asarray(Y_vals)
    Y_vals=Y_vals.flatten()
    plotting.plot_scatter_by_class(axes[num], X_vals, Y_vals,yhat=sorted_array)
    plotting.annotate(axes[num], xlabel="Epoch", ylabel="Top-K Overlap Score on "+ ("Train" if type_data=='tr' else "Test")+" Set", title="")
    num+=1
plt.legend(handles=lines,labels=["emotion","sst","hate","rotten_tomatoes"],bbox_to_anchor=(0.3, -0.3), ncol=4,)
plotting.adjust_gridspec()
plotting.show_gridspec()
axes[0].figure.savefig('./tvd.pdf')

    