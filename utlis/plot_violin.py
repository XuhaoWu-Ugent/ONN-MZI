#this code is used to draw Fig.2 in our paper.
#please remove the plotting.py.forViolin's file suffix before use this code.
import numpy as np
import matplotlib.pyplot as plt
import plotting
import pickle
data='rotten_tomatoes'
#emotion,sst,hate,rotten_tomatoes
#if you want to get the data,please modify the commented out code in attention/model/Binary_Classification.py/train_ours
with open('./1/ours_jsd_list_'+data+'.pkl', 'rb') as file:
#with open('./1/ours_jsd_list_'+data+'.pkl', 'rb') as file:
    X_vals = pickle.load(file)
with open('./1/ours_max_att_list_'+data+'.pkl', 'rb') as file:
    Y_vals = pickle.load(file) 
with open('./1/ours_label_list_'+data+'.pkl', 'rb') as file:
    yhat = pickle.load(file)
fig, axes = plotting.init_gridspec(1, 1, 1)

plotting.plot_violin_by_class(axes[0], Y_vals, np.abs(X_vals), yhat, xlim=[0,1],bins=4)

# plot_scatter_by_class(axes[0], X_vals, Y_vals, yhat)
plotting.annotate(axes[0], xlabel="", ylabel="", title="")


plotting.adjust_gridspec()
plotting.show_gridspec()

# save_axis_in_file(fig, axes[0], 'File Path', 'filename')

plt.show()
