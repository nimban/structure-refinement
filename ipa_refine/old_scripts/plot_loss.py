import matplotlib.pyplot as plt
import json
import numpy as np
import os

base_path = '/home/nn122/remote_runtime/newfold/ipa_refine/outputs/'
exec_id = '2022-01-07 16:39:23'
batch_size = 4

# 2022-01-07 16:39:23       # initial;
# 2022-01-24 03:24:43       #26030
# 2022-01-24 03:24:10       #31504 - 1e-3,0.5
# 2022-01-24 03:21:35       #15214

#            2022-01-24 10:10:05               #13917    scale=1
# 2022-01-24 10:10:50                          #49237    scale=0.1

# 2022-01-24 10:12:35                          #41253    no_blocks=10
# 2022-01-24 10:15:38                   #19332    no_blocks=6

with open(os.path.join(base_path, exec_id, "loss_curve.json")) as file:
    losses = json.load(file)

plt.plot(losses['val_fape'], label='val_fape')
plt.plot(losses['train_fape'], label='train_fape')
# plt.plot(losses['val_drmsd'], label='val_drmsd')
# plt.plot(losses['train_drmsd'], label='train_drmsd')

plt.legend()
plt.show()