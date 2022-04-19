import matplotlib.pyplot as plt
import json
import numpy as np

with open("/home/nn122/remote_runtime/newfold/ipa_refine/outputs/val_loss_curve.json") as file:
    val_data = json.load(file)

with open("/home/nn122/remote_runtime/newfold/ipa_refine/outputs/train_loss_curve.json") as file:
    train_data = json.load(file)

# i = [i for i, d in enumerate(val_data)]
# batch_size = 1118

# train_batched = np.reshape(train_data, (int(len(train_data)/batch_size), batch_size))
# train_mean_loss = np.mean(train_batched, axis=-1)
plt.plot(val_data, label='val')
plt.plot(train_data, label='train')
plt.legend()
plt.show()
