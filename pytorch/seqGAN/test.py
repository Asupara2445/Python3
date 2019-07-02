import os
import glob
import matplotlib.pyplot as plt
import numpy as np

script_path = os.path.dirname(__file__)
result_path = os.path.join(script_path + "/result")

loss_files = glob.glob(result_path + "/*.npy")

loss_list = []
for i, path in enumerate(loss_files):
    arr = np.load(path)
    loss_list.append(arr)


fig = plt.figure()

ax1_1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1_2 = ax1_1.twinx()
ax1_1.plot(loss_list[0][0], "r-", label="accuracy")
ax1_2.plot(loss_list[0][1], "g-", label="loss")
h1, l1 = ax1_1.get_legend_handles_labels()
h2, l2 = ax1_2.get_legend_handles_labels()
ax1_1.legend(h1+h2, l1+l2, loc='lower right')

ax1_1.set_xlabel("Epochs")
ax1_1.set_ylabel("accuracy")
ax1_2.set_ylabel("loss")

ax2.plot(loss_list[1][0], "g-", label="loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("loss")

ax2.legend()
plt.show()