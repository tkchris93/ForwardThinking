import sys
import pickle
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

fwd_acc = []
fwd_val_acc = []
fwd_times = []

back_acc = []
back_val_acc = []
back_times = []

for i in xrange(1,4):
    fwd = np.load('layer{}_mnist_results.npz'.format(i))
    fwd_acc.append(fwd['acc'])
    fwd_val_acc.append(fwd['val_acc'])
    fwd_times.append(fwd['times'])

fwd_acc = np.concatenate(fwd_acc)
fwd_val_acc = np.concatenate(fwd_val_acc)
fwd_times = np.concatenate(fwd_times)

back = np.load('mnist_backprop_results.npz'.format(i))
back_acc = back['acc']
back_val_acc = back['val_acc']
back_times = back['times']


sns.set_context("paper")
sns.set(font='serif')
sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]})

fig = plt.figure()
fig.set_size_inches(3.5, 4)
ax = plt.subplot(211)
fig.dpi = 100

c1 = '#008000'
c2 = '#000073'

# create training accuracy plot
ax.plot(fwd_times / 60., 100 * (np.array(fwd_acc)), label = 'Forward Thinking', c=c1, linewidth=2)
ax.plot(back_times / 60., 100 * (np.array(back_acc)), label = 'Backpropagation', c=c2, alpha=.5)
plt.yticks(np.arange(91,101))
plt.title('Training Accuracy')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('Minutes')
plt.ylabel('Accuracy (%)')

# create testing accuracy plot
ax = plt.subplot(212)
ax.plot(fwd_times / 60., 100 * (np.array(fwd_val_acc)), label = 'Forward Thinking', c=c1, linewidth=2)
ax.plot(back_times / 60., 100 * (np.array(back_val_acc)), label = 'Backpropagation', c=c2, alpha=.5)
plt.yticks(np.arange(98,100,.25))
plt.title('Test Accuracy')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('Minutes')
plt.ylabel('Accuracy (%)')

plt.savefig('accuracy.png')
plt.show()
