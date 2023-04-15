from glob import glob
from importlib import reload
from useis.ai import model
from useis.ai import dataset

from tqdm import tqdm
import matplotlib.pyplot as plt

reload(model)
reload(dataset)

input_files = '/data_1/pick_dataset/*.pickle'

file_list = glob(input_files)

picker_dataset = dataset.PickingDataset(file_list)

train_dataset, test_dataset = picker_dataset.split(split_fraction=0.8)

picker = model.AIPicker()

epoch = 100
train_losses = []
test_losses = []
batch_size = 2000
for e in tqdm(range(0, epoch)):
    picker.train(train_dataset, batch_size=batch_size)
    predictions, test_loss_mean, test_loss_std = picker.validate(
        test_dataset, batch_size=batch_size)
    test_losses.append(test_loss_mean)
    predictions, train_loss_mean, train_loss_std = picker.validate(
        train_dataset, batch_size=batch_size)
    train_losses.append(train_loss_mean)

    print(f'test loss: {test_loss_mean:0.2f}\ntrain loss: {train_loss_mean:0.2f}')

picker.save('/data_1/ai_models/ai_picker.pickle')

plt.figure(1)
plt.clf()
plt.plot(train_losses, label='train losses')
plt.plot(test_losses, label='test losses')
plt.legend()
plt.show()
plt.pause(0.1)
