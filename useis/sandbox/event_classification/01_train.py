from useis.processors import classifier
from importlib import reload
from useis.ai import model
from uquake.core.logging import logger
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
reload(classifier)
reload(model)

torch.cuda.empty_cache()

classifier_project = classifier.Classifier('/data_1/projects/', 'classifier', 'OT',
                                           reset_training=False)

train, test, validation = classifier_project.split_dataset(use_synthetic=False)


ec = model.EventClassifier(train.num_classes)

losses = []
accuracies = []

batch_size = 500

for epoch in tqdm(range(0, 100)):
    ec.train(train, batch_size=batch_size)
    accuracy = ec.validate(test, batch_size=250)
    accuracy = [float(a) for a in accuracy]

    text = f'bound:[{np.min(accuracy), np.max(accuracy)}\n' \
           f'mean: {np.mean(accuracy)}\n' \
           f'median: {np.median(accuracy)}\n' \
           f'std: {np.std(accuracy)}'

    print(text)

    logger.info(f'Average iteration loss: {np.mean(ec.losses): 0.3f} '
                f'+/- {np.std(ec.losses): 0.3f}')
    logger.info(f'Average iteration accuracy: {np.mean(ec.accuracies):0.3f} '
                f'+/- {np.std(ec.accuracies): 0.3f}')

    losses.append(np.mean(ec.losses))
    accuracies.append(np.mean(accuracy))
    # accuracies.append(np.mean(ec.accuracies))

    if epoch % 10 == 0:
        # ec.model.eval()
        ec.save(f'classifier_model_{epoch + 1}_epoch.pickle')

    plt.figure(1)
    plt.clf()
    plt.ion()
    # plt.plot(np.arange(epoch + 1), losses)
    plt.plot(np.arange(epoch + 1), accuracies)
    plt.xlabel('epoch')
    plt.ylabel('loss/accuracy')
    plt.pause(0.1)
    plt.show()

