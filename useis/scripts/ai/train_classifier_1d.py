import dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from importlib import reload
reload(dataset)
from tqdm import tqdm
import model
import pickle
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import confusion_matrix
from plotcm import plot_confusion_matrix
import model
reload(model)

input_directory = '/data_1/classification_dataset_1D/'
suffix = ''
extension = 'pickle'

# file_list = glob(os.path.join(input_directory, '*', f'*{suffix}.{extension}'))

# training, test = spectrogram_dataset.split_dataset(input_directory, split=0.8,
#                                                    seed=1)

file_list = dataset.FileList(input_directory, extension=extension)

ec = model.EventClassifier1D(len(file_list.category_list))

losses = []
accuracies = []

batch_size = 1000
for epoch in tqdm(range(0, 100)):
    logger.info('selecting data')
    training_dataset = file_list.select1d(1e5)
    test_dataset = file_list.select1d(1e4)
    ec.train(training_dataset, batch_size=batch_size)
    accuracy = ec.validate(test_dataset, batch_size=batch_size)

    logger.info(f'Average iteration loss: {np.mean(ec.losses): 0.3f} '
                f'+/- {np.std(ec.losses): 0.3f}')
    logger.info(f'Average iteration accuracy: {np.mean(ec.accuracies):0.3f} '
                f'+/- {np.std(ec.accuracies): 0.3f}')

    losses.append(np.mean(ec.losses))
    accuracies.append(accuracy)
    # accuracies.append(np.mean(ec.accuracies))

    if epoch % 10 == 0:
        # ec.model.eval()
        ec.save(f'classifier_model_1d_{epoch + 1}_epoch.pickle')

    plt.figure(1)
    plt.clf()
    plt.ion()
    # plt.plot(np.arange(epoch + 1), losses)
    plt.plot(np.arange(epoch + 1), accuracies)
    plt.xlabel('epoch')
    plt.ylabel('loss/accuracy')
    plt.pause(0.1)
    plt.show()




# # pickle.dump(training_dataset, open('training_dataset.pickle', 'wb'))
# # training_dataset = pickle.load(open('training_dataset.pickle', 'rb'))
# # training_dataset = pickle.load(open('training_dataset.pickle', 'rb'))
#
# # test_dataset = spectrogram_dataset.SpectrogramDataset(input_directory,
# #                                                       max_image_sample=4e4)
# # pickle.dump(test_dataset, open('test_dataset.pickle', 'wb'))
# # test_dataset = pickle.load(open('test_dataset.pickle', 'rb'))
#
# nb_pixel = training_dataset.nb_pixel
# nb_categories = training_dataset.nb_categories
#
#
#
# # model = model.CNN(nb_categories)
# # VGG16
# # model = models.vgg16(pretrained=False, progress=True)
# # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1,
# #                                dilation=1, groups=1, bias=True)]
# # first_conv_layer.extend(list(model.features))
# # model.features = nn.Sequential(*first_conv_layer)
#
# # resnet 50
# # model = models.resnet34(pretrained=False)
# # weights = model.conv1.weight.clone()
# # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# # model.conv1.weight[:, :1] = weights
# # model.conv1.weight[:, 3] = model.conv1.weight[:, 0]
#
#
# # model = nn.Sequential(
# #     nn.Linear(nb_pixel, 128),
# #     nn.ReLU(),
# #     nn.Linear(128, nb_categories)
# # )
#
#
# batch_size = 2000
#
# training_loader = DataLoader(training_dataset, batch_size=batch_size,
#                              shuffle=True)
#
# n_epochs = 25
#
# # Stuff to store
# train_losses = []  # np.zeros(n_epochs)
# train_accuracies = []
#
# # plt.ion()
# # fig, ax = plt.subplots()
# # xlim = 1000
# # # plt.xlim([0, xlim])
# # plt.ylim([0, 1])
# # plt.axhline(0.95, color='k', ls='--')
# # plt.axhline(0.98, color='k', ls='--')
# #
# # fig2, ax2 = plt.subplots()
# # plt.xlim([0, 2300])
# # plt.ylim([0, 2])
# # plt.show()
#
# for it in tqdm(range(n_epochs)):
#
#     for inputs, targets in tqdm(training_loader):
#         inputs = inputs.view(inputs.size()[0], -1, inputs.size()[1],
#                              inputs.size()[2])
#         inputs = inputs.to(device)
#         targets = targets.to(device)
#
#         optimizer.zero_grad()
#
#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         tmp_choices = [np.argmax(out.cpu().detach().numpy())
#                        for out in outputs]
#         total_accuracy = np.sum(tmp_choices ==
#                                 targets.cpu().detach().numpy()) \
#                          / len(tmp_choices)
#
#         train_accuracy.append(total_accuracy)
#
#         # Backward and optimize
#         loss.backward()
#         optimizer.step()
#
#         # print(loss.item())
#         train_loss.append(loss.item())
#         iterations.append(i)
#
#         if i % 20 == 19:
#             logger.info(f'accuracy: '
#                         f'{np.mean(np.array(train_accuracy)[-100:-1]): 0.3f} '
#                         f'+/- '
#                         f'{np.std(np.array(train_accuracy)[-100:-1]):0.3f}')
#             cm = confusion_matrix(targets.cpu().detach().numpy(),
#                                   np.array(tmp_choices))
#
#             print(cm)
#
#             plt.figure(10)
#             plt.clf()
#             plot_confusion_matrix(cm, training_dataset.category_list,
#                                   normalize=True)
#             plt.show()
#
#
#             # test_loader = DataLoader(test_dataset, batch_size=1000,
#             #                          shuffle=True)
#             #
#             # test_accuracies = []
#             # for t_inputs, t_targets in tqdm(test_loader):
#             #     model = model.to(device)
#             #     inps = t_inputs.view(t_inputs.size()[0], -1,
#             #                          t_inputs.size()[1],
#             #                          t_inputs.size()[2])
#             #     # inps = inps.to(device)
#             #     # targs = t_targets.to(device)
#             #     targs = t_targets.to(device)
#             #     #
#             #     outs = model(inps.to(device))
#             #
#             #     tmp_choices = [np.argmax(out.cpu().detach().numpy())
#             #                    for out in outs]
#             #     test_accuracies.append(np.sum(tmp_choices ==
#             #                                 targs.cpu().detach().numpy()) \
#             #                            / len(tmp_choices))
#             #
#             # logger.info(f'accuracy: {np.mean(test_accuracies)} +/- '
#             #             f'{np.std(test_accuracies)}')
#
#         # if i == 0:
#         #     sc, = ax.plot(iterations, train_loss)
#         #     plt.show()
#         #     sc2, = ax.plot(iterations, train_accuracy)
#         # # if i == 200:
#         # #     optimizer.param_groups[0]['lr'] = \
#         # #         optimizer.param_groups[0]['lr'] * 10
#         # # if i == 400:
#         # #     optimizer.param_groups[0]['lr'] = \
#         # #         optimizer.param_groups[0]['lr'] * 10
#         # else:
#         #     # plt.figure(1)
#         #     # plt.clf()
#         #     # plt.plot(iterations, train_losses)
#         #     # plt.show()
#         #     # ax.plot(iterations, train_losses)
#         #     if np.max(iterations) > xlim:
#         #         xlim += 1000
#         #         plt.xlim([0, xlim])
#         #     sc.set_data(iterations, train_loss)
#         #     sc2.set_data(iterations, train_accuracy)
#         #
#         #     ax = plt.gca()
#         #
#         #     # recompute the ax.dataLim
#         #     ax.relim()
#         #     # update ax.viewLim using the new dataLim
#         #     ax.autoscale_view()
#         #     plt.draw()
#         #
#         #     fig.canvas.draw_idle()
#         #     # fig2.canvas.draw_idle()
#         #     try:
#         #         fig.canvas.flush_events()
#         #         # fig2.canvas.flush_events()
#         #     except NotImplementedError:
#         #         pass
#         i += 1
#         train_losses.append(train_loss)
#         train_accuracies.append(train_accuracy)
#
#     # to avoid over-fitting the model
#     # training_dataset = spectrogram_dataset.SpectrogramDataset(input_directory,
#     #                                                           max_image_sample
#     #                                                           =2e5)
#     training_dataset = file_list.select(1e5)
#     training_loader = DataLoader(training_dataset, batch_size=batch_size,
#                                  shuffle=True)
#
#     # train_losses.append(np.min(train_loss))
#     # train_accuracy.append(torch.sum())
#     # if it == 0:
#     #     sc2, = ax2.plot(it, train_losses)
#     #     plt.plot()
#     # else:
#     #     sc2.set_data(np.arange(len(train_losses)), train_losses)
#     #     fig.canvas.draw_idle()
#     #     try:
#     #         fig.canvas.flush_events()
#     #     except NotImplementedError:
#     #         pass
#
# pickle.dump(model, open('model.pickle', 'rb'))
#
# # cpu_model = model.cpu()
# # test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
# # i = 0
# # acc = 0
# # for t_inputs, t_targets in tqdm(test_loader):
# #     model = model.to(device)
# #     inps = t_inputs.view(t_inputs.size()[0], -1, t_inputs.size()[1],
# #                          t_inputs.size()[2])
# #     # inps = inps.to(device)
# #     # targs = t_targets.to(device)
# #     targs = t_targets.to(device)
# #     #
# #     outputs = model(inps.to(device))
# #
# #     tmp_choices = [np.argmax(out.cpu().detach().numpy())
# #                    for out in outputs]
# #     total_accuracy = np.sum(tmp_choices ==
# #                             targs.cpu().detach().numpy()) \
# #                      / len(tmp_choices)
# #
# #     acc += total_accuracy
# #     i += 1
#     # input(total_accuracy)
#
#
#
