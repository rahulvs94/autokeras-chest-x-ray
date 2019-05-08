import os
import torch
import pprint
import datetime
import numpy as np
from autokeras import CnnModule
from nas.grid import GridSearcher
import torch.backends.cudnn as cudnn
from nas.greedy import GreedySearcher
from torch.utils.data import DataLoader
from autokeras.nn.metric import Accuracy
import torchvision.transforms as transforms
from DatasetGenerator import DatasetGenerator
from sklearn.metrics.ranking import roc_auc_score
from autokeras.nn.generator import ResNetGenerator
from torch.utils.data.sampler import SubsetRandomSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

parameters = {'transResize': 256,
                  'imgTransCrop': 224,
                  'batchSize': 48,
                  'numberOfWorkers': 16,
                  'numClasses': 14,
                  'validSize': 0.15,
                  'normalize': transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225]),
                  'BCEloss': torch.nn.BCEWithLogitsLoss(reduction='mean'),
                  'timeLimit': 1 * 5 * 60
                  }


def computeAUROC(target, prediction, classCount):
    outAUROC = []

    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()

    for i in range(classCount):
        outAUROC.append(roc_auc_score(target[:, i], prediction[:, i]))

    return outAUROC


def data(params):
    currentDirectory = os.getcwd()

    pathTrainData = os.path.join(currentDirectory, 'dataset/small_train/')
    pathTestData = os.path.join(currentDirectory, 'dataset/small_test/')

    pathLabelsTrain = os.path.join(currentDirectory, 'dataset/small_train_w_labels.txt')
    pathLabelsTest = os.path.join(currentDirectory, 'dataset/small_test_w_labels.txt')

    # pathTrainData = os.path.join(currentDirectory, 'dataset/train/')
    # pathTestData = os.path.join(currentDirectory, 'dataset/test/')
    #
    # pathLabelsTrain = os.path.join(currentDirectory, 'dataset/train_w_labels.txt')
    # pathLabelsTest = os.path.join(currentDirectory, 'dataset/test_w_labels.txt')

    # pathTrainData = os.path.join(currentDirectory, 'dataset/experiment_train/')
    # pathTestData = os.path.join(currentDirectory, 'dataset/experiment_test/')
    #
    # pathLabelsTrain = os.path.join(currentDirectory, 'dataset/experiment_train_w_labels.txt')
    # pathLabelsTest = os.path.join(currentDirectory, 'dataset/experiment_test_w_labels.txt')

    pp = pprint.PrettyPrinter(indent=2)
    print('\n==> Parameters used...')
    pp.pprint(params)

    print('\n==> Printing directory paths...')
    print('Current directory: ', currentDirectory)
    print('train directory: ', pathTrainData)
    print('test directory: ', pathTestData)

    print('\n==> Preparing data augmentation...')

    transform_train = []
    transform_train.append(transforms.RandomResizedCrop(size=params['imgTransCrop'],
                                                        ratio=(0.5, 1)))
    transform_train.append(transforms.ToTensor())
    transform_train.append(params['normalize'])
    transformTrain = transforms.Compose(transform_train)

    transform_test = []
    # transform_test.append(transforms.Resize(transResize))
    transform_test.append(transforms.Resize(params['imgTransCrop']))
    transform_test.append(transforms.ToTensor())
    transform_test.append(params['normalize'])
    transformTest = transforms.Compose(transform_test)

    print('\n==> Preparing dataset generator...')

    trainSet = DatasetGenerator(
        pathImageDirectory=pathTrainData,
        pathDatasetFile=pathLabelsTrain,
        transform=transformTrain
    )

    testSet = DatasetGenerator(
        pathImageDirectory=pathTestData,
        pathDatasetFile=pathLabelsTest,
        transform=transformTest
    )

    print('\n==> Preparing validation set...')

    numberOfTrain = len(trainSet)
    indices = list(range(numberOfTrain))
    np.random.shuffle(indices)
    split = int(np.floor(params['validSize'] * numberOfTrain))
    train_idx, valid_idx = indices[split:], indices[:split]

    # https://pytorch.org/docs/stable/data.html
    print('\n==> Preparing train and validation sampler...')

    trainSampler = SubsetRandomSampler(train_idx)
    validSampler = SubsetRandomSampler(valid_idx)

    print('\n==> Preparing dataloader...')

    trainLoader = DataLoader(
        dataset=trainSet,
        batch_size=params['batchSize'],
        sampler=trainSampler,
        num_workers=params['numberOfWorkers'],
        pin_memory=True
    )

    validationLoader = DataLoader(
        dataset=trainSet,
        batch_size=params['batchSize'],
        sampler=validSampler,
        shuffle=False,
        num_workers=params['numberOfWorkers'],
        pin_memory=True
    )

    testLoader = DataLoader(
        dataset=testSet,
        batch_size=params['batchSize'],
        shuffle=False,
        num_workers=params['numberOfWorkers'],
        pin_memory=True
    )

    print("\nSize of train set: ", len(trainLoader) * params['batchSize'])
    print("Size of validation set: ", len(validationLoader) * params['batchSize'])
    print("Size of test set: ", len(testLoader) * params['batchSize'])

    return trainSet, testSet, trainLoader, validationLoader, testLoader


def train(params):
    trainSet, testSet, trainLoader, validationLoader, testLoader = data(params)

    print('\n==> Preparing input size...')

    (image, target) = trainSet[0]

    # print("\nTarget: ", target)
    # print("\nSize of traget: ", target.size())
    # print("\nType of image returned from data generator: ", type(image))
    # print("\nSize of image returned from data generator: ", image.size())

    # Used for randomResizedCrop
    image = image.permute(2, 1, 0)

    # # Used for tenCrop
    # image = image.permute(2, 1, 0)

    # print("\nSize of image after manipulation: ", image.size())

    input_shape = np.expand_dims(image, axis=0).shape
    print("\nInput shape: ", input_shape)

    # Auto-keras uses default multi-gpus: https://autokeras.com/start/#enable-multi-gpu-training

    # https://github.com/jhfjhfj1/autokeras/blob/master/autokeras/net_module.py  {line 152}
    # https://github.com/jhfjhfj1/autokeras/blob/master/autokeras/search.py

    print('\n==> Preparing NAS searcher with Bayesian Optimization...')
    resnetModule = CnnModule(loss=params['BCEloss'],
                             metric=Accuracy,
                             searcher_args={'generators': ResNetGenerator},  # add more generator in list
                             verbose=True)

    print("\n>>>>>>>>>> Start time: ", datetime.datetime.now())
    resnetModule.fit(n_output_node=params['numClasses'],
                     input_shape=input_shape,
                     train_data=trainLoader,
                     test_data=validationLoader,
                     time_limit=params['timeLimit'])
    print("\n>>>>>>>>>> Stop time: ", datetime.datetime.now())

    resnetModule.save_best_model()

    print('\n==> Predicting output...')
    prediction, targetData = resnetModule.custom_predict(test_loader=testLoader)

    print('\n==> Computing AUC...')
    aurocIndividual = computeAUROC(targetData, prediction, params['numClasses'])
    aurocMean = np.array(aurocIndividual).mean()

    print("AUC ROC for each class: ", aurocIndividual)
    print("AUC ROC mean: ", aurocMean)


def load_model_and_evaluate(params):

    _, _, _, _, testLoader = data(params)

    cudnn.benchmark = True

    model = torch.load('/home/cougarnet.uh.edu/rvsawan3/PycharmProjects/autokeras_chest/best_model/final_model_20190508-15-08-09.h5')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    targetDatas = torch.FloatTensor().cuda()
    outputs = torch.FloatTensor().cuda()

    with torch.no_grad():
        for index, (input, target) in enumerate(testLoader):
            target = target.cuda(non_blocking = True)
            input = input.cuda(non_blocking = True)

            targetDatas = torch.cat((targetDatas, target), 0)
            outputs = torch.cat((outputs, model(input)), 0)

    print('\n==> Computing AUC...')
    aurocIndividual = computeAUROC(targetDatas, outputs, params['numClasses'])
    aurocMean = np.array(aurocIndividual).mean()

    print("AUC ROC for each class: ", aurocIndividual)
    print("AUC ROC mean: ", aurocMean)


if __name__ == '__main__':
    # train(parameters)
    load_model_and_evaluate(parameters)

