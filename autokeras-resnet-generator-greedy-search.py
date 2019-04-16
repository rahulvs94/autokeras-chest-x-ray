import os
import torch
import datetime
import numpy as np
from autokeras import CnnModule
import torchvision.models as models
from nas.greedy import GreedySearcher
from torch.utils.data import DataLoader
from autokeras.nn.metric import Accuracy
import torchvision.transforms as transforms
from DatasetGenerator import DatasetGenerator
from autokeras.nn.generator import ResNetGenerator
from autokeras.nn.model_trainer import ModelTrainer
from torch.utils.data.sampler import SubsetRandomSampler

# Paper for reference: https://pubs.rsna.org/doi/10.1148/radiol.2017162326

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

if __name__ == '__main__':

    currentDirectory = os.getcwd()
    # pathTrainData = os.path.join(currentDirectory, 'small dataset/small_train/')
    # pathTestData = os.path.join(currentDirectory, 'small dataset/small_test/')
    #
    # pathLabelsTrain = os.path.join(currentDirectory, 'small dataset/small_train_w_labels.txt')
    # pathLabelsTest = os.path.join(currentDirectory, 'small dataset/small_test_w_labels.txt')

    pathTrainData = os.path.join(currentDirectory, 'train/')
    pathTestData = os.path.join(currentDirectory, 'test/')

    pathLabelsTrain = os.path.join(currentDirectory, 'train_w_labels.txt')
    pathLabelsTest = os.path.join(currentDirectory, 'test_w_labels.txt')

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    transResize = 256
    imgTransCrop = 224
    batchSize = 48
    numberOfWorkers = 24
    num_classes = 14
    validSize = 0.15

    BCEloss = torch.nn.BCEWithLogitsLoss(reduction = 'mean')

    print('#-------------------- Directory paths --------------------#')
    print('Current directory: ', currentDirectory)
    print('train directory: ', pathTrainData)
    print('test directory: ', pathTestData)

    print('\n==> Preparing data augmentation...')

    transform_train = []
    transform_train.append(transforms.RandomResizedCrop(size = imgTransCrop,
                                                        ratio = (0.5, 1)))
    transform_train.append(transforms.ToTensor())
    transform_train.append(normalize)
    transformTrain = transforms.Compose(transform_train)

    transform_test = []
    transform_test.append(transforms.Resize(transResize))
    transform_test.append(transforms.ToTensor())
    transform_test.append(normalize)
    transformTest = transforms.Compose(transform_test)

    print('\n==> Preparing train dataset generator...')

    trainSet = DatasetGenerator(
        pathImageDirectory = pathTrainData,
        pathDatasetFile = pathLabelsTrain,
        transform = transformTrain
    )

    print('\n==> Preparing test dataset generator...')

    testSet = DatasetGenerator(
        pathImageDirectory = pathTestData,
        pathDatasetFile = pathLabelsTest,
        transform = transformTest
    )

    print('\n==> Preparing validation set...')

    numberOfTrain = len(trainSet)
    indices = list(range(numberOfTrain))
    np.random.shuffle(indices)
    split = int(np.floor(validSize * numberOfTrain))
    train_idx, valid_idx = indices[split:], indices[:split]
    print('\n15% validation set created...')

    # https://pytorch.org/docs/stable/data.html
    print('\n==> Preparing train and validation sampler...')

    trainSampler = SubsetRandomSampler(train_idx)
    validSampler = SubsetRandomSampler(valid_idx)

    print('\n==> Preparing train dataloader...')

    trainLoader = DataLoader(
        dataset = trainSet,
        batch_size = batchSize,
        sampler = trainSampler,
        num_workers = numberOfWorkers,
        pin_memory = True
    )

    print("len of trainLoader: ", len(trainLoader))

    print('\n==> Preparing validation dataloader...')

    validationLoader = DataLoader(
        dataset = trainSet,
        batch_size = batchSize,
        sampler = validSampler,
        shuffle = False,
        num_workers = numberOfWorkers,
        pin_memory = True
    )

    print('\n==> Preparing test dataloader...')

    testLoader = DataLoader(
        dataset = testSet,
        batch_size = batchSize,
        shuffle = False,
        num_workers = numberOfWorkers,
        pin_memory = True
    )

    print('\n==> Preparing input size...')

    (image, target) = trainSet[0]

    print("\nTarget: ", target)
    print("\nSize of traget: ", target.size())
    print("\nType of image returned from data generator: ", type(image))
    print("\nSize of image returned from data generator: ", image.size())

    # Used for randomResizedCrop
    image = image.permute(2, 1, 0)

    # # Used for tenCrop
    # image = image.permute(2, 1, 0)

    print("\nSize of image after manipulation: ", image.size())

    input_shape = np.expand_dims(image, axis = 0).shape

    #input_shape = np.array(image).shape
    print("\nInput shape: ", input_shape)

    # Auto-keras uses default multi-gpus: https://autokeras.com/start/#enable-multi-gpu-training

    print('#-------------------- NAS Searcher --------------------#')

    # https://github.com/jhfjhfj1/autokeras/blob/master/autokeras/net_module.py  {line 152}
    # https://github.com/jhfjhfj1/autokeras/blob/master/autokeras/search.py

    print('\n==> Preparing NAS searcher with Greedy search...')
    resnetModule = CnnModule(loss = BCEloss,
                          metric = Accuracy,
                          searcher_args = {'generators': ResNetGenerator},
                          verbose = True,
                          search_type = GreedySearcher)

    print('\n==> Started NAS searcher...')
    print("\nTime search started: ", datetime.datetime.now())
    resnetModule.fit(n_output_node = num_classes,
                  input_shape = input_shape,
                  train_data = trainLoader,
                  test_data = validationLoader,
                  time_limit = 15 * 60 * 60)

    '''print('\n==> Final training (not initialized weights) after finding the best neural architecture...')
    resnetModule.final_fit(train_data = trainLoader,
                        test_data = testLoader,
                        # do not keep trainer_args None
                        # Exception occurred at train(): train_model() argument after ** must be a mapping, not NoneType
                        trainer_args = {'max_iter_num': 200,
                                        'max_no_improvement_num': 5},
                        retrain = False)'''

    resnetModule.save_model()

    # prediction = resnetModule.predict(test_loader = testLoader)