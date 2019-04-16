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
from autokeras.nn.generator import CnnGenerator
from autokeras.nn.model_trainer import ModelTrainer
from torch.utils.data.sampler import SubsetRandomSampler

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

if __name__ == '__main__':

    # train_on_gpu = torch.cuda.is_available()

    currentDirectory = os.getcwd()
    pathTrainData = os.path.join(currentDirectory, 'dataset/train/')
    pathTestData = os.path.join(currentDirectory, 'dataset/test/')

    pathLabelsTrain = os.path.join(currentDirectory, 'dataset/train_w_labels.txt')
    pathLabelsTest = os.path.join(currentDirectory, 'dataset/test_w_labels.txt')

    tempModelPath = os.path.join(currentDirectory, 'models/')

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    transResize = 256
    imgTransCrop = 224
    batchSize = 32
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

    # transform_train = []
    # transform_train.append(transforms.Resize(transResize))
    # transform_train.append(transforms.TenCrop(imgTransCrop))
    # transform_train.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
    # transform_train.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
    # transformTrain = transforms.Compose(transform_train)

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



    # ################################ for TenCrop ################################
    # temp = torch.FloatTensor().cuda()
    #
    # with torch.no_grad():
    #     for i, (input, target) in enumerate(trainLoader):
    #         target = target.cuda(non_blocking = True)
    #         input = input.cuda(non_blocking = True)
    #         print("\n", i)
    #         print(input.size())
    #         print(target.size())
    #         bs, ncrops, c, h, w = input.size()
    #         result = input.view(-1, c, h, w)  # fuse batch size and ncrops
    #         print("result: ", result.size())
    #         result_avg = result.view(bs, ncrops, -1).mean(1)  # avg over crops
    #         print("result_avg: ", result_avg.size())
    #
    #         temp = torch.cat((temp, result_avg.data), 0)
    #         print("Size of temp: ", temp.size())




    # Auto-keras uses default multi-gpus: https://autokeras.com/start/#enable-multi-gpu-training

    '''print('#-------------------- ResNet-101 --------------------#')

    model = models.resnet101(pretrained = False)

    resnet101 = torch.nn.DataParallel(model).cuda()

    model_trainer = ModelTrainer(model = resnet101,
                                 path = tempModelPath,
                                 loss_function = BCEloss,
                                 metric = Accuracy,
                                 train_data = trainLoader,
                                 test_data = testLoader,
                                 verbose = True)

    loss_values, metric_value = model_trainer.train_model(max_iter_num = 100,
                                                          max_no_improvement_num = None,
                                                          timeout = None)

    print("\nLoss values: ", loss_values)
    print("\nMetric values: ", metric_value)'''

    print('#-------------------- NAS Searcher --------------------#')

    # https://github.com/jhfjhfj1/autokeras/blob/master/autokeras/net_module.py  {line 152}
    # https://github.com/jhfjhfj1/autokeras/blob/master/autokeras/search.py

    print('\n==> Preparing NAS searcher with Greedy search...')
    cnnModule = CnnModule(loss = BCEloss,
                          metric = Accuracy,
                          searcher_args = {'generators': CnnGenerator,
                                           'path': 'models/autokeras-CnnGenerator-greedy-searcher.h5'},
                          verbose = True,
                          search_type = GreedySearcher)

    print('\n==> Started NAS searcher...')
    print("\nTime search started: ", datetime.datetime.now())
    cnnModule.fit(n_output_node = num_classes,
                  input_shape = input_shape,
                  train_data = trainLoader,
                  test_data = validationLoader,
                  time_limit = 2 * 60 * 60)

    print('\n==> Final training (not initialized weights) after finding the best neural architecture...')
    cnnModule.final_fit(train_data = trainLoader,
                        test_data = testLoader,
                        # do not keep trainer_args None
                        # Exception occurred at train(): train_model() argument after ** must be a mapping, not NoneType
                        trainer_args = {'max_iter_num': 200,
                                        'max_no_improvement_num': 5},
                        retrain = False)

    best_graph = cnnModule.best_model()
    cnnModule.save_model(trainer_args = {'max_no_improvement_num': 5})

    # prediction = cnnModule.predict(test_loader = testLoader)