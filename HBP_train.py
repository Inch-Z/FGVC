import torch
from torch import nn
from torch.autograd import Variable
import os
from utils import saveModel,loadModel,chooseData,writeHistory,writeLog, get_parameter_number
import time
from models.classification_network.HBP import HBP
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
def train(modelConfig,dataConfig,logConfig):
    """
    训练
    :param modelConfig: 模型配置
    :param dataConfig: 数据配置
    :param logConfig:  日志配置
    :return:
    """
    # 模型配置
    model = modelConfig['model']
    criterion = modelConfig['criterion']
    optimzer = modelConfig['optimzer']
    epochs =  modelConfig['epochs']
    device = modelConfig['device']

    #数据加载器
    trainLoader = dataConfig['trainLoader']
    validLoader = dataConfig['validLoader']
    trainLength =  dataConfig['trainLength']
    validLength = dataConfig['validLength']

    # 日志及模型保存
    modelPath = logConfig['modelPath']
    historyPath = logConfig['historyPath']
    logPath = logConfig['logPath']
    lastModelPath = logConfig['lastModelPath']


    trainLosses = []
    trainAcces = []
    validLosses = []
    validAcces = []
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('train is starting in ' + now)
    bestAcc = 0.
    best_train_Acc = 0.
    best_epoch = 0

    for epoch in range(epochs):
        print("Epoch{}/{}".format(epoch, epochs))
        print("-" * 10)

        trainLoss, trainAcc = oneEpoch_train(model,trainLoader,optimzer,criterion,device)
        validLoss, validAcc = oneEpoch_valid(model,validLoader,criterion,device)

        trainLoss = trainLoss / len(trainLoader)
        trainAcc =  trainAcc / trainLength
        validLoss = validLoss / len(validLoader)
        validAcc = validAcc / validLength

        # trainLosses.append(trainLoss)
        # trainAcces.append(trainAcc)
        #
        # validLosses.append(validLoss)
        # validAcces.append(validAcc)
        # 模型验证有进步时,保存模型
        if validAcc > bestAcc:
            bestAcc = validAcc
            best_train_Acc = trainAcc
            best_epoch = epoch
            # saveModel(model,modelPath)

        # 训练日志
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        trainLog = now + " Train loss is :{:.4f},Train accuracy is:{:.4f}%\n".format(trainLoss, 100 * trainAcc)
        validLog = now + " Valid loss is :{:.4f},Valid accuracy is:{:.4f}%\n".format(validLoss, 100 * validAcc)
        best_val_log = now + ' best val Acc is {:.4f}%\n'.format(100 * bestAcc)
        best_train_log = now + ' best train Acc is {:.4f}%\n'.format(100 * best_train_Acc)
        best_epoch_log = now + ' bestAcc is : ' + str(best_epoch)
        log = trainLog + validLog + best_train_log + best_val_log + best_epoch_log

        print(log)

        # 训练历史 每个EPOCH都覆盖一次
        # history = {
        #     'trainLosses':trainLosses,
        #     'trainAcces':trainAcces,
        #     'validLosses':validLosses,
        #     'validAcces':validAcces
        # }

        # writeLog(logPath,log)
        # writeHistory(historyPath,history)

        # 保存最新一次模型
        # saveModel(model,lastModelPath)

def oneEpoch_train(model,dataLoader,optimzer,criterion,device):
    """
    训练一次 或者 验证/测试一次
    :param model: 模型
    :param dataLoader: 数据加载器
    :param optimzer: 优化器
    :param criterion: loss计算函数
    :return: loss acc
    """
    # 模式

    model.train()
    loss = 0.
    acc = 0.
    for (inputs, labels) in dataLoader:
        # 使用某个GPU加速图像 label 计算
        inputs, labels = inputs.to(f'cuda:{model.device_ids[0]}'), labels.to(f'cuda:{model.device_ids[0]}')
        inputs, labels = Variable(inputs), Variable(labels)

        # 梯度设为零，求前向传播的值
        optimzer.zero_grad()
        outputs = model(inputs)
        _loss = criterion(outputs, labels)

        # 反向传播
        _loss.backward()
        # 更新网络参数
        optimzer.step()

        _, preds = torch.max(outputs.data, 1)
        loss += _loss.item()
        acc += torch.sum(preds == labels).item()

    return loss,acc

def oneEpoch_valid(model,dataLoader,criterion,device):
    """
    训练一次 或者 验证/测试一次
    :param model: 模型
    :param dataLoader: 数据加载器
    :param criterion: loss计算函数
    :return: loss acc
    """
    with torch.no_grad():
        model.eval()
        loss = 0.
        acc = 0.
        for (inputs, labels) in dataLoader:
            inputs, labels = inputs.to(f'cuda:{model.device_ids[0]}'), labels.to(f'cuda:{model.device_ids[0]}')
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            _loss = criterion(outputs, labels)


            _, preds = torch.max(outputs.data, 1)
            loss += _loss.item()
            acc += torch.sum(preds == labels).item()

    return loss,acc


def _CUB200():
    """
    CUB200数据集
    :return:
    """
    # 定义模型 定义评价 优化器等
    lr = 1e-4
    class_num = 200
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    print("cuda:1,2,3")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = HBP()
    device_ids = [0, 1, 2]
    model = nn.DataParallel(model, device_ids=device_ids).cuda(0)
    model.to(f'cuda:{model.device_ids[0]}')
    criterion = torch.nn.CrossEntropyLoss()

    optimzer = torch.optim.SGD([
        {'params': model.parameters(), 'lr': lr * 1},
    ],
        lr=lr, momentum=0.9, weight_decay=5e-4)

    # torch.optim.lr_scheduler.StepLR(optimzer, 10, gamma=0.94, last_epoch=-1)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max=10)
    epochs = 200
    batchSize = 4
    worker = 4
    modelConfig = {
        'model': model,
        'criterion': criterion,
        'optimzer': optimzer,
        'epochs': epochs,
        'device': device
    }

    from torchvision import transforms as T
    # 自定义数据增强方式
    trainTransforms = T.Compose([
        T.Resize(550),
        T.RandomCrop(448, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testTransforms = T.Compose([
        T.Resize(550),
        T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('CUB200', batchSize, worker,
                                                                                            trainTransforms,
                                                                                            testTransforms)

    # 没有验证集，所以使用测试集来做验证集
    dataConfig = {
        'trainLoader': trainLoader,
        'validLoader': testLoader,
        'trainLength': trainLength,
        'validLength': testLength
    }

    modelPath = os.path.join(os.getcwd(), 'checkpoints', '_CUB200.pth')
    lastModelPath = os.path.join(os.getcwd(), 'checkpoints', '_CUB200_last.pth')
    historyPath = os.path.join(os.getcwd(), 'historys', '_CUB200.npy')
    logPath = os.path.join(os.getcwd(), 'logs', '_CUB200.txt')

    logConfig = {
        'modelPath': modelPath,
        'historyPath': historyPath,
        'logPath': logPath,
        'lastModelPath': lastModelPath
    }

    train(modelConfig, dataConfig, logConfig)


if __name__ == '__main__':
    print(torch.__version__)
    _CUB200()