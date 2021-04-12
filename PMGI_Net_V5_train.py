import torch
from torch import nn
from torch.autograd import Variable
import os
from utils import saveModel, loadModel, chooseData, writeHistory, writeLog, jigsaw_generator
import time
from models.backbone import resnet_for_pmg
from models.classification_network.PMGI_Net import PMGI
from models.classification_network.PMGI_Net_Extend import PMGI_Extend
from models.classification_network.PMGI_Net_V3 import PMGI_V3
from models.classification_network.PMGI_Net_V3_Extend import PMGI_V3_Extend
from models.classification_network.PMGI_Net_V4 import PMGI_V4

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


class Net(nn.Module):
    def __init__(self, model, CLASS=102):
        super(Net, self).__init__()
        # 选择resnet 除最后一层的全连接，改为CLASS输出
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.pmg = PMGI_V4(model, feature_size=8192, classes_num=CLASS)

    def forward(self, x, train_flag='train'):
        x1, x2, x3, x_concat= self.pmg(x, train_flag)
        return x1, x2, x3, x_concat


def train(modelConfig, dataConfig, logConfig):
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
    epochs = modelConfig['epochs']
    device = modelConfig['device']

    # 数据加载器
    trainLoader = dataConfig['trainLoader']
    validLoader = dataConfig['validLoader']
    trainLength = dataConfig['trainLength']
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

    best_L1_Acc = 0.
    best_L2_Acc = 0.
    best_L3_Acc = 0.
    best_concat_Acc = 0.
    best_com_Acc = 0.
    best_epoch = 0

    for epoch in range(epochs):
        print("Epoch{}/{}".format(epoch, epochs))
        print("-" * 10)

        loss_1, loss_2, loss_3, loss_concat, loss, acc_1, acc_2, acc_3, acc_com \
            = oneEpoch_train(model, trainLoader, optimzer, criterion, device)

        val_loss_1, val_acc_1, val_loss_2, val_acc_2, val_loss_3, val_acc_3, val_loss_concat, val_acc_concat, val_loss_com, val_acc_com\
            = oneEpoch_valid(model, validLoader, criterion, device)


        loss_1 = loss_1 / len(trainLoader)
        loss_2 = loss_2 / len(trainLoader)
        loss_3 = loss_3 / len(trainLoader)
        loss_concat = loss_concat / len(trainLoader)
        loss = loss / len(trainLoader)

        acc_1 = acc_1 / trainLength
        acc_2 = acc_2 / trainLength
        acc_3 = acc_3 / trainLength
        acc_concat = acc_com / trainLength

        val_loss_1 = val_loss_1 / len(validLoader)
        val_loss_2 = val_loss_2 / len(validLoader)
        val_loss_3 = val_loss_3 / len(validLoader)
        val_loss_concat = val_loss_concat / len(validLoader)
        val_loss_com = val_loss_com / len(validLoader)

        val_acc_1 = val_acc_1 / validLength
        val_acc_2 = val_acc_2 / validLength
        val_acc_3 = val_acc_3 / validLength
        val_acc_concat = val_acc_concat / validLength
        val_acc_com = val_acc_com / validLength

        # 模型验证有进步时,保存模型
        if val_acc_1 > best_L1_Acc:
            best_L1_Acc = val_acc_1

        if val_acc_2 > best_L2_Acc:
            best_L2_Acc = val_acc_2

        if val_acc_3 > best_L3_Acc:
            best_L3_Acc = val_acc_3

        if val_acc_concat > best_concat_Acc:
            best_concat_Acc = val_acc_concat

        if val_acc_com > best_com_Acc:
            best_epoch = epoch
            best_com_Acc = val_acc_com
            # saveModel(model,modelPath)

        # 训练日志
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        train_L1_Log = now + " Train L1 loss is :{:.4f},Train accuracy is:{:.4f}%\n".format(loss_1, 100 * acc_1)
        train_L2_Log = now + " Train L2 loss is :{:.4f},Train accuracy is:{:.4f}%\n".format(loss_2, 100 * acc_2)
        train_L3_Log = now + " Train L3 loss is :{:.4f},Train accuracy is:{:.4f}%\n".format(loss_3, 100 * acc_3)
        train_concat_Log = now + " Train concat loss is :{:.4f},Train accuracy is:{:.4f}%\n".format(loss_concat,
                                                                                                    100 * acc_concat)
        train_total_Log = now + " Train total loss is :{:.4f}\n\n".format(loss)

        val_L1_log = now + " Valid L1 loss is :{:.4f},Valid accuracy is:{:.4f}%\n".format(val_loss_1, 100 * val_acc_1)
        val_L2_log = now + " Valid L2 loss is :{:.4f},Valid accuracy is:{:.4f}%\n".format(val_loss_2, 100 * val_acc_2)
        val_L3_log = now + " Valid L3 loss is :{:.4f},Valid accuracy is:{:.4f}%\n".format(val_loss_3, 100 * val_acc_3)
        val_concat_log = now + " Valid concat loss is :{:.4f},Valid accuracy is:{:.4f}%\n".format(val_loss_concat,
                                                                                                  100 * val_acc_concat)
        val_com_log = now + " Valid com loss is :{:.4f},Valid accuracy is:{:.4f}%\n\n".format(val_loss_com,
                                                                                              100 * val_acc_com)

        best_L1_log = now + ' best L1 Acc is {:.4f}%\n'.format(100 * best_L1_Acc)
        best_L2_log = now + ' best L2 Acc is {:.4f}%\n'.format(100 * best_L2_Acc)
        best_L3_log = now + ' best L3 Acc is {:.4f}%\n'.format(100 * best_L3_Acc)
        best_concat_log = now + ' best concat Acc is {:.4f}%\n'.format(100 * best_concat_Acc)
        best_com_log = now + ' best com Acc is {:.4f}%\n'.format(100 * best_com_Acc)
        best_epoch_log = now + ' best Acc epoch is :' + str(best_epoch) + "\n\n"

        train_log = train_L1_Log + train_L2_Log + train_L3_Log + train_concat_Log + train_total_Log
        val_log = val_L1_log + val_L2_log + val_L3_log + val_concat_log + val_com_log
        best_log = best_L1_log + best_L2_log + best_L3_log + best_concat_log + best_com_log + best_epoch_log

        print(train_log + val_log + best_log)


        # 训练历史 每个EPOCH都覆盖一次
        # history = {
        #     'trainLosses':trainLosses,
        #     'trainAcces':trainAcces,
        #     'validLosses':validLosses,
        #     'validAcces':validAcces
        # }

        writeLog(logPath, train_log + val_log + best_log)
        # writeHistory(historyPath,history)

        # 保存最新一次模型
        # saveModel(model,lastModelPath)


def oneEpoch_train(model, dataLoader, optimzer, criterion, device):
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
    loss_1 = 0.
    loss_2 = 0.
    loss_3 = 0.
    loss_concat = 0.
    acc_1 = 0.
    acc_2 = 0.
    acc_3 = 0.
    acc_concat = 0.

    for (inputs, labels) in dataLoader:
        # 使用某个GPU加速图像 label 计算
        inputs, labels = inputs.to(f'cuda:{model.device_ids[0]}'), labels.to(f'cuda:{model.device_ids[0]}')
        inputs, labels = Variable(inputs), Variable(labels)
        inputs_1 = inputs
        inputs_2 = jigsaw_generator(inputs, 2)
        inputs_4 = jigsaw_generator(inputs, 4)


        # 梯度设为零，求前向传播的值
        # step 1
        optimzer.zero_grad()
        output_1, _, _, _ = model(x1=inputs_1, x2=inputs_2, x3=inputs_4, train_flag="train")
        _loss_1 = criterion(output_1, labels)
        _loss_1.backward()
        optimzer.step()

        # step 2
        optimzer.zero_grad()
        _, output_2, _, _ = model(x1=inputs_1, x2=inputs_2, x3=inputs_4, train_flag="train")
        _loss_2 = criterion(output_2, labels)
        _loss_2.backward()
        optimzer.step()

        # step 3
        optimzer.zero_grad()
        _, _, output_3, _ = model(x1=inputs_1, x2=inputs_2, x3=inputs_4, train_flag="train")
        _loss_3 = criterion(output_3, labels)
        _loss_3.backward()
        optimzer.step()

        # step 4
        optimzer.zero_grad()
        _, _, _, output_4 = model(x1=inputs_1, x2=inputs_2, x3=inputs_4, train_flag="train")
        _loss_concat = criterion(output_4, labels) * 2
        _loss_concat.backward()
        optimzer.step()

        _, preds_1 = torch.max(output_1.data, 1)
        _, preds_2 = torch.max(output_2.data, 1)
        _, preds_3 = torch.max(output_3.data, 1)
        _, preds = torch.max(output_4.data, 1)

        loss += (_loss_1.item() + _loss_2.item() + _loss_3.item() + _loss_concat.item())
        loss_1 += _loss_1.item()
        loss_2 += _loss_2.item()
        loss_3 += _loss_3.item()
        loss_concat += _loss_concat.item()

        acc_1 += torch.sum(preds_1 == labels).item()
        acc_2 += torch.sum(preds_2 == labels).item()
        acc_3 += torch.sum(preds_3 == labels).item()
        acc_concat += torch.sum(preds == labels).item()

    return loss_1, loss_2, loss_3, loss_concat, loss, acc_1, acc_2, acc_3, acc_concat


def oneEpoch_valid(model, dataLoader, criterion, device):
    """
    训练一次 或者 验证/测试一次
    :param model: 模型
    :param dataLoader: 数据加载器
    :param criterion: loss计算函数
    :return: loss acc
    """
    with torch.no_grad():
        model.eval()
        loss_1 = 0.
        loss_2 = 0.
        loss_3 = 0.
        loss_concat = 0.
        loss_com = 0.

        acc_1 = 0.
        acc_2 = 0.
        acc_3 = 0.
        acc_concat = 0.
        acc_com = 0.
        for (inputs, labels) in dataLoader:
            inputs, labels = inputs.to(f'cuda:{model.device_ids[0]}'), labels.to(f'cuda:{model.device_ids[0]}')
            inputs, labels = Variable(inputs), Variable(labels)

            outputs1, outputs2, outputs3, outputs_concat = model(x1=inputs, x2=inputs, x3=inputs,train_flag="val")

            outputs_com = outputs1 + outputs2 + outputs3 + outputs_concat

            _loss_1 = criterion(outputs1, labels)
            _loss_2 = criterion(outputs2, labels)
            _loss_3 = criterion(outputs3, labels)
            _loss_concat = criterion(outputs_concat, labels)
            _loss_com = criterion(outputs_com, labels)

            _, preds_1 = torch.max(outputs1.data, 1)
            _, preds_2 = torch.max(outputs2.data, 1)
            _, preds_3 = torch.max(outputs3.data, 1)
            _, preds_concat = torch.max(outputs_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)

            loss_1 += _loss_1.item()
            loss_2 += _loss_2.item()
            loss_3 += _loss_3.item()
            loss_concat += _loss_concat.item()
            loss_com += _loss_com.item()

            acc_1 += torch.sum(preds_1 == labels).item()
            acc_2 += torch.sum(preds_2 == labels).item()
            acc_3 += torch.sum(preds_3 == labels).item()
            acc_concat += torch.sum(preds_concat == labels).item()
            acc_com += torch.sum(predicted_com == labels).item()

    return loss_1, acc_1, loss_2, acc_2, loss_3, acc_3, loss_concat, acc_concat, loss_com, acc_com


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
    print("cuda:0,1,2,3")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = Net(resnet_for_pmg.resnet50(pretrained=True), class_num)
    device_ids = [0, 1, 2 ,3]
    model = nn.DataParallel(model, device_ids=device_ids).cuda(0)
    model.to(f'cuda:{model.device_ids[0]}')
    criterion = torch.nn.CrossEntropyLoss()

    optimzer = torch.optim.SGD([
        {'params': model.module.pmg.features.parameters(), 'lr': lr * 1},
        {'params': model.module.pmg.classifier_concat.parameters(), 'lr': lr * 10},
        {'params': model.module.pmg.classifier1.parameters(), 'lr': lr * 10},
        {'params': model.module.pmg.classifier2.parameters(), 'lr': lr * 10},
        {'params': model.module.pmg.classifier3.parameters(), 'lr': lr * 10},
        {'params': model.module.pmg.conv_block1.parameters(), 'lr': lr * 10},
        {'params': model.module.pmg.conv_block2.parameters(), 'lr': lr * 10},
        {'params': model.module.pmg.conv_block3.parameters(), 'lr': lr * 10},
        # {'params': model.module.pmg.map1.parameters(), 'lr': lr * 10},
        # {'params': model.module.pmg.map2.parameters(), 'lr': lr * 10},
        # {'params': model.module.pmg.fc.parameters(), 'lr': lr * 10},
    ],
        lr=lr, momentum=0.9, weight_decay=5e-4)

    # torch.optim.lr_scheduler.StepLR(optimzer, 10, gamma=0.94, last_epoch=-1)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimzer, T_max=10)
    epochs = 200
    batchSize = 16
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
    # _stanfordCars()
    # _stanfordDogs()
    _CUB200()

# nohup python PMGI_Net_train.py>historys/PMGI_base_resnet_50__in_CUB.log 2>&1 &
