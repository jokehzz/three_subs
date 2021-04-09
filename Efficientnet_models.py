import os
import time
import torch
import numpy as np
import pandas as pd
import warnings
from sklearn.utils import shuffle
from PIL import Image
from torch import nn
from sklearn.metrics import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms, models
from torch import optim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)
warnings.filterwarnings('ignore')
use_gpu = torch.cuda.is_available()

class my_data_set():
    def __init__(self,pic_path,label_path,input_size,sign):
        super(my_data_set, self).__init__()
        self.pic_path = pic_path
        self.label_path = label_path
        self.input_size = input_size
        self.sign = sign

    def get_data(self):
        pic_list = os.listdir(self.pic_path)
        label_dic = pd.read_csv(self.label_path).to_dict(orient="list")
        datas = list()
        targets = list()
        print("读取数据！")
        for pic in pic_list:
            print("读入的数据为{}".format(pic))
            pic_name = os.path.join(self.pic_path, pic)
            img = Image.open(pic_name).convert("RGB")
            label = label_dic[pic.split(".")[0]]
            image, label_result = self.transform(img,label,self.sign)
            datas.append(image)
            targets.append(label_result)
        return np.array(datas), np.array(targets)

    # 对输入的图像进行裁剪放缩等，以及将label转化为Tensor并且返回
    def transform(self, img, label, sign):
        if sign == 'train':
            trans = transforms.Compose([transforms.Resize(self.input_size),
                                        transforms.CenterCrop(self.input_size),
                                        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            trans = transforms.Compose([transforms.Resize(self.input_size),
                                        transforms.CenterCrop(self.input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img_tran = trans(img)
        label = np.array(label)
        label = [float(i) for i in label]
        # 需要将label转化为float类型的，因为BCELoss只接受float类型的
        # label = torch.FloatTensor(label)
        return img_tran.numpy(),label

def train_model(data_type,model_directory,model,criterion,optimizer,scheduler,data_loader,data_length,num_epochs=20):
    Sigmoid_fun = nn.Sigmoid()
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print("-" * 30)
        # 每训练一个epoch，验证一下网络模型
        running_loss = 0.0
        # 统计自定义的准确率，召回率数值
        running_precision = 0.0
        running_recall = 0.0
        # 利用sklearn框架计算precision、recall、f1_score
        running_pre_score = 0.0
        running_rec_score = 0.0
        running_fscore = 0.0

        batch_num = 0
        # 学习率更新的模式
        # scheduler.step()
        model.train()
        for data in data_loader:
            inputs, labels = data
            print(labels.numpy().shape)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            # 梯度清零
            optimizer.zero_grad()
            # 网络前向传播
            outputs = model(inputs)
            # 计算Loss值
            # print(outputs)
            # print(type(outputs))
            loss = criterion(Sigmoid_fun(outputs.float()), labels.float())
            # 根据需求选择模型预测结果准确率的函数
            precision, recall, pre_score, rec_score, fscore = calculate_accyracy_mode_one(
                Sigmoid_fun(outputs.to(torch.float32)), labels.float())
            # 将返回5个值，其中分别是基于自定义的precision、recall，sklearn框架下的pre_score、rec_score、fscore
            running_precision += precision
            running_recall += recall

            running_pre_score += pre_score
            running_rec_score += rec_score
            running_fscore += fscore

            batch_num += 1
            # 方向梯度
            loss.backward()
            # 更新权重
            optimizer.step()
            # 计算一个epoch的Loss值与准确率
            running_loss += loss.item() * inputs.size(0)
        # 按照提示将学习率更新模式放在权重更新之后
        # scheduler.step()
        epoch_loss = running_loss / data_length
        print("epoch_loss", epoch_loss)
        scheduler.step(epoch_loss)
        epoch_precision = running_precision / batch_num
        print("epoch_precision", epoch_precision)
        epoch_recall = running_recall / batch_num
        print("epoch_recall", epoch_recall)

        epoch_pre_score = running_pre_score / batch_num
        print("epoch_pre_score", epoch_pre_score)
        epoch_rec_score = running_rec_score / batch_num
        print("epoch_rec_score", epoch_rec_score)
        epoch_fscore = running_fscore / batch_num
        print("epcch_f_score", epoch_fscore)
    time_end = time.time() - since
    print(time_end)
    print("保存模型！")
    if data_type == '12':
        save_path = os.path.join(model_directory,'Efficientnet_b0_model_12.pkl')
        torch.save(model,save_path)
    elif data_type == '6':
        save_path = os.path.join(model_directory,'Efficientnet_b0_model_6.pkl')
        torch.save(model,save_path)

# 设定一个阈值，大于阈值则判断属于这类
def calculate_accyracy_mode_one(model_pred, labels):
    # model_pred是经过sigmoid处理的，sigmoid处理之后㐓视为这一类的概率
    # accuracy为阈值
    accuracy = 0.5
    # 判断概率是否大于阈值
    pred_result = model_pred > accuracy
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0, 0, 0, 0
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    # 模型预测结果中有多少个是正确的
    precision = true_predict_num / pred_one_num
    # 模型预测正确的结果中，占真实标签的数量
    recall = true_predict_num / target_one_num

    # 运用sklearn中已经封装好的方法进行计算
    pre_score = precision_score(labels.cuda().data.cpu().numpy(), pred_result.cpu().numpy(), average="samples")
    rec_score = recall_score(labels.cuda().data.cpu().numpy(), pred_result.cpu().numpy(), average="samples")
    f_score = f1_score(labels.cuda().data.cpu().numpy(), pred_result.cpu().numpy(), average="samples")
    return precision.item(), recall.item(), pre_score, rec_score, f_score


def eff_train(num_class,model_directory,pic_path,label_path,data_type):
    #input_size = 380
    input_size = 450
    # 输出类别数
    #num_class = 27
    # 批处理尺寸
    batch_size = 10
    # 训练多少个批次
    epochs = 45
    # 如果是ture则特征提取，False代表是微调
    feature_extract = False
    # 学习率
    LR = 1e-3
    # pic_path = 'pic(no_time_step_RGB)'
    # label_path = 'label_one_hot.csv'
    sign = 'train'
    print(num_class)
    datas, labels = my_data_set(pic_path,label_path,input_size,sign).get_data()
    datasets = TensorDataset(torch.from_numpy(datas), torch.from_numpy(labels))
    data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True,num_workers=1)

    model = EfficientNet.from_pretrained('efficientnet-b0')
    feature = model._fc.in_features
    model._fc = nn.Linear(in_features=feature, out_features=num_class, bias=True)
    if use_gpu:
        model = model.cuda()
    #model = torch.nn.DataParallel(model)
    # 获取神经网络各个部分参数
    params_to_data = model.parameters()
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(params_to_data, lr=LR, betas=(0.9, 0.999), eps=1e-9)
    # 由于选用的指标是epochloss，所以mode选择min，这样当loss不再降低时，就降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)
    print("模型训练！")
    train_model(data_type,model_directory,model,criterion,optimizer,scheduler,data_loader,len(datas),epochs)

def main():
    pass

if __name__ == '__main__':
    main()
