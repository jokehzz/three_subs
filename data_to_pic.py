import numpy as np, os, sys, joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from helper_code import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer

#主要负责将数据转化为图片,header里面存储了标签类信息，recording里面存储了原始数据，leads获取的信息数量，pir_dir是用于存储图片的路径
#主要目的是生成图片以及得到对应的one-hot编码label，图片是存储到目录中，独立热编码后的label是存储到json中
def get_train_data(header_files,recording_files,leads,pic_dir,label_path):
    label_list = list()
    for i in range(len(recording_files)):
        #得到标签和数据，分别是str格式和数组格式
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])

        available_leads = get_leads(header)
        #获取其中的导联序号(下标)
        indices = list()
        for lead in leads:
            index_tem = available_leads.index(lead)
            indices.append(index_tem)
        #recording是根据导联序号防护的数据
        recording = recording[indices,:]
        imputer = SimpleImputer().fit(recording)
        recording_transform = imputer.transform(recording)
        datas = data_truncation(recording_transform)

        if len(datas.shape) == 1:
            datas = np.array([datas])

        fig = plt.figure()
        for j in range(len(datas)):
            plt.plot(range(datas.shape[1]),datas[j])
        filename = os.path.join(pic_dir,str(i)+'.jpg')
        plt.savefig(filename)
        plt.close(fig)

        label = get_labels(header)
        label_list.append(label)
    mlb = MultiLabelBinarizer()
    label_one_hot = mlb.fit_transform(label_list)
    classes = mlb.classes_
    label_dic = dict()
    for i in range(len(recording_files)):
        label_dic[str(i)] = label_one_hot[i]
    df = pd.DataFrame(label_dic)
    df.to_csv(label_path,index=False,encoding='utf-8')
    return classes

#主要是对图片进行截断处理，使得输入的维度都一样
def data_truncation(recording_transform):
    datas = np.array(recording_transform)
    dimension = 5000
    
    if len(datas.shape) == 1:
        datas = np.array([datas])

    if datas.shape[1]>=dimension:
        result_data = datas[:,:dimension]
    else:
        result_data = list()
        for data in datas:
            if len(data) >= 5000:
                sub_data = data[:5000]
            else:
                sub_data = np.pad(data,(0,5000-len(data)),'constant')
            result_data.append(sub_data)
    return np.array(result_data)

def get_pic_test_data(header,recording,leads,test_sign):
    available_leads = get_leads(header)
    # 获取其中的导联序号(下标)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    # recording是根据导联序号的数据
    recording = recording[indices, :]
    imputer = SimpleImputer().fit(recording)
    recording_transform = imputer.transform(recording)
    datas = data_truncation(recording_transform)

    if len(datas.shape) == 1:
        datas = np.array([datas])

    pic_test_path = 'test_pic_data'
    if not os.path.isdir(pic_test_path):
        os.mkdir(pic_test_path)
    fig = plt.figure()
    for j in range(len(datas)):
        plt.plot(range(datas.shape[1]), datas[j])
    filename = os.path.join(pic_test_path,'tem'+str(test_sign)+'.jpg')
    plt.savefig(filename)
    plt.close(fig)


def main():
    pass

if __name__ == '__main__':
    main()

