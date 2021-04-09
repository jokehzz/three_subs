import torch
from helper_code import *
from extra_models import *
import numpy as np, os, sys, joblib
from PIL import Image
from torch import nn
from sklearn.impute import SimpleImputer
from Efficientnet_models import *
from data_to_pic import *
from sklearn.preprocessing import MultiLabelBinarizer

twelve_lead_model_filename = '12_lead_model.sav'
six_lead_model_filename = '6_lead_model.sav'
three_lead_model_filename = '3_lead_model.sav'
two_lead_model_filename = '2_lead_model.sav'
pic_path1 = 'pic_data_12'
pic_path2 = 'pic_data_6'
label_path = 'label_one_hot.csv'
################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)
    if not os.path.isdir(pic_path1):
        os.mkdir(pic_path1)
    if not os.path.isdir(pic_path2):
        os.mkdir(pic_path2)

    # Extract classes from dataset.
    print('Extracting classes...')

    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)

    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    # data = np.zeros((num_recordings, 14), dtype=np.float32) # 14 features: one feature for each lead, one feature for age, and one feature for sex
    # labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes

    #datas与labels都是numpy数组形式的，labels是经过独立热编码的
    datas,labels,classes,mlb = get_features(header_files, recording_files, twelve_leads)

    t = list()
    for data in datas:
        imputer = SimpleImputer().fit(data)
        data = imputer.transform(data)
        if data.shape != (12,502):
            if len(data.shape) == 1:
                data = np.pad(data,(0,502-len(data)),'constant')
                data = np.repeat(np.array([data]),12,axis=0)
            else:
                data = np.random.rand(12,502)
        t.append(data)
    datas = np.array(t)
    #获取各个导联上的数据，返回的是一个列表，12个导联上的数据都存储在data_list中
    data_list = get_data_for_every_daolian(datas)
    trains(data_list,labels,model_directory,classes,imputer,mlb)

    #上述代码是训练纯粹的树模型的，下面的代码是训练神经网络模型的
    classes1 = get_train_data(header_files, recording_files, twelve_leads,pic_path1,label_path)
    classes2 = get_train_data(header_files, recording_files, six_leads, pic_path2, label_path)
    eff_train(len(classes1),model_directory,pic_path1,label_path,'12')
    eff_train(len(classes1),model_directory,pic_path1,label_path,'6')

    # Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')


################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
    joblib.dump(d, filename, protocol=0)

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    # return load_model(filename)
    return 'twelve_'+model_directory

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    # return load_model(filename)
    return 'six_'+model_directory

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    # return load_model(filename)
    return 'three_'+model_directory

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    # return load_model(filename)
    return 'two_'+model_directory

# Generic function for loading a model.
def load_model(filename):
    return joblib.load(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(sign, header, recording):
    model_dir = sign.split('_')[1]
    model_name1 = os.path.join(model_dir,'extra_model1.pkl')
    model_name2 = os.path.join(model_dir,'extra_model2.pkl')
    model_name3 = os.path.join(model_dir,'extra_model3.pkl')
    model_name4 = os.path.join(model_dir,'extra_model4.pkl')
    model_name5 = os.path.join(model_dir,'extra_model5.pkl')
    model_name6 = os.path.join(model_dir,'extra_model6.pkl')
    model_name7 = os.path.join(model_dir,'extra_model7.pkl')
    model_name8 = os.path.join(model_dir,'extra_model8.pkl')
    model_name9 = os.path.join(model_dir,'extra_model9.pkl')
    model_name10 = os.path.join(model_dir,'extra_model10.pkl')
    model_name11 = os.path.join(model_dir,'extra_model11.pkl')
    model_name12 = os.path.join(model_dir,'extra_model12.pkl')
    model1 = joblib.load(model_name1)
    model2 = joblib.load(model_name2)
    model3 = joblib.load(model_name3)
    model4 = joblib.load(model_name4)
    model5 = joblib.load(model_name5)
    model6 = joblib.load(model_name6)
    model7 = joblib.load(model_name7)
    model8 = joblib.load(model_name8)
    model9 = joblib.load(model_name9)
    model10 = joblib.load(model_name10)
    model11 = joblib.load(model_name11)
    model12 = joblib.load(model_name12)
    classes = model1['classes']
    imputer = model1['imputer']
    mlb = model1['mlb']
    classifier1 = model1['classifier']
    classifier2 = model2['classifier']
    classifier3 = model3['classifier']
    classifier4 = model4['classifier']
    classifier5 = model5['classifier']
    classifier6 = model6['classifier']
    classifier7 = model7['classifier']
    classifier8 = model8['classifier']
    classifier9 = model9['classifier']
    classifier10 = model10['classifier']
    classifier11 = model11['classifier']
    classifier12 = model12['classifier']
    datas = get_test_data(header, recording, twelve_leads)
    datas = imputer.transform(datas)
    proba1 = get_proba_by_dimension1(classifier1.predict_proba([datas[0]]))
    proba2 = get_proba_by_dimension1(classifier2.predict_proba([datas[1]]))
    proba3 = get_proba_by_dimension1(classifier3.predict_proba([datas[2]]))
    proba4 = get_proba_by_dimension1(classifier4.predict_proba([datas[3]]))
    proba5 = get_proba_by_dimension1(classifier5.predict_proba([datas[4]]))
    proba6 = get_proba_by_dimension1(classifier6.predict_proba([datas[5]]))
    proba7 = get_proba_by_dimension1(classifier7.predict_proba([datas[6]]))
    proba8 = get_proba_by_dimension1(classifier8.predict_proba([datas[7]]))
    proba9 = get_proba_by_dimension1(classifier9.predict_proba([datas[8]]))
    proba10 = get_proba_by_dimension1(classifier10.predict_proba([datas[9]]))
    proba11 = get_proba_by_dimension1(classifier11.predict_proba([datas[10]]))
    proba12 = get_proba_by_dimension1(classifier12.predict_proba([datas[11]]))
    pro_mean = np.mean([proba1,proba2,proba3,proba4,proba5,proba6,proba7,proba8,proba9,proba10,proba11,proba12],axis=0)

    #神经网络模型名称
    eff_model12_name = os.path.join(model_dir,'Efficientnet_b0_model_12.pkl')
    eff_model12 = torch.load(eff_model12_name)
    #将数据写入文件中
    get_pic_test_data(header,recording,twelve_leads,'12')
    pic_name = os.path.join('test_pic_data','tem12.jpg')

    img = Image.open(pic_name).convert("RGB")
    trans = transforms.Compose([transforms.Resize(450),
                                transforms.CenterCrop(450),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_train = trans(img)
    img_trains = img_train.numpy()
    Sigmoid_fun = nn.Sigmoid()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn_inputs = torch.from_numpy(np.array([img_trains]))
    input_value = nn_inputs.to(device)
    outputs = eff_model12(input_value)
    pred = Sigmoid_fun(outputs.to(torch.float32))
    predicts = pred.cuda().data.cpu().numpy()
    final_predicts = np.array(predicts[0])+np.array(pro_mean)*1.3
    labels = [1 if i>=0.35 else 0 for i in final_predicts]
    return classes,labels,final_predicts


# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(sign, header, recording):
    model_dir = sign.split('_')[1]
    model_number = [twelve_leads.index(i) for i in six_leads]
    model_name1 = os.path.join(model_dir, 'extra_model'+str(model_number[0]+1)+'.pkl')
    model_name2 = os.path.join(model_dir, 'extra_model'+str(model_number[1]+1)+'.pkl')
    model_name3 = os.path.join(model_dir, 'extra_model'+str(model_number[2]+1)+'.pkl')
    model_name4 = os.path.join(model_dir, 'extra_model'+str(model_number[3]+1)+'.pkl')
    model_name5 = os.path.join(model_dir, 'extra_model'+str(model_number[4]+1)+'.pkl')
    model_name6 = os.path.join(model_dir, 'extra_model'+str(model_number[5]+1)+'.pkl')
    model1 = joblib.load(model_name1)
    model2 = joblib.load(model_name2)
    model3 = joblib.load(model_name3)
    model4 = joblib.load(model_name4)
    model5 = joblib.load(model_name5)
    model6 = joblib.load(model_name6)
    classes = model1['classes']
    imputer = model1['imputer']
    mlb = model1['mlb']
    classifier1 = model1['classifier']
    classifier2 = model2['classifier']
    classifier3 = model3['classifier']
    classifier4 = model4['classifier']
    classifier5 = model5['classifier']
    classifier6 = model6['classifier']
    # datas,labels,classes = get_features(header, recording, twelve_leads)
    datas = get_test_data(header, recording, six_leads)
    datas = imputer.transform(datas)
    proba1 = np.array(get_proba_by_dimension1(classifier1.predict_proba([datas[0]])))
    proba2 = np.array(get_proba_by_dimension1(classifier2.predict_proba([datas[1]])))
    proba3 = np.array(get_proba_by_dimension1(classifier3.predict_proba([datas[2]])))
    proba4 = np.array(get_proba_by_dimension1(classifier4.predict_proba([datas[3]])))
    proba5 = np.array(get_proba_by_dimension1(classifier5.predict_proba([datas[4]])))
    proba6 = np.array(get_proba_by_dimension1(classifier6.predict_proba([datas[5]])))
    probas = proba1+proba2+proba3+proba4+proba5+proba6
    pro_mean = np.mean([proba1, proba2, proba3, proba4, proba5, proba6], axis=0)


    # 神经网络模型名称
    eff_model6_name = os.path.join(model_dir, 'Efficientnet_b0_model_6.pkl')
    eff_model6 = torch.load(eff_model6_name)
    # 将数据写入文件中
    get_pic_test_data(header, recording, twelve_leads,'6')
    pic_name = os.path.join('test_pic_data', 'tem6.jpg')
    img = Image.open(pic_name).convert("RGB")
    trans = transforms.Compose([transforms.Resize(450),
                                transforms.CenterCrop(450),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_train = trans(img)
    img_trains = img_train.numpy()
    Sigmoid_fun = nn.Sigmoid()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn_inputs = torch.from_numpy(np.array([img_trains]))
    input_value = nn_inputs.to(device)
    outputs = eff_model6(input_value)
    pred = Sigmoid_fun(outputs.to(torch.float32))
    predicts = pred.cuda().data.cpu().numpy()
    final_predicts = np.array(predicts[0])+np.array(pro_mean)*1.2
    labels = [1 if i>=0.35 else 0 for i in final_predicts]
    return classes,labels,final_predicts

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(sign, header, recording):
    model_dir = sign.split('_')[1]
    model_number = [twelve_leads.index(i) for i in three_leads]
    model_name1 = os.path.join(model_dir, 'extra_model' + str(model_number[0]+1)+'.pkl')
    model_name2 = os.path.join(model_dir, 'extra_model' + str(model_number[1]+1)+'.pkl')
    model_name3 = os.path.join(model_dir, 'extra_model' + str(model_number[2]+1)+'.pkl')
    model1 = joblib.load(model_name1)
    model2 = joblib.load(model_name2)
    model3 = joblib.load(model_name3)
    classes = model1['classes']
    imputer = model1['imputer']
    mlb = model1['mlb']
    classifier1 = model1['classifier']
    classifier2 = model2['classifier']
    classifier3 = model3['classifier']

    datas = get_test_data(header,recording,three_leads)
    datas = imputer.transform(datas)

    proba1 = np.array(get_proba_by_dimension1(classifier1.predict_proba([datas[0]])))
    proba2 = np.array(get_proba_by_dimension1(classifier2.predict_proba([datas[1]])))
    proba3 = np.array(get_proba_by_dimension1(classifier3.predict_proba([datas[2]])))
    probas = proba1+proba2+proba3
    labels = [1 if i>=0.3 else 0 for i in probas]
    return classes,labels,probas

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(sign, header, recording):
    model_dir = sign.split('_')[1]
    model_number = [twelve_leads.index(i) for i in two_leads]
    model_name1 = os.path.join(model_dir, 'extra_model'+str(model_number[0]+1)+'.pkl')
    model_name2 = os.path.join(model_dir, 'extra_model'+str(model_number[1]+1)+'.pkl')
    model1 = joblib.load(model_name1)
    model2 = joblib.load(model_name2)
    classes = model1['classes']
    imputer = model1['imputer']
    mlb = model1['mlb']
    classifier1 = model1['classifier']
    classifier2 = model2['classifier']
    datas, labels, classes = get_features(header, recording, two_leads)
    datas = imputer.transform(datas)
    proba1 = np.array(get_proba_by_dimension1(classifier1.predict_proba([datas[0]])))
    proba2 = np.array(get_proba_by_dimension1(classifier2.predict_proba([datas[1]])))
    probas = proba1+proba2
    labels = [1 if i>=0.2 else 0 for i in probas]
    return classes,labels,probas

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
#header是一个字符串，包含的是label类的数据，然后recording是导联数据，格式为numpy数组,都是单个用户值
def get_features(header_files,recording_files,leads):
    datas = list()
    targets = list()
    tem_age_sex = list()
    #主要是获取维度，便于之后截断处理
    dimensions = list()
    for i in range(len(recording_files)):
        #得到标签和数据，分别是str格式和数组格式
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])

        # Extract age.
        age = get_age(header)
        if age is None:
            age = float('nan')
        # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
        sex = get_sex(header)
        if sex in ('Female', 'female', 'F', 'f'):
            sex = 0
        elif sex in ('Male', 'male', 'M', 'm'):
            sex = 1
        else:
            sex = float('nan')
        tem_age_sex.append([age,sex])

        # Reorder/reselect leads in recordings.
        #主要是获取所有存在的导联
        available_leads = get_leads(header)
        #获取其中的导联序号(下标)
        indices = list()
        for lead in leads:
            i = available_leads.index(lead)
            indices.append(i)
        #recording是返回得到的导联形式的数据
        recording = recording[indices, :]

        #获取归一化后的数据
        data_scaler = np.zeros_like(recording)
        for i in range(len(leads)):
            data_scaler[i,:] = scaler(recording[i,:])

        #之后对于每一行数据以窗口为10，步长为10节奏进行采样，采样的数据是均值与标准差
        window = 10
        num_windows = int(data_scaler.shape[1]/window)
        mean_std_values = list()
        for i in range(len(data_scaler)):
            data = data_scaler[i,:]
            #主要是存储各个采样区间的均值与方差
            tem_message = list()
            #获取各个窗口内的均值与标准差，并且存储在tem_message中
            for j in range(num_windows):
                sub_data = data[j*window:(j+1)*window]
                mean_value = np.mean(sub_data)
                std_value = np.std(sub_data,ddof=1)
                tem_message.append(mean_value)
                tem_message.append(std_value)
            mean_std_values.append(tem_message)
        mean_std_arr = np.array(mean_std_values)
        #获取维度
        dimensions.append(mean_std_arr.shape[1])
        #获取数据
        datas.append(mean_std_arr)
        #获取label
        current_labels = get_labels(header)
        targets.append(current_labels)

    mlb = MultiLabelBinarizer()
    label_one_hot = mlb.fit_transform(targets)
    classes = mlb.classes_

    result_data = truncation(datas,dimensions)
    #将数据与用户的性别与年龄等合并在一起
    # result_datas = np.concatenate((result_data,np.array(tem_age_sex)),axis=1)
    result_datas = concat_data(result_data,tem_age_sex)
    return result_datas,label_one_hot,classes,mlb

def get_test_data(header,recording,leads):
    tem_age_sex = list()
    dimensions = list()
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')
    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')
    tem_age_sex.append([age,sex])

    # Reorder/reselect leads in recordings.
    #主要是获取所有存在的导联
    available_leads = get_leads(header)
    #获取其中的导联序号(下标)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    #recording是返回得到的导联形式的数据
    recording = recording[indices, :]

    #获取归一化后的数据
    data_scaler = np.zeros_like(recording)
    for i in range(len(leads)):
        data_scaler[i,:] = scaler(recording[i,:])

    #之后对于每一行数据以窗口为10，步长为10节奏进行采样，采样的数据是均值与标准差
    window = 10
    num_windows = int(data_scaler.shape[1]/window)
    mean_std_values = list()
    for i in range(len(data_scaler)):
        data = data_scaler[i,:]
        #主要是存储各个采样区间的均值与方差
        tem_message = list()
        #获取各个窗口内的均值与标准差，并且存储在tem_message中
        for j in range(num_windows):
            sub_data = data[j*window:(j+1)*window]
            mean_value = np.mean(sub_data)
            std_value = np.std(sub_data,ddof=1)
            tem_message.append(mean_value)
            tem_message.append(std_value)
        mean_std_values.append(tem_message)
    mean_std_arr = np.array(mean_std_values)
    #获取维度
    dimensions.append(mean_std_arr.shape[1])
    result_data = truncation(mean_std_arr,dimensions)
    #将数据与用户的性别与年龄等合并在一起
    # result_datas = np.concatenate((result_data,np.array(tem_age_sex)),axis=1)
    result_datas = concat_data(result_data,tem_age_sex)
    return result_datas

#讲数据与用户性别及年龄合并在一起
def concat_data(result_data,tem_age_sex):
    datas = list()
    tem_AS = [i for i in tem_age_sex]
    for i in range(len(result_data)):
        tem_data = result_data[i]
        if len(np.array(tem_data).shape) == 2:
            tem_age_sex = np.repeat([tem_AS[i]],tem_data.shape[0],axis=0)
            con_data = np.concatenate((tem_data,tem_age_sex),axis=1)
            datas.append(con_data)
        elif len(np.array(tem_data).shape) == 1:
            tem_age_sex = np.repeat(tem_age_sex,result_data.shape[0],axis=0)
            con_data = np.concatenate((result_data,tem_age_sex),axis=1)
            datas = con_data
            break
    return np.array(datas)

#对数据进行截断处理
def truncation(datas,dimensions):
    s = [np.array(i).shape for i in datas]
    min_dimension = min(dimensions)
    if min_dimension>=500:
        result_datas = list()
        for data in datas:
            if len(data.shape) == 2:
                result_datas.append(data[:,:500])
            elif len(data.shape) == 1:
                result_datas.append(data[:500])
    else:
        result_datas = list()
        for data in datas:
            if len(data.shape) == 2:
                if data.shape[1]>=500:
                    sub_data = data[:500]
                else:
                    sub_data = np.pad(data,(0,500-len(data)),'constant')
                result_datas.append(sub_data)
            elif len(data.shape) == 1:
                result_datas.append(data[:500])
    return np.array(result_datas)

#是对数据进行归一化处理
def scaler(data):
    min_value = min(data)
    max_value = max(data)
    if float(max_value) == 0.0 and float(min_value) == 0.0:
        result = data
    else:
        max_min_inerval = max_value-min_value
        result = [(i-min_value)/max_min_inerval for i in data]
    return result

#将数据中各个导联上的数据分隔开
def get_data_for_every_daolian(datas):
    list1 = list();list2 = list();list3 = list();list4 = list()
    list5 = list();list6 = list();list7 = list();list8 = list()
    list9 = list();list10 = list();list11 = list();list12 = list()
    for data in datas:
        list1.append(data[0])
        list2.append(data[1])
        list3.append(data[2])
        list4.append(data[3])
        list5.append(data[4])
        list6.append(data[5])
        list7.append(data[6])
        list8.append(data[7])
        list9.append(data[8])
        list10.append(data[9])
        list11.append(data[10])
        list12.append(data[11])
    return [list1,list2,list3,list4,list5,list6,list7,list8,list9,list10,list11,list12]

#分别对12导联上的数据进行预测，并将对应模型进行存储
def trains(data_list,labels,model_directory,classes,imputer,mlb):
    for i in range(len(data_list)):
        script = i+1
        data = np.array(data_list[i])
        name = 'extra_model'+str(script)+'.pkl'
        filename = os.path.join(model_directory,name)
        if script == 1:
            model = extra_model1()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)
        elif script == 2:
            model = extra_model2()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)
        elif script == 3:
            model = extra_model3()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)
        elif script == 4:
            model = extra_model4()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)
        elif script == 5:
            model = extra_model5()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)
        elif script == 6:
            model = extra_model6()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)
        elif script == 7:
            model = extra_model7()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)
        elif script == 8:
            model = extra_model8()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)
        elif script == 9:
            model = extra_model9()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)
        elif script == 10:
            model = extra_model10()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)
        elif script == 11:
            model = extra_model11()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)
        elif script == 12:
            model = extra_model12()
            model.fit(data,labels)
            dic = {'classes': classes, 'imputer': imputer, 'classifier': model,'mlb':mlb}
            joblib.dump(dic,filename)

#由于独立热编码返回的维度比较复杂，通过该函数返回对应label“1”对应的概率
def get_proba_by_dimension1(probas):
    result = list()
    for pro in probas:
        for i in pro:
            result.append(i[1])
    return result
