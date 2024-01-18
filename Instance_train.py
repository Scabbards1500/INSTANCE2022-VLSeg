import pandas as pd
import torch
import os
from model import *
import numpy as np

from model.modelUnet import BinaryUNet3dModel_ori

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def trainbinaryunet3d():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\traindata.csv')
    maskdatasource = csvdata.iloc[:, 1].values
    # print(maskdatasource) #D:\\python\\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\\dataset\\train\\label\\013.npy
    imagedatasource = csvdata.iloc[:, 0].values
    csvdataaug = pd.read_csv('dataprocess\\data\\trainaugdata.csv')
    maskdataaug = csvdataaug.iloc[:, 1].values
    imagedataaug = csvdataaug.iloc[:, 0].values
    imagedata = np.concatenate((imagedatasource, imagedataaug), axis=0)
    maskdata = np.concatenate((maskdatasource, maskdataaug), axis=0)
    # 文本部分！
    textdata = pd.read_csv(r"D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataprocess\data\Train_text.csv")

    # 这里就先不打乱了,放进dataloader里面一起打乱
    trainimages = imagedata
    trainlabels = maskdata
    #textdata 等到放进去之后再匹配好了
    traintext = textdata

    data_dir2 = 'dataprocess\data\\validata.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values
    valtext = pd.read_csv(r"D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataprocess\data\Val_text.csv")

    unet3d = BinaryUNet3dModel(image_depth=32, image_height=320, image_width=256, image_channel=1, numclass=1,bert_type=r'D:\model\biobert_base_cased_v1_2',
                               batch_size=1, loss_name='BinaryDiceLoss')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, traintext, valtext,
                        model_dir='log/instance/dice/Unet',
                        bert_dir=r'D:\model\biobert_base_cased_v1_2',
                        epochs=200, showwind=[4, 8])




# def trainbinaryVnet3d():
#     # Read  data set (Train data from CSV file)
#     csvdata = pd.read_csv('dataprocess\\data\\traindata.csv')
#     maskdatasource = csvdata.iloc[:, 1].values
#     imagedatasource = csvdata.iloc[:, 0].values
#     csvdataaug = pd.read_csv('dataprocess\\data\\trainaugdata.csv')
#     maskdataaug = csvdataaug.iloc[:, 1].values
#     imagedataaug = csvdataaug.iloc[:, 0].values
#     imagedata = np.concatenate((imagedatasource, imagedataaug), axis=0)
#     maskdata = np.concatenate((maskdatasource, maskdataaug), axis=0)
#     # 文本部分！
#     textdata = pd.read_csv(r"D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataprocess\data\Train_text.csv")
#     # shuffle imagedata and maskdata together
#     perm = np.arange(len(imagedata))
#     np.random.shuffle(perm)
#     trainimages = imagedata[perm]
#     trainlabels = maskdata[perm]
#     traintext = textdata[perm]
#
#     csv_data2 = pd.read_csv('dataprocess\data\\validata.csv')
#     valimages = csv_data2.iloc[:, 0].values
#     vallabels = csv_data2.iloc[:, 1].values
#     valtext = pd.read_csv(r"D:\python\INSTANCE2022-INtracranial-hemorrhage-SegmenTAtioN-ChallengE-main\dataprocess\data\Val_text.csv")
#
#     vnet3d = BinaryVNet3dModel(image_depth=32, image_height=320, image_width=256, image_channel=1, numclass=1,
#                                batch_size=1, loss_name='BinaryDiceLoss')
#     vnet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, traintext, valtext,
#                         model_dir='log/instance/dice/Vnet',
#                         bert_dir= r'D:\model\biobert_base_cased_v1_2',
#                         epochs=200, showwind=[4, 8])

def trainbinaryunet3d_ori():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\traindata.csv')
    maskdatasource = csvdata.iloc[:, 1].values
    imagedatasource = csvdata.iloc[:, 0].values
    csvdataaug = pd.read_csv('dataprocess\\data\\trainaugdata.csv')
    maskdataaug = csvdataaug.iloc[:, 1].values
    imagedataaug = csvdataaug.iloc[:, 0].values
    imagedata = np.concatenate((imagedatasource, imagedataaug), axis=0)
    maskdata = np.concatenate((maskdatasource, maskdataaug), axis=0)
    # 文本部分！

    # 这里就先不打乱了,放进dataloader里面一起打乱
    trainimages = imagedata
    trainlabels = maskdata

    data_dir2 = 'dataprocess\data\\validata.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = BinaryUNet3dModel_ori(image_depth=32, image_height=320, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryDiceLoss')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/instance/dice/Unet',
                        epochs=200, showwind=[4, 8])



if __name__ == '__main__':

    trainbinaryunet3d()
    # trainbinaryunet3d_ori()
    # trainbinaryVnet3d()
