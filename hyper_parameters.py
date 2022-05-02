import os
import platform

import torch

'''
This file is the hyper_parameters in this program.
Alter it to your own when you want to run it on your own pc or server.
'''

'''
This part creates some necessary directory.
'''
# part:result file path of this project, the pth file will be saved under it.
# result file in this project
if not os.path.exists("./results"):
    os.mkdir("./results")
Result_Root = "./results"
# a chache of the intermidiate result
if not os.path.exists("./save_weights"):
    os.mkdir("./save_weights")

'''
parameters
In this part , all the hyper parameters are listed here.
'''
# which gpu to use, 0 default. Multi-gpu is not supported in this project.
Which_GPU = "0"
# which dataset to use
Data_Name = "DRIVE"
# which backbone to use
Back_Bone = "unet"
# FCN use Aux.
Aux = True if Back_Bone == "fcn" else False
# learning rate initially, it will decrease when training
Initial_Learning_Rate = 0.01
# batchsize
Batch_Size = 2
# epoch number
Epoch = 300
# use when debugging, it's to print messages on the console
Print_Frequency = 1
# device
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use cuda.amp?
Cuda_Amp = False
# loss function
Loss_Function = "dice"
Level_Set_Coe = 1e-6

# pretrained?
# 如果加载pretrained模型，这里直接填写模型的pth文件路径
Pretrained = ""
# resume?
# 是否是resume，如果是resume恢复训练，则resume下填写恢复路径（是一个pth文件）
Resume = ""
Start_Epoch = 0
if Resume != "":
    Start_Epoch = torch.load(Resume, map_location='cpu')['epoch'] + 1
'''
paths and files
In this part , all the directory of this projects are listed here.
'''
# which dataset to use
if Data_Name == "voc2012":
    # part:VOC 2012 Pascal Dataset
    # os judge
    if platform.system() == "Windows":
        # VOCdevkit file location
        VOC_Dataset_Root = "E:/Datasets/Pascal_voc_2012"
    else:
        # VOCdevkit file location
        VOC_Dataset_Root = "/Data20T/data20t/data20t/Liuyifei/Datasets"
    # some sub-dir
    # VOC2012 file location
    VOC2012_Dataset_Path = os.path.join(VOC_Dataset_Root, "VOCdevkit", "VOC2012")
    # voc2012 picture location
    VOC2012_Pic = os.path.join(VOC2012_Dataset_Path, "JPEGImages")
    # voc2012 mask location(seg class, not instance)
    VOC2012_Mask = os.path.join(VOC2012_Dataset_Path, "SegmentationClass")

    Data_Root = VOC_Dataset_Root
    # VOC2012 has 20 classes to detect.
    Class_Num = 20
elif Data_Name == "DRIVE":
    # part:DRIVE Dataset
    if platform.system() == "Windows":
        Data_Path = "E:/Datasets/"
    else:
        # Data_Path = "/Data20T/data20t/data20t/Liuyifei/Datasets"
        Data_Path = "/root/autodl-tmp"
    Data_Root = Data_Path
    Class_Num = 1

# part:pretrained FCN with ResNet50&101 by the COCO dataset
# os judge
if platform.system() == "Windows":
    # FCN COCO Resnet pretrained model location
    FCN_ResNet_COCOTrained_Path = "E:/Datasets/fcn_trained"
else:
    # FCN COCO Resnet pretrained model location
    FCN_ResNet_COCOTrained_Path = "/Data20T/data20t/data20t/Liuyifei/Datasets/fcn_trained"
# FCN COCO Resnet pretrained model
# download them and rename them as follow
# download at: "https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth",
# "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth"
FCN_ResNet50_COCO = os.path.join(FCN_ResNet_COCOTrained_Path, "fcn_resnet50_coco.pth")
FCN_ResNet101_COCO = os.path.join(FCN_ResNet_COCOTrained_Path, "fcn_resnet101_coco.pth")
