import os
import platform

import torch

'''
This file is the hyper_parameters in this program.
Alter it to your own when you want to run it on your own pc or server.
'''


class HyperParameters(object):
    def __init__(self, level_set_coe=1e-6, data_name='RITE', back_bone='attunet', epoch=200, batch_size=2,
                 which_gpu='1', step=1,
                 pretrained=''):
        self.generate_dir()
        self.Result_Root = "./results"
        # which gpu to use, 0 default. Multi-gpu is not supported in this project.
        self.Which_GPU = which_gpu
        # which dataset to use
        self.Data_Name = data_name
        # which backbone to use
        self.Back_Bone = back_bone
        # FCN use Aux.
        self.Aux = True if self.Back_Bone == "fcn" else False
        # Step
        # step1:unsupervised learning
        # step2:supervised learning
        self.Step = step
        # learning rate initially, it will decrease when training
        self.Initial_Learning_Rate = 0.01
        # batchsize
        self.Batch_Size = batch_size
        # epoch number
        self.Epoch = epoch
        # use when debugging, it's to print messages on the console
        self.Print_Frequency = 1
        # use cuda.amp?
        self.Cuda_Amp = False
        # 是否使用Dice损失
        self.Dice = "dice"
        # 0代表不使用水平集损失
        self.Level_Set_Coe = level_set_coe
        # loss weight
        self.Loss_Weight = [1.0, 2.0]
        # resume?
        # 是否是resume，如果是resume恢复训练，则resume下填写恢复路径（是一个pth文件）
        self.Resume = ""
        # pretrained?
        # 如果加载pretrained模型，这里直接填写模型的pth文件路径
        self.Pretrained = pretrained
        # Pretrained = "experimental_data/DRIVE/model-unet-coe-5e-6-best_dice-0.821.pth"

        self.Device = self.judge_device()

        self.Start_Epoch = self.judge_resume()

        self.Class_Num = 0
        self.Data_Root = ''
        self.judge_dataroot()

    def generate_dir(self):
        '''
        This part creates some necessary directory.
        '''
        # part:result file path of this project, the pth file will be saved under it.
        # result file in this project
        if not os.path.exists("./results"):
            os.mkdir("./results")

        # a chache of the intermidiate result
        if not os.path.exists("./save_weights"):
            os.mkdir("./save_weights")

    def judge_device(self):
        if torch.cuda.is_available():
            if self.Which_GPU != '0':
                device_name = 'cuda' + ':' + self.Which_GPU
            else:
                device_name = 'cuda'
        else:
            device_name = 'cpu'
        return torch.device(device_name)

    def judge_resume(self):
        if self.Resume != "":
            return torch.load(self.Resume, map_location='cpu')['epoch'] + 1
        else:
            return 0

    def judge_pretrained(self):
        if self.Step == 2 and self.Pretrained == '':
            raise ValueError('When do step 2, pretrained should not be empty.')

    def judge_dataroot(self):
        if self.Data_Name == "voc2012":
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

            self.Data_Root = VOC_Dataset_Root
            # VOC2012 has 20 classes to detect.
            self.Class_Num = 20
        elif self.Data_Name == "DRIVE":
            # part:DRIVE Dataset
            if platform.system() == "Windows":
                Data_Path = "E:/Datasets/"
            else:
                Data_Path = "/Data20T/data20t/data20t/Liuyifei/Datasets"
                # Data_Path = "/root/autodl-tmp"
            self.Data_Root = Data_Path
            self.Class_Num = 1
        elif self.Data_Name == 'Chase_db1':
            # part:DRIVE Dataset
            if platform.system() == "Windows":
                Data_Path = "E:/Datasets/"
            else:
                Data_Path = "/Data20T/data20t/data20t/Liuyifei/Datasets"
                # Data_Path = "/root/autodl-tmp"

            self.Data_Root = Data_Path
            self.Class_Num = 1
        elif self.Data_Name == 'RITE':
            # part:DRIVE Dataset
            if platform.system() == "Windows":
                Data_Path = "E:/Datasets/"
            else:
                Data_Path = "/Data20T/data20t/data20t/Liuyifei/Datasets"
                # Data_Path = "/root/autodl-tmp"

            self.Data_Root = Data_Path
            self.Class_Num = 1
        elif self.Data_Name == 'ISIC2018':
            # part:DRIVE Dataset
            if platform.system() == "Windows":
                Data_Path = "E:/Datasets/"
            else:
                Data_Path = "/Data20T/data20t/data20t/Liuyifei/Datasets"
                # Data_Path = "/root/autodl-tmp"

            self.Data_Root = Data_Path
            self.Class_Num = 1


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

if __name__ == '__main__':
    my_hyper_params = HyperParameters()
    print(my_hyper_params)
