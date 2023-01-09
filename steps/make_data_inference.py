import argparse
import time

import torch.utils.data as data
from PIL import Image

from src.get_transforms import *
from utils.handy_functions import *
from utils.timer import Timer


# 把下面这行打开，即可看到pytorch封装的VOCSegmentation类，这里把它重写了
# import torchvision
# voc_train_dataset = torchvision.datasets.VOCSegmentation(root='', year='2012', image_set='train', )
# 重写的要求：1. 将数据存成一个list能随时找到，label也要对应存好
# 2. 必须有__getitem__函数和__len__函数，分别返回一条数据和数据集整体长度

# VOC not used
class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        """
        这个是VOCSegmentation的重写，具体见原来的VOCSegmentation类
        :param voc_root:str VOCdevkit的上级目录，此路径下一定要有VOCdevkit文件夹
        :param year:str 2007或者2012
        :param transforms:function 图像尺寸更改函数，用于对图像进行预先处理，解决比如尺寸不够、尺寸过大等等
        :param txt_name:str 训练集的索引txt名称，一般位于/VOCdevkit/VOC2012/ImageSets/Segmentation那里
        """
        super(VOCSegmentation, self).__init__()
        # 断言年份必须是2007或者2012，这是pascal数据集的特性
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 拼合路径VOC2012/2007
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        # 断言检查路径是否合法
        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        # 图片路径、遮罩路径、索引txt路径
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        # 断言检查路径是否合法
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        # 一行一行读取txt中的数据，并删去每行数据结尾的换行符
        # strip()函数是copy一份字符串，然后返回那个字符串去掉首尾空格和换行（非字符串中间的空格）或指定字符
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        # 给每个图片和遮罩名字
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        # 断言二者必须相等
        assert (len(self.images) == len(self.masks))

        self.transforms = transforms

    def __getitem__(self, index):
        # 如果要重写VOCSegmentation类，则__getitem__和__len__必须重写，否则报错
        """
        getitem函数的意义就是给一个index能取出数据集里的那个item
        :param index: index (int): Index
        :return: tuple: (image, target) where target is the image segmentation.
        """
        # 图片要转成RGB，而目标则只是mask
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """
        len函数就是求该类的length
        :return:int 数据集的长度
        """
        return len(self.images)

    @staticmethod
    def cat_list(images, fill_value=0):
        """
        一个batch中的尺寸统一化
        :param images: Tensor
        :param fill_value: 填充值
        :return: 一个尺寸统一化的batch
        """
        # 计算该batch数据中，channel, h, w的最大值
        # img.shape for img in images这里for img in images是把images[0]作为循环体了
        # 比如images.shape=16*3*375*500（这里图像尺寸一致了，其实不一定一致，这就是它的作用）
        # 即batchsize16，3通道RGB，高度375，宽度500，那么for img in images就是16个循环，把每个图遍历一遍
        # zip将其打包，结果就是(3,3,...)(375,375,...)(500,500,...)各16个，因为zip输入的参数是16个list，每个list里有cwh
        # 接下来一个for循环循环数是3，因为是3个zip，输出是一个迭代器，对应的是3，375和500，然后强制转化成tuple，得到(3,375,500)
        # 所以这两句话的意思就是把这个batch中的最大channel和最大h最大w求出来
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        # 再把batchsize加上，+是拼接，于是(3,375,500)拼接成了(16,3,375,500)而且这时候后面三项统一为该batch中的最大值了
        batch_shape = (len(images),) + max_size

        # images[0]没有实际意义，本句的意思就相当于new一个新的tensor，大小是batchshape，填充fill_value
        batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
        # 然后就是把images里的数据拷进去
        # :img.shape[-2]的意思是0~倒数2位-1，比如:7就是0-6共7个，copy方法就是把每张img拷贝进去
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        return batched_imgs

    @staticmethod
    def collate_fn(batch):
        """
        :param batch:
        :return:
        """
        # zip把batch元组打包
        images, targets = list(zip(*batch))
        # imgs尺寸统一化处理填0
        batched_imgs = VOCSegmentation.cat_list(images, fill_value=0)
        # mask则填255
        batched_targets = VOCSegmentation.cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


class DriveDataset(data.Dataset):
    def __init__(self, root: str, train: bool, transforms=None, manual_type=1):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]

        if self.flag == 'training':
            self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                           for i in img_names]
        elif self.flag == 'test':
            if manual_type == 1:
                self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                               for i in img_names]
            elif manual_type == 2:
                self.manual = [os.path.join(data_root, "2nd_manual", i.split("_")[0] + "_manual2.gif")
                               for i in img_names]

        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # check files
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        # /255即是manual中黑色为0，血管处白色为1
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # 取反色，将中间置黑0，四周置白255
        roi_mask = 255 - np.array(roi_mask)
        # clip函数，设定numpy的上下界，这里是将manual+roi_mask的上界限设为255，下界限设为0，存在mask中
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def cat_list(images, fill_value=0):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        return batched_imgs

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = DriveDataset.cat_list(images, fill_value=0)
        batched_targets = DriveDataset.cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


class Chase_db1Dataset(data.Dataset):
    def __init__(self, root: str, transforms=None, manual_type=1):
        super(Chase_db1Dataset, self).__init__()
        data_root = os.path.join(root, "CHASE_DB1")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, )) if i.endswith(".jpg")]
        self.img_list = [os.path.join(data_root, i) for i in img_names]
        if manual_type == 1:
            self.manual = [os.path.join(data_root, os.path.splitext(i)[0]) + "_1stHO.png"
                           for i in img_names]
        elif manual_type == 2:
            self.manual = [os.path.join(data_root, os.path.splitext(i)[0]) + "_2ndHO.png"
                           for i in img_names]

        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.manual[idx]).convert('L')
        # /255即是manual中黑色为0，血管处白色为1
        mask = np.array(mask) / 255

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def cat_list(images, fill_value=0):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        return batched_imgs

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = Chase_db1Dataset.cat_list(images, fill_value=0)
        batched_targets = Chase_db1Dataset.cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


class MakeData:
    def __init__(self, args):
        self.dataset_manual_1 = None
        self.dataset_manual_2 = None
        self.loader_manual_1 = None
        self.loader_manual_2 = None

        if args.dataset == "voc2012":
            self.make_voc2012(args=args)
        elif args.dataset == "DRIVE":
            self.train_dataset_manual_1 = None
            self.test_dataset_manual_1 = None
            self.test_dataset_manual_2 = None

            self.train_loader_manual_1 = None
            self.test_loader_manual_1 = None
            self.test_loader_manual_2 = None
            self.make_DRIVE(args=args)
            self.loader_manual_1 = self.test_loader_manual_1
            self.loader_manual_2 = self.test_loader_manual_2
        elif args.dataset == 'Chase_db1':
            self.make_Chase_db1(args=args)

    # voc2012 not used
    def make_voc2012(self, args):
        print("Start making voc2012 data...")
        timer = Timer('Stage: Make Data ')
        # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
        # 此处指定了transform，使用函数get_transform指定，对train和val使用不同的transform
        self.dataset_manual_1 = VOCSegmentation(args.data_path,
                                                year="2012",
                                                transforms=voc_get_transform(train=True),
                                                txt_name="train.txt")

        # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
        self.dataset_manual_2 = VOCSegmentation(args.data_path,
                                                year="2012",
                                                transforms=voc_get_transform(train=False),
                                                txt_name="val.txt")

        # 这里把workernum以cpu数量，batchsize和8中的最小值进行选取
        # numworkers主要用于数据导入，数量恰好才是最高效的
        num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
        self.loader_manual_1 = torch.utils.data.DataLoader(self.dataset_manual_1,
                                                           batch_size=args.batch_size,
                                                           num_workers=num_workers,
                                                           shuffle=True,
                                                           pin_memory=True,
                                                           collate_fn=self.dataset_manual_1.collate_fn)

        self.loader_manual_2 = torch.utils.data.DataLoader(self.dataset_manual_2,
                                                           batch_size=1,
                                                           num_workers=num_workers,
                                                           pin_memory=True,
                                                           collate_fn=self.dataset_manual_2.collate_fn)

        print("Making data finished at: " + str(time.ctime(timer.get_current_time())))
        print("Time used: " + str(timer.get_stage_elapsed()))
        print('Done.')

    def make_DRIVE(self, args):
        print("Start making DRIVE data...")
        timer = Timer('Stage: Make Data ')
        self.train_dataset_manual_1 = DriveDataset(args.data_path,
                                                   train=True,
                                                   transforms=drive_get_transform(train=False))

        self.test_dataset_manual_1 = DriveDataset(args.data_path,
                                                  train=False,
                                                  manual_type=1,
                                                  transforms=drive_get_transform(train=False))

        self.test_dataset_manual_2 = DriveDataset(args.data_path,
                                                  train=False,
                                                  manual_type=2,
                                                  transforms=drive_get_transform(train=False))
        num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
        self.train_loader_manual_1 = torch.utils.data.DataLoader(self.train_dataset_manual_1,
                                                                 batch_size=1,
                                                                 num_workers=num_workers,
                                                                 pin_memory=True,
                                                                 collate_fn=self.train_dataset_manual_1.collate_fn)

        self.test_loader_manual_1 = torch.utils.data.DataLoader(self.test_dataset_manual_1,
                                                                batch_size=1,
                                                                num_workers=num_workers,
                                                                pin_memory=True,
                                                                collate_fn=self.test_dataset_manual_1.collate_fn)

        self.test_loader_manual_2 = torch.utils.data.DataLoader(self.test_dataset_manual_2,
                                                                batch_size=1,
                                                                num_workers=num_workers,
                                                                pin_memory=True,
                                                                collate_fn=self.test_dataset_manual_2.collate_fn)
        print("Making data finished at: " + str(time.ctime(timer.get_current_time())))
        print("Time used: " + str(timer.get_stage_elapsed()))
        print('Done.')

    def make_Chase_db1(self, args):
        print("Start making Chase_db1 data...")
        timer = Timer('Stage: Make Data ')
        self.dataset_manual_1 = Chase_db1Dataset(args.data_path,
                                                 manual_type=1,
                                                 transforms=chase_db_get_transform(train=True))

        self.dataset_manual_2 = Chase_db1Dataset(args.data_path,
                                                 manual_type=2,
                                                 transforms=chase_db_get_transform(train=False))
        num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
        self.loader_manual_1 = torch.utils.data.DataLoader(self.dataset_manual_2,
                                                           batch_size=1,
                                                           num_workers=num_workers,
                                                           pin_memory=True,
                                                           collate_fn=self.dataset_manual_2.collate_fn)

        self.loader_manual_2 = torch.utils.data.DataLoader(self.dataset_manual_2,
                                                           batch_size=1,
                                                           num_workers=num_workers,
                                                           pin_memory=True,
                                                           collate_fn=self.dataset_manual_2.collate_fn)
        print("Making data finished at: " + str(time.ctime(timer.get_current_time())))
        print("Time used: " + str(timer.get_stage_elapsed()))
        print('Done.')


def parse_arguments():
    parser = argparse.ArgumentParser(description="inference your model.")
    # batchsize是做数据的时候判断使用多少CPU核心时用的，其实在做验证集时并不需要
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--data-path", default='E:/Datasets', help="data root")
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--dataset", default='DRIVE', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    train_loader = MakeData(args=args).loader_manual_1
    for img, mask in train_loader:
        val_range(img, 'img')
        val_range(mask, 'mask')
        img_show(format_convert(mask))
    pass
