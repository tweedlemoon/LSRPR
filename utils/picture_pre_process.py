from utils import transforms as Trans


class VOCSegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """

        :param base_size:int 基础尺寸
        :param crop_size:int 裁剪尺寸
        :param hflip_prob: float 是一个可能性的值，0-1之间，表示图像翻转的概率
        :param mean:
        :param std:
        """
        # 最小0.5*520=260，最大2.0*520=1040
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        # transform是一个list，第一个元素是RandomResize
        trans = [Trans.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(Trans.RandomHorizontalFlip(hflip_prob))
        # 列表拼元素：append；列表拼列表：extend
        trans.extend([
            Trans.RandomCrop(crop_size),
            Trans.ToTensor(),
            Trans.Normalize(mean=mean, std=std),
        ])
        self.transforms = Trans.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class VOCSegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Trans.Compose([
            Trans.RandomResize(base_size, base_size),
            Trans.ToTensor(),
            Trans.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def voc_get_transform(train):
    # 用于判定是train还是eval，如果train=True则返回SegmentationPresetTrain函数否则SegmentationPresetEval
    # 首先说明，voc2012数据集绝大多数图片尺寸都是500*375，当然还有一些别的大小的，但是属于少数
    # 基础尺寸520，裁剪尺寸480
    base_size = 520
    crop_size = 480

    return VOCSegmentationPresetTrain(base_size, crop_size) if train else VOCSegmentationPresetEval(base_size)


# ---------------------------------------------------------------------------------------------------------------
class DriveSegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [Trans.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(Trans.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(Trans.RandomVerticalFlip(vflip_prob))
        trans.extend([
            Trans.RandomCrop(crop_size),
            Trans.ToTensor(),
            Trans.Normalize(mean=mean, std=std),
        ])
        self.transforms = Trans.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DriveSegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Trans.Compose([
            Trans.ToTensor(),
            Trans.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def drive_get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # 此数据集的每张图片都是565*584形状的图片。
    base_size = 565
    # 裁剪尺寸为480*480，也就是说Unet输入的图片大小固定为480*480
    crop_size = 480

    if train:
        return DriveSegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return DriveSegmentationPresetEval(mean=mean, std=std)
