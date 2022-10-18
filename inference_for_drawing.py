import argparse

from PIL import Image

from hyper_parameters import *
from utils.handy_functions import *
from models.unet import create_unet_model
from models.r2unet import *
from models.sa_unet import SA_Unet
from models.attunetplus import AttU_Net_Plus
import PIL
from torchvision import transforms

from utils.eval_utils import ConfusionMatrix
from utils.timer import Timer
from steps.make_data import MakeData as originmk
from steps.make_data_inference import MakeData as infmk
from utils.color_palette import generate_color_img

Model_path = ''
Input_pic = ''


# 494,361
# 522,389

def parse_arguments():
    parser = argparse.ArgumentParser(description='Do one prediction.')
    # batchsize是做数据的时候判断使用多少CPU核心时用的，其实在做验证集时并不需要
    parser.add_argument("-b", "--batch-size", default=Batch_Size, type=int)
    parser.add_argument("--which-gpu", default=Which_GPU, type=str, help="which gpu to use")
    parser.add_argument("--data-path", default=Data_Root, help="data root")
    parser.add_argument("--device", default=str(Device), type=str, help="training device")
    parser.add_argument('--model_path', default=Model_path, help="the best trained model root")
    parser.add_argument("--back-bone", default='attunet', type=str,
                        choices=["fcn", "unet", "r2unet", "attunet", "r2attunet", 'saunet', 'attunetplus'])
    parser.add_argument("--num-classes", default=Class_Num, type=int)
    parser.add_argument("--dataset", default='DRIVE', type=str, choices=["DRIVE", 'Chase_db1'],
                        help="which dataset to use")

    parser.add_argument('--pic', default=Input_pic)

    return parser.parse_args()


def create_model(args):
    if args.back_bone == 'unet':
        return create_unet_model(num_classes=args.num_classes + 1)
    elif args.back_bone == 'r2unet':
        return R2U_Net(output_ch=args.num_classes + 1)
    elif args.back_bone == 'attunet':
        return AttU_Net(output_ch=args.num_classes + 1)
    elif args.back_bone == 'r2attunet':
        return R2AttU_Net(output_ch=args.num_classes + 1)
    elif args.back_bone == 'saunet':
        return SA_Unet(base_size=16)
    elif args.back_bone == 'saunet64':
        return SA_Unet(base_size=64)
    elif args.back_bone == 'attunetplus':
        return AttU_Net_Plus(output_ch=args.num_classes + 1, sa=True)


def predict_one_pic(args):
    img = Image.open(args.pic).convert('RGB')
    device = args.device
    model = create_model(args=args).to(device)

    pth = torch.load(args.model_path, map_location=device)
    model.load_state_dict(pth['model'])
    net_output = model(img)['out']
    val_range('net output', net_output)


def generate_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    args = parse_arguments()
    if not os.path.exists(args.model_path):
        raise ValueError('model path does not exist!')
    args.back_bone = args.model_path.split('/')[-1:][0].split('-')[1]
    args.dataset = args.model_path.split('/')[-2:][0]
    if args.model_path.split('/')[-1:][0].split('-')[3] == '0' or \
            args.model_path.split('/')[-1:][0].split('-')[3] == '0.0':
        args.is_mine = 'origin'
    else:
        args.is_mine = 'mine'

    # 当显存不够时使用
    # args.device = 'cpu'
    os.environ["OMP_NUM_THREADS"] = '1'
    if args.device == 'cuda':
        # use which GPU and initial
        os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpu

    predict_one_pic(args=args)
