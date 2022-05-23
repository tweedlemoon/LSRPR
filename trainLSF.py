import argparse

from hyper_parameters import *
from steps.make_data import MakeData
from steps.make_net import MakeNet
from steps.train_eval_model import train_eval_model
from utils.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="LSF for segmentation")

    parser.add_argument("--which-gpu", default=Which_GPU, type=str, help="which gpu to use")
    parser.add_argument("--dataset", default=Data_Name, type=str, choices=["voc2012", "DRIVE", 'Chase_db1'],
                        help="which dataset to use")
    parser.add_argument("--data-path", default=Data_Root, help="data root")
    parser.add_argument("--num-classes", default=Class_Num, type=int)
    parser.add_argument("--device", default=str(Device), type=str, help="training device")
    parser.add_argument("--back-bone", default=Back_Bone, type=str,
                        choices=["fcn", "unet", "r2unet", "attunet", "r2attunet"])
    parser.add_argument("--pretrained", "-p", default=Pretrained, type=str, help="if load pretrained model, add here.")
    # Here is the auxilier loss, which is used in the pytorch official source code.
    parser.add_argument("--aux", default=Aux, type=bool, help="auxilier loss")
    parser.add_argument("--dice", default=Dice, type=str,
                        help="choose which loss function to use.", choices=["dice", ])
    parser.add_argument("--level-set-coe", default=Level_Set_Coe, type=float)
    # loss weight for every class, input should be the same amount of the num_classes+1
    parser.add_argument("--loss-weight", default=Loss_Weight, type=float, nargs='+')

    parser.add_argument("-b", "--batch-size", default=Batch_Size, type=int)
    parser.add_argument("--epochs", default=Epoch, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=Initial_Learning_Rate, type=float, help='initial learning rate')
    # momentum是梯度下降的一个数值，称之为动量，可以优化梯度下降，帮助其跳出鞍点
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # 这里用到了权重衰减，可以更好的走到最小值处
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=Print_Frequency, type=int, help='print frequency')
    # 下面两个参数是继续训练时用到
    parser.add_argument('--resume', default=Resume, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=Start_Epoch, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=Cuda_Amp, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--result-root", default=Result_Root, type=str, help="Save the result of the program")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # +1这个1是背景background
    args.num_classes += 1
    if args.device == 'cuda':
        # use which GPU and initial
        os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpu
    print(args)
    timer = Timer("Work begin.")

    # Step1: Make train & val datasets.
    print("----------------------------------\n" + "STAGE 1: Start making datasets.")
    Data = MakeData(args=args)

    # Step2: Create backbone network.
    print("----------------------------------\n" + "STAGE 2: Start making network.")
    Net = MakeNet(args=args, train_loader_lenth=len(Data.train_loader))

    # Step3: Train.
    print("----------------------------------\n" + "STAGE 3: Start training.")
    train_eval_model(args=args, Data=Data, Net=Net)
