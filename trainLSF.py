import argparse
from hyper_parameters import *
from steps.make_data import MakeData
from steps.make_net import MakeNet
from steps.train_eval_model import train_eval_model
from utils.timer import Timer

my_params = HyperParameters()
# 限制CPU进程数
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu


def parse_args():
    parser = argparse.ArgumentParser(description="LSF for segmentation")

    parser.add_argument("--which-gpu", default=my_params.Which_GPU, type=str, help="which gpu to use")
    parser.add_argument("--dataset", default=my_params.Data_Name, type=str,
                        choices=["DRIVE", 'Chase_db1', 'RITE', 'ISIC2018'],
                        help="which dataset to use")
    parser.add_argument("--data-path", default=my_params.Data_Root, help="data root")
    parser.add_argument("--num-classes", default=my_params.Class_Num, type=int)
    parser.add_argument("--device", default=str(my_params.Device), type=str, help="training device")
    parser.add_argument("--back-bone", default=my_params.Back_Bone, type=str,
                        # laddernet/fanet may be unusable.
                        choices=["fcn", "unet", "r2unet", "attunet", "r2attunet", 'saunet', 'saunet64', 'attunetplus',
                                 'kiunet', 'laddernet', 'fanet'])
    parser.add_argument("--pretrained", "-p", default=my_params.Pretrained, type=str,
                        help="if load pretrained model, add here.")
    parser.add_argument('--step', default=my_params.Step, type=int,
                        help='Step1:unsupervised learning, Step2:supervised learning',
                        choices=[1, 2])
    # Here is the auxilier loss, which is used in the pytorch official source code.
    parser.add_argument("--aux", default=my_params.Aux, type=bool, help="auxilier loss")
    parser.add_argument("--dice", default=my_params.Dice, type=str,
                        help="choose which loss function to use.", choices=["dice", ])
    parser.add_argument("--level-set-coe", default=my_params.Level_Set_Coe, type=float)
    # loss weight for every class, input should be the same amount of the num_classes+1
    parser.add_argument("--loss-weight", default=my_params.Loss_Weight, type=float, nargs='+')

    parser.add_argument("-b", "--batch-size", default=my_params.Batch_Size, type=int)
    parser.add_argument("--epochs", default=my_params.Epoch, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=my_params.Initial_Learning_Rate, type=float, help='initial learning rate')
    # momentum是梯度下降的一个数值，称之为动量，可以优化梯度下降，帮助其跳出鞍点
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # 这里用到了权重衰减，可以更好的走到最小值处
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=my_params.Print_Frequency, type=int, help='print frequency')
    # 下面两个参数是继续训练时用到
    parser.add_argument('--resume', default=my_params.Resume, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=my_params.Start_Epoch, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=my_params.Cuda_Amp, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--result-root", default=my_params.Result_Root, type=str, help="Save the result of the program")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # +1这个1是背景background
    args.num_classes += 1

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
