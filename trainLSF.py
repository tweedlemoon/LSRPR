from hyper_parameters import *
import argparse
import steps.make_data as make_data
from utils.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--which-gpu", default=Which_GPU, type=str, help="which gpu to use")
    parser.add_argument("--data-path", default=VOC_Dataset_Root, help="VOCdevkit root")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default=str(Device), type=str, help="training device")
    parser.add_argument("-b", "--batch-size", default=Batch_Size, type=int)
    parser.add_argument("--epochs", default=Epoch, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=Initial_Learning_Rate, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=Print_Frequency, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--result-root", default=Result_Root, type=str, help="Save the result of the program")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    make_data.MakeData(args=args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpu

    pass

    # main(args)
