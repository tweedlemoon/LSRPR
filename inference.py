import argparse

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
Manual = 'manual2'


def parse_arguments():
    parser = argparse.ArgumentParser(description="inference your model.")
    # batchsize是做数据的时候判断使用多少CPU核心时用的，其实在做验证集时并不需要
    parser.add_argument("-b", "--batch-size", default=Batch_Size, type=int)
    parser.add_argument("--which-gpu", default=Which_GPU, type=str, help="which gpu to use")
    parser.add_argument("--data-path", default=Data_Root, help="data root")
    parser.add_argument("--device", default=str(Device), type=str, help="training device")
    parser.add_argument('--model_path', default=Model_path, help="the best trained model root")
    parser.add_argument("--back-bone", default='unet', type=str,
                        choices=["fcn", "unet", "r2unet", "attunet", "r2attunet", 'saunet', 'attunetplus'])
    parser.add_argument("--num-classes", default=Class_Num, type=int)
    parser.add_argument("--dataset", default='DRIVE', type=str, choices=["DRIVE", 'Chase_db1', 'RITE', 'ISIC2018'],
                        help="which dataset to use")
    parser.add_argument("--is_val", default='val', type=str, choices=['train', 'val'],
                        help='Use test or train to inference. Only for DRIVE dataset.')
    parser.add_argument("--manual", default=Manual, type=str, choices=['manual1', 'manual2'],
                        help='Use which manual to inference.')

    parser.add_argument('--show', default='no', type=str, choices=['yes', 'no'],
                        help='Whether to output the result one by one.')
    parser.add_argument('--visualization', '-v', default='all', type=str, choices=['all', 'none'],
                        help='Whether to generate all the result of the network.')
    parser.add_argument('--is_mine', default='origin', type=str, choices=['origin', 'mine'],
                        help='If the model has levelset function.')

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


def compute_index(args):
    matrix = ConfusionMatrix(num_classes=args.num_classes + 1)

    device = args.device
    model = create_model(args=args).to(device)

    pth = torch.load(args.model_path, map_location=device)
    model.load_state_dict(pth['model'])
    loader = None

    # 多张图片测试，直接制作dataloader
    if args.dataset == 'DRIVE':
        if args.is_val == 'val':
            if args.manual == 'manual1':
                loader = infmk(args=args).loader_manual_1
            elif args.manual == 'manual2':
                loader = infmk(args=args).loader_manual_2
        elif args.is_val == 'train':
            loader = infmk(args=args).train_dataset_manual_1
    elif args.dataset == 'Chase_db1':
        if args.manual == 'manual1':
            loader = infmk(args=args).loader_manual_1
        elif args.manual == 'manual2':
            loader = infmk(args=args).loader_manual_2
    elif args.dataset == 'RITE':
        loader = originmk(args=args).val_loader
    elif args.dataset == 'ISIC2018':
        loader = originmk(args=args).val_loader

    all_f1_score = 0.0
    all_accuracy = 0.0
    all_miou = 0.0
    print('--------------------------------')
    timer = Timer('Evaluating...')
    model.eval()
    with torch.no_grad():
        for idx, (img, real_result) in enumerate(loader, start=0):
            ground_truth = loader.dataset.manual[idx]
            if args.dataset != 'ISIC2018':
                ground_truth = transforms.ToTensor()(PIL.Image.open(ground_truth).convert('1')).to(torch.int64)
            if args.dataset == 'ISIC2018':
                ground_truth = transforms.Resize(512)(PIL.Image.open(ground_truth))
                ground_truth = transforms.ToTensor()(ground_truth.convert('1')).to(torch.int64)
            ground_truth = ground_truth.to(device)

            img = img.to(device)
            net_output = model(img)['out']
            # 进行argmax操作
            argmax_output = net_output.argmax(1)

            # 1-的意思是，正例给血管
            matrix.update(1 - ground_truth, 1 - argmax_output)
            matrix.prf_compute()
            all_accuracy += matrix.accuracy
            # 这里是二分类，定义血管白色为正例，所以precision和recall都是第一个，从而F1score也是第一个，故带下标0
            all_f1_score += matrix.f1_score[0]
            all_miou += matrix.miou
            matrix.reset()

    accuracy = all_accuracy / loader.__len__()
    f1_score = all_f1_score / loader.__len__()
    miou = all_miou / loader.__len__()
    print("Time used: " + str(timer.get_stage_elapsed()))

    print('Report:')
    print('Accuracy:', accuracy.item())
    print('F1 Score:', f1_score.item())
    print('mIoU:', miou.item())
    print('--------------------------------')


def run_inference(args):
    device = args.device
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # load model
    # model = create_unet_model(num_classes=args.num_classes + 1).to(device)
    model = create_model(args=args).to(device)

    pth = torch.load(args.model_path, map_location=device)
    model.load_state_dict(pth['model'])
    loader = None

    # 多张图片测试，直接制作dataloader
    if args.dataset == 'DRIVE':
        if args.is_val == 'val':
            if args.manual == 'manual1':
                loader = infmk(args=args).loader_manual_1
            elif args.manual == 'manual2':
                loader = infmk(args=args).loader_manual_2
        elif args.is_val == 'train':
            loader = infmk(args=args).train_dataset_manual_1
    elif args.dataset == 'Chase_db1':
        if args.manual == 'manual1':
            loader = infmk(args=args).loader_manual_1
        elif args.manual == 'manual2':
            loader = infmk(args=args).loader_manual_2
    elif args.dataset == 'RITE':
        loader = originmk(args=args).val_loader
    elif args.dataset == 'ISIC2018':
        loader = originmk(args=args).val_loader

    model.eval()
    if args.dataset == 'DRIVE':
        with torch.no_grad():
            for idx, (img, real_result) in enumerate(loader, start=0):
                original_img = loader.dataset.img_list[idx]
                ground_truth = loader.dataset.manual[idx]
                roi_mask = loader.dataset.roi_mask[idx]
                original_img = PIL.Image.open(original_img)
                ground_truth = PIL.Image.open(ground_truth)
                # double_img_show(format_convert(original_img), format_convert(ground_truth))

                img = img.to(device)
                net_output = model(img)['out']
                # val_range("Network output", net_output)
                # 进行argmax操作
                argmax_output = net_output.argmax(1)
                # val_range("Argmax output", argmax_output)
                # 注意此处必须先把tensor从gpu中拿到cpu才能转numpy
                np_argmax_output = np.array(argmax_output.cpu())
                # 0是黑，255是白，故要乘以255，0依旧是0,1则变成255
                np_argmax_output = np_argmax_output.astype(np.uint8).squeeze(0) * 255
                # val_range("Numpy argmax output", np_argmax_output)

                # 把周围跟mask处理一下
                roi_img = PIL.Image.open(roi_mask).convert('L')
                roi_img = np.array(roi_img)
                np_argmax_output[roi_img == 0] = 0

                # 生成带颜色的图片
                color_img = generate_color_img(
                    ground_truth=transforms.ToTensor()(ground_truth.convert('1')).to(torch.int64),
                    prediction=argmax_output.cpu())

                if args.show == 'yes':
                    triple_img_show(original_img=format_convert(original_img),
                                    original_mask=format_convert(ground_truth),
                                    predicted_img=format_convert(np_argmax_output))
                    img_show(img=color_img)

                if args.visualization == 'all':
                    parent_dir = os.path.join('predict_pic', args.back_bone + '_' + args.is_mine + '_' + args.dataset)
                    generate_path(parent_dir)
                    predicted_img = format_convert(np_argmax_output)
                    this_img = os.path.basename(loader.dataset.img_list[idx])
                    this_img = this_img.split('.')[0]
                    save_img_name = os.path.join(parent_dir, this_img + '_' + args.back_bone + '_prediciton' + '.png')
                    save_img_name_color = os.path.join(parent_dir,
                                                       this_img + '_' + args.back_bone + '_prediciton_color' + '.png')
                    predicted_img.save(save_img_name)
                    color_img.save(save_img_name_color)
                    print('Have saved ' + save_img_name)
                    print('Have saved ' + save_img_name_color)
                print('Done ' + '[' + str(idx + 1) + '/' + str(loader.dataset.__len__()) + ']')

    elif args.dataset == 'Chase_db1':
        with torch.no_grad():
            for idx, (img, real_result) in enumerate(loader, start=0):
                original_img = loader.dataset.img_list[idx]
                ground_truth = loader.dataset.manual[idx]
                original_img = PIL.Image.open(original_img)
                ground_truth = PIL.Image.open(ground_truth)
                # double_img_show(format_convert(original_img), format_convert(ground_truth))

                img = img.to(device)
                net_output = model(img)['out']
                # val_range("Network output", net_output)
                # 进行argmax操作
                argmax_output = net_output.argmax(1)
                # val_range("Argmax output", argmax_output)
                # 注意此处必须先把tensor从gpu中拿到cpu才能转numpy
                np_argmax_output = np.array(argmax_output.cpu())
                # 0是黑，255是白，故要乘以255，0依旧是0,1则变成255
                np_argmax_output = np_argmax_output.astype(np.uint8).squeeze(0) * 255
                # val_range("Numpy argmax output", np_argmax_output)

                # 生成带颜色的图片
                color_img = generate_color_img(
                    ground_truth=transforms.ToTensor()(ground_truth.convert('1')).to(torch.int64),
                    prediction=argmax_output.cpu())

                if args.show == 'yes':
                    triple_img_show(original_img=format_convert(original_img),
                                    original_mask=format_convert(ground_truth),
                                    predicted_img=format_convert(np_argmax_output))
                    img_show(img=color_img)

                if args.visualization == 'all':
                    parent_dir = os.path.join('predict_pic', args.back_bone + '_' + args.is_mine + '_' + args.dataset)
                    generate_path(parent_dir)
                    predicted_img = format_convert(np_argmax_output)
                    this_img = os.path.basename(loader.dataset.img_list[idx])
                    this_img = this_img.split('.')[0]
                    save_img_name = os.path.join(parent_dir, this_img + '_' + args.back_bone + '_prediciton' + '.png')
                    save_img_name_color = os.path.join(parent_dir,
                                                       this_img + '_' + args.back_bone + '_prediciton_color' + '.png')
                    predicted_img.save(save_img_name)
                    color_img.save(save_img_name_color)
                    print('Have saved ' + save_img_name)
                    print('Have saved ' + save_img_name_color)
                print('Done ' + '[' + str(idx + 1) + '/' + str(loader.dataset.__len__()) + ']')

    elif args.dataset == 'RITE':
        with torch.no_grad():
            for idx, (img, real_result) in enumerate(loader, start=0):
                original_img = loader.dataset.img_list[idx]
                ground_truth = loader.dataset.manual[idx]
                original_img = PIL.Image.open(original_img)
                ground_truth = PIL.Image.open(ground_truth)
                # double_img_show(format_convert(original_img), format_convert(ground_truth))

                img = img.to(device)
                net_output = model(img)['out']
                # val_range("Network output", net_output)
                # 进行argmax操作
                argmax_output = net_output.argmax(1)
                # val_range("Argmax output", argmax_output)
                # 注意此处必须先把tensor从gpu中拿到cpu才能转numpy
                np_argmax_output = np.array(argmax_output.cpu())
                # 0是黑，255是白，故要乘以255，0依旧是0,1则变成255
                np_argmax_output = np_argmax_output.astype(np.uint8).squeeze(0) * 255
                # val_range("Numpy argmax output", np_argmax_output)

                # 生成带颜色的图片
                color_img = generate_color_img(
                    ground_truth=transforms.ToTensor()(ground_truth.convert('1')).to(torch.int64),
                    prediction=argmax_output.cpu())

                if args.show == 'yes':
                    triple_img_show(original_img=format_convert(original_img),
                                    original_mask=format_convert(ground_truth),
                                    predicted_img=format_convert(np_argmax_output))
                    img_show(img=color_img)

                if args.visualization == 'all':
                    parent_dir = os.path.join('predict_pic', args.back_bone + '_' + args.is_mine + '_' + args.dataset)
                    generate_path(parent_dir)
                    predicted_img = format_convert(np_argmax_output)
                    this_img = os.path.basename(loader.dataset.img_list[idx])
                    this_img = this_img.split('.')[0]
                    save_img_name = os.path.join(parent_dir, this_img + '_' + args.back_bone + '_prediciton' + '.png')
                    save_img_name_color = os.path.join(parent_dir,
                                                       this_img + '_' + args.back_bone + '_prediciton_color' + '.png')
                    predicted_img.save(save_img_name)
                    color_img.save(save_img_name_color)
                    print('Have saved ' + save_img_name)
                    print('Have saved ' + save_img_name_color)
                print('Done ' + '[' + str(idx + 1) + '/' + str(loader.dataset.__len__()) + ']')

    elif args.dataset == 'ISIC2018':
        with torch.no_grad():
            for idx, (img, real_result) in enumerate(loader, start=0):
                original_img = loader.dataset.img_list[idx]
                ground_truth = loader.dataset.manual[idx]
                original_img = PIL.Image.open(original_img)
                ground_truth = PIL.Image.open(ground_truth)
                # double_img_show(format_convert(original_img), format_convert(ground_truth))

                img = img.to(device)
                net_output = model(img)['out']
                # val_range("Network output", net_output)
                # 进行argmax操作
                argmax_output = net_output.argmax(1)
                # val_range("Argmax output", argmax_output)
                # 注意此处必须先把tensor从gpu中拿到cpu才能转numpy
                np_argmax_output = np.array(argmax_output.cpu())
                # 0是黑，255是白，故要乘以255，0依旧是0,1则变成255
                np_argmax_output = np_argmax_output.astype(np.uint8).squeeze(0) * 255
                # val_range("Numpy argmax output", np_argmax_output)

                ground_truth = transforms.Resize(512)(ground_truth)
                ground_truth = transforms.ToTensor()(ground_truth.convert('1')).to(torch.int64)
                # 生成带颜色的图片
                color_img = generate_color_img(
                    ground_truth=ground_truth,
                    prediction=argmax_output.cpu())

                if args.show == 'yes':
                    triple_img_show(original_img=format_convert(original_img),
                                    original_mask=format_convert(ground_truth),
                                    predicted_img=format_convert(np_argmax_output))
                    img_show(img=color_img)

                if args.visualization == 'all':
                    parent_dir = os.path.join('predict_pic', args.back_bone + '_' + args.is_mine + '_' + args.dataset)
                    generate_path(parent_dir)
                    predicted_img = format_convert(np_argmax_output)
                    this_img = os.path.basename(loader.dataset.img_list[idx])
                    this_img = this_img.split('.')[0]
                    save_img_name = os.path.join(parent_dir, this_img + '_' + args.back_bone + '_prediciton' + '.png')
                    save_img_name_color = os.path.join(parent_dir,
                                                       this_img + '_' + args.back_bone + '_prediciton_color' + '.png')
                    predicted_img.save(save_img_name)
                    color_img.save(save_img_name_color)
                    print('Have saved ' + save_img_name)
                    print('Have saved ' + save_img_name_color)
                print('Done ' + '[' + str(idx + 1) + '/' + str(loader.dataset.__len__()) + ']')


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
    # 预测图存储位置
    generate_path('predict_pic/')

    compute_index(args=args)
    run_inference(args=args)
