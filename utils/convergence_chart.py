from matplotlib import pyplot as plt
import re

myreport1 = '../results/train-result-model-attunet-coe-0-time-20220517-1.txt'
myreport2 = '../results/train-result-model-attunet-coe-1e-06-time-20220523-1.txt'


def analysis(f):
    dice_list = []

    message = []
    lines = f.readlines()
    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if re.search('epoch', line):
            message = []
            result = re.search('\d+', line).span()
            epoch = int(line[int(result[0]):int(result[1])])
            message.append(line_num)
            message.append(epoch)
        if re.search('dice coefficient', line):
            result = re.search('\d+.\d+', line).span()
            cur_dice = float(line[int(result[0]):int(result[1])])
            dice_list.append(1 - cur_dice)
            message.append(cur_dice)
        if re.search('global correct', line):
            result = re.search('\d+.\d+', line).span()
            cur_correct = float(line[int(result[0]):int(result[1])])
            message.append(cur_correct)
        if re.search('mean IoU', line):
            result = re.search('\d+.\d+', line).span()
            cur_iou = float(line[int(result[0]):int(result[1])])
            message.append(cur_iou)
    return dice_list


if __name__ == '__main__':
    file = open(myreport1, 'r')
    y = analysis(file)[0:200]
    plt.plot(y, label='with LS-Loss')
    file = open(myreport2, 'r')
    z = analysis(file)[0:200]
    plt.plot(z, label='without LS-Loss')
    plt.xlabel('epoch')
    plt.ylabel('Dice Loss')
    plt.legend()
    plt.show()
