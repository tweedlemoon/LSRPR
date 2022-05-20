import re

# Filename = '../experimental_data/1e-6levelset/1/results20220420-194802.txt'
# Filename = '../experimental_data/unet/1e-6levelset/results20220502-114425.txt'
# Filename = '../experimental_data/5e-6levelset/results20220503-101549.txt'
Filename = '../experimental_data/r2unet/train-result-model-r2unet-coe-0-time-20220517-095604.txt'


class ResultDescription(object):
    def __init__(self):
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.accuracy = 0.0
        self.epoch = 0
        self.line_num = 1

    def update(self, line_num, epoch, best_dice, accuracy, best_iou):
        self.best_dice = best_dice
        self.best_iou = best_iou
        self.accuracy = accuracy
        self.epoch = epoch
        self.line_num = line_num

    def __str__(self):
        ret = 'Report\n' + \
              '-------------------------\n' + \
              'Best Dice: ' + str(self.best_dice) + '\n' + \
              'Best mean IoU:' + str(self.best_iou) + '\n' + \
              'In Epoch:' + str(self.epoch) + '\n' + \
              'In Line:' + str(self.line_num) + '\n' + \
              '-------------------------\n'
        return ret


def analysis(f):
    message = []
    best_dice = 0.0
    cur_dice = 0.0
    description = ResultDescription()
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
            message.append(cur_dice)
        if re.search('global correct', line):
            result = re.search('\d+.\d+', line).span()
            cur_correct = float(line[int(result[0]):int(result[1])])
            message.append(cur_correct)
        if re.search('mean IoU', line):
            result = re.search('\d+.\d+', line).span()
            cur_iou = float(line[int(result[0]):int(result[1])])
            message.append(cur_iou)
            if cur_dice >= best_dice:
                description.update(line_num=message[0],
                                   epoch=message[1],
                                   best_dice=message[2],
                                   accuracy=message[3],
                                   best_iou=message[4],
                                   )

    print(description)


if __name__ == '__main__':
    file = open(Filename, 'r')
    analysis(file)
    file.close()
