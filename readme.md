# 使用水平集改良的血管分割(Improved vessel segmentation approach using the level-set function)

## 简介 Introduction

coming soon...



## 更新日志 Updates

2022/4/25: basic version (containing unet model & DRIVE dataset)

2022/5/2: add mum-ford loss

2022/5/13: add support of r2unet, attetion-unet, r2attention-unet

2022/5/20: add support of Chase_db1 dataset

More...



## 环境 Requirements

本项目需要的环境极简。

This project requires a very minimal setting.

- Pytorch (Including Torchvision)
- Pillow
- matplotlib



## 结果及权重 Results & Checkpoints

本表格记录了目前使用此种方法在该数据集上本人训练出的最好结果。

This table shows the greatest results I've gotten so far using my training strategy on the datasets.

| Dataset | Method      | Accuracy/F1 score/mIoU | Checkpoint     | Log            |
| ------- | ----------- | ---------------------- | -------------- | -------------- |
| DRIVE   | Unet (2015) |                        | coming soon... | coming soon... |
|         |             |                        |                |                |
|         |             |                        |                |                |
|         |             |                        |                |                |
|         |             |                        |                |                |
|         |             |                        |                |                |
|         |             |                        |                |                |



## 运行方法



```bash
# 后台运行
nohup bash run.sh > run.log 2>&1 &

# 查找该进程
ps -aux | grep run.sh
```

