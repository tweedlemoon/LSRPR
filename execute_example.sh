python trainLSF.py \
  --which-gpu 0 \
  --device cuda \
  --dataset DRIVE \
  --back-bone attunetplus \
  --data-path /root/autodl-tmp/liuyifei \
  --level-set-coe 0 \
  --loss-weight 1.0 2.0 \
  --batch-size 2 \
  --epochs 300 \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 1e-4

python inference.py \
  --which-gpu 0 \
  --device cuda \
  --model_path experimental_data/DRIVE/model-r2attunet-coe-0-time-20220523-1-best_dice-0.7942858338356018.pth

shutdown -h now
