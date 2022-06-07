python trainLSF.py \
  --which-gpu 0 \
  --device cuda \
  --dataset DRIVE \
  --back-bone attunetplus \
  --data-path "/root/autodl-tmp/liuyifei" \
  --level-set-coe 0 \
  --loss-weight 1.0 2.0 \
  --batch-size 2 \
  --epochs 300 \
  --lr 0.01

shutdown -h now
