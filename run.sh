cd /liuyifei/LSFforSeg

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  --dataset DRIVE \
  -b 2 \
  --epochs 200 \
  --level-set-coe 0 \
  --loss-weight 1.0 2.0 \
  --back-bone attunetplus

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  --dataset DRIVE \
  -b 2 \
  --epochs 300 \
  --level-set-coe 0.000001 \
  --loss-weight 1.0 2.0 \
  --back-bone attunetplus

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  --dataset Chase_db1 \
  -b 2 \
  --epochs 200 \
  --level-set-coe 0 \
  --loss-weight 1.0 2.0 \
  --back-bone attunetplus

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  --dataset Chase_db1 \
  -b 2 \
  --epochs 300 \
  --level-set-coe 0.000001 \
  --loss-weight 1.0 2.0 \
  --back-bone attunetplus

python utils/emailSender.py

shutdown -h now
