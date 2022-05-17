cd /liuyifei/LSFforSeg

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  -b 2 \
  --epochs 300 \
  --loss-weight 1.0 2.0 \
  --back-bone unet

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  -b 2 \
  --epochs 300 \
  --level-set-coe 0.000001 \
  --loss-weight 1.0 2.0 \
  --back-bone unet

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  -b 2 \
  --epochs 300 \
  --loss-weight 1.0 2.0 \
  --back-bone r2unet

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  -b 2 \
  --epochs 300 \
  --level-set-coe 0.000001 \
  --loss-weight 1.0 2.0 \
  --back-bone r2unet

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  -b 2 \
  --epochs 300 \
  --loss-weight 1.0 2.0 \
  --back-bone attunet

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  -b 2 \
  --epochs 300 \
  --level-set-coe 0.000001 \
  --loss-weight 1.0 2.0 \
  --back-bone attunet

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  -b 2 \
  --epochs 300 \
  --loss-weight 1.0 2.0 \
  --back-bone r2attunet

python trainLSF.py \
  --data-path "/root/autodl-tmp/liuyifei" \
  -b 2 \
  --epochs 300 \
  --level-set-coe 0.000001 \
  --loss-weight 1.0 2.0 \
  --back-bone r2attunet

python utils/emailSender.py

sudo shutdown -h now
