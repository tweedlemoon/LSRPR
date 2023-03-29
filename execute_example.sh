CUDA_VISIBLE_DEVICES=0 python trainLSF.py \
--which-gpu 0 \
--device cuda \
--dataset DRIVE \
--back-bone attunet \
--data-path /root/datasets \
--step 1 \
--level-set-coe 0 \
--loss-weight 1.0 2.0 \
--batch-size 4 \
--epochs 100 \
--lr 0.01 \
--momentum 0.9 \
--weight-decay 1e-4

CUDA_VISIBLE_DEVICES=0 python inference.py \
--which-gpu 0 \
--device cuda \
--data-path /root/datasets \
--is_val val \
--manual manual2 \
--visualization none \
--show no \
--model_path experimental_data/DRIVE/model-r2attunet-coe-0-time-20220523-1-best_dice-0.7942858338356018.pth
