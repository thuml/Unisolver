export CUDA_VISIBLE_DEVICES='1,2,3'
export NCCL_P2P_LEVEL=NVL

data_path=YOUR_DATA_PATH

torchrun --nproc_per_node 3 exp_HeterNS_train.py \
  --data-path $data_path \
  --in_dim 10 \
  --out_dim 1 \
  --h 64 \
  --w 64 \
  --h-down 1 \
  --w-down 1 \
  --T-in 1 \
  --T-out 19 \
  --learning-rate 0.001 \
  --model Unisolver_HeterNS\
  --model-save-path ./checkpoints/ns \
  --model-save-name Unisolver_HeterNS \
  --log-path ns_logs/Unisolver_HeterNS \
  --batch-size 40 \
  --epochs 301 \
  --maxup_ratio 0.0 \
  --maxup 0