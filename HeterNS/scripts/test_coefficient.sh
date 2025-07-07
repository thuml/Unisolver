export CUDA_VISIBLE_DEVICES=1

data_path=YOUR_DATA_PATH
checkpoint_path=YOUR_CHECKPOINT_PATH

python exp_ns_test_coefficients.py \
  --data-path $data_path \
  --ntrain 1000 \
  --ntest 200 \
  --ntotal 1200 \
  --in_dim 10 \
  --out_dim 1 \
  --h 64 \
  --w 64 \
  --h-down 1 \
  --w-down 1 \
  --T-in 10 \
  --T-out 10 \
  --batch-size 20 \
  --learning-rate 0.0005 \
  --model Unisolver_HeterNS \
  --model-pretrain-path $checkpoint_path \
  --d-model 64 \
  --patch-size 4,4 \
  --padding 0,0 \
  --log-path null
