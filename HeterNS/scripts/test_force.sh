export CUDA_VISIBLE_DEVICES=6
export NCCL_P2P_LEVEL=NVL

model_name=Unisolver_HeterNS
data_path=YOUR_DATA_PATH
pretrain_path=$YOUR_CHECKPOINT_PATH

python exp_ns_test_force_0.5.py \
  --data-path $data_path \
  --in_dim 10 \
  --out_dim 1 \
  --ntest 200 \
  --h 256 \
  --w 256 \
  --h-down 4 \
  --w-down 4 \
  --T-in 10 \
  --T-out 10 \
  --learning-rate 0.001 \
  --model $model_name \
  --log-path None \
  --batch-size 20 \
  --epochs 201 \
  --model-pretrain-path $pretrain_path

python exp_ns_test_force_1.5.py \
  --data-path $data_path/ \
  --in_dim 10 \
  --out_dim 1 \
  --ntest 200 \
  --h 256 \
  --w 256 \
  --h-down 4 \
  --w-down 4 \
  --T-in 10 \
  --T-out 10 \
  --learning-rate 0.001 \
  --model $model_name \
  --log-path None \
  --batch-size 20 \
  --epochs 201 \
  --model-pretrain-path $pretrain_path

python exp_ns_test_force_2.5.py \
  --data-path $data_path \
  --in_dim 10 \
  --out_dim 1 \
  --ntest 200 \
  --h 256 \
  --w 256 \
  --h-down 4 \
  --w-down 4 \
  --T-in 10 \
  --T-out 10 \
  --learning-rate 0.001 \
  --model $model_name \
  --log-path None \
  --batch-size 20 \
  --epochs 201 \
  --model-pretrain-path $pretrain_path

python exp_ns_test_force_3.5.py \
  --data-path $data_path \
  --in_dim 10 \
  --out_dim 1 \
  --ntest 200 \
  --h 256 \
  --w 256 \
  --h-down 4 \
  --w-down 4 \
  --T-in 10 \
  --T-out 10 \
  --learning-rate 0.001 \
  --model $model_name \
  --log-path None \
  --batch-size 20 \
  --epochs 201 \
  --model-pretrain-path $pretrain_path

