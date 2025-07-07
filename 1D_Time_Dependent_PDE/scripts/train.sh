export CUDA_VISIBLE_DEVICES='2,3,4,5'
export NCCL_P2P_LEVEL=NVL

torchrun --nproc_per_node 4 exp_1D_PDEs.py \
        --in_dim 1 \
        --out_dim 1 \
        --h 256 \
        --h-down 1 \
        --learning-rate 0.001 \
        --model Unisolver_1D \
        --model-save-path ./checkpoints/burgers \
        --model-save-name ViT_1D_INR_global_embed_differ_layer_gate_polyinr_L_llm_subspace1_3000k_no_clip \
        --log-path ns_logs/ViT_1D_INR_global_embed_differ_layer_gate_polyinr_L_llm_subspace1_3000k_no_clip \
        --batch-size 32 \
        --epochs 501 \
        --resolution 256 \
        --step-size 10