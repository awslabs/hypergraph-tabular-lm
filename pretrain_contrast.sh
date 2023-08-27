# This script is tested on a AWS G5 node with 8 A10 GPUs; 
# Please change it according to your own environment.
# pretrain with contrastive objective
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore run_pretrain.py --data_path './data/pretrain/' --contrast_bipartite_edge True --gradient_clip_val 2.0 --accelerator "gpu" --devices 8 --replace_sampler_ddp False --accumulate_grad_batches 4 --base_learning_rate 1e-4
