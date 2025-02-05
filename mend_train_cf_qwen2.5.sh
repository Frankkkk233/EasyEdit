# hparams/TRAINING/MEND/qwen2.5-instruct.yaml:
#set:results_dir: ./results/InstructEdit/cf
#max_iters: 100000
# log_interval: 1000
# val_interval: 2500
# early_stop_patience: 10000
CUDA_VISIBLE_DEVICES=0 python model_train.py --ds cf --alg MEND --model qwen2.5-instruct