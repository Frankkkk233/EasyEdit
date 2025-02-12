# hparams/TRAINING/MEND/llama3.1-instruct.yaml:
#set:results_dir: ./results/InstructEdit/evoke
#max_iters: 10000
# log_interval: 100
# val_interval: 250
# early_stop_patience: 1000
CUDA_VISIBLE_DEVICES=0 python model_train.py --ds evoke --alg MEND --model llama3.1-instruct