from model_train import edit_train
from model_edit import edit_test
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds",default='mq',type=str,help="data")
    
    # 解析命令行参数
    args = parser.parse_args()
    edit_train(model='llama3.1-instruct',layer='none',train_ds=args.ds,alg='MEND')
    edit_test(model='llama3.1-instruct',layer='none',edit_ds=args.ds,alg='MEND')

if __name__ == '__main__':
    main()