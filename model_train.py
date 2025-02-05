import os.path
import sys
import argparse
import json
import random
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams, LoRAHyperParams, QLoRAHyperParams, \
    GraceHyperParams, PMETHyperParams,MELOHyperParams, MALMENTrainingHparams, MALMENHyperParams, WISEHyperParams, R_ROMEHyperParams, EMMETHyperParams, \
    DeepEditApiHyperParams, DPOHyperParams
from easyeditor import BaseEditor,EditTrainer
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset,CounterFactDataset,MQuAKEDataset
from easyeditor.dataset.counterfact import adjust_cf
from easyeditor.dataset.MQuAKE import adjust_mq
# def multi_model_edit(hparams,train_ds,val_ds):
import os     
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"





def train(editor,data_dict):
    

    metrics, edited_model, _ = editor.edit(**data_dict,
        train_ds=None,
        keep_original_weight=True
    )
    return metrics




def check_file(path):
    if os.path.exists(path):pass
    else:
        print("-------------file doesn't exit---------------")
        print('path:',path)
        sys.exit()   



def edit_train(model:str=None,layer:str=None,train_ds:str=None,val_ds:str=None,alg:str=None):
    print('model:',model)
    print('layer:',layer)
    print('data:',train_ds)
    print('alg:',alg)


#alg_set
    if alg=='MEND':
        config_path='./hparams/TRAINING/'+alg+'/'+model+'.yaml'
        training_hparams = MENDTrainingHparams    
    training_hparams=training_hparams.from_hparams(config_path)   

#ds_set
    if train_ds=='cf':
        #just for debug
        # train_ds = CounterFactDataset('./data/counterfact/counterfact-testing.json',config=training_hparams)
        # val_ds = CounterFactDataset('./data/counterfact/counterfact-testing.json',config=training_hparams)
        #train
        train_ds = CounterFactDataset('./data/counterfact/counterfact-train.json',config=training_hparams)
        val_ds = CounterFactDataset('./data/counterfact/counterfact-val.json',config=training_hparams)
    if train_ds=='mq':
        #just for debug
        # train_ds = MQuAKEDataset('./data/mquake/MQuAKE-CF-testing.json',config=training_hparams)
        # val_ds = MQuAKEDataset('./data/mquake/MQuAKE-CF-testing.json',config=training_hparams)
        #train
        train_ds = MQuAKEDataset('./data/mquake/MQuAKE-CF-3k-v2_train.json',config=training_hparams)
        val_ds = MQuAKEDataset('./data/mquake/MQuAKE-CF-3k-v2_test.json',config=training_hparams)



    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=val_ds
    )



    print(f'---------------use cuda:{training_hparams.device}-----------------------')


    trainer.run()
    # json.dump(metrics, open('./logs/'+model+'_'+alg+'_'+train_ds+'.json', 'w'), indent=4)
    
    # train(hparams,train_ds,val_ds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds",default='cf',type=str,help="test_data")
    parser.add_argument("--alg",default='MEND',type=str,help="algorithm")
    parser.add_argument("--model",default='qwen2.5-instruct',type=str)
    
    # 解析命令行参数
    args = parser.parse_args()
    edit_train(model=args.model,layer='none',train_ds=args.ds,alg=args.alg)

if __name__ == '__main__':
    main()