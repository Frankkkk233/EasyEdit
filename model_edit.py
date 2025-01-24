import os.path
import sys
import argparse
import json
import random
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MELOHyperParams,
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    DeepEditApiHyperParams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset,CounterFactDataset,MQuAKEDataset
from easyeditor.dataset.counterfact import adjust_cf
from easyeditor.dataset.MQuAKE import adjust_mq
import os           

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# def multi_model_edit(hparams,train_ds,val_ds):





def edit(editor,data_dict):
    

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



def edit_test(model:str=None,layer:str=None,edit_ds:str=None,val_ds:str=None,alg:str=None,suffix:str=None):
    print('model:',model)
    print('layer:',layer)
    print('data:',edit_ds)
    print('alg:',alg)


#alg_set
    if alg=='FT':
        editing_hparams = FTHyperParams
    if alg=='MEND':
        editing_hparams = MENDHyperParams 
    if alg=='MEMIT':
        editing_hparams = MEMITHyperParams
    if alg=='DeepEdit_Api':
        editing_hparams = DeepEditApiHyperParams
    if alg=='MELO':
        editing_hparams = MELOHyperParams 
    config_path='./hparams/'+alg+'/'+model+'.yaml'
#ds_set
    if edit_ds=='cf':
        #just for debug
        # ds = CounterFactDataset('./data/counterfact/counterfact-testing.json')
        #edit
        ds = CounterFactDataset('./data/counterfact/counterfact-edit.json')
        ds=adjust_cf(ds)
    if edit_ds=='mq':
        #just for debug
        # ds = MQuAKEDataset('./data/mquake/MQuAKE-CF-testing.json')
        #edit
        ds = MQuAKEDataset('./data/mquake/MQuAKE-CF-3k-v2.json')
        ds=adjust_mq(ds)


    hparams=editing_hparams.from_hparams(config_path)
    editor=BaseEditor.from_hparams(hparams)


    print(f'---------------use cuda:{hparams.device}-----------------------')


    metrics=edit(editor,ds)
    json.dump(metrics, open('./logs/'+model+'_'+alg+'_'+edit_ds+suffix+'.json', 'w+'), indent=4)
    
    # train(hparams,train_ds,val_ds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds",default='mq',type=str,help="test_data")
    parser.add_argument("--alg",default='MEMIT',type=str,help="algorithm")
    
    # 解析命令行参数
    args = parser.parse_args()
    edit_test(model='llama3.1-instruct',layer='none',edit_ds=args.ds,alg=args.alg,suffix='')

if __name__ == '__main__':
    main()