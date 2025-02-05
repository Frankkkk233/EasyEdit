import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, AutoTokenizer

from ..util.globals import *
from ..trainer.utils import dict_to


class MQuAKEDataset(Dataset):
    """
    Dataset of new factual knowledge based on MQuAKE.
    """

    def __init__(self, data_dir: str, size: typing.Optional[int] = None, config=None, *args, **kwargs):
        data_dir = Path(data_dir)
        mquake_loc = data_dir

        if config is not None:
            self.config = config
        if config is not None and hasattr(config, 'max_length'):
            self.max_length = config.max_length
        else:
            self.max_length = 40

        # For Meta Training
        if config is not None and hasattr(config, 'tokenizer_name'):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.model.name
            )
            # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
            if self.config.use_customized:
                tokenizer = AutoTokenizer.from_pretrained(tok_name)
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else: 

                tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name,trust_remote_code=True
                )
                tokenizer.pad_token_id = tokenizer.eos_token_id
            if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('GPTTokenizer Detected, Set pad token id and left padding!!!')
            elif isinstance(tokenizer, LlamaTokenizer):
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('LlamaTokenizer Detected, Set pad token id and left padding!!!')
            elif 'qwen' in config.model_name.lower():
                tokenizer.eos_token='<|endoftext|>'
                tokenizer.pad_token='<|endoftext|>'
                tokenizer.unk_token='<|endoftext|>'
                # tokenizer.padding_side = 'left'
                # print('QwenTokenizer Detected, Set pad token id and left padding!!!')
            elif 'mistral' in config.model_name.lower():
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = 'left'
                print('MistralTokenizer Detected, Set pad token id and left padding!!!')
            self.tok = tokenizer

        with open(mquake_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            prompt = ""
            subject = ""
            target_new = ""
            rephrase_prompt = ""
            if len(record["requested_rewrite"])!=1:continue
            for x in record["requested_rewrite"]:
                prompt = prompt + x["prompt"].format(x["subject"]) + "?"
                subject = subject + x["subject"] + ","
                target_new = target_new + x["target_new"]["str"] + ","
                rephrase_prompt = rephrase_prompt + x["question"]
            subject = subject[:-1] if subject.endswith(',') else subject
            target_new = target_new[:-1] if target_new.endswith(',') else target_new
            data.append(
                {
                    "case_id": i,
                    "prompt": prompt,
                    "subject": subject,
                    "target_new": target_new,
                    "rephrase_prompt": rephrase_prompt,
                    "portability_prompt": record["questions"],
                    "portability_ground_truth": [record["new_answer"]]*len(record["questions"]),
                }
            )

        if size is not None:
            data = data[:size]
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def collate_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        cond = [b["cond"] for b in batch]
        rephrase = [b["rephrase_prompt"] for b in batch]
        loc = [b["locality_prompt"] for b in batch]
        loc_ans = [b["locality_ground_truth"] for b in batch]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": cond,
                "rephrase": rephrase,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"])

        edit_inner["labels"] = edit_labels

        edit_rephrase = {}
        edit_rephrase["input_ids"] = batches["rephrase_input_ids"]
        edit_rephrase["attention_mask"] = batches["rephrase_attention_mask"]
        edit_rephrase["labels"] = edit_labels

        # loc
        loc = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )

        loc_ans = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )
        loc["decoder_attention_mask"] = loc_ans["attention_mask"]
        loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])

        # portability TODO

        cond = {k[5:]: v for k, v in batches.items() if k.startswith("cond")}
        batch = {
            "edit_inner": edit_inner,
            "edit_rephrase": edit_rephrase,
            "loc": loc,
            "cond": cond,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)

    def collate_gpt_fn(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        cond = [b["cond"] for b in batch]
        rephrase = [b["rephrase_prompt"] for b in batch]
        loc = [b["locality_prompt"] for b in batch]
        loc_ans = [b["locality_ground_truth"] for b in batch]

        # if (hasattr(self.config, 'alg') and self.config.alg == 'SERAC') or \
        #         (hasattr(self.config, 'alg_name') and self.config.alg_name == 'SERAC'):
        #     def flatten(nested_list: typing.List[typing.List]):
        #         return [item for nested_list_ in nested_list for item in nested_list_]
        #
        #     trg = [' ' + trg_ for trg_ in trg]
        #     loc_ans = [' ' + loc_ans_ for loc_ans_ in loc_ans]
        #     src = [[src_ + self.tok.decode(self.tok(trg_, truncation=True, max_length=self.config.max_length)['input_ids'][:i])
        #             for i in range(len(self.tok(trg_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for src_, trg_ in zip(src, trg)]
        #     rephrase = [[rephrase_ + self.tok.decode(self.tok(trg_, truncation=True, max_length=self.config.max_length)['input_ids'][:i])
        #             for i in range(len(self.tok(trg_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for rephrase_, trg_ in zip(rephrase, trg)]
        #     loc = [[loc_ + self.tok.decode(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)['input_ids'][:i])
        #             for i in range(len(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for loc_, loc_ans_ in zip(loc, loc_ans)]
        #     trg = [[self.tok.decode(self.tok(trg_, truncation=True, max_length=self.config.max_length)['input_ids'][i])
        #             for i in range(len(self.tok(trg_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for src_, trg_ in zip(src, trg)]
        #     loc_ans = [[self.tok.decode(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)['input_ids'][i])
        #             for i in range(len(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for loc_, loc_ans_ in zip(loc, loc_ans)]
        #
        #     src, rephrase, trg, loc, loc_ans = flatten(src), flatten(rephrase), flatten(trg), flatten(loc), flatten(loc_ans)
        #
        # else:
        src = [src_ + ' ' + trg_ for src_, trg_ in zip(src, trg)]
        rephrase = [rephrase_ + ' ' + trg_ for rephrase_, trg_ in zip(rephrase, trg)]
        loc = [loc_ + ' ' + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]

        if 'gpt' in self.config.tokenizer_class.lower():
            trg = [' ' + t for t in trg]
            loc_ans = [' ' + t for t in loc_ans]
            
        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": cond,
                "rephrase": rephrase,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"])

        edit_inner["labels"] = edit_labels

        edit_rephrase = {}
        edit_rephrase["input_ids"] = batches["rephrase_input_ids"]
        edit_rephrase["attention_mask"] = batches["rephrase_attention_mask"]
        edit_rephrase["labels"] = edit_labels

        # loc
        loc = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )

        loc_ans = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        )
        loc["decoder_attention_mask"] = loc_ans["attention_mask"]
        loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])

        # portability TODO

        cond = {k[5:]: v for k, v in batches.items() if k.startswith("cond")}
        batch = {
            "edit_inner": edit_inner,
            "edit_rephrase": edit_rephrase,
            "loc": loc,
            "cond": cond,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)
    
    def collate_customized(self, batch):
        src = [b["prompt"] for b in batch]
        trg = [b["target_new"] for b in batch]
        cond = [b.get("cond",None) for b in batch]
        rephrase = [b["rephrase_prompt"] for b in batch]
        loc = [b.get("locality_prompt",None) for b in batch]
        loc_ans = [b.get("locality_ground_truth",None) for b in batch]
        port = [b.get("portability_prompt",None) for b in batch]
        port_ans = [b.get("portability_ground_truth",None) for b in batch]

        # if (hasattr(self.config, 'alg') and self.config.alg == 'SERAC') or \
        #         (hasattr(self.config, 'alg_name') and self.config.alg_name == 'SERAC'):
        #     def flatten(nested_list: typing.List[typing.List]):
        #         return [item for nested_list_ in nested_list for item in nested_list_]
        #
        #     trg = [' ' + trg_ for trg_ in trg]
        #     loc_ans = [' ' + loc_ans_ for loc_ans_ in loc_ans]
        #     src = [[src_ + self.tok.decode(self.tok(trg_, truncation=True, max_length=self.config.max_length)['input_ids'][:i])
        #             for i in range(len(self.tok(trg_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for src_, trg_ in zip(src, trg)]
        #     rephrase = [[rephrase_ + self.tok.decode(self.tok(trg_, truncation=True, max_length=self.config.max_length)['input_ids'][:i])
        #             for i in range(len(self.tok(trg_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for rephrase_, trg_ in zip(rephrase, trg)]
        #     loc = [[loc_ + self.tok.decode(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)['input_ids'][:i])
        #             for i in range(len(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for loc_, loc_ans_ in zip(loc, loc_ans)]
        #     trg = [[self.tok.decode(self.tok(trg_, truncation=True, max_length=self.config.max_length)['input_ids'][i])
        #             for i in range(len(self.tok(trg_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for src_, trg_ in zip(src, trg)]
        #     loc_ans = [[self.tok.decode(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)['input_ids'][i])
        #             for i in range(len(self.tok(loc_ans_, truncation=True, max_length=self.config.max_length)["input_ids"]))]
        #            for loc_, loc_ans_ in zip(loc, loc_ans)]
        #
        #     src, rephrase, trg, loc, loc_ans = flatten(src), flatten(rephrase), flatten(trg), flatten(loc), flatten(loc_ans)
        #
        # else:

        src = [src_ + ' ' + trg_ for src_, trg_ in zip(src, trg)]
        rephrase = [rephrase_ + ' ' + trg_ for rephrase_, trg_ in zip(rephrase, trg)]
        trg = [' ' + t for t in trg]
        if False:
            loc = [loc_ + ' ' + loc_ans_ for loc_, loc_ans_ in zip(loc, loc_ans)]
            loc_ans = [' ' + t for t in loc_ans]
        port =   [[port__+' '+ port_ans__ for port__, port_ans__ in zip(sub_port, sub_port_ans)]
        for sub_port, sub_port_ans in zip(port, port_ans)]
        port_ans = [[' ' + t for t in sub_port_ans] for sub_port_ans in port_ans]
        if self.config.batch_size==1:
            port=port[0]
            port_ans=port_ans[0]
        if 'gpt' in self.config.tokenizer_class.lower():
            trg = [' ' + t for t in trg]
            loc_ans = [' ' + t for t in loc_ans]
            
        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "rephrase": rephrase,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch

        # edit_inner
        edit_inner = {}
        edit_inner["input_ids"] = batches["src_input_ids"]
        edit_inner["attention_mask"] = batches["src_attention_mask"]
        edit_labels = self.get_edit_labels(batches["trg_input_ids"])
        edit_labels = edit_labels[:,1:]  if 'llama' in self.config.model_class.lower() else edit_labels  #去掉bos

        edit_inner["labels"] = edit_labels

        edit_rephrase = {}
        edit_rephrase["input_ids"] = batches["rephrase_input_ids"]
        edit_rephrase["attention_mask"] = batches["rephrase_attention_mask"]
        edit_rephrase["labels"] = edit_labels

        # loc
        if False:
            loc = dict(
            self.tok(
                loc,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
                )
            )

            loc_ans = dict(
            self.tok(
                loc_ans,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
                )
            )

            loc["decoder_attention_mask"] = loc_ans["attention_mask"]
            loc["labels"] = self.get_edit_labels(loc_ans["input_ids"])
            loc['labels'] = loc['labels'][:,1:]  #去掉bos
            loc["decoder_attention_mask"]=loc["decoder_attention_mask"][:,1:]
        # port
        #这里三个问题长短不一，有padding，会体现在attn_mask里面


        port = dict(
        self.tok(
            port,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
            )
        )

        port_ans = dict(
        self.tok(
            port_ans,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
            )
        )

        port["decoder_attention_mask"] = port_ans["attention_mask"]
        port["labels"] = self.get_edit_labels(port_ans["input_ids"])
        port['labels'] = port['labels'][:,1:] if 'llama' in self.config.model_class.lower() else port['labels'] #去掉bos
        port["decoder_attention_mask"]=port["decoder_attention_mask"][:,1:] if 'llama' in self.config.model_class.lower() else port['decoder_attention_mask']




        batch = {
            "edit_inner": edit_inner,
            "edit_rephrase": edit_rephrase,
            "port": port,
            "raw": batch,
        }
        return dict_to(batch, self.config.device)


def adjust_mq(test_data):
# 假设数据记录 test_data 和 record 已经定义

# 先创建各个字段的列表
    case_ids = [i for i in range(len(test_data))]  # 假设 i 是按顺序生成的
    prompts = [test_data_['prompt'] for test_data_ in test_data]
    subjects = [test_data_['subject'] for test_data_ in test_data]
    targets = [test_data_['target_new'] for test_data_ in test_data]
    rephrase_prompts = [test_data_['rephrase_prompt'] for test_data_ in test_data]
    portability_prompts = [test_data_["portability_prompt"] for test_data_ in test_data]
    portability_ground_truths = [test_data_["portability_ground_truth"] for test_data_ in test_data]
    portability_inputs = {
        'neighborhood':{
            'prompt': portability_prompts,
            'ground_truth': portability_ground_truths
        },
    }

    # 打包成字典
    data_dict =  {
            'ds_name':'mq',
            "case_id": case_ids,
            "prompts": prompts,
            "subject": subjects,
            "target_new": targets,
            "rephrase_prompts": rephrase_prompts,
            "portability_inputs": portability_inputs
        }
    return data_dict

