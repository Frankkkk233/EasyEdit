import torch
import torch.nn.functional as F


def kl_loc_loss(pre, post, mask=None):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.contiguous().view(-1, pre.shape[-1])
    post_ = post.contiguous().view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (
                pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))
            ).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError
def slice_list(matrix,start_indices,left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[:,start_indices[0]-1:-1]
        else:
            return matrix[:,start_indices[0]:]

def binary_log_probs(pred, targ):
    neg_mask = torch.ones_like(pred)
    neg_mask[targ == 0] *= -1
    pred = pred * neg_mask
    log_probs = F.logsigmoid(pred)
    acc = (log_probs.exp() > 0.5).float().mean()
    return {
        "acc": acc,
        "log_prob": log_probs.mean(),
        "prob": log_probs.exp().mean(),
        "nll": -log_probs.mean(),
        "n_tokens": log_probs.shape[0],
    }

def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()

def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels

def multiclass_log_probs( pred, targ, shift=False, eps=torch.finfo(torch.float32).eps, exact_match=False, **kwargs):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        if "inner_sent" in kwargs or "personality" in kwargs or "multimodal" in kwargs:
            targ = targ[:, 1:]
        else:
            pred = pred[:, -targ.size(1):]
        # targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
    
    # debug
    # print(pred.shape, targ.shape)
    # if pred.size(1) > targ.size(1):
    #     pred = pred[:, :targ.size(1)]

    if exact_match:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        if pred.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()
        acc = correct.sum() / num_non_padding
    
    if "inner_sent" in kwargs or "inner_per" in kwargs:
        same_sent_mask = kwargs["same_mask"]
        good_mask = mask * same_sent_mask.unsqueeze(-1)
        bad_mask = mask * (~same_sent_mask.unsqueeze(-1))

        good_log_prob = masked_mean(unmasked_log_probs, good_mask)
        bad_log_prob = masked_mean((1 - unmasked_log_probs.exp() + eps).log(), bad_mask)

        n_tokens = good_mask.float().sum()
        log_prob = good_log_prob
        prob = log_prob.exp()

        if kwargs["unlikelihood"]:
            nll = -good_log_prob - bad_log_prob
        else:
            nll = -good_log_prob
    else:
        n_tokens = mask.float().sum()
        log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
        prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
        
        nll = -log_prob
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": nll,
        "pred_ids": pred_ids,
    }


def masked_log_probs(config, pred, targ, shift=False, exact_match=False, **kwargs):
    pred = pred.to(torch.float32)

    if not (pred.dim() == 2 or pred.dim() == 3):
        raise RuntimeError(f"Expected pred to have 2 or 3 dimensions, got {pred.shape}")

    if pred.shape[-1] == 1:
        return binary_log_probs(pred, targ)
    else:
        return multiclass_log_probs(config, pred, targ, shift=shift, exact_match=exact_match, **kwargs)



        
        
def compute_prob(model, tok, prompts, targets, device , generate=False):
    if type(prompts)!=list:
        prompts=[prompts]
    if type(targets)!=list:
        targets=[targets]
    if len(prompts)!=len(targets):
        targets=targets*len(prompts)
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    before_padding_side = tok.padding_side
    tok.padding_side = 'left'
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    tok.padding_side = before_padding_side
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        if generate:    
            answers = model.generate(prompt_tok['input_ids'],attention_mask=prompt_tok['attention_mask'], max_new_tokens=prompt_target_tok['input_ids'].size(1)-prompt_tok['input_ids'].size(1) )
        else:
            outputs = model(**prompt_target_tok)
    if not generate:
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1)
    labels = prompt_target_tok['input_ids']
    if generate:
        answers= slice_list(answers,prompt_len,left=False)
    else:
        answers = slice_list(answers,prompt_len,left=True)
    labels = slice_list(labels,prompt_len,left=False)
    prob_list=[]
    answer_list=[]
    for i in range(labels.size(0)):
        if not generate:
            result=multiclass_log_probs(logits[i].unsqueeze(0), labels[i].unsqueeze(0), shift=True)
            prob_list.append(result['prob'].item())
        answer=''.join(tok.batch_decode(answers[i]))
        if answer[0]==' ':
            answer=answer[1:]
        answer_list.append(answer)
    
    return {'prob':prob_list,'pred':answer_list}
    
def overfitting_metric(model, tok, requests, device,return_metric):
    prompt=requests.get('prompt',[])
    label=requests.get('label',[])     #answer for the question
    edit_target=requests.get('edit_target',[])   #target after editing
    if 'OAP' in return_metric and not any(requests['orignal_answer']):
        origin=compute_prob(model, tok, prompt, label, device, True)['pred']    #original answer
        requests['orignal_answer']=origin
    origin=requests.get('orignal_answer',[])

    
    if 'CAP' in return_metric:
        cap=compute_prob(model, tok, prompt, label, device)['prob']
    if 'OAP' in return_metric:
        oap=compute_prob(model, tok, prompt, origin, device)['prob']
    if 'DP' in return_metric:
        dp=compute_prob(model, tok, prompt, edit_target, device)['prob']
    if 'EOS' in return_metric:
        eos=[int(cap_>dp_) for cap_,dp_ in zip(cap,dp)]
    if 'AMS' in return_metric:
        ams=[int(cap_>oap_) for cap_,oap_ in zip(cap,oap)]
    return {
        'CAP':cap if 'CAP' in return_metric else None,
        'OAP':oap if 'OAP' in return_metric else None,
        'original_answer':origin if 'OAP' in return_metric else None,
        'DP':dp if 'DP' in return_metric else None,
        'EOS':eos if 'EOS' in return_metric else None,
        'AMS':ams if 'AMS' in return_metric else None,
    }
def compute_overfitting_score(model, tok, requests, device):
    #reliability score
    if 'prompt' in requests.keys() and any(requests['prompt']):
        reli_request={
        'prompt':requests.get('prompt',[]),
        'label':requests.get('target_new',[])
        }
        reli_score=overfitting_metric(model, tok, reli_request, device, return_metric=['CAP'])
    else:
        reli_score=None
    #rephrase score
    if 'rephrase_prompt' in requests.keys() and any(requests['rephrase_prompt']):
        rephrase_request={
            'prompt':requests.get('rephrase_prompt',[]),
            'label':requests.get('target_new',[])
        }
        rephrase_score=overfitting_metric(model, tok, rephrase_request, device, return_metric=['CAP'])
    else:
        rephrase_score=None
    #prot score
    if 'port_prompt' in requests.keys() and any(requests['port_prompt']):
        port_request={
            'prompt':requests.get('port_prompt',[]),
        'label':requests.get('port_answer',[]),
        'edit_target':requests.get('target_new',[]),
        'orignal_answer':requests.get('port_orignal_answer',[])
        }
        port_score=overfitting_metric(model, tok, port_request, device, return_metric=['CAP','OAP','DP','EOS','AMS'])
        requests['port_orignal_answer']=port_score['original_answer']
    else:
        port_score=None
    #relation score
    if 'relation_prompt' in requests.keys() and any(requests['relation_prompt']):
        relation_request={
            'prompt':requests.get('relation_prompt',[]),
        'label':requests.get('ground_truth',[]),
        'edit_target':requests.get('target_new',[]),
        }
        relation_score=overfitting_metric(model, tok, relation_request, device, return_metric=['CAP','DP','EOS'])
    else:
        relation_score=None
    #subjective score
    if 'subject_prompt' in requests.keys() and any(requests['subject_prompt']):
        subject_request={
            'prompt':requests.get('subject_prompt',[]),
        'label':requests.get('subject_answer',[]),
        'edit_target':requests.get('target_new',[]),
        }
        subject_score=overfitting_metric(model, tok, subject_request, device, return_metric=['CAP','DP','EOS'])
    else:
        subject_score=None
    #prefix score
    if 'prefix_prompt' in requests.keys() and any(requests['prefix_prompt']):
        prefix_request={
            'prompt':requests.get('prefix_prompt',[]),
        'label':requests.get('ground_truth',[]),
        'edit_target':requests.get('target_new',[]),
        }
        prefix_score=overfitting_metric(model, tok, prefix_request, device, return_metric=['CAP','DP','EOS'])
    else:
        prefix_score=None
    return {
        'reliability':reli_score,
        'rephrase':rephrase_score,
        'portability':port_score,
        'relation':relation_score,
        'subject':subject_score,
        'prefix':prefix_score,
    }
    



