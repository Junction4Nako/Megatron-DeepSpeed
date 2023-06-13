"""
split the a BLOOM pre-trained models from huggingface to fit the Megatron-Deepspeed checkpoint loading
"""
from transformers import AutoConfig, BloomForCausalLM
import argparse
import torch
import os
from collections import defaultdict

fp16 = False

def split_weight(k, v, tp=1):
    if 'layernorm' in k:
        # layernorm will not be splitted
        if 'word_embeddings' in k:
            new_k = 'word_embeddings.norm.{}'.format(k.split('.')[-1])
        else:
            new_k = '.'.join(k.split('.')[3:])
        return new_k, [v]*tp
    elif 'ln_f' in k:
        new_k = k.split('.')[-1]
        return new_k, [v]*tp
    elif 'word_embeddings.weight' in k:
        # embedding layer
        new_k = 'word_embeddings.weight'
        source_dim = v.shape[0]
        target_dim = source_dim // tp
        return new_k, torch.split(v, target_dim, dim=0)
    else:
        # transformer layers
        if 'query_key_value' in k or 'dense_h_to_4h' in k:
            split_form = 'col'
        elif 'self_attention.dense' in k or 'dense_4h_to_h' in k:
            split_form = 'row'
        else:
            raise NotImplementedError('found unsupported module name {}, please check'.format(k))
        if split_form == 'col':
            source_dim = v.shape[0]
            target_dim = source_dim // tp
            new_v = torch.split(v, target_dim, dim=0)
        else:
            if 'bias' in k:
                new_v = [v] * tp
            else:
                source_dim = v.shape[1]
                target_dim = source_dim // tp
                new_v = torch.split(v, target_dim, dim=1)
        new_k = '.'.join(k.split('.')[3:])
        return new_k, new_v

def merge_weight(k, v):
    if k == 'weight' or k == 'bias':
        # the final layernorm without merge
        print('found the weight for ln_f')
        return k, v[0]
    if 'word_embeddings.norm' in k:
        # word_embddings norm
        new_k = 'word_embeddings_layernorm.{}'.format(k.split('.')[-1])
        return new_k, v[0]
    if 'layernorm' in k:
        # layernorm will not be splitted
        return k, v[0]
    elif 'word_embeddings.weight' in k:
        # embedding layer
        new_k = 'word_embeddings.weight'
        return new_k, torch.cat(v, dim=0)
    else:
        # transformer layers
        if 'query_key_value' in k or 'dense_h_to_4h' in k:
            split_form = 'col'
        elif 'self_attention.dense' in k or 'dense_4h_to_h' in k:
            split_form = 'row'
        else:
            raise NotImplementedError('found unsupported module name {}, please check'.format(k))
        if split_form == 'col':
            new_v = torch.cat(v, dim=0)
        else:
            if 'bias' in k:
                new_v = v[0]
            else:
                new_v = torch.cat(v, dim=1)
        new_k = k
        return new_k, new_v


def merge_layer(items, target_sd, key_format='{}', tp=1):
    split_dict = [torch.load(item, map_location='cpu') for item in sorted(items)] # sorted is important here!
    assert len(split_dict) == tp, 'the size of state dict list should be the same as tp size'
    # processing
    for k in split_dict[0].keys():
        # all split dict must have the same keys
        merged_key, merged_weight = merge_weight(k, [sd[k] for sd in split_dict])
        target_sd[key_format.format(merged_key)] = merged_weight
    return True


def merge_model(n_layer, all_checkpoints, save_path, tp=1):
    merged_state_dict = {}
    # embedding layer
    print('processing the embedding layer')
    emb_module_list = all_checkpoints[1]
    merge_layer(emb_module_list, merged_state_dict, 'transformer.{}', tp)
    merged_state_dict['lm_head.weight'] = merged_state_dict['transformer.word_embeddings.weight'] # tie the lm head with word embeddings
    # encoder
    for i in range(n_layer):
        print('processing the {}-th encoder layer'.format(i))
        layer_id = 3 + i
        layer_state_dict_list = all_checkpoints[layer_id]
        layer_format = 'transformer.h.{}'.format(i) + '.{}'
        merge_layer(layer_state_dict_list, merged_state_dict, layer_format, tp)
    # final ln
    print('processing the last layernorm layer')
    fln_module_list = all_checkpoints[n_layer + 4]
    merge_layer(fln_module_list, merged_state_dict, 'transformer.ln_f.{}', tp)
    
    print('saving the model')
    torch.save(merged_state_dict, save_path)
    return None




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_model', type=str, default=None, help='model to be splitted')
    parser.add_argument('--tp_size', type=int, default=None, help='the tensor parallelism size')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save weights to')
    parser.add_argument('--half', action='store_true', help='whether to transform weights to fp16')
    args = parser.parse_args()

    global fp16
    fp16 = args.half

    # check the checkpoint
    all_ckpts = defaultdict(list)
    all_layer_no = set()
    all_split_no = set()
    for fn in os.listdir(args.source_model):
        if fn.startswith('layer_'):
            info = fn.split('-')
            layer_no = int(info[0][-2:])
            split_no = int(info[1][-2:])
            all_layer_no.add(layer_no)
            all_split_no.add(split_no)
            all_ckpts[layer_no].append(os.path.join(args.source_model, fn))
    
    tp_size = max(all_split_no) + 1
    num_layers = max(all_layer_no) - 4
    if args.tp_size is not None:
        assert args.tp_size == tp_size, 'the tp size of the checkpoint ({}) is not matched with your definition ({})'.format(tp_size, args.tp_size)
    print('founded {} layers with tp size as {}'.format(num_layers, tp_size))

    merge_model(num_layers, all_ckpts, args.save_path, tp_size)
    # split_model(state_dict, config, args.save_path, args.tp_size)

if __name__=='__main__':
    main()

    