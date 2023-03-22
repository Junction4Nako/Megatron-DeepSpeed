"""
split the a BLOOM pre-trained models from huggingface to fit the Megatron-Deepspeed checkpoint loading
"""
from transformers import AutoConfig, BloomForCausalLM
import argparse
import torch
import os

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


def split_layer(items, save_name, tp=1):
    split_dict = [{} for i in range(tp)]
    # processing
    for item in items:
        split_out = split_weight(item[0], item[1], tp)
        assert len(split_out[1]) == tp, 'the split operation output != tp size'
        for i in range(tp):
            split_dict[i][split_out[0]] = split_out[1][i]

    # saving
    for i in range(tp):
        target_dict = split_dict[i]
        torch.save(target_dict, save_name.format(i))
    return None


def split_model(model_state_dict, model_cfg, save_path, tp=1):
    n_layer = model_cfg.n_layer
    # embedding layer
    print('processing the embedding layer')
    emb_module = [(k, v) for k,v in model_state_dict.items() if 'word_embeddings' in k]
    emb_format = os.path.join(save_path, 'layer_01-model_{:02d}-model_states.pt')
    split_layer(emb_module, emb_format, tp)
    # encoder
    for i in range(n_layer):
        print('processing the {}-th encoder layer'.format(i))
        layer_module = [(k, v) for k,v in model_state_dict.items() if 'transformer.h.{}.'.format(i) in k]
        layer_id = 3 + i
        layer_format = os.path.join(save_path, 'layer_{:02d}'.format(layer_id) + '-model_{:02d}-model_states.pt')
        split_layer(layer_module, layer_format, tp)
    # final ln
    print('processing the last layernorm layer')
    fln_module = [(k, v) for k,v in model_state_dict.items() if 'ln_f' in k]
    fln_format = os.path.join(save_path, 'layer_{:02d}'.format(n_layer+4) + '-model_{:02d}-model_states.pt')
    split_layer(fln_module, fln_format, tp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_model', type=str, default=None, help='model to be splitted')
    parser.add_argument('--tp_size', type=int, default=1, help='the tensor parallelism size')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save weights to')
    parser.add_argument('--half', action='store_true', help='whether to transform weights to fp16')
    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        print('creating the new directory')
        os.mkdir(args.save_path)

    global fp16
    fp16 = args.half

    # build a model on CPU and load the parameters
    print('loading the model in CPU')
    full_model = BloomForCausalLM.from_pretrained(args.source_model)
    config = AutoConfig.from_pretrained(args.source_model)

    state_dict = full_model.state_dict()
    if fp16:
        state_dict = {k:v.half() for k,v in state_dict.items()}
    split_model(state_dict, config, args.save_path, args.tp_size)

if __name__=='__main__':
    main()

    