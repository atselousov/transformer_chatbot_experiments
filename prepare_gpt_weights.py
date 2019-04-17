import argparse
import json
import wget
from pathlib import Path
import shutil

import numpy as np
import torch
import tensorflow as tf


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights_dir', type=Path, help='')
    parser.add_argument('--output_dir', type=Path, help='')
    parser.add_argument('--model', type=str, choices=['gpt', 'gpt2'], help='')

    return parser


def load_weights(weights_dir, gpt2=False):
    if gpt2:
        base_url = 'https://storage.googleapis.com/gpt-2/models/117M/'
        file_names = ['checkpoint', 'encoder.json', 'hparams.json', 'model.ckpt.data-00000-of-00001',
                      'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']
    else:
        base_url = 'https://github.com/openai/finetune-transformer-lm/blob/master/model/'
        file_names = ['encoder_bpe_40000.json', 'params_shapes.json', 'vocab_40000.bpe'] + \
                     [f'params_{i}.npy' for i in range(10)]

    weights_dir.mkdir(parents=True, exist_ok=True)
    for name in file_names:
        output_path = weights_dir / name
        if not output_path.exists():
            url = base_url + name
            wget.download(url, str(output_path))


def prepare_gpt_weights(weights_dir):
    load_weights(weights_dir, gpt2=False)

    parameters_shapes_path = weights_dir / 'parameters_shapes.json'
    parameters_weights_paths = [weights_dir / f'params_{i}.npy' for i in range(10)]

    with open(parameters_shapes_path, 'r') as parameters_shapes_file:
        parameters_shapes = json.load(parameters_shapes_file)

    parameters_weights = [np.load(path) for path in parameters_weights_paths]
    parameters_offsets = np.cumsum([np.prod(shape) for shape in parameters_shapes])
    parameters_weights = np.split(np.concatenate(parameters_weights, 0), parameters_offsets)[:-1]
    parameters_weights = [p.reshape(s) for p, s in zip(parameters_weights, parameters_shapes)]

    pos_embedding = torch.from_numpy(parameters_weights[0])
    tok_embedding = torch.from_numpy(parameters_weights[1])
    parameters_weights = parameters_weights[2:]

    transformer_state = {}
    for layer_id in range(12):
        offset = 12 * layer_id
        current_state = {f'layers.{layer_id}.attn.qkv_proj.weight': torch.from_numpy(parameters_weights[offset + 0].squeeze(dim=0).transpose((1, 0))),
                         f'layers.{layer_id}.attn.qkv_proj.bias': torch.from_numpy(parameters_weights[offset + 1]),
                         f'layers.{layer_id}.attn.out_proj.weight': torch.from_numpy(parameters_weights[offset + 2].squeeze(dim=0).transpose((1, 0))),
                         f'layers.{layer_id}.attn.out_proj.bias': torch.from_numpy(parameters_weights[offset + 3]),
                         f'layers.{layer_id}.attn_norm.weight': torch.from_numpy(parameters_weights[offset + 4]),
                         f'layers.{layer_id}.attn_norm.bias': torch.from_numpy(parameters_weights[offset + 5]),
                         f'layers.{layer_id}.ff.layer_1.weight': torch.from_numpy(parameters_weights[offset + 6].squeeze(dim=0).transpose((1, 0))),
                         f'layers.{layer_id}.ff.layer_1.bias': torch.from_numpy(parameters_weights[offset + 7]),
                         f'layers.{layer_id}.ff.layer_2.weight': torch.from_numpy(parameters_weights[offset + 8].squeeze(dim=0).transpose((1, 0))),
                         f'layers.{layer_id}.ff.layer_2.bias': torch.from_numpy(parameters_weights[offset + 9]),
                         f'layers.{layer_id}.ff_norm.weight': torch.from_numpy(parameters_weights[offset + 10]),
                         f'layers.{layer_id}.ff_norm.bias': torch.from_numpy(parameters_weights[offset + 11])}

        transformer_state.update(current_state)

    state = {'pos_embedding': pos_embedding,
             'tok_embedding': tok_embedding,
             'transformer_state': transformer_state}

    return state


def prepare_gpt2_weights(weights_dir):
    def get_weights(name):
        return tf.train.load_variable(weights_dir, name)

    load_weights(weights_dir, gpt2=True)

    pos_embedding = torch.from_numpy(get_weights('model/wpe'))
    tok_embedding = torch.from_numpy(get_weights('model/wte'))

    transformer_state = {'final_norm.weight': torch.from_numpy(get_weights('model/ln_f/g')),
                         'final_norm.bias': torch.from_numpy(get_weights('model/ln_f/b'))}
    for layer_id in range(12):
        current_state = {f'layers.{layer_id}.attn.qkv_proj.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/attn/c_attn/w').squeeze(dim=0).transpose((1, 0))),
                         f'layers.{layer_id}.attn.qkv_proj.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/attn/c_attn/b')),
                         f'layers.{layer_id}.attn.out_proj.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/attn/c_proj/w').squeeze(dim=0).transpose((1, 0))),
                         f'layers.{layer_id}.attn.out_proj.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/attn/c_proj/b')),
                         f'layers.{layer_id}.attn_norm.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/ln_1/g')),
                         f'layers.{layer_id}.attn_norm.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/ln_1/b')),
                         f'layers.{layer_id}.ff.layer_1.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/mlp/c_fc/w').squeeze(dim=0).transpose((1, 0))),
                         f'layers.{layer_id}.ff.layer_1.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/mlp/c_fc/b')),
                         f'layers.{layer_id}.ff.layer_2.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/mlp/c_proj/w').squeeze(dim=0).transpose((1, 0))),
                         f'layers.{layer_id}.ff.layer_2.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/mlp/c_proj/b')),
                         f'layers.{layer_id}.ff_norm.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/ln_2/g')),
                         f'layers.{layer_id}.ff_norm.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/ln_2/b'))}

        transformer_state.update(current_state)

    state = {'pos_embedding': pos_embedding,
             'tok_embedding': tok_embedding,
             'transformer_state': transformer_state}

    return state


def main(args):
    state_path = args.output_dir / 'model_state.pt'
    new_bpe_vocab_path = args.output_dir / 'bpe.vocab'
    new_bpe_codes_path = args.output_dir / 'bpe.code'

    if args.model == 'gpt':
        state = prepare_gpt_weights(args.weights_dir)
        bpe_vocab_path = args.weights_dir / 'encoder_bpe_40000.json'
        bpe_codes_path = args.weights_dir / 'vocab_40000.bpe'
    elif args.model == 'gpt2':
        state = prepare_gpt2_weights(args.weights_dir)
        bpe_vocab_path = args.weights_dir / 'encoder.json'
        bpe_codes_path = args.weights_dir / 'vocab.bpe'
    else:
        assert False

    torch.save(state, state_path)
    shutil.copy(str(bpe_vocab_path), str(new_bpe_vocab_path))
    shutil.copy(str(bpe_codes_path), str(new_bpe_codes_path))


if __name__ == '__main__':
    arg_parser = get_parser()
    args = arg_parser.parse_known_args()[0]
    main(args)
