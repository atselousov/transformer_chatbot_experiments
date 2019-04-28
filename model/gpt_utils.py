import json
import urllib.request
import shutil
import copy
import os

import numpy as np
import torch
import tensorflow as tf
import torch.nn as nn
from attrdict import AttrDict
from scipy.interpolate import RectBivariateSpline


MODEL_INFO = {
    'gpt': {
        'base_url': 'https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/',
        'weights': ['params_shapes.json'] + [f'params_{i}.npy' for i in range(10)],
        'bpe_vocab': 'encoder_bpe_40000.json',
        'bpe_codes': 'vocab_40000.bpe',
        'config': {
            'n_layers': 12,
            'n_embeddings': 40478,
            'n_pos_embeddings': 512,
            'embeddings_size': 768,
            'n_heads': 12,
            'dropout': 0.1,
            'embed_dropout': 0.1,
            'attn_dropout': 0.1,
            'ff_dropout': 0.1,
            'normalize_before': False
        }
    },
    'gpt2': {
        'base_url': 'https://storage.googleapis.com/gpt-2/models/117M/',
        'weights': ['checkpoint', 'hparams.json', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta'],
        'bpe_vocab': 'encoder.json',
        'bpe_codes': 'vocab.bpe',
        'config': {
            'n_layers': 12,
            'n_embeddings': 50257,
            'n_pos_embeddings': 1024,
            'embeddings_size': 768,
            'n_heads': 12,
            'dropout': 0.1,
            'embed_dropout': 0.1,
            'attn_dropout': 0.1,
            'ff_dropout': 0.1,
            'normalize_before': True
        }
    }
}


def _download_file(file_url, output_path):
    if not os.path.exists(output_path):
        urllib.request.urlretrieve(file_url, output_path)


def _get_gpt_weights(params_dir):
    parameters_shapes_path = os.path.join(params_dir, MODEL_INFO['gpt']['weights'][0])
    parameters_weights_paths = [os.path.join(params_dir, file) for file in MODEL_INFO['gpt']['weights'][1:]]

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
        current_state = {f'layers.{layer_id}.attn.qkv_proj.weight': torch.from_numpy(parameters_weights[offset + 0].squeeze(0).transpose((1, 0))),
                         f'layers.{layer_id}.attn.qkv_proj.bias': torch.from_numpy(parameters_weights[offset + 1]),
                         f'layers.{layer_id}.attn.out_proj.weight': torch.from_numpy(parameters_weights[offset + 2].squeeze(0).transpose((1, 0))),
                         f'layers.{layer_id}.attn.out_proj.bias': torch.from_numpy(parameters_weights[offset + 3]),
                         f'layers.{layer_id}.attn_norm.weight': torch.from_numpy(parameters_weights[offset + 4]),
                         f'layers.{layer_id}.attn_norm.bias': torch.from_numpy(parameters_weights[offset + 5]),
                         f'layers.{layer_id}.ff.layer_1.weight': torch.from_numpy(parameters_weights[offset + 6].squeeze(0).transpose((1, 0))),
                         f'layers.{layer_id}.ff.layer_1.bias': torch.from_numpy(parameters_weights[offset + 7]),
                         f'layers.{layer_id}.ff.layer_2.weight': torch.from_numpy(parameters_weights[offset + 8].squeeze(0).transpose((1, 0))),
                         f'layers.{layer_id}.ff.layer_2.bias': torch.from_numpy(parameters_weights[offset + 9]),
                         f'layers.{layer_id}.ff_norm.weight': torch.from_numpy(parameters_weights[offset + 10]),
                         f'layers.{layer_id}.ff_norm.bias': torch.from_numpy(parameters_weights[offset + 11])}

        transformer_state.update(current_state)

    state = {'pos_embedding': pos_embedding,
             'tok_embedding': tok_embedding,
             'transformer_state': transformer_state}

    return state


def _get_gpt2_weights(params_dir):
    def get_weights(name):
        return tf.train.load_variable(params_dir, name)

    pos_embedding = torch.from_numpy(get_weights('model/wpe'))
    tok_embedding = torch.from_numpy(get_weights('model/wte'))

    transformer_state = {'final_norm.weight': torch.from_numpy(get_weights('model/ln_f/g')),
                         'final_norm.bias': torch.from_numpy(get_weights('model/ln_f/b'))}
    for layer_id in range(12):
        current_state = {f'layers.{layer_id}.attn.qkv_proj.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/attn/c_attn/w').squeeze(0).transpose((1, 0))),
                         f'layers.{layer_id}.attn.qkv_proj.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/attn/c_attn/b')),
                         f'layers.{layer_id}.attn.out_proj.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/attn/c_proj/w').squeeze(0).transpose((1, 0))),
                         f'layers.{layer_id}.attn.out_proj.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/attn/c_proj/b')),
                         f'layers.{layer_id}.attn_norm.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/ln_1/g')),
                         f'layers.{layer_id}.attn_norm.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/ln_1/b')),
                         f'layers.{layer_id}.ff.layer_1.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/mlp/c_fc/w').squeeze(0).transpose((1, 0))),
                         f'layers.{layer_id}.ff.layer_1.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/mlp/c_fc/b')),
                         f'layers.{layer_id}.ff.layer_2.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/mlp/c_proj/w').squeeze(0).transpose((1, 0))),
                         f'layers.{layer_id}.ff.layer_2.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/mlp/c_proj/b')),
                         f'layers.{layer_id}.ff_norm.weight': torch.from_numpy(get_weights(f'model/h{layer_id}/ln_2/g')),
                         f'layers.{layer_id}.ff_norm.bias': torch.from_numpy(get_weights(f'model/h{layer_id}/ln_2/b'))}

        transformer_state.update(current_state)

    state = {'pos_embedding': pos_embedding,
             'tok_embedding': tok_embedding,
             'transformer_state': transformer_state}

    return state


def prepare_gpt_weights(output_path, model='gpt', params_dir='gpt_params_tmp'):
    supported_models = list(MODEL_INFO.keys())
    if model not in supported_models:
        raise ValueError(f'Wrong model: expected {supported_models}, got {model}')

    os.makedirs(params_dir, exist_ok=True)
    for file in MODEL_INFO[model]['weights']:
        file_url = MODEL_INFO[model]['base_url'] + file
        file_path = os.path.join(params_dir, file)
        _download_file(file_url, file_path)

    if model == 'gpt':
        weights = _get_gpt_weights(params_dir)
    elif model == 'gpt2':
        weights = _get_gpt2_weights(params_dir)
    else:
        assert False

    torch.save(weights, output_path)
    shutil.rmtree(params_dir, ignore_errors=True)


def prepare_bpe_vocab(output_path, model='gpt'):
    supported_models = list(MODEL_INFO.keys())
    if model not in supported_models:
        raise ValueError(f'Wrong model: expected {supported_models}, got {model}')
    
    file_url = MODEL_INFO[model]['base_url'] + MODEL_INFO[model]['bpe_vocab']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _download_file(file_url, output_path)


def prepare_bpe_codes(output_path, model='gpt'):
    supported_models = list(MODEL_INFO.keys())
    if model not in supported_models:
        raise ValueError(f'Wrong model: expected {supported_models}, got {model}')

    file_url = MODEL_INFO[model]['base_url'] + MODEL_INFO[model]['bpe_codes']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _download_file(file_url, output_path)


def load_gpt_weights(gpt_model, state, n_special_tokens=0):
    if isinstance(gpt_model.embedding.pos_embedding, nn.Embedding):
        if gpt_model.embedding.pos_embedding.num_embeddings - 1 > state['pos_embedding'].shape[0]:
            xx = np.linspace(0, state['pos_embedding'].shape[0], gpt_model.embedding.pos_embedding.num_embeddings - 1)
            new_kernel = RectBivariateSpline(np.arange(state['pos_embedding'].shape[0]),
                                             np.arange(state['pos_embedding'].shape[1]),
                                             state['pos_embedding'])
            state['pos_embedding'] = new_kernel(xx, np.arange(state['pos_embedding'].shape[1]))

        state['pos_embedding'] = state['pos_embedding'][:gpt_model.embedding.pos_embedding.num_embeddings - 1]
        gpt_model.embedding.pos_embedding.weight.data[1:] = state['pos_embedding']

    state['tok_embedding'] = state['tok_embedding'][:gpt_model.embedding.tok_embedding.num_embeddings - n_special_tokens]
    gpt_model.embedding.tok_embedding.weight.data[:n_special_tokens] = state['tok_embedding'].mean(dim=0)
    gpt_model.embedding.tok_embedding.weight.data[n_special_tokens:] = state['tok_embedding']

    gpt_model.load_state_dict(state['transformer_state'], strict=False)

    # Initialize shared attention layer is necessary
    for layer in gpt_model.layers:
        attn_state = layer.attn.state_dict()
        for context_attn in layer.context_attns:
            context_attn.load_state_dict(copy.deepcopy(attn_state))
