import json
import urllib.request
import tempfile
import copy
import os

import numpy as np
import torch
import tensorflow as tf
import torch.nn as nn
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
    'gpt2_small': {
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
    }, 
    'gpt2_medium': {
        'base_url': 'https://storage.googleapis.com/gpt-2/models/345M/',
        'weights': ['checkpoint', 'hparams.json', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta'],
        'bpe_vocab': 'encoder.json',
        'bpe_codes': 'vocab.bpe',
        'config': {
            'n_layers': 24,
            'n_embeddings': 50257,
            'n_pos_embeddings': 1024,
            'embeddings_size': 1024,
            'n_heads': 16,
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


def _get_gpt_weights(params_dir, model):
    parameters_shapes_path = os.path.join(params_dir, MODEL_INFO[model]['weights'][0])
    parameters_weights_paths = [os.path.join(params_dir, file) for file in MODEL_INFO[model]['weights'][1:]]

    with open(parameters_shapes_path, 'r') as parameters_shapes_file:
        parameters_shapes = json.load(parameters_shapes_file)

    parameters_weights = [np.load(path) for path in parameters_weights_paths]
    parameters_offsets = np.cumsum([np.prod(shape) for shape in parameters_shapes])
    parameters_weights = np.split(np.concatenate(parameters_weights, 0), parameters_offsets)[:-1]
    parameters_weights = [p.reshape(s) for p, s in zip(parameters_weights, parameters_shapes)]

    def get_weights(idx, transpose=False):
        weights = parameters_weights[idx]
        if transpose:
            weights = weights.squeeze(0).transpose((1, 0))
        return torch.from_numpy(weights)

    pos_embedding = get_weights(0)
    tok_embedding = get_weights(1)

    n_layers = MODEL_INFO[model]['config']['n_layers']
    transformer_state = {}
    for layer_id in range(n_layers):
        offset = 2 + n_layers * layer_id
        current_state = {f'layers.{layer_id}.attn.qkv_proj.weight': get_weights(offset + 0, transpose=True),
                         f'layers.{layer_id}.attn.qkv_proj.bias': get_weights(offset + 1),
                         f'layers.{layer_id}.attn.out_proj.weight': get_weights(offset + 2, transpose=True),
                         f'layers.{layer_id}.attn.out_proj.bias': get_weights(offset + 3),
                         f'layers.{layer_id}.attn_norm.weight': get_weights(offset + 4),
                         f'layers.{layer_id}.attn_norm.bias': get_weights(offset + 5),
                         f'layers.{layer_id}.ff.layer_1.weight': get_weights(offset + 6, transpose=True),
                         f'layers.{layer_id}.ff.layer_1.bias': get_weights(offset + 7),
                         f'layers.{layer_id}.ff.layer_2.weight': get_weights(offset + 8, transpose=True),
                         f'layers.{layer_id}.ff.layer_2.bias': get_weights(offset + 9),
                         f'layers.{layer_id}.ff_norm.weight': get_weights(offset + 10),
                         f'layers.{layer_id}.ff_norm.bias': get_weights(offset + 11)}

        transformer_state.update(current_state)

    state = {'pos_embedding': pos_embedding,
             'tok_embedding': tok_embedding,
             'transformer_state': transformer_state}

    return state


def _get_gpt2_weights(params_dir, model):
    def get_weights(name, transpose=False):
        weights = tf.train.load_variable(params_dir, name)
        if transpose:
            weights = weights.squeeze(0).transpose((1, 0))
        return torch.from_numpy(weights)

    pos_embedding = get_weights('model/wpe')
    tok_embedding = get_weights('model/wte')

    n_layers = MODEL_INFO[model]['config']['n_layers']
    transformer_state = {'final_norm.weight': get_weights('model/ln_f/g'),
                         'final_norm.bias': get_weights('model/ln_f/b')}
    for layer_id in range(n_layers):
        layer_state = {f'layers.{layer_id}.attn.qkv_proj.weight': get_weights(f'model/h{layer_id}/attn/c_attn/w', transpose=True),
                       f'layers.{layer_id}.attn.qkv_proj.bias': get_weights(f'model/h{layer_id}/attn/c_attn/b'),
                       f'layers.{layer_id}.attn.out_proj.weight': get_weights(f'model/h{layer_id}/attn/c_proj/w', transpose=True),
                       f'layers.{layer_id}.attn.out_proj.bias': get_weights(f'model/h{layer_id}/attn/c_proj/b'),
                       f'layers.{layer_id}.attn_norm.weight': get_weights(f'model/h{layer_id}/ln_1/g'),
                       f'layers.{layer_id}.attn_norm.bias': get_weights(f'model/h{layer_id}/ln_1/b'),
                       f'layers.{layer_id}.ff.layer_1.weight': get_weights(f'model/h{layer_id}/mlp/c_fc/w', transpose=True),
                       f'layers.{layer_id}.ff.layer_1.bias': get_weights(f'model/h{layer_id}/mlp/c_fc/b'),
                       f'layers.{layer_id}.ff.layer_2.weight': get_weights(f'model/h{layer_id}/mlp/c_proj/w', transpose=True),
                       f'layers.{layer_id}.ff.layer_2.bias': get_weights(f'model/h{layer_id}/mlp/c_proj/b'),
                       f'layers.{layer_id}.ff_norm.weight': get_weights(f'model/h{layer_id}/ln_2/g'),
                       f'layers.{layer_id}.ff_norm.bias': get_weights(f'model/h{layer_id}/ln_2/b')}

        transformer_state.update(layer_state)

    state = {'pos_embedding': pos_embedding,
             'tok_embedding': tok_embedding,
             'transformer_state': transformer_state}

    return state


def _check_supported_models(model):
    supported_models = list(MODEL_INFO.keys())
    if model not in supported_models:
        raise ValueError(f'Wrong model: expected {supported_models}, got {model}')


def prepare_gpt_weights(output_path, model):
    _check_supported_models(model)

    with tempfile.TemporaryDirectory() as params_dir:
        for file in MODEL_INFO[model]['weights']:
            file_url = MODEL_INFO[model]['base_url'] + file
            file_path = os.path.join(params_dir, file)
            _download_file(file_url, file_path)

        if model == 'gpt':
            weights = _get_gpt_weights(params_dir, model)
        elif model == 'gpt2_small' or model == 'gpt2_medium':
            weights = _get_gpt2_weights(params_dir, model)
        else:
            assert False

        torch.save(weights, output_path)


def prepare_bpe_vocab(output_path, model):
    _check_supported_models(model)
    
    file_url = MODEL_INFO[model]['base_url'] + MODEL_INFO[model]['bpe_vocab']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _download_file(file_url, output_path)


def prepare_bpe_codes(output_path, model):
    _check_supported_models(model)

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
