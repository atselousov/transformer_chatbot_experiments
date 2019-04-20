import git
import torch
from attrdict import AttrDict
from model.gpt_utils import MODEL_INFO
from decouple import Csv, config as env_config


model = env_config('MODEL', default='gpt2', cast=str)


def get_model_config():
    supported_models = list(MODEL_INFO.keys())
    if model not in supported_models:
        raise ValueError(f'Wrong model: expected {supported_models}, got {model}')

    default_config = AttrDict(MODEL_INFO[model]['config'])
    config = AttrDict({# weights and vocab
                       'weights_path': './checkpoints/last_checkpoint.pt',
                       'vocab_path': f'./{model}_parameters/bpe.vocab',
                       'codes_path': f'./{model}_parameters/bpe.codes',

                       # decoding
                       'max_seq_len': 128,
                       'beam_size': env_config('BEAM_SIZE', default=3, cast=int),
                       'sample_best_beam': False,
                       'diversity_groups': env_config('DIVERSITY_GROUP', default=1, cast=int),
                       'diversity_coef': env_config('DIVERSITY_COEF', default=0, cast=int),
                       'annealing_proba': env_config('ANNEALING_PROBA', default=0, cast=float),
                       'annealing_topk': env_config('ANNEALING_TOPK', default=None, cast=lambda x: x if x is None else int(x)),
                       'length_penalty': env_config('LENGTH_PENALTY', default=0.6, cast=float),

                       # embeddings
                       'constant_pos_embedding': env_config('CONSTANT_POS_EMBEDDING', default=False, cast=bool),
                       'sparse_embeddings': env_config('SPARSE_EMBEDDINGS', default=False, cast=bool),
                       'dialog_embeddings': env_config('DIALOG_EMBEDDINGS', default=True, cast=bool),
                       'use_start_end': env_config('USE_START_END', default=False, cast=bool),

                       # model type
                       'model': model,
                       'opt_level': env_config('OPT_LEVEL', default=None, cast=lambda x: x if x is None else str(x)),  # 'O0', 'O1', 'O2', 'O3'
                       'single_input': env_config('SINGLE_INPUT', default=False, cast=bool),
                       'zero_shot': env_config('ZERO_SHOT', default=False, cast=bool),
                       'shared_attention': env_config('SHARED_ATTENTION', default=True, cast=bool),
                       'successive_attention': env_config('SUCCESSIVE_ATTENTION', default=False, cast=bool),
                       'shared_enc_dec': env_config('SHARED_ENC_DEC', default=False, cast=bool),
                       
                       # gpt
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': default_config.n_pos_embeddings,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout})

    return config


def get_trainer_config():
    supported_models = list(MODEL_INFO.keys())
    if model not in supported_models:
        raise ValueError(f'Wrong model: expected {supported_models}, got {model}')
    
    repo = git.Repo(search_parent_directories=True)
    default_config = AttrDict(MODEL_INFO[model]['config'])
    config = AttrDict({# datasets
                       'train_datasets': env_config('TRAIN_DATASETS', default='datasets/ConvAI2/train_self_original.txt', cast=Csv(str)),
                       'limit_train_size': env_config('LIMIT_TRAIN_SIZE', default=-1, cast=int),
                       'limit_test_size': env_config('LIMIT_TEST_SIZE', default=-1, cast=int),
                       'test_datasets': env_config('TEST_DATASETS', default='datasets/ConvAI2/valid_self_original.txt', cast=Csv(str)),
                       'persona_augment': env_config('PERSONA_AUGMENT', default=False, cast=bool),
                       'persona_aug_syn_proba': env_config('PERSONA_AUG_SYN_PROBA', default=0.0, cast=float),

                       # weights
                       'load_checkpoint': env_config('LOAD_CHECKPOINT', default='', cast=str),
                       'gpt_weights_path': f'{model}_parameters/model.pt',

                       # optimization
                       'n_epochs': env_config('N_EPOCHS', default=20, cast=int),
                       'train_batch_size': env_config('TRAIN_BATCH_SIZE', default=128, cast=int),
                       'train_batch_split': env_config('TRAIN_BATCH_SPLIT', default=32, cast=int),
                       'test_batch_size': env_config('TEST_BATCH_SIZE', default=8, cast=int),
                       'lr': 6.25e-5,
                       'lr_warmup': 0.002,  # a fraction of total training (epoch * train_set_length) if linear_schedule == True
                       'linear_schedule': env_config('LINEAR_SCHEDULE', default=True, cast=bool),
                       'weight_decay': 0.01,
                       'clip_grad': None,
                       
                       # losses
                       's2s_weight': env_config('S2S_WEIGHT', default=1, cast=float),
                       'lm_weight': env_config('LM_WEIGHT', default=0, cast=float),
                       'label_smoothing': env_config('LABEL_SMOOTHING', default=0.1, cast=float),
                       'risk_weight': env_config('RISK_WEIGHT', default=0, cast=float),
                       'risk_metric': env_config('RISK_METRIC', default='f1', cast=str),
                       'hits_weight': env_config('HITS_WEIGHT', default=0, cast=float),
                       'negative_samples': env_config('NEGATIVE_SAMPLES', default=0, cast=int),
                       'loss_scale': env_config('LOSS_SCALE', default=None, cast=str),  # e.g. '128', 'dynamic'

                       # info
                       'writer_comment': env_config('WRITER_COMMENT', default='', cast=str),
                       'repo_id': str(repo),
                       'repo_sha': str(repo.head.object.hexsha),
                       'repo_branch': str(repo.active_branch),
                       
                       # other
                       'n_jobs': 4,
                       'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                       'test_period': 1,
                       'evaluate_full_sequences': env_config('EVALUATE_FULL_SEQUENCES', default=True, cast=bool)})

    return config
