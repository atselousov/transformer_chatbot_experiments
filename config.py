from attrdict import AttrDict
from copy import deepcopy
import torch
from model.utils import openai_transformer_config
import git
from decouple import config as env_config
from decouple import Csv


repo = git.Repo(search_parent_directories=True)


def cast2(type_):
    return lambda val: val if val is None else type_(val)


def get_model_config():
    default_config = openai_transformer_config()
    config = AttrDict({'bpe_vocab_path': './parameters/bpe.vocab',
                       'bpe_codes_path': './parameters/bpe.code',
                       'checkpoint_path': './checkpoints/last_checkpoint',  # Keep the checpoint folder for the checkpoints of the agents
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'normalize_embeddings': env_config('NORMALIZE_EMBEDDINGS', default=False, cast=bool),
                       'max_seq_len': 128,
                       'beam_size': env_config('BEAM_SIZE', default=3, cast=int),
                       'diversity_coef': env_config('DIVERSITY_COEF', default=0, cast=int),
                       'diversity_groups': env_config('DIVERSITY_GROUP', default=1, cast=int),
                       'annealing_topk': env_config('ANNEALING_TOPK', default=None, cast=cast2(int)),
                       'annealing': env_config('ANNEALING', default=0, cast=float),
                       'length_penalty': env_config('LENGTH_PENALTY', default=0.6, cast=float),
                       'n_segments': None,
                       'constant_embedding': env_config('CONSTANT_EMBEDDINGS', default=False, cast=bool),
                       'multiple_choice_head': env_config('MULTIPLE_CHOICE_HEAD', default=False, cast=bool),
                       'share_models': env_config('SHARE_MODELS', default=True, cast=bool),
                       'successive_attention': env_config('SUCCESSIVE_ATTENTION', default=False, cast=bool),
                       'sparse_embeddings': env_config('SPARSE_EMBEDDINGS', default=True, cast=bool),
                       'shared_attention': env_config('SHARED_ATTENTION', default=True, cast=bool),
                       'dialog_embeddings': env_config('DIALOG_EMBEDDINGS', default=True, cast=bool),
                       'single_input': env_config('SINGLE_INPUT', default=False, cast=bool),
                       'use_start_end': env_config('USE_START_END', default=False, cast=bool),
                       'apex_level': env_config('APEX_LEVEL', default=None, cast=cast2(str)),  # 'O0', 'O1', 'O2', 'O3',
                       'bs_temperature': env_config('BS_TEMPERATURE', default=1, cast=float),
                       'bs_nucleus_p': env_config('BS_NUCLEUS_P', default=0, cast=float)
                       })

    return config


def get_trainer_config():
    config = AttrDict({'n_epochs': env_config('N_EPOCHS', default=3, cast=int),
                       'writer_comment': env_config('WRITER_COMMENT', default='', cast=str),
                       'train_batch_size': env_config('TRAIN_BATCH_SIZE', default=128, cast=int),
                       'batch_split': env_config('BATCH_SPLIT', default=32, cast=int),
                       'test_batch_size': env_config('TEST_BATCH_SIZE', default=8, cast=int),
                       'lr': 6.25e-5,
                       'lr_warmup': 0.002,  # a fraction of total training (epoch * train_set_length) if linear_schedule == True
                       'weight_decay': 0.01,
                       's2s_weight': env_config('S2S_WEIGHT', default=1, cast=float),
                       'lm_weight': env_config('LM_WEIGHT', default=0, cast=float),
                       'risk_weight': env_config('RISK_WEIGHT', default=0, cast=float),
                       'hits_weight': env_config('HITS_WEIGHT', default=0, cast=float),
                       'negative_samples': env_config('NEGATIVE_SAMPLES', default=0, cast=int),
                       'n_jobs': 4,
                       'label_smoothing': env_config('LABEL_SMOOTHING', default=0.1, cast=float),
                       'clip_grad': None,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'zero_shot': env_config('ZERO_SHOT', default=False, cast=bool),
                       'persona_augment': env_config('PERSONA_AUGMENT', default=False, cast=bool),
                       'persona_aug_syn_proba': env_config('PERSONA_AUG_SYN_PROBA', default=0.0, cast=float),
                       'apex_loss_scale': env_config('APEX_LOSS_SCALE', default=None, cast=cast2(str)), # e.g. '128', 'dynamic'
                       'linear_schedule': env_config('LINEAR_SCHEDULE', default=True, cast=bool),
                       'evaluate_full_sequences': env_config('EVALUATE_FULL_SEQUENCES', default=True, cast=bool),
                       'limit_eval_size': env_config('LIMIT_EVAL_TIME', default=-1, cast=int),
                       'limit_train_size': env_config('LIMIT_TRAIN_TIME', default=-1, cast=int),
                       'risk_metric': env_config('RISK_METRIC', default='f1', cast=str),
                       'load_last': env_config('LOAD_LAST', default='', cast=str), #./checkpoints/last_checkpoint',  # Now that we save several experiments you can put the path of the checpoint file you want to load here
                       'repo_id': str(repo),
                       'repo_sha': str(repo.head.object.hexsha),
                       'repo_branch': str(repo.active_branch),
                       'openai_parameters_dir': './parameters',
                       'last_checkpoint_path': 'last_checkpoint',  # there are now in the ./runs/XXX/ experiments folders
                       'eval_references_file': 'eval_references_file',
                       'eval_predictions_file': 'eval_predictions_file',
                       'interrupt_checkpoint_path': 'interrupt_checkpoint',  # there are now in the ./runs/XXX/ experiments folders
                       'train_datasets': env_config('TRAIN_DATASETS', default='datasets/ConvAI2/train_self_original.txt', cast=Csv(str)),
                       'train_datasets_cache': 'train_cache.bin',
                       'test_datasets': env_config('TEST_DATASETS', default='datasets/ConvAI2/valid_self_original.txt', cast=Csv(str)),
                       'test_datasets_cache': 'test_cache.bin'
                       })

    local_config = deepcopy(config)
    local_config.train_batch_size = 2
    local_config.batch_split = 1
    local_config.test_batch_size = 3
    local_config.negative_samples = 2
    local_config.n_jobs = 0
    local_config.device = 'cpu'
    local_config.risk_weight = 1
    local_config.zero_shot = True
    local_config.fp16 = False
    local_config.train_datasets_cache = './datasets/train_datasets_cache.bin'
    local_config.test_datasets_cache = './datasets/test_datasets_cache.bin'

    return config if torch.cuda.is_available() else local_config
