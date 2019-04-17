import git
import torch
from attrdict import AttrDict
from model.utils import gpt_config, gpt2_config
from decouple import Csv, config as env_config


def get_model_config(gpt2=False):
    default_config = gpt2_config() if gpt2 else gpt_config()

    config = AttrDict({ # weights
                        'weights_path': './checkpoints/last_checkpoint.pt',

                        # decoding
                       'max_seq_len': 128,
                       'beam_size': env_config('BEAM_SIZE', default=3, cast=int),
                       'diversity_groups': env_config('DIVERSITY_GROUP', default=1, cast=int),
                       'diversity_coef': env_config('DIVERSITY_COEF', default=0, cast=int),
                       'annealing_prob': env_config('ANNEALING_PROB', default=0, cast=float),
                       'annealing_topk': env_config('ANNEALING_TOPK', default=None, cast=int),
                       'length_penalty': env_config('LENGTH_PENALTY', default=0.6, cast=float),

                        # embeddings
                       'constant_pos_embeddings': env_config('CONSTANT_POS_EMBEDDINGS', default=False, cast=bool),
                       'sparse_embeddings': env_config('SPARSE_EMBEDDINGS', default=False, cast=bool),
                       'dialog_embeddings': env_config('DIALOG_EMBEDDINGS', default=True, cast=bool),
                       'use_start_end': env_config('USE_START_END', default=False, cast=bool),

                        # model type
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
                       'ff_dropout': default_config.ff_dropout,})

    return config


def get_trainer_config(gpt2=False):
    repo = git.Repo(search_parent_directories=True)

    config = AttrDict({# datasets
                       'train_datasets': env_config('TRAIN_DATASETS', default='datasets/ConvAI2/train_self_original.txt', cast=Csv(str)),
                       'test_datasets': env_config('TEST_DATASETS', default='datasets/ConvAI2/valid_self_original.txt', cast=Csv(str)),
                       'persona_augment': env_config('PERSONA_AUGMENT', default=False, cast=bool),
                       'persona_aug_syn_proba': env_config('PERSONA_AUG_SYN_PROBA', default=0.0, cast=float),

                       # weights
                       'load_checkpoint': env_config('LOAD_CHECKPOINT', default='', cast=str),
                       'gpt_parameters_dir': './gpt2_parameters' if gpt2 else './gpt_parameters',

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
                       'fp16': env_config('FP16', default=False, cast=bool),
                       'loss_scale': env_config('LOSS_SCALE', default=0, cast=float),

                       # info
                       'writer_comment': env_config('WRITER_COMMENT', default='', cast=str),
                       'repo_id': str(repo),
                       'repo_sha': str(repo.head.object.hexsha),
                       'repo_branch': str(repo.active_branch),
                       
                       # other
                       'seed': 0,
                       'n_jobs': 4,
                       'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                       'test_period': 1,
                       'evaluate_full_sequences': env_config('EVALUATE_FULL_SEQUENCES', default=True, cast=bool),
                       'limit_eval_size': env_config('LIMIT_EVAL_TIME', default=-1, cast=int),
                       'limit_train_size': env_config('LIMIT_TRAIN_TIME', default=-1, cast=int),})

    return config
