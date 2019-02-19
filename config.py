from attrdict import AttrDict
import torch
from model.utils import openai_transformer_config
import git

repo = git.Repo(search_parent_directories=True)

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
                       'max_seq_len': 256,
                       'beam_size': 3,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': None,
                       'annealing': 0,
                       'length_penalty': 0.6,
                       'n_segments': None,
                       'multiple_choice_head': True})

    return config


def get_trainer_config():
    config = AttrDict({'n_epochs': 0,
                       'batch_size': 256,
                       'batch_split': 64,
                       'lr': 6.25e-5,
                       'lr_warmup': 0.002,  # a fraction of total training (epoch * train_set_length) if linear_schedule == True
                       'weight_decay': 0.01,
                       's2s_weight': 1,
                       'lm_weight': 0.5,
                       'risk_weight': 0,
                       'hits_weight': 1,
                       'negative_samples': 2,
                       'single_input': True,
                       'dialog_embeddings': True,
                       'n_jobs': 0,
                       'label_smoothing': 0.1,
                       'clip_grad': None,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'persona_augment': False,
                       'persona_aug_syn_proba': 0.0,
                       'fp16': False,
                       'loss_scale': 0,
                       'linear_schedule': True,
                       'load_last': '',  # Now that we save several experiments you can put the path of the checpoint file you want to load here
                       'repo_id': str(repo),
                       'repo_sha': str(repo.head.object.hexsha),
                       'repo_branch': str(repo.active_branch),
                       'openai_parameters_dir': './parameters',
                       'last_checkpoint_path': 'last_checkpoint',  # there are now in the ./runs/XXX/ experiments folders
                       'eval_references_file': 'eval_references_file',
                       'eval_predictions_file': 'eval_predictions_file',
                       'interrupt_checkpoint_path': 'interrupt_checkpoint',  # there are now in the ./runs/XXX/ experiments folders
                       'train_datasets': ['./datasets/ConvAI2/train_self_original_no_cands.txt',],
                                          # './datasets/ConvAI2/train_self_revised_no_cands.txt',
                                          # './datasets/DailyDialog/train_dailydialog.txt'],
                       'train_datasets_cache': './datasets/train_datasets_cache.bin',
                       'test_datasets': ['./datasets/ConvAI2/valid_self_original_no_cands.txt',],
                                         # './datasets/ConvAI2/valid_self_revised_no_cands.txt',
                                         # './datasets/DailyDialog/valid_dailydialog.txt'],
                       'test_datasets_cache': './datasets/test_datasets_cache.bin'})

    local_config = AttrDict({'n_epochs': 100,
                       'batch_size': 2,
                       'batch_split': 1,
                       'lr': 6.25e-5,
                       'lr_warmup': 0.002,  # a fraction of total training (epoch * train_set_length) if linear_schedule == True
                       'weight_decay': 0.01,
                       's2s_weight': 1,
                       'lm_weight': 0.5,
                       'risk_weight': 0,
                       'hits_weight': 1,
                       'negative_samples': 2,
                       'single_input': True,
                       'dialog_embeddings': True,
                       'n_jobs': 0,
                       'label_smoothing': 0.1,
                       'clip_grad': None,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cpu',
                       'persona_augment': False,
                       'persona_aug_syn_proba': 0.0,
                       'load_last': True, 
                       'fp16': False,
                       'loss_scale': 0,
                       'linear_schedule': True,
                       'openai_parameters_dir': './parameters',
                       'last_checkpoint_path': './checkpoints/last_checkpoint',
                       'eval_references_file': './evaluation_files/eval_references_file',
                       'eval_predictions_file': './evaluation_files/eval_predictions_file',
                       'interrupt_checkpoint_path': './checkpoints/interrupt_checkpoint',
                       'train_datasets': # ['./datasets/ConvAI2/train_self_revised_no_cands.txt',],
                                          ['./datasets/ConvAI2/train_self_original_no_cands.txt',],
                                        #   './datasets/DailyDialog/train_dailydialog.txt'],
                       'load_last': '', 
                       'repo_id': str(repo),
                       'repo_sha': str(repo.head.object.hexsha),
                       'repo_branch': str(repo.active_branch),
                       'openai_parameters_dir': './parameters',
                       'last_checkpoint_path': 'last_checkpoint',
                       'eval_references_file': 'eval_references_file',
                       'eval_predictions_file': 'eval_predictions_file',
                       'interrupt_checkpoint_path': 'interrupt_checkpoint',
                       'train_datasets': ['./datasets/ConvAI2/train_self_revised_no_cands.txt',
                                          './datasets/ConvAI2/train_self_original_no_cands.txt',
                                          './datasets/DailyDialog/train_dailydialog.txt'],
                       'train_datasets_cache': './datasets/train_datasets_cache.bin',
                       'test_datasets': ['./datasets/ConvAI2/valid_self_revised_no_cands.txt',
                                         './datasets/ConvAI2/valid_self_original_no_cands.txt',
                                         './datasets/DailyDialog/valid_dailydialog.txt'],
                       'test_datasets_cache': './datasets/test_datasets_cache.bin'})

    return config if torch.cuda.is_available() else local_config
