#!/usr/bin/env python3

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from tensorboardX import SummaryWriter
import torch

from config import get_model_config, get_trainer_config
from metrics import nlp_metrics, specified_nlp_metric
from model.dataset import FacebookDataset
from model.text import get_vocab
from model.trainer import Trainer
from model.transformer_model import TransformerModel
from model.utils import f1_score, set_seed
from model.gpt_utils import prepare_gpt_weights, prepare_bpe_vocab, prepare_bpe_codes, load_gpt_weights


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=-1, help="Distributed training.")
    parser.add_argument('--server_ip', type=str, default='', help="Used for debugging on GPU machine.")
    parser.add_argument('--server_port', type=str, default='', help="Used for debugging on GPU machine.")

    return parser


def _create_writer(local_rank, writer_comment):
    class DummyWriter:
        """ Used for distributed training (from NVIDIA apex example).
            A dummy logger used so that only the main process write and log informations.
        """
        def __init__(self, *input, **kwargs):
            self.log_dir = "runs/dummy_logs/"

        def add_scalar(self, *input, **kwargs):
            pass

    # Log only on main process
    if local_rank not in [-1, 0]:
        sys.stdout = open(f"runs/log_distributed_{local_rank}", "w")  # dump sdtout
        writer = DummyWriter()
    else:
        writer = SummaryWriter(comment=writer_comment)

    return writer


def _create_model(model_config, trainer_config, logger):
    vocab_path, codes_path = Path(model_config.vocab_path), Path(model_config.codes_path)
    if not vocab_path.exists():
        logger.info('Downloading bpe vocabulary')
        prepare_bpe_vocab(vocab_path, model=model_config.model)
    if not codes_path.exists():
        logger.info('Downloading bpe codes')
        prepare_bpe_codes(codes_path, model=model_config.model)

    vocab = get_vocab(vocab_path=vocab_path,
                      codes_path=codes_path,
                      tokenizer_type=model_config.model,
                      zero_shot_special_tokens=model_config.zero_shot)

    model = TransformerModel(n_layers=model_config.n_layers,
                             n_embeddings=len(vocab),
                             n_pos_embeddings=model_config.n_pos_embeddings,
                             embeddings_size=model_config.embeddings_size,
                             padding_idx=vocab.pad_id,
                             n_heads=model_config.n_heads,
                             dropout=model_config.dropout,
                             embed_dropout=model_config.embed_dropout,
                             attn_dropout=model_config.attn_dropout,
                             ff_dropout=model_config.ff_dropout,
                             bos_id=vocab.bos_id,
                             eos_id=vocab.eos_id,
                             sent_dialog_id=vocab.sent_dialog_id,
                             max_seq_len=model_config.max_seq_len,
                             sample_best_beam=model_config.sample_best_beam,
                             beam_size=model_config.beam_size,
                             length_penalty=model_config.length_penalty,
                             annealing_topk=model_config.annealing_topk,
                             annealing_proba=model_config.annealing_proba,
                             diversity_coef=model_config.diversity_coef,
                             diversity_groups=model_config.diversity_groups,
                             multiple_choice_head=(trainer_config.hits_weight > 0 and trainer_config.negative_samples > 0),
                             constant_pos_embedding=model_config.constant_pos_embedding,
                             single_input=model_config.single_input,
                             dialog_embeddings=model_config.dialog_embeddings,
                             shared_enc_dec=model_config.shared_enc_dec,
                             successive_attention=model_config.successive_attention,
                             sparse_embeddings=model_config.sparse_embeddings,
                             shared_attention=model_config.shared_attention)

    if not trainer_config.load_checkpoint:
        gpt_weights_path = Path(trainer_config.gpt_weights_path)
        if not gpt_weights_path.exists():
            logger.info('Downloading weights')
            prepare_gpt_weights(gpt_weights_path, model=model_config.model)
        gpt_weights = torch.load(gpt_weights_path)

        load_gpt_weights(model.decoder, gpt_weights, n_special_tokens=vocab.n_special_tokens)
        load_gpt_weights(model.encoder,gpt_weights, n_special_tokens=vocab.n_special_tokens)

        logger.info(f'GPT weights loaded from {trainer_config.gpt_weights_path}')

    return vocab, model


def _create_datasets(model_config, trainer_config, vocab, logger):
    logger.info('Loading datasets')

    train_dataset = FacebookDataset(paths=trainer_config.train_datasets,
                                    vocab=vocab,
                                    max_lengths=(model_config.n_pos_embeddings - 1) // (3 if model_config.single_input else 1),  # A bit restrictive here
                                    dialog_embeddings=model_config.dialog_embeddings,
                                    cache='train_cache.bin',
                                    use_start_end=model_config.use_start_end,
                                    negative_samples=trainer_config.negative_samples,
                                    augment=trainer_config.persona_augment,
                                    aug_syn_proba=trainer_config.persona_aug_syn_proba,
                                    limit_size=trainer_config.limit_train_size)
    test_dataset = FacebookDataset(paths=trainer_config.test_datasets,
                                   vocab=vocab,
                                   max_lengths=(model_config.n_pos_embeddings - 1) // (3 if model_config.single_input else 1),  # A bit restrictive here
                                   dialog_embeddings=model_config.dialog_embeddings,
                                   cache='test_cache.bin',
                                   use_start_end=model_config.use_start_end,
                                   negative_samples=-1,  # Keep all negative samples
                                   augment=False,
                                   limit_size=trainer_config.limit_test_size)

    logger.info(f'Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}')

    return train_dataset, test_dataset


def _create_trainer(model_config, trainer_config, model, vocab, train_dataset, test_dataset, writer, logger):
    device = torch.device(trainer_config.device)
    model_trainer = Trainer(model=model,
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            writer=writer,
                            train_batch_size=trainer_config.train_batch_size,
                            batch_split=trainer_config.train_batch_split,
                            test_batch_size=trainer_config.test_batch_size,
                            lr=trainer_config.lr,
                            lr_warmup=trainer_config.lr_warmup,
                            weight_decay=trainer_config.weight_decay,
                            s2s_weight=trainer_config.s2s_weight,
                            label_smoothing=trainer_config.label_smoothing,
                            lm_weight=trainer_config.lm_weight,
                            risk_weight=trainer_config.risk_weight,
                            hits_weight=trainer_config.hits_weight,
                            single_input=model_config.single_input,
                            n_jobs=trainer_config.n_jobs,
                            clip_grad=trainer_config.clip_grad,
                            device=device,
                            ignore_idxs=vocab.special_tokens_ids,
                            local_rank=args.local_rank,
                            apex_level=model_config.opt_level,
                            apex_loss_scale=trainer_config.loss_scale,
                            linear_schedule=trainer_config.linear_schedule,
                            n_epochs=trainer_config.n_epochs,
                            evaluate_full_sequences=trainer_config.evaluate_full_sequences,
                            checkpoints_dir=Path(writer.log_dir))

    if trainer_config.load_checkpoint:
        state = torch.load(trainer_config.load_checkpoint, map_location=device)
        model_trainer.load_state_dict(state)
        logger.info('Trainer checkpoint loaded from {}'.format(trainer_config.load_checkpoint))

    return model_trainer


def external_metrics_func(full_references, full_predictions, epoch, metric=None):
    references_file_path = Path(writer.log_dir) / f'references_file_{epoch}'
    predictions_file_path = Path(writer.log_dir) / f'predictions_file_{epoch}'
    with open(references_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_references))
    with open(predictions_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_predictions))

    if metric is not None:
        return specified_nlp_metric([references_file_path], predictions_file_path, metric)

    nist, bleu, meteor, entropy, div, avg_len = nlp_metrics([references_file_path], predictions_file_path)

    metrics = {'meteor': meteor, 'avg_len': avg_len}
    for name, metric in (('nist', nist), ('entropy', entropy), ('div', div), ('bleu', bleu)):
        for i, m in enumerate(metric, 1):
            metrics['{}_{}'.format(name, i)] = m

    return metrics


def _create_after_epoch_funcs(model_trainer, test_dataset, vocab, test_period, n_samples=0):
    def sample_text_func(epoch):
        model_trainer.model.eval()
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target, _ in samples:
            contexts = [torch.tensor([c], dtype=torch.long, device=model_trainer.device) for c in [persona_info, dialog]
                        if len(c) > 0]
            prediction = model_trainer.model.predict(contexts)[0]

            persona_info_str = vocab.ids2string(persona_info[1:-1])
            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(vocab.talker1_eos, '').replace(vocab.talker2_eos, '')
            target_str = vocab.ids2string(target[1:-1])
            prediction_str = vocab.ids2string(prediction)

            logger.info('\n')
            logger.info('Persona info:\n\t{}'.format(persona_info_str))
            logger.info('Dialog:{}'.format(dialog_str))
            logger.info('Target:\n\t{}'.format(target_str))
            logger.info('Prediction:\n\t{}'.format(prediction_str))

    def test_func(epoch):
        if (epoch + 1) % test_period == 0:
            metric_funcs = {'f1_score': f1_score}
            model_trainer.test(metric_funcs, external_metrics_func, epoch)

    return [sample_text_func, test_func]

def _create_risk_metric_func(risk_metric, vocab):
    """ risk_metric selected in:
        f1, meteor, avg_len, nist_{1, 2, 3, 4}, entropy_{1, 2, 3, 4}, div_{1, 2}, bleu_{1, 2, 3, 4}
    """

    def f1_risk(predictions, targets):
        scores = f1_score(predictions, targets, average=False)
        assert all([0 <= s <= 1.0 for s in scores])
        return [1 - s for s in scores]

    def external_metric_risk(predictions, targets):
        string_targets = list(vocab.ids2string(t) for t in targets)
        string_predictions = list(vocab.ids2string(t) for t in predictions)
        metrics = [external_metrics_func([t], [p], epoch=-1, metric=risk_metric) for p, t in
                    zip(string_predictions, string_targets)]

        if any([s in risk_metric for s in ['entropy', 'nist', 'avg_len']]):
            return [-m for m in metrics]

        assert all([0 <= s <= 1.0 for s in metrics]), metrics

        return [1 - m for m in metrics]

    if risk_metric == 'f1':
        return f1_risk

    return external_metric_risk

def main(args, logger):
    set_seed(seed=0)

    model_config, trainer_config = get_model_config(), get_trainer_config()
    writer = _create_writer(args.local_rank, trainer_config.writer_comment)
    vocab, model = _create_model(model_config, trainer_config, logger)
    train_dataset, test_dataset = _create_datasets(model_config, trainer_config, vocab, logger)
    model_trainer = _create_trainer(model_config, trainer_config, model, vocab, train_dataset, test_dataset, writer, logger)

    after_epoch_funcs = _create_after_epoch_funcs(model_trainer, test_dataset, vocab, trainer_config.test_period, n_samples=0)
    risk_func = _create_risk_metric_func(trainer_config.risk_metric, vocab)

    logger.info("model config: {}".format(model_config))
    logger.info("trainer config: {}".format(trainer_config))
    logger.info(f"Logging to {writer.log_dir}")  # Let's save everything on an experiment in the ./runs/XXX/directory

    if args.local_rank in [-1, 0]:
        with open(Path(writer.log_dir) / "model_config.json", "w") as f:
            json.dump(model_config, f)
        with open(Path(writer.log_dir) / "trainer_config.json", "w") as f:
            json.dump(trainer_config, f)

    model_trainer.train(after_epoch_funcs=after_epoch_funcs, risk_func=risk_func, save_last=True, save_interrupted=True)


if __name__ == '__main__':
    arg_parser = get_parser()
    args = arg_parser.parse_known_args()[0]

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.ERROR)
    logger = logging.getLogger(__file__)

    if args.server_ip and args.server_port and args.local_rank in [-1, 0]:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    main(args, logger)
