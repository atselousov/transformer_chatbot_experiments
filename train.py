import os
import torch
import random
import logging
import argparse
import json
import sys
from tensorboardX import SummaryWriter

from model.utils import load_openai_weights, set_seed, f1_score, open, unicode
from model.transformer_model import TransformerModel
from model.trainer import Trainer
from model.text import BPEVocab
from model.dataset import FacebookDataset
from config import get_model_config, get_trainer_config
from metrics import nlp_metrics

class DummyWriter:
    """ Used for distributed training (from NVIDIA apex example).
        A dummy logger used so that only the main process write and log informations.
    """
    def __init__(self, *input, **kwargs):
        self.log_dir = "runs/dummy_logs/"
    def add_scalar(self, *input, **kwargs):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help="Distributed training.")
    parser.add_argument('--server_ip', type=str, default='', help="Used for debugging on GPU machine.")
    parser.add_argument('--server_port', type=str, default='', help="Used for debugging on GPU machine.")
    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.ERROR)
    logger = logging.getLogger(__file__)
    if args.server_ip and args.server_port and args.local_rank in [-1, 0]:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Log only on main process
    if args.local_rank not in [-1, 0]:
        sys.stdout = open(f"./runs/log_distributed_{args.local_rank}", "w")  # dump sdtout
        writer = DummyWriter()
    else:
        writer = SummaryWriter()

    model_config = get_model_config()
    trainer_config = get_trainer_config()

    logger.info("model config: {}".format(model_config))
    logger.info("trainer config: {}".format(trainer_config))
    log_dir = writer.log_dir
    interrupt_checkpoint_path = os.path.join(log_dir, trainer_config.interrupt_checkpoint_path)
    last_checkpoint_path = os.path.join(log_dir, trainer_config.last_checkpoint_path)
    logger.info("Logging to {}".format(log_dir))  # Let's save everything on an experiment in the ./runs/XXX/directory
    if args.local_rank in [-1, 0]:
        with open(os.path.join(log_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f)
        with open(os.path.join(log_dir, "trainer_config.json"), "w") as f:
            json.dump(trainer_config, f)

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)

    vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)

    transformer = TransformerModel(n_layers=model_config.n_layers,
                                   n_embeddings=len(vocab),
                                   n_pos_embeddings=model_config.n_pos_embeddings,
                                   embeddings_size=model_config.embeddings_size,
                                   padding_idx=vocab.pad_id,
                                   n_heads=model_config.n_heads,
                                   dropout=model_config.dropout,
                                   embed_dropout=model_config.embed_dropout,
                                   attn_dropout=model_config.attn_dropout,
                                   ff_dropout=model_config.ff_dropout,
                                   normalize_embeddings=model_config.normalize_embeddings,
                                   bos_id=vocab.bos_id,
                                   eos_id=vocab.eos_id,
                                   sent_dialog_id=vocab.sent_dialog_id,
                                   max_seq_len=model_config.max_seq_len,
                                   beam_size=model_config.beam_size,  
                                   length_penalty=model_config.length_penalty,
                                   n_segments=model_config.n_segments,
                                   annealing_topk=model_config.annealing_topk,
                                   annealing=model_config.annealing,
                                   diversity_coef=model_config.diversity_coef,
                                   diversity_groups=model_config.diversity_groups,
                                   multiple_choice_head=model_config.multiple_choice_head,
                                   constant_embedding=model_config.constant_embedding,
                                   single_input=trainer_config.single_input,
                                   dialog_embeddings=trainer_config.dialog_embeddings,
                                   share_models=model_config.share_models,
                                   successive_attention=model_config.successive_attention,
                                   sparse_embeddings=model_config.sparse_embeddings,
                                   vocab=None)  # for beam search debugging

    if not trainer_config.load_last:
        load_openai_weights(transformer.transformer_module, 
                            trainer_config.openai_parameters_dir,
                            n_special_tokens=vocab.n_special_tokens)
        if not model_config.share_models:
            load_openai_weights(transformer.encoder_module, 
                                trainer_config.openai_parameters_dir,
                                n_special_tokens=vocab.n_special_tokens)
        logger.info('OpenAI weights loaded from {}, model shared: {}'.format(
                        trainer_config.openai_parameters_dir, model_config.share_models))

    logger.info('loading datasets')
    train_dataset = FacebookDataset(trainer_config.train_datasets, vocab,
                                    max_lengths=(transformer.n_pos_embeddings - 1) // (3 if trainer_config.single_input else 1),  # A bit restrictive here
                                    dialog_embeddings=trainer_config.dialog_embeddings,
                                    cache=trainer_config.train_datasets_cache,
                                    use_start_end=trainer_config.use_start_end,
                                    negative_samples=trainer_config.negative_samples,
                                    augment=trainer_config.persona_augment,
                                    aug_syn_proba=trainer_config.persona_aug_syn_proba,
                                    limit_size=trainer_config.limit_train_size)
    test_dataset = FacebookDataset(trainer_config.test_datasets, vocab,
                                   max_lengths=(transformer.n_pos_embeddings - 1) // (3 if trainer_config.single_input else 1),  # A bit restrictive here
                                   dialog_embeddings=trainer_config.dialog_embeddings,
                                   cache=trainer_config.test_datasets_cache,
                                   use_start_end=trainer_config.use_start_end,
                                   negative_samples=-1,  # Keep all negative samples
                                   augment=False,
                                   aug_syn_proba=0.0,
                                   limit_size=trainer_config.limit_eval_size)
    logger.info(f'train dataset {len(train_dataset)} test dataset {(test_dataset)}')
    model_trainer = Trainer(transformer,
                            train_dataset,
                            writer,
                            test_dataset,
                            train_batch_size=trainer_config.train_batch_size,
                            batch_split=trainer_config.batch_split,
                            test_batch_size=trainer_config.test_batch_size,
                            lr=trainer_config.lr,
                            lr_warmup=trainer_config.lr_warmup,
                            weight_decay=trainer_config.weight_decay,
                            s2s_weight=trainer_config.s2s_weight,
                            lm_weight=trainer_config.lm_weight,
                            risk_weight=trainer_config.risk_weight,
                            hits_weight=trainer_config.hits_weight,
                            single_input=trainer_config.single_input,
                            n_jobs=trainer_config.n_jobs,
                            clip_grad=trainer_config.clip_grad,
                            device=device,
                            ignore_idxs=vocab.special_tokens_ids,
                            local_rank=args.local_rank,
                            fp16=trainer_config.fp16,
                            loss_scale=trainer_config.loss_scale,
                            linear_schedule=trainer_config.linear_schedule,
                            n_epochs=trainer_config.n_epochs,
                            evaluate_full_sequences=trainer_config.evaluate_full_sequences)

    if trainer_config.load_last:
        state_dict = torch.load(trainer_config.load_last, map_location=device)
        model_trainer.load_state_dict(state_dict)
        logger.info('Weights loaded from {}'.format(trainer_config.load_last))


    # helpers -----------------------------------------------------
    def external_metrics_func(full_references, full_predictions):
        references_file_path = os.path.join(writer.log_dir, trainer_config.eval_references_file)
        predictions_file_path = os.path.join(writer.log_dir, trainer_config.eval_predictions_file)
        with open(references_file_path, 'w', encoding='utf-8') as f:
            f.write(unicode('\n'.join(full_references)))
        with open(predictions_file_path, 'w', encoding='utf-8') as f:
            f.write(unicode('\n'.join(full_predictions)))

        nist, bleu, meteor, entropy, div, avg_len = nlp_metrics([references_file_path], predictions_file_path)

        metrics = {'meteor': meteor, 'avg_len': avg_len}
        for name, metric in (('nist', nist), ('entropy', entropy), ('div', div), ('bleu', bleu)):
            for i, m in enumerate(metric):
                metrics['{}_{}'.format(name, i+1)] = m
        return metrics

    def save_func(epoch):
        if epoch != -1:
            torch.save(model_trainer.state_dict(), last_checkpoint_path)

    def sample_text_func(epoch):
        n_samples = 0
        model_trainer.model.eval()
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target, _ in samples:
            contexts = [torch.tensor([c], dtype=torch.long, device=model_trainer.device) for c in [persona_info, dialog] if len(c) > 0]
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
        if (epoch+1) % trainer_config.test_period == 0:
            metric_funcs = {'f1_score': f1_score}
            model_trainer.test(metric_funcs, external_metrics_func, epoch)

    def f1_risk(predictions, targets):
        scores = f1_score(predictions, targets, average=False)
        return [1-s for s in scores]

    # helpers -----------------------------------------------------


    try:
        model_trainer.train(after_epoch_funcs=[save_func, sample_text_func, test_func], risk_func=f1_risk)
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(model_trainer.state_dict(), interrupt_checkpoint_path)
        raise e


if __name__ == '__main__':
    main()
