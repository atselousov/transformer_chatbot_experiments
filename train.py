import torch
import random
import logging
import argparse
from model.utils import load_openai_weights, set_seed, f1_score, open, unicode
from model.transformer_model import TransformerModel
from model.trainer import Trainer
from model.text import BPEVocab
from model.dataset import FacebookDataset
from config import get_model_config, get_trainer_config
from metrics import nlp_metrics

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__file__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help="Distributed training.")
    args = parser.parse_args()

    model_config = get_model_config()
    trainer_config = get_trainer_config()

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
                                   bos_id=vocab.bos_id,
                                   eos_id=vocab.eos_id,
                                   max_seq_len=model_config.max_seq_len,
                                   beam_size=model_config.beam_size,  
                                   length_penalty=model_config.length_penalty,
                                   n_segments=model_config.n_segments,
                                   annealing_topk=model_config.annealing_topk,
                                   annealing=model_config.annealing,
                                   diversity_coef=model_config.diversity_coef,
                                   diversity_groups=model_config.diversity_groups,
                                   multiple_choice_head=model_config.multiple_choice_head)

    if not trainer_config.load_last:
        load_openai_weights(transformer.transformer_module, 
                            trainer_config.openai_parameters_dir,
                            n_special_tokens=vocab.n_special_tokens)
        logger.info('OpenAI weights loaded from {}'.format(trainer_config.openai_parameters_dir))

    logger.info('loading datasets')
    train_dataset = FacebookDataset(trainer_config.train_datasets, vocab, transformer.n_pos_embeddings - 1,
                                    dialog_embeddings=trainer_config.dialog_embeddings, cache=trainer_config.train_datasets_cache)
    test_dataset = FacebookDataset(trainer_config.test_datasets, vocab, transformer.n_pos_embeddings - 1,
                                   dialog_embeddings=trainer_config.dialog_embeddings, cache=trainer_config.test_datasets_cache)

    model_trainer = Trainer(transformer,
                            train_dataset,
                            test_dataset,
                            batch_size=trainer_config.batch_size,
                            batch_split=trainer_config.batch_split,
                            lr=trainer_config.lr,
                            lr_warmup=trainer_config.lr_warmup,
                            lm_weight=trainer_config.lm_weight,
                            risk_weight=trainer_config.risk_weight,
                            hits_weight=trainer_config.hits_weight,
                            negative_samples=trainer_config.negative_samples,
                            single_input=trainer_config.single_input,
                            n_jobs=trainer_config.n_jobs,
                            clip_grad=trainer_config.clip_grad,
                            device=device,
                            ignore_idxs=vocab.special_tokens_ids,
                            local_rank=args.local_rank,
                            fp16=trainer_config.fp16,
                            loss_scale=trainer_config.loss_scale,
                            linear_schedule=trainer_config.linear_schedule,
                            n_epochs=trainer_config.n_epochs)

    if trainer_config.load_last:
        state_dict = torch.load(trainer_config.last_checkpoint_path, map_location=device)
        model_trainer.load_state_dict(state_dict)
        logger.info('Weights loaded from {}'.format(trainer_config.last_checkpoint_path))


    # helpers -----------------------------------------------------
    def external_metrics_func(full_references, full_predictions):
        with open(trainer_config.eval_references_file, 'w', encoding='utf-8') as f:
            f.write(unicode('\n'.join(full_references)))
        with open(trainer_config.eval_predictions_file, 'w', encoding='utf-8') as f:
            f.write(unicode('\n'.join(full_predictions)))

        nist, bleu, meteor, entropy, div, avg_len = nlp_metrics([trainer_config.eval_references_file],
                                                                trainer_config.eval_predictions_file)
        return {'nist': nist, 'bleu': bleu, 'meteor': meteor, 'entropy': entropy, 'div': div, 'avg_len': avg_len}

    def save_func(epoch):
        torch.save(model_trainer.state_dict(), trainer_config.last_checkpoint_path)

    def sample_text_func(epoch):
        n_samples = 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target in samples:
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
            model_trainer.test(metric_funcs, external_metrics_func)

    def f1_risk(predictions, targets):
        scores = f1_score(predictions, targets, average=False)
        return [1-s for s in scores]

    # helpers -----------------------------------------------------


    try:
        model_trainer.train(after_epoch_funcs=[save_func, sample_text_func, test_func], risk_func=f1_risk)
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(model_trainer.state_dict(), trainer_config.interrupt_checkpoint_path)
        raise e


if __name__ == '__main__':
    main()
