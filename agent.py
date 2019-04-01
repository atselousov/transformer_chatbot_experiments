import torch
import torch.nn.functional as F
from collections import deque
from parlai.core.agents import Agent
from model.transformer_model import TransformerModel
from model.text import BPEVocab
from model.utils import pad_sequence
from model.postprocessing import ngram_replaser, ReplyChecker, detokenize, syntax_fix
from model.sentiment import pick_emoji, clean_emoji
from config import get_model_config
import random


class TransformerAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        agent_args = argparser.add_argument_group('Agent parameters')
        agent_args.add_argument('-gpu', '--gpu', type=int, default=-1, 
                                help='which GPU to use')
        agent_args.add_argument('--no-cuda', type=bool, default=False,
                                help='disable GPUs even if available. otherwise, will use GPUs if '
                                     'available on the device.')
        agent_args.add_argument('--rank_candidates', type=bool, default=False,
                                help='Whether the model should parse candidates for ranking.')
        agent_args.add_argument('--sample', type=bool, default=False,
                                help='Sampling of beam from beam search')
        agent_args.add_argument('--clean_emoji', type=bool, default=True,
                                help='')
        agent_args.add_argument('--check_grammar', type=bool, default=True,
                                help='')

        agent_args.add_argument('--max_seq_len', type=int, default=128,
                                help='')
        agent_args.add_argument('--beam_size', type=int, default=1,
                                help='')
        agent_args.add_argument('--diversity_coef', type=float, default=0,
                                help='')
        agent_args.add_argument('--diversity_groups', type=int, default=1,
                                help='')
        agent_args.add_argument('--annealing_topk', type=float, default=None,
                                help='')
        agent_args.add_argument('--annealing', type=float, default=0.0,
                                help='')
        agent_args.add_argument('--length_penalty', type=float, default=0.6,
                                help='')
        
        return argparser

    def __init__(self, opt, shared=None):
        super(TransformerAgent, self).__init__(opt, shared)

        self.use_cuda = not self.opt.get('no_cuda') and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(self.opt['gpu'])

        torch.set_grad_enabled(False)

        model_config = get_model_config()
        self.vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)

        self.clean_emoji = self.opt['clean_emoji']
        self.check_grammar = self.opt['check_grammar']
        self.dialog_embeddings = model_config.dialog_embeddings
        self.use_start_end = model_config.use_start_end
        self.single_input = model_config.single_input

        # 'max_seq_len': 128,
        # 'beam_size': 1,
        # 'diversity_coef': 0,
        # 'diversity_groups': 1,
        # 'annealing_topk': None,
        # 'annealing': 0,
        # 'length_penalty': 0.6,

        self.vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)

        if self.opt['annealing_topk'] is not None:
            assert self.opt['annealing_topk'] > self.opt['beam_size']

        assert self.opt['diversity_coef'] >= 0
        assert self.opt['beam_size'] % self.opt['diversity_groups'] == 0

        if shared is None:
            self.model = TransformerModel(n_layers=model_config.n_layers,
                                          n_embeddings=len(self.vocab),
                                          n_pos_embeddings=model_config.n_pos_embeddings,
                                          embeddings_size=model_config.embeddings_size,
                                          padding_idx=self.vocab.pad_id,
                                          n_heads=model_config.n_heads,
                                          dropout=model_config.dropout,
                                          embed_dropout=model_config.embed_dropout,
                                          attn_dropout=model_config.attn_dropout,
                                          ff_dropout=model_config.ff_dropout,
                                          bos_id=self.vocab.bos_id,
                                          eos_id=self.vocab.eos_id,
                                          sent_dialog_id=self.vocab.sent_dialog_id,
                                          max_seq_len=self.opt['max_seq_len'],
                                          beam_size=self.opt['beam_size'],
                                          length_penalty=self.opt['length_penalty'],
                                          n_segments=model_config.n_segments,
                                          sample=self.opt['sample'],
                                          annealing_topk=self.opt['annealing_topk'],
                                          annealing=self.opt['annealing'],
                                          diversity_coef=self.opt['diversity_coef'],
                                          diversity_groups=self.opt['diversity_groups'],
                                          normalize_embeddings=model_config.normalize_embeddings,
                                          multiple_choice_head=model_config.multiple_choice_head,
                                          constant_embedding=model_config.constant_embedding,
                                          vocab=self.vocab,
                                          single_input=model_config.single_input,
                                          dialog_embeddings=model_config.dialog_embeddings,
                                          share_models=model_config.share_models,
                                          successive_attention=model_config.successive_attention,
                                          sparse_embeddings=model_config.sparse_embeddings,
                                          shared_attention=model_config.sparse_embeddings
                                          )

            state_dict = torch.load(model_config.checkpoint_path, map_location=lambda storage, loc: storage)
            if 'model' in state_dict:
                state_dict = state_dict['model']

            self.model.load_state_dict(state_dict)
            print('Weights loaded from {}'.format(model_config.checkpoint_path))

            if self.use_cuda:
                self.model = self.model.cuda()

            self.model.eval()

        else:
            self.model = shared['model']

        self.reset()

    def _preprocess_text(self, text):
        if self.clean_emoji:
            text = clean_emoji(text)

        if self.check_grammar:
            text = syntax_fix(text).lower()

        return text

    def _parse(self, text):
        # todo: fix grammar mistakes?
        persona_info = []
        dialog = []
        for subtext in text.split('\n'):
            subtext = subtext.strip()
            
            if subtext.startswith('your persona:'):
                subtext = subtext.replace('your persona:', '').strip()
                subtext = self._preprocess_text(subtext).strip()
                persona_info.append(subtext)
            else:
                subtext = self._preprocess_text(subtext).strip()
                dialog.append(subtext)

        return persona_info, dialog

    def _process_info(self, info):
        info = self._add_start_end(info[:self.model.n_pos_embeddings - (2 if self.use_start_end else 0)],
                                   self.vocab.info_bos_id,
                                   self.vocab.info_eos_id)
        info = self._add_dialog_embeddings(info, self.vocab.info_dialog_id)

        return info

    def _process_1st_replica(self, replica):
        replica = self._add_start_end(replica, self.vocab.talker1_bos_id, self.vocab.talker1_eos_id)
        replica = self._add_dialog_embeddings(replica, self.vocab.talker1_dialog_id)
        return replica

    def _process_2nd_replica(self, replica):
        replica = self._add_start_end(replica, self.vocab.talker2_bos_id, self.vocab.talker2_eos_id)
        replica = self._add_dialog_embeddings(replica, self.vocab.talker2_dialog_id)
        return replica

    def _add_dialog_embeddings(self, toks, dialog_tok):
        if self.dialog_embeddings:
            toks = [[t, dialog_tok] for t in toks]
        return toks

    def _add_start_end(self, toks, start, end):
        if self.use_start_end:
            toks = [start] + toks + [end]
        return toks

    def observe(self, observation):
        if self.episode_done:
            self.reset()

        if 'text' in observation:
            text = observation['text']
            info, dialog = self._parse(text)

            info = sum([self.vocab.string2ids(i) for i in info], [])
            if info:
                prev_info = [h[0] for h in self.history['info']] if self.dialog_embeddings else self.history['info']
                self.history['info'] = self._process_info(prev_info[1:-1] + info)

            for i, replica in enumerate(dialog, 1):
                replica = self.vocab.string2ids(replica)
                replica = self._process_1st_replica(replica) if i % 2 == 1 else self._process_2nd_replica(replica)
                self.history['dialog'].extend(replica)

        observation['agent'] = self        

        self.episode_done = observation['episode_done']
        self.observation = observation
        
        return observation
    
    def act(self):
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):
        def is_valid_history(history):
            return len(history['dialog'])

        def to_tensor(string):
            ids = [self.vocab.bos_id] + self.vocab.string2ids(string) + [self.vocab.eos_id]
            ids = self._add_dialog_embeddings(ids, self.vocab.sent_dialog_id)
            return torch.tensor(ids, dtype=torch.long)

        def to_cuda(data):
            if not self.use_cuda:
                return data

            if isinstance(data, (list, tuple)):
                return list(map(lambda x: x.cuda(), data))

            return data.cuda()

        batch_reply = [{'id': self.getID(), 'text': '', 'text_candidates': []} for _ in range(len(observations))]
        valid_ids = [i for i, obs in enumerate(observations) if is_valid_history(obs['agent'].history)]
        batch_size = len(valid_ids)

        if batch_size == 0:
            return batch_reply

        try:
            valid_observations = [observations[i] for i in valid_ids]

            infos = [obs['agent'].history['info'] for obs in valid_observations]
            dialogs = [list(obs['agent'].history['dialog'])[-self.model.n_pos_embeddings+1:] for obs in valid_observations]
            contexts = []

            if max(map(len, infos)) > 0:
                infos = [torch.tensor(i, dtype=torch.long) for i in infos]
                contexts.append(infos)

            if max(map(len, dialogs)) > 0:
                dialogs = [torch.tensor(d, dtype=torch.long) for d in dialogs]
                contexts.append(dialogs)

            if self.single_input:
                contexts = [torch.cat(c, dim=0) for c in zip(*contexts)]
                contexts = pad_sequence(contexts, batch_first=True, padding_value=self.model.padding_idx, left=True)
            else:
                contexts = map(lambda x: pad_sequence(x, batch_first=True, padding_value=self.model.padding_idx),
                               contexts)

            contexts = to_cuda(contexts)

            pred_texts = self.model.predict(contexts)

            for i in range(batch_size):
                pred_toks = self._process_2nd_replica(pred_texts[i])
                valid_observations[i]['agent'].history['dialog'].extend(pred_toks)
                batch_reply[valid_ids[i]]['text'] = self.vocab.ids2string(pred_texts[i])
                batch_reply[valid_ids[i]]['episode_done'] = valid_observations[i]['agent'].episode_done

            if self.opt['rank_candidates']:
                enc_contexts = [self.model.encode(c) for c in contexts] if not self.single_input else []

                candidates = [list(obs.get('label_candidates', [])) for obs in valid_observations]
                lens_candidates = [len(c) for c in candidates]

                if max(lens_candidates) > 0:
                    candidates = [c + ['' for _ in range(max(lens_candidates) - len(c))] for c in candidates]
                    scores = [[] for _ in range(len(candidates))]

                    for i in range(max(lens_candidates)):
                        current_cands = [to_tensor(c[i])[:self.model.n_pos_embeddings-1] for c in candidates]
                        current_cands = to_cuda(current_cands)

                        lens = map(lambda x: x.size(0), current_cands) if self.single_input else None
                        current_cands = pad_sequence(current_cands, batch_first=True,
                                                     padding_value=self.model.padding_idx)
                        if self.single_input:
                            current_cands = torch.cat((contexts, current_cands), dim=1)[-self.model.n_pos_embeddings:]

                        logits = self.model.decode(current_cands[:, :-1], enc_contexts)

                        if current_cands.dim() == 3:
                            current_cands = current_cands[:, :, 0]

                        log_probas = F.log_softmax(logits, dim=-1)
                        log_probas = torch.gather(log_probas, -1, current_cands[:, 1:].unsqueeze(-1)).squeeze(-1)

                        if self.single_input:
                            # zero context
                            for j, l in enumerate(lens):
                                current_cands[j, :-l+1] = self.model.padding_idx

                        log_probas.masked_fill_(current_cands[:, 1:].eq(self.model.padding_idx), 0)

                        current_lens = current_cands[:, 1:].ne(self.model.padding_idx).float().sum(dim=-1)
                        current_scores = log_probas.sum(dim=-1) / current_lens

                        for k, s in enumerate(current_scores):
                            if i < lens_candidates[k]:
                                scores[k].append(s.item())

                    ranked_ids = [sorted(range(len(s)), key=lambda k: s[k], reverse=True) for s in scores]
                    ranked_strings = [[c[i] for i in ids] for ids, c in zip(ranked_ids, candidates)]

                    for i in range(batch_size):
                        batch_reply[valid_ids[i]]['text_candidates'] = ranked_strings[i]

        except Exception as e:
            # raise e
            print(e)

        return batch_reply

    def share(self):
        shared = super(TransformerAgent, self).share()
        shared['opt'] = self.opt
        shared['model'] = self.model

        return shared

    def reset(self):
        self.history = {'info': [], 'dialog': deque(maxlen=self.model.n_pos_embeddings-1)}
        self.episode_done = True
        self.observation = None
