import torch
from torch.nn import functional as F
from projects.convai2.eval_ppl import setup_args, eval_ppl
from projects.convai2.build_dict import build_dict
from agent import TransformerAgent

class TransformerAgentPpl(TransformerAgent):
    def __init__(self, opt, shared=None):
        super(TransformerAgentPpl, self).__init__(opt, shared)
        if shared:
            self.prefix2words = shared['prefix2words']
        else:
            print("Build prefix conversion map between convai dict and our bpe dict")
            convai_dict = build_dict()
            assert len(convai_dict) == 19304
            self.prefix2words = self.vocab.get_prefix2words(convai_dict)

    def share(self):
        shared = super(TransformerAgentPpl, self).share()
        shared['prefix2words'] = self.prefix2words()
        return shared

    def next_word_probability(self, partial_out):
        """Return probability distribution over next words given an input and
        partial true output. This is used to calculate the per-word perplexity.
        """
        def is_valid_history(history):
            return len(history['dialog'])

        def to_tensor(string):
            ids = [self.vocab.bos_id] + self.vocab.string2ids(string) + [self.vocab.eos_id]
            return torch.tensor(ids, dtype=torch.long)

        obs = self.observation
        infos = obs['agent'].history['info'][:self.model.n_pos_embeddings-3]
        infos = ([self.vocab.info_bos_id] + infos + [self.vocab.info_eos_id] if len(infos) else infos)
        dialogs = list(obs['agent'].history['dialog'])[-self.model.n_pos_embeddings+1:]
        prevs = [self.vocab.bos_id] + partial_out
        contexts = []

        infos = torch.tensor([infos], dtype=torch.long)
        if self.use_cuda:
            infos = infos.cuda()
        contexts.append(infos)

        dialogs = torch.tensor([dialogs], dtype=torch.long)
        if self.use_cuda:
            dialogs = dialogs.cuda()
        contexts.append(dialogs)

        enc_contexts = [self.model.encode(c) for c in contexts]

        prevs = torch.tensor([prevs], dtype=torch.long)
        if self.use_cuda:
            prevs = prevs.cuda()

        logits = self.model.decode(prevs, enc_contexts)
        probs = F.softmax(logits[0, -1], dim=0)

        dist = {}
        for prefix, words in self.prefix2words.items():
            for word, ratio in words.items():
                dist[word] = probs[self.vocab.token2id[prefix]].item() * ratio
        return dist

if __name__ == '__main__':
    parser = setup_args()

    parser.set_defaults(model='eval_ppl:TransformerAgentPpl',
                        batchsize=1,
                        rank_candidates=False,
                        sample=False,
                        wild_mode=False,
                        clean_emoji=False,
                        check_grammar=False,
                        max_seq_len=256,
                        beam_size=1,
                        annealing_topk=None,
                        annealing=0,
                        length_penalty=0.6)
    opt = parser.parse_args()
    eval_ppl(opt)

