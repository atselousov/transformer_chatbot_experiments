#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import pad_sequence
from .optim import Adam, NoamOpt
from .loss import LabelSmoothingLoss

logger = logging.getLogger(__file__)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset=None, batch_size=8,
                 batch_split=1, lm_weight=0.5, risk_weight=0, hits_weight=0, lr=6.25e-5, lr_warmup=2000, 
                 n_jobs=0, clip_grad=None, label_smoothing=0, device=torch.device('cuda'),
                 ignore_idxs=[], negative_samples=0, single_input=False):
        self.model = model.to(device)
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx).to(device)
        self.hits_criterion = nn.CrossEntropyLoss().to(device)
        self.criterion = LabelSmoothingLoss(n_labels=self.model.n_embeddings, smoothing=label_smoothing, ignore_index=self.model.padding_idx).to(device)
        base_optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.embeddings_size, 1, lr_warmup, base_optimizer)

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size//batch_split, shuffle=True, 
                                           num_workers=n_jobs, collate_fn=self.collate_func)
        if test_dataset is not None:
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size//batch_split, shuffle=False, 
                                            num_workers=n_jobs, collate_fn=self.collate_func)
        self.vocab = train_dataset.vocab

        self.batch_split = batch_split
        self.lm_weight = lm_weight
        self.risk_weight = risk_weight
        self.hits_weight = hits_weight
        self.clip_grad = clip_grad
        self.device = device
        self.ignore_idxs = ignore_idxs
        self.negative_samples = negative_samples
        self.single_input = single_input

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def collate_func(self, data):
        persona_info, h, y = zip(*data)

        contexts = []

        if max(map(len, persona_info)) > 0:
            persona_info = [torch.tensor(d, dtype=torch.long) for d in persona_info]
            persona_info = pad_sequence(persona_info, batch_first=True, padding_value=self.model.padding_idx)
            contexts.append(persona_info)

        if max(map(len, h)) > 0:
            h = [torch.tensor(d, dtype=torch.long) for d in h]
            h = pad_sequence(h, batch_first=True, padding_value=self.model.padding_idx)
            contexts.append(h)

        y_out = [torch.tensor(d, dtype=torch.long) for d in y]
        y_out = pad_sequence(y_out, batch_first=True, padding_value=self.model.padding_idx)

        if self.negative_samples > 0 and len(y) > 1:
            # sample self.negative_samples as distractors for each instance.
            # We sample negative_samples inside the batch (a bit different from Huggingface original implementation)
            probs = (1 - np.identity(len(y)))/(len(y) - 1)  # don't sample our target as negative sample
            distractors = [np.random.choice(y, size=self.negative_samples, replace=False, p=probs[i]) for i in range(len(y))]
            distractors = np.stack(distractors).transpose().flatten()   # ordered as: self.negative_samples * bsz, seq_length
            distractors = [torch.tensor(d, dtype=torch.long) for d in distractors]
            distractors = pad_sequence(distractors, batch_first=True, padding_value=self.model.padding_idx)
        else:
            distractors = None

        if self.single_input:
            # context in y and distractors
            y_out = torch.cat(contexts + [y_out], dim=-1)
            distractors = torch.cat([t.repeat(self.negative_samples, 1, 1).view(-1, t.size(-1)) for t in contexts] + [distractors], dim=-1)
            contexts = []

        return contexts, y_out, distractors

    def _eval_train(self, epoch, risk_func=None):
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        loss = 0
        lm_loss = 0
        risk_loss = 0
        for i, (contexts, targets, distractors) in enumerate(tqdm_data):
            contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)

            enc_contexts = []

            # lm loss
            batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            for context in contexts:
                enc_context = self.model.encode(context.clone())
                enc_contexts.append(enc_context)

                if self.lm_weight > 0:
                    context_outputs = self.model.generate(enc_context[0])
                    ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                    context.masked_fill_(ignore_mask, self.model.padding_idx)
                    prevs, nexts = context_outputs[:, :-1, :].contiguous(), context[:, 1:].contiguous()
                    batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / len(contexts))

            # s2s loss
            if self.hits_weight > 0 and self.negative_samples > 0:
                # Keep the hidden states
                nexts = targets[:, 1:].contiguous()
                hidden_state, padding_mask = self.model.transformer_module(targets, enc_contexts)
                outputs = self.model.generate(hidden_state[:, :-1].contiguous())
            else:
                prevs, nexts = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
                outputs = self.model.decode(prevs, enc_contexts)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            # hits@1 loss
            if self.hits_weight > 0 and self.negative_samples > 0:
                extended_contexts = tuple(tuple(t.repeat([self.negative_samples]+[1]*(t.dim()-1)) for t in c) for c in enc_contexts)
                neg_logits = self.model.decode_classify(distractors, extended_contexts)
                true_logits = self.model.classify(hidden_state, padding_mask)
                clf_logits = torch.cat((true_logits.view(-1, 1), neg_logits.view(self.negative_samples, -1).transpose(0, 1)), dim=1)
                clf_labels = torch.tensor([0] * len(true_logits), dtype=torch.long, device=self.device)
                hits_loss = self.hits_criterion(clf_logits, clf_labels)

            # risk loss
            batch_risk_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            if risk_func is not None and self.risk_weight > 0:

                beams, beam_lens = self.model.beam_search(enc_contexts, return_beams=True)

                target_lens = targets.ne(self.model.padding_idx).sum(dim=-1)
                targets = [target[1:length-1].tolist() for target, length in zip(targets, target_lens)]
                batch_risks = []
                for b in range(beams.shape[1]):
                    predictions = [b[1:l-1].tolist() for b, l in zip(beams[:, b, :], beam_lens[:, b])]
                    risks = torch.tensor(risk_func(predictions, targets), dtype=torch.float, device=self.device)
                    batch_risks.append(risks)
                batch_risks = torch.stack(batch_risks, dim=-1)

                batch_probas = []
                for b in range(beams.shape[1]):
                    logits = self.model.decode(beams[:, b, :-1], enc_contexts)
                    probas = F.log_softmax(logits, dim=-1)
                    probas = torch.gather(probas, -1, beams[:, b, 1:].unsqueeze(-1)).squeeze(-1)
                    probas = probas.sum(dim=-1) / beam_lens[:, b].float()
                    batch_probas.append(probas)
                batch_probas = torch.stack(batch_probas, dim=-1)
                batch_probas = F.softmax(batch_probas, dim=-1)

                batch_risk_loss = torch.mean((batch_risks * batch_probas).sum(dim=-1))

            # optimization
            full_loss = (batch_lm_loss * self.lm_weight + self.risk_weight * batch_risk_loss + batch_loss) / self.batch_split
            full_loss.backward()

            if (i + 1) % self.batch_split == 0:
                if self.clip_grad is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

                self.optimizer.step()
                self.optimizer.zero_grad()

            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)
            risk_loss = (i * risk_loss + batch_risk_loss.item()) / (i + 1)

            tqdm_data.set_postfix({'lm_loss': lm_loss, 'loss': loss, 'risk_loss': risk_loss})

    def _eval_test(self, metric_funcs={}, external_metrics_func=None):
        self.model.eval()

        tqdm_data = tqdm(self.test_dataloader, desc='Test')
        loss = 0
        lm_loss = 0
        metrics = {name: 0 for name in metric_funcs.keys()}
        full_references, full_predictions = [], []
        for i, (contexts, targets) in enumerate(tqdm_data):
            contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)

            enc_contexts = []

            # lm loss
            batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            for context in contexts:
                enc_context = self.model.encode(context.clone())
                enc_contexts.append(enc_context)

                if self.lm_weight > 0:
                    context_outputs = self.model.generate(enc_context[0])
                    ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                    context.masked_fill_(ignore_mask, self.model.padding_idx)
                    prevs, nexts = context_outputs[:, :-1, :].contiguous(), context[:, 1:].contiguous()
                    batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / len(contexts))

            # s2s loss
            prevs, nexts = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            predictions = self.model.beam_search(enc_contexts)
            target_lens = targets.ne(self.model.padding_idx).sum(dim=-1)
            targets = [t[1:l-1].tolist() for t, l in zip(targets, target_lens)]

            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)
            for name, func in metric_funcs.items():
                score = func(predictions, targets)
                metrics[name] = (metrics[name] * i + score) / (i + 1)

            if external_metrics_func:
                # Store text strings for external metrics
                string_targets = list(self.vocab.ids2string(t) for t in targets)
                string_predictions = list(self.vocab.ids2string(t) for t in predictions)
                full_references.extend(string_targets)
                full_predictions.extend(string_predictions)

            tqdm_data.set_postfix(dict({'lm_loss': lm_loss, 'loss': loss}, **metrics))

        if external_metrics_func:
            external_metrics = external_metrics_func(full_references, full_predictions)
            metrics.update(external_metrics)

        logger.info(metrics)

    def test(self, metric_funcs={}, external_metrics_func=None):
        if hasattr(self, 'test_dataloader'):
            self._eval_test(metric_funcs, external_metrics_func)

    def train(self, epochs, after_epoch_funcs=[], risk_func=None):
        for epoch in range(epochs):
            self._eval_train(epoch, risk_func)

            for func in after_epoch_funcs:
                func(epoch)
