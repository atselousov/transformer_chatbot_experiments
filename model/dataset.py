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

import os
import random
import torch
from torch.utils.data import Dataset
from .postprocessing import augment_replica


class FacebookDataset(Dataset):
    @staticmethod
    def parse_data(path):
        with open(path, 'r', encoding='utf-8') as file:
            data = []
            for line in file.readlines():
                line = line.strip()

                if len(line) == 0:
                    continue

                space_idx = line.find(' ')
                if space_idx == -1:
                    dialog_idx = int(line)
                else:
                    dialog_idx = int(line[:space_idx])

                if int(dialog_idx) == 1:
                    data.append({'persona_info': [], 'dialog': []})

                dialog_line = line[space_idx + 1:].split('\t')
                dialog_line = [l.strip() for l in dialog_line]

                if dialog_line[0].startswith('your persona:'):
                    persona_info = dialog_line[0].replace('your persona: ', '')
                    data[-1]['persona_info'].append(persona_info)

                elif len(dialog_line) > 1:
                    data[-1]['dialog'].append(dialog_line[0])
                    data[-1]['dialog'].append(dialog_line[1])

            return data

    @staticmethod
    def make_dataset(data, vocab, max_lengths):
        dataset = []
        for chat in data:
            persona_info = [vocab.string2ids(s) for s in chat['persona_info']]
            dialog = [vocab.string2ids(s) for s in chat['dialog']]

            if len(dialog) % 2 == 1:
                dialog = dialog[:-1]

            dataset.append((persona_info, dialog))

        return dataset

    def __init__(self, paths, vocab, *, max_lengths=2048, min_infos=2, cache=None, augment=False,
                 syn_proba=0.1):
        assert min_infos > 0

        if isinstance(paths, str):
            paths = [paths]

        self.augment = augment
        self.syn_proba = syn_proba

        self.vocab = vocab
        self.max_lengths = max_lengths
        self.min_infos = min_infos

        if cache and os.path.exists(cache):
            self.data = torch.load(cache)
        else:
            parsed_data = sum([FacebookDataset.parse_data(path) for path in paths], [])
            self.data = FacebookDataset.make_dataset(parsed_data, vocab, max_lengths)
            if cache:
                torch.save(self.data, cache)

    def __len__(self):
        return len(self.data)

    def _augment(self, sentences, info=False):
        if not self.augment:
            return sentences

        if info:
            n_info_samples = max(self.min_infos, random.randint(1, len(sentences)))
            n_info_samples = min(n_info_samples, len(sentences))
            sentences = random.sample(sentences, n_info_samples)
            random.shuffle(sentences)
        else:
            begin = random.randrange(0, len(sentences) // 2, 2)
            end = random.randrange(begin + 2, len(sentences) + 1, 2)

            sentences = sentences[begin:end]

        def _try2augment(sent):
            if random.uniform(0, 1) < self.syn_proba:
                sent = self.vocab.ids2string(sent)
                sent = augment_replica(sent)
                sent = self.vocab.string2ids(sent)
            return sent

        sentences = map(_try2augment, sentences)

        return list(sentences)

    def __getitem__(self, idx):
        persona_info, dialog = self.data[idx]

        if len(persona_info):
            persona_info = self._augment(persona_info, info=True)
            persona_info = sum(persona_info, [])
            persona_info = [self.vocab.info_bos_id] + persona_info[:self.max_lengths-2] + [self.vocab.info_eos_id]

        dialog = self._augment(dialog)

        h = []
        for i, ids in enumerate(dialog[:-1], 1):
            if i % 2 == 1:
                ids = [self.vocab.talker1_bos_id] + ids + [self.vocab.talker1_eos_id]
            else:
                ids = [self.vocab.talker2_bos_id] + ids + [self.vocab.talker2_eos_id]
            h.extend(ids)
        h = h[-self.max_lengths:]

        y = [self.vocab.bos_id] + dialog[-1] + [self.vocab.eos_id]
        y = y[:self.max_lengths]

        return persona_info, h, y
