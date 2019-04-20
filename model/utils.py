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

import random
from collections import Counter

import torch


def set_seed(seed=0):
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def repeat_along_dim1(obj, repetitions):
    """ repeat (a possibly nested object of) tensors from (batch, ...) to (batch * repetitions, ...) """
    if isinstance(obj, tuple):
        return tuple(repeat_along_dim1(o, repetitions) for o in obj)
    if isinstance(obj, list):
        return list(repeat_along_dim1(o, repetitions) for o in obj)

    obj = obj.unsqueeze(1).repeat([1, repetitions] + [1] * len(obj.size()[1:]))
    return obj.view(-1, *obj.size()[2:])


def pad_sequence(sequences, batch_first=False, padding_value=0, left=False):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    if not len(sequences):
        return torch.empty(0)
    trailing_dims = sequences[0].size()[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        s_slice = slice(-length, None) if left else slice(None, length)
        s_slice = (i, s_slice) if batch_first else (s_slice, i)
        out_tensor[s_slice] = tensor

    return out_tensor


def f1_score(predictions, targets, average=True):
    def f1_score_items(pred_items, gold_items):
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())

        if num_same == 0:
            return 0

        precision = num_same / len(pred_items)
        recall = num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    scores = [f1_score_items(p, t) for p, t in zip(predictions, targets)]

    if average:
        return sum(scores) / len(scores)

    return scores
