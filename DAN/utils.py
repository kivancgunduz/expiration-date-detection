# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/DAN/utils.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 5267 bytes
import torch, torch.nn.functional as F
from datetime import datetime

def flatten_label(target):
    label_flatten = []
    label_length = []
    for i in range(0, target.size()[0]):
        cur_label = target[i].tolist()
        label_flatten += cur_label[:cur_label.index(0) + 1]
        label_length.append(cur_label.index(0) + 1)

    label_flatten = torch.LongTensor(label_flatten)
    label_length = torch.IntTensor(label_length)
    return (label_flatten, label_length)


class cha_encdec:

    def __init__(self, dict_file, case_sensitive=True):
        self.dict = []
        self.case_sensitive = case_sensitive
        lines = open(dict_file, 'r').readlines()
        for line in lines:
            self.dict.append(line.replace('\n', ''))

    def encode(self, label_batch):
        max_len = max([len(s) for s in label_batch])
        out = torch.zeros(len(label_batch), max_len + 1).long()
        for i in range(0, len(label_batch)):
            if not self.case_sensitive:
                cur_encoded = torch.tensor([self.dict.index(char.lower()) if char.lower() in self.dict else len(self.dict) for char in label_batch[i]]) + 1
            else:
                cur_encoded = torch.tensor([self.dict.index(char) if char in self.dict else len(self.dict) for char in label_batch[i]]) + 1
            out[i][0:len(cur_encoded)] = cur_encoded

        return out

    def decode(self, net_out, length):
        out = []
        out_prob = []
        net_out = F.softmax(net_out, dim=1)
        for i in range(0, length.shape[0]):
            current_idx_list = net_out[int(length[:i].sum()):int(length[:i].sum() + length[i])].topk(1)[1][:, 0].tolist()
            current_text = ''.join([self.dict[(_ - 1)] if _ <= len(self.dict) else '' for _ in current_idx_list if _ > 0])
            current_probability = net_out[int(length[:i].sum()):int(length[:i].sum() + length[i])].topk(1)[0][:, 0]
            current_probability = torch.exp(torch.log(current_probability).sum() / current_probability.size()[0])
            out.append(current_text)
            out_prob.append(current_probability)

        return (
         out, out_prob)


class Attention_AR_counter:

    def __init__(self, display_string, dict_file, case_sensitive):
        self.correct = 0
        self.correct_full = 0
        self.two = 0
        self.three = 0
        self.total_images = 0
        self.total_samples = 0
        self.distance_C = 0
        self.total_C = 0.0
        self.distance_W = 0
        self.total_W = 0.0
        self.display_string = display_string
        self.case_sensitive = case_sensitive
        self.de = cha_encdec(dict_file, case_sensitive)

    def clear(self):
        self.correct_full = 0
        self.total_samples = 0
        self.distance_C = 0
        self.total_C = 0.0
        self.distance_W = 0
        self.total_W = 0.0

    def add_iter(self, output, out_length, predictions):
        self.total_images += 1
        PRD_texts, prdt_prob = self.de.decode(output, out_length)
        if len(''.join(PRD_texts)) == 6:
            if len(PRD_texts) == 3:
                if ''.join(PRD_texts).isdigit():
                    predictions = check_special_format(PRD_texts, predictions)
        else:
            if PRD_texts is not None:
                PRD_texts = ' '.join(PRD_texts).upper()
            else:
                PRD_texts = None
        return (
         PRD_texts, predictions)

    def show(self):
        print(self.display_string)
        if self.total_samples == 0:
            pass
        print('DMY Accuracy: {:.4f}\tCorrect/Total: {}/{}'.format(self.correct / self.total_samples, self.correct, self.total_samples))
        print('EXP Accuracy: {:.4f}\tCorrect/Total: {}/{}'.format(self.correct_full / self.total_images, self.correct_full, self.total_images))
        self.clear()


class Loss_counter:

    def __init__(self, display_interval):
        self.display_interval = display_interval
        self.total_iters = 0.0
        self.loss_sum = 0

    def add_iter(self, loss):
        self.total_iters += 1
        self.loss_sum += float(loss)

    def clear(self):
        self.total_iters = 0
        self.loss_sum = 0

    def get_loss(self):
        loss = self.loss_sum / self.total_iters if self.total_iters > 0 else 0
        self.total_iters = 0
        self.loss_sum = 0
        return loss


def check_special_format(date, prds):
    c_year = datetime.today().year - 2000
    date = torch.tensor([int(d) for d in date])
    if abs(date[0] - c_year) > abs(date[2] - c_year):
        prds.extra_fields['labels'] = torch.tensor([1, 2, 3])
    else:
        if abs(date[0] - c_year) < abs(date[2] - c_year):
            prds.extra_fields['labels'] = torch.tensor([3, 2, 1])
    return prds
# okay decompiling ./utils.pyc
