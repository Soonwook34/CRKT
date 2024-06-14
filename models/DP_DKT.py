import torch
import torch.nn as nn
from torch.nn import (Embedding, GRU, Dropout, Sequential, Linear, ReLU, Sigmoid)
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)
from torch.nn.functional import one_hot


class DP_DKT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "DP_DKT"
        self.num_q = args.question
        self.num_c = args.concept
        self.num_r = 2
        self.num_o = args.option
        self.dim_q = args.dim_q
        self.dim_c = args.dim_c

        # embedding
        self.question_emb = Embedding(self.num_q + 1, self.dim_q, padding_idx=-1)
        self.concept_emb = Embedding(self.num_c + 1, self.dim_c, padding_idx=-1)
        self.score_emb = Embedding(self.num_r + 1, self.dim_c, padding_idx=-1)
        self.option_emb = Embedding(self.num_o + 1, self.dim_c, padding_idx=-1)

        # main module
        input_feature = self.dim_q + self.dim_c * 4
        self.hidden_feature = input_feature // 2
        self.rnn = GRU(input_size=input_feature, hidden_size=self.hidden_feature, batch_first=True, bidirectional=False)

        # prediction module
        self.pred_in_feature = self.hidden_feature + self.dim_q + self.dim_c * 2
        self.mlp_pred = Sequential(
            Linear(self.pred_in_feature, self.pred_in_feature), ReLU(), Dropout(args.dropout),
            Linear(self.pred_in_feature, self.pred_in_feature), ReLU(), Dropout(args.dropout))
        self.output_layer = Linear(self.pred_in_feature, self.num_o)
        self.dropout = Dropout(args.dropout)
        self.sigmoid = Sigmoid()

    def forward(self, question, concept, score, option, answer):
        mask = torch.ne(score, -1).int()
        data_length = torch.sum(mask, dim=1).detach().cpu().tolist()

        question = torch.where(question >= 0, question, self.num_q)
        concept_mask = torch.ne(concept, -1)  # (batch, seq, max_concept_len)
        concept = torch.where(concept >= 0, concept, self.num_c)
        score = torch.where(score >= 0, score, self.num_r)
        option = torch.where(option >= 0, option, self.num_o)
        answer_pred = torch.where(answer >= 0, answer, 0)[:, 1:]
        answer = torch.where(answer >= 0, answer, self.num_o)

        question_emb = self.question_emb(question)
        cnt_c = torch.sum(concept_mask, dim=-1).unsqueeze(-1)  # (batch, seq, 1)
        cnt_c = torch.where(cnt_c > 0, cnt_c, 1)
        concept_emb = torch.sum(self.concept_emb(concept) * concept_mask.unsqueeze(-1), dim=2) / cnt_c  # (batch, seq, dim)
        score_emb = self.score_emb(score)
        option_emb = self.option_emb(option)
        answer_emb = self.option_emb(answer)

        lstm_input = [question_emb, concept_emb,
                      score_emb, option_emb, answer_emb]
        lstm_input = self.dropout(torch.cat(lstm_input, dim=-1))
        packed_data = pack_padded_sequence(lstm_input, lengths=data_length, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_data)
        ht, _ = pad_packed_sequence(packed_output, batch_first=True)

        ht_pred = ht[:, :-1, :]
        pred_input = [ht_pred, question_emb[:, 1:, :], concept_emb[:, 1:, :], answer_emb[:, 1:, :]]
        pred_input = self.dropout(torch.cat(pred_input, dim=-1))
        output = self.output_layer(self.mlp_pred(pred_input) + pred_input)
        # output = self.output_layer(pred_input)

        kt_one_hot = one_hot(answer_pred, num_classes=self.num_o)
        kt_output = self.sigmoid(torch.sum(output * kt_one_hot, dim=-1))
        ot_output = torch.softmax(output, dim=-1)

        return kt_output, ot_output

