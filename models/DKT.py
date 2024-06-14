import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot


class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid'):
        super().__init__()
        self.name = "DKT"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2 + 1, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)

    def forward(self, q, r):
        emb_type = self.emb_type
        if emb_type == "qid":
            x = torch.where(q >= 0, q * 2 + r, self.num_c * 2)
            xemb = self.interaction_emb(x)

        h, _ = self.lstm_layer(xemb)
        h = h[:, :-1]
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        question_pred = torch.where(q >= 0, q, 0)[:, 1:]
        pred_mask = one_hot(question_pred, num_classes=self.num_c)
        ouptut = torch.sum(y * pred_mask, dim=-1)

        return ouptut
