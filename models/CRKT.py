from math import sqrt

import torch
from torch import nn
from torch.nn import (Embedding, Sequential, Linear, Sigmoid, ReLU, Dropout, LayerNorm, Softplus, CosineSimilarity)
from torch.nn.functional import (one_hot, softmax)
from torch_geometric.nn import GCN
from torch_geometric.utils import to_edge_index, add_remaining_self_loops


class CRKT(nn.Module):
    def __init__(self, num_c, num_q, num_o, dim_c, dim_q, dim_g,
                 num_heads, layer_g, top_k, lamb, map, option_list, dropout, bias, version):
        super().__init__()
        self.name = "CRKT"

        ### Config
        self.num_c = num_c
        self.num_q = num_q
        self.num_o = num_o
        self.dim_c = dim_c
        self.dim_q = dim_q
        self.dim_g = dim_g
        self.top_k = top_k
        self.lamb = lamb
        self.concept_map = nn.Parameter(map, requires_grad=False)
        self.num_edge = to_edge_index(self.concept_map.to_sparse())[0].shape[1]
        self.option_mask = nn.Parameter(torch.zeros((self.num_q + 1, self.num_o), dtype=torch.int),
                                        requires_grad=False)
        for q, max_option in enumerate(option_list):
            self.option_mask[q, :max_option] = 1
        self.option_mask[-1, :2] = 1  # padding for chosen and valid unchosen response
        self.option_range = nn.Parameter(torch.tensor([[i for i in range(self.num_o)] for _ in range(self.num_q + 1)]),
                                         requires_grad=False)

        # Embedding
        self.concept_emb = Embedding(self.num_c, self.dim_c)
        self.question_emb = Embedding(self.num_q + 1, self.dim_q)
        self.response_emb = Embedding(self.num_q * self.num_o + 2, self.dim_q, padding_idx=-1)

        # 4.A Disentangled Response Encoder
        dim_h1 = self.dim_q // 2
        self.enc_correct = Sequential(
            Linear(self.dim_q, dim_h1), ReLU(), Dropout(dropout),
            Linear(dim_h1, self.dim_q))
        self.enc_wrong = Sequential(
            Linear(self.dim_q, dim_h1), ReLU(), Dropout(dropout),
            Linear(dim_h1, self.dim_q))
        self.enc_unchosen = Sequential(
            Linear(self.dim_q, dim_h1), ReLU(), Dropout(dropout),
            Linear(dim_h1, self.dim_q))
        self.attn_response = CRKTLayer(self.dim_q, self.dim_q, num_heads, dropout, kq_same=True)

        # 4,B Knowledge Retriever
        self.attn_question = CRKTLayer(self.dim_q, self.dim_q, num_heads, dropout, kq_same=True)
        self.attn_state = CRKTLayer(self.dim_q, self.dim_q, num_heads, dropout, kq_same=True)

        # 4.C Concept Map Encoder
        dim_h2 = (self.dim_q + self.dim_c) // 2
        self.enc_concept = Sequential(
            Linear(self.dim_q + self.dim_c, dim_h2), ReLU(), Dropout(dropout),
            Linear(dim_h2, self.dim_g))
        dim_h3 = (self.dim_q + self.dim_c * 2) // 4
        self.enc_intensity = Sequential(
            Linear(self.dim_q + self.dim_c * 2, dim_h3), ReLU(), Dropout(dropout),
            Linear(dim_h3, 1))
        self.gcn = GCN(in_channels=self.dim_g, hidden_channels=self.dim_g // 2, num_layers=layer_g,
                       out_channels=1, dropout=dropout, bias=bias)

        # 4.D IRT-based Prediction
        self.w_relevance = Linear(self.dim_q, self.dim_c)
        self.mlp_diff = Sequential(
            Linear(self.dim_q, dim_h1), ReLU(), Dropout(dropout),
            Linear(dim_h1, 1))
        self.sigmoid = Sigmoid()
        self.relu = ReLU()
        self.softplus = Softplus()

        self.cossim = CosineSimilarity(dim=-1)
        self.temp = 0.05

    def forward(self, question, concept, score, option, unchosen,
                pos_score=None, pos_option=None, neg_score=None, neg_option=None, inference=False):
        self.batch_size, self.seq_len = question.shape

        seq_mask = torch.ne(score, -1)[:, :-1]
        qt_idx, c_qt_idx, ot_idx, ut_idx, ut_mask = self.get_index(question, concept, score, option, unchosen)

        # get question embedding
        qt = self.question_emb(qt_idx)  # (batch, seq, dim_q)

        # 4.A Disentangled Response Encoder
        dt_hat = self.encode_disentangled_response(ot_idx, score, ut_idx, ut_mask)  # (batch, seq, dim_q)

        # 4.B Knowledge Retriever
        qt_hat, ht = self.get_knowledge_state(qt, dt_hat)  # (batch, seq-1, dim_q)

        # 4.C Concept Map Encoder
        mt = self.get_concept_mastery(ht)  # (batch, seq-1, concept, dim_g)

        q_target = qt[:, 1:, :]  # (batch, seq-1, dim_q)
        edge_index, edge_weight = self.get_concept_map(q_target)

        c_target_idx = c_qt_idx[:, 1:, :]  # (batch, seq-1, max_c)
        mt_hat, g_target = self.update_concept_mastery(mt, edge_index, edge_weight, c_target_idx)  # (batch, seq-1, concept)

        # 4.D IRT-based Prediction
        r_target, topk_r_target = self.get_concept_weight(q_target)  # (batch, seq-1, concept)

        output = self.predict(mt_hat, topk_r_target, q_target)

        # 4.E Model Training (Contrastive Learning)
        _, _, pos_ot_idx, pos_ut_idx, pos_ut_mask = self.get_index(question, concept, pos_score, pos_option, unchosen)
        _, _, neg_ot_idx, neg_ut_idx, neg_ut_mask = self.get_index(question, concept, neg_score, neg_option, unchosen)
        pos_dt_hat = self.encode_disentangled_response(pos_ot_idx, pos_score, pos_ut_idx, pos_ut_mask)  # (batch, seq, dim_q)
        _, pos_ht = self.get_knowledge_state(qt, pos_dt_hat)  # (batch, seq-1, dim_q)
        neg_dt_hat = self.encode_disentangled_response(neg_ot_idx, neg_score, neg_ut_idx, neg_ut_mask)  # (batch, seq, dim_q)
        _, neg_ht = self.get_knowledge_state(qt, neg_dt_hat)  # (batch, seq-1, dim_q)

        seq_count = torch.sum(seq_mask, dim=-1).unsqueeze(-1)
        pooled_score = torch.sum(ht * seq_mask.unsqueeze(-1), dim=1) / seq_count  # (batch, dim_q)
        pooled_pos_score = torch.sum(pos_ht * seq_mask.unsqueeze(-1), dim=1) / seq_count  # (batch, dim_q)
        pooled_neg_score = torch.sum(neg_ht * seq_mask.unsqueeze(-1), dim=1) / seq_count  # (batch, dim_q)
        pos_cossim = self.cossim(pooled_score.unsqueeze(1), pooled_pos_score.unsqueeze(0)) / self.temp  # (batch, batch)
        neg_cossim = self.cossim(pooled_score.unsqueeze(1), pooled_neg_score.unsqueeze(0)) / self.temp  # (batch, batch)
        neg_weights = torch.eye(neg_cossim.shape[0], device=neg_cossim.device)
        neg_cossim = neg_cossim + neg_weights
        inter_cossim = torch.cat([pos_cossim, neg_cossim], dim=1)  # (batch, batch*2)
        inter_label = torch.arange(inter_cossim.shape[0]).long().to(inter_cossim.device)

        return output, r_target, g_target, inter_cossim, inter_label

    def get_index(self, question, concept, score, option, unchosen):
        qt_idx = torch.where(question >= 0, question, self.num_q)  # (batch, seq)
        c_qt_idx = torch.where(concept >= 0, concept, self.num_c)  # (batch, seq, max_c)
        opt = torch.where(option >= 0, option, 0)  # (batch, seq)
        opt_one_hot = one_hot(opt, num_classes=self.num_o)
        ot_idx = torch.where(
            option >= 0,
            qt_idx * self.num_o + option,
            self.num_q * self.num_o
        )  # (batch, seq)
        # use all unchosen responses
        ut_mask = self.option_mask[qt_idx] - opt_one_hot
        ut_idx = torch.where(
            ut_mask > 0,
            (qt_idx * self.num_o).unsqueeze(-1) + self.option_range[qt_idx],
            self.num_q * self.num_o + 1
        )  # (batch, seq, num_o)

        return qt_idx, c_qt_idx, ot_idx, ut_idx, ut_mask

    def encode_disentangled_response(self, ot_idx, score, ut_idx, ut_mask):
        ot = self.response_emb(ot_idx)  # (batch, seq, dim_q)
        ut = self.response_emb(ut_idx)  # (batch, seq, num_o, dim_q)

        correct_mask = torch.eq(score, 1)
        wrong_mask = torch.eq(score, 0)

        ot_prime = ot
        ot_prime[correct_mask] = self.enc_correct(ot[correct_mask])
        ot_prime[wrong_mask] = self.enc_wrong(ot[wrong_mask])

        ut_prime = self.enc_unchosen(ut)

        # use all unchosen responses
        ut_count = torch.sum(ut_mask, dim=-1, keepdim=True)
        ut_prime = torch.sum(ut_prime * ut_mask.unsqueeze(-1), dim=2) / ut_count

        dt = ot_prime - self.lamb * ut_prime
        dt_hat = self.attn_response(dt, dt, dt, self.seq_len, maxout=False)

        return dt_hat

    def get_knowledge_state(self, qt, dt_hat):
        qt_hat = self.attn_question(qt, qt, qt, self.seq_len, maxout=False)  # (batch, seq-1, dim_q)

        ht = self.attn_state(qt_hat, qt_hat, dt_hat, self.seq_len, maxout=True)  # (batch, seq-1, dim_q)

        return qt_hat[:, :-1, :], ht[:, :-1, :]

    def get_concept_mastery(self, ht):
        ht_concept = (
            ht.unsqueeze(2)
            .expand(-1, -1, self.num_c, -1)
            .contiguous()
        )  # (batch, seq-1, concept, dim_q)

        ci = self.concept_emb.weight  # (concept, dim_c)
        ci_batch = (
            ci[None, None, :, :]
            .expand(self.batch_size, self.seq_len - 1, -1, -1)
            .contiguous()
        )  # (batch, seq-1, concept, dim_c)

        mt = self.enc_concept(torch.cat([ht_concept, ci_batch], dim=-1))  # (batch, seq-1, concept, dim_g)

        return mt

    def get_concept_map(self, q_target):
        batch_adj = (
            self.concept_map.expand(self.batch_size * (self.seq_len - 1), -1, -1)
            .contiguous()
            .to_sparse()
        )  # (batch*seq-1, concept, concept)
        batch_edge_index, edge_weight = to_edge_index(batch_adj)
        batch_index = batch_edge_index[0]
        edge_index = batch_edge_index[1:] + (batch_index * self.num_c)

        # Target-specific Edge Weight
        q_target_edge = (
            q_target.unsqueeze(2)
            .expand(-1, -1, self.num_edge, -1)
            .contiguous()
        )  # (batch, seq-1, edge, dim_q)
        cij_idx = to_edge_index(self.concept_map.to_sparse())[0]  # (2, edge)
        cij = self.concept_emb(cij_idx)  # (2, edge, dim_c)
        cij_concat = torch.cat([cij[0, :, :], cij[1, :, :]], dim=-1)  # (edge, dim_c*2)
        cij_batch = (
            cij_concat[None, None, :, :]
            .expand(self.batch_size, self.seq_len - 1, -1, -1)
            .contiguous()
        )  # (batch, seq-1, edge, dim_c*2)

        edge_weight = self.relu(self.enc_intensity(torch.cat([q_target_edge, cij_batch], dim=-1)))
        edge_weight = edge_weight.flatten()

        num_nodes = self.batch_size * (self.seq_len - 1) * self.num_c
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight,
                                                           fill_value=0.5, num_nodes=num_nodes)

        return edge_index, edge_weight

    def update_concept_mastery(self, mt, edge_index, edge_weight, c_target_idx):
        g_target = torch.sum(one_hot(c_target_idx, num_classes=self.num_c + 1).float(),
                             dim=2)[:, :, :-1]  # (batch, seq-1, concept)
        g_target_flat = g_target.flatten().unsqueeze(-1)

        mt = mt.view(-1, self.dim_g)
        mt_hat = self.gcn(x=mt, edge_index=edge_index, edge_weight=edge_weight)
        # mt_hat = self.sigmoid(self.gcn(x=mt, edge_index=edge_index, edge_weight=edge_weight))
        mt_hat = mt_hat.view(self.batch_size, self.seq_len - 1, self.num_c)  # (batch, seq-1, concept)

        return mt_hat, g_target

    def get_concept_weight(self, q_target):
        ck = self.concept_emb.weight  # (concept, dim_c)
        ck_batch = (
            ck[None, None, :, :]
            .expand(self.batch_size, self.seq_len - 1, -1, -1)
            .contiguous()
        )  # (batch, seq-1, concept, dim_c)

        q_target_batch = (
            # q_target
            self.w_relevance(q_target)
            .unsqueeze(2)
            .expand(-1, -1, self.num_c, -1)
            .contiguous()
        )  # (batch, seq-1, concept, dim_c)

        r_target = torch.sum(q_target_batch * ck_batch, dim=-1)  # (batch, seq-1, concept)
        # r_target = self.sigmoid(torch.sum(q_target_batch * ck_batch, dim=-1))  # (batch, seq-1, concept)

        # Top-K Concept
        topk_values, _ = torch.topk(r_target, k=self.top_k, dim=-1, largest=True, sorted=True)  # (batch, seq-1, k)
        topk_mask = r_target >= topk_values[:, :, -1].unsqueeze(-1)
        topk_r_target = r_target.masked_fill(topk_mask == 0, -1e+32)

        topk_r_target = torch.softmax(topk_r_target, dim=-1)
        topk_r_target = topk_r_target.masked_fill(topk_mask == 0, 0)
        r_target = self.sigmoid(r_target)

        return r_target, topk_r_target

    def predict(self, mt_hat, r_target, q_target):
        ability = torch.sum(r_target * mt_hat, dim=2)  # (batch, seq-1)

        difficulty = self.mlp_diff(q_target).squeeze(-1)  # (batch, seq-1)
        # difficulty = self.sigmoid(self.mlp_diff(q_target).squeeze(-1))  # (batch, seq-1)

        output = self.sigmoid(ability - difficulty)  # (batch, seq-1)

        return output


class CRKTLayer(nn.Module):
    def __init__(self, dim, dim_out, num_heads, dropout, kq_same=True):
        super().__init__()
        self.attn_kt = KTAttention(dim, dim_out, num_heads, kq_same)

        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(dim_out)

    def forward(self, query, key, value, seq_len, maxout=False):
        attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device), diagonal=0).bool()

        attn_output = self.attn_kt(query, key, value, attn_mask, maxout)

        output = self.layer_norm(self.dropout(attn_output))

        return output


class KTAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, kq_same, bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.d_k = dim // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(self.dim, self.dim, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(self.dim, self.dim, bias=bias)
        self.v_linear = nn.Linear(self.dim, self.dim, bias=bias)

        self.decay_rate = nn.Parameter(torch.zeros(num_heads, 1, 1))
        nn.init.xavier_uniform_(self.decay_rate)
        self.out_proj = nn.Linear(self.dim, self.dim_out, bias=bias)

        self.relu = ReLU()

    def forward(self, query, key, value, mask, maxout=False):
        self.batch_size, self.seq_len, _ = query.shape

        # perform linear operation and split into h heads
        q = self.q_linear(query).view(self.batch_size, self.seq_len, self.num_heads, self.d_k)
        k = self.k_linear(key).view(self.batch_size, self.seq_len, self.num_heads, self.d_k)
        v = self.v_linear(value).view(self.batch_size, self.seq_len, self.num_heads, self.d_k)

        q = q.transpose(1, 2)  # (batch, head, seq, d_K)
        k = k.transpose(1, 2)  # (batch, head, seq, d_K)
        v = v.transpose(1, 2)  # (batch, head, seq, d_K)

        attn_output = self.attention(q, k, v, mask, maxout)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(self.batch_size, self.seq_len, self.dim)
        )

        output = self.out_proj(attn_output)

        return output

    def attention(self, q, k, v, mask, maxout):
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.d_k)  # (batch, head, seq, seq)

        t = torch.arange(self.seq_len).float().expand(self.seq_len, -1).to(q.device)
        tau = t.transpose(0, 1).contiguous()

        # calculate context-aware attention score
        with torch.no_grad():
            # $|t - \tau|$
            temporal_dist = torch.abs(t - tau)[None, None, :, :]  # (1, 1, seq, seq)

            # $\gamma_{t, t^\prime}$
            masked_attn_score = softmax(attn_score.masked_fill(mask == 0, -1e+32), dim=-1)

            # $\sum_{t^\prime=\tau+1}^{t}{t, t^\prime}$
            tau_sum_score = torch.cumsum(masked_attn_score, dim=-1)
            t_sum_score = torch.sum(masked_attn_score, dim=-1, keepdim=True)

            dist = self.relu(temporal_dist * (t_sum_score - tau_sum_score))
            dist = dist.detach()

        total_effect = torch.exp(-self.decay_rate.abs().unsqueeze(0) * dist)
        attn_score *= total_effect

        # normalize attention score
        alpha = softmax(attn_score.masked_fill(mask == 0, -1e+32), dim=-1)
        alpha = alpha.masked_fill(mask == 0, 0)

        # maxout scale
        if maxout:
            scale = torch.clamp(1.0 / alpha.max(dim=-1, keepdim=True)[0], max=5.0)
            alpha *= scale

        output = torch.matmul(alpha, v)

        return output

