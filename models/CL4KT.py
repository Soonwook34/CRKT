import numpy as np
import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Parameter,
    Embedding,
    Linear,
    LayerNorm,
    Dropout,
    Softplus,
    ModuleList,
    Sequential
)
import torch.nn.functional as F
from torch.nn.modules.activation import GELU
from torch.nn.init import xavier_uniform_, constant_

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class CL4KT(Module):
    def __init__(self, num_skills, num_questions, seq_len, args):
        super(CL4KT, self).__init__()
        self.name = "CL4KT"
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.seq_len = seq_len
        self.hidden_size = args.dim_c
        self.num_blocks = args.n_blocks
        self.num_attn_heads = args.num_heads
        self.kq_same = True
        self.final_fc_dim = 512
        self.d_ff = args.d_ff
        self.l2 = 0.0
        self.dropout = args.dropout
        self.reg_cl = 0.1
        self.negative_prob = 1.0
        self.hard_negative_weight = 1.0

        self.question_embed = Embedding(
            self.num_skills + 2, self.hidden_size, padding_idx=0
        )
        self.interaction_embed = Embedding(
            2 * (self.num_skills + 2), self.hidden_size, padding_idx=0
        )
        self.sim = Similarity(temp=0.05)

        self.question_encoder = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.interaction_encoder = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.knoweldge_retriever = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out = Sequential(
            Linear(2 * self.hidden_size, self.final_fc_dim),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim, self.final_fc_dim // 2),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim // 2, 1),
        )

        self.cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.loss_fn = nn.BCELoss(reduction="mean")

    def forward(self, batch):
        if self.training:
            q_i, q_j, q = batch["skills"]  # augmented q_i, augmented q_j and original q
            r_i, r_j, r, neg_r = batch[
                "responses"
            ]  # augmented r_i, augmented r_j and original r
            attention_mask_i, attention_mask_j, attention_mask = batch["attention_mask"]

            ques_i_embed = self.question_embed(q_i)
            ques_j_embed = self.question_embed(q_j)
            inter_i_embed = self.get_interaction_embed(q_i, r_i)
            inter_j_embed = self.get_interaction_embed(q_j, r_j)
            if self.negative_prob > 0:
                # inter_k_embed = self.get_negative_interaction_embed(q, r) # hard negative
                inter_k_embed = self.get_interaction_embed(q, neg_r)

            # mask=2 means bidirectional attention of BERT
            ques_i_score, ques_j_score = ques_i_embed, ques_j_embed
            inter_i_score, inter_j_score = inter_i_embed, inter_j_embed

            # BERT
            for block in self.question_encoder:
                ques_i_score, _ = block(
                    mask=2,
                    query=ques_i_score,
                    key=ques_i_score,
                    values=ques_i_score,
                    apply_pos=False,
                )
                ques_j_score, _ = block(
                    mask=2,
                    query=ques_j_score,
                    key=ques_j_score,
                    values=ques_j_score,
                    apply_pos=False,
                )

            for block in self.interaction_encoder:
                inter_i_score, _ = block(
                    mask=2,
                    query=inter_i_score,
                    key=inter_i_score,
                    values=inter_i_score,
                    apply_pos=False,
                )
                inter_j_score, _ = block(
                    mask=2,
                    query=inter_j_score,
                    key=inter_j_score,
                    values=inter_j_score,
                    apply_pos=False,
                )
                if self.negative_prob > 0:
                    inter_k_score, _ = block(
                        mask=2,
                        query=inter_k_embed,
                        key=inter_k_embed,
                        values=inter_k_embed,
                        apply_pos=False,
                    )

            pooled_ques_i_score = (ques_i_score * attention_mask_i.unsqueeze(-1)).sum(
                1
            ) / attention_mask_i.sum(-1).unsqueeze(-1)
            pooled_ques_j_score = (ques_j_score * attention_mask_j.unsqueeze(-1)).sum(
                1
            ) / attention_mask_j.sum(-1).unsqueeze(-1)

            ques_cos_sim = self.sim(
                pooled_ques_i_score.unsqueeze(1), pooled_ques_j_score.unsqueeze(0)
            )
            # Hard negative should be added

            ques_labels = torch.arange(ques_cos_sim.size(0)).long().to(q_i.device)
            question_cl_loss = self.cl_loss_fn(ques_cos_sim, ques_labels)
            # question_cl_loss = torch.mean(question_cl_loss)

            pooled_inter_i_score = (inter_i_score * attention_mask_i.unsqueeze(-1)).sum(
                1
            ) / attention_mask_i.sum(-1).unsqueeze(-1)
            pooled_inter_j_score = (inter_j_score * attention_mask_j.unsqueeze(-1)).sum(
                1
            ) / attention_mask_j.sum(-1).unsqueeze(-1)

            inter_cos_sim = self.sim(
                pooled_inter_i_score.unsqueeze(1), pooled_inter_j_score.unsqueeze(0)
            )

            if self.negative_prob > 0:
                pooled_inter_k_score = (
                    inter_k_score * attention_mask.unsqueeze(-1)
                ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                neg_inter_cos_sim = self.sim(
                    pooled_inter_i_score.unsqueeze(1), pooled_inter_k_score.unsqueeze(0)
                )
                inter_cos_sim = torch.cat([inter_cos_sim, neg_inter_cos_sim], 1)
                # print(inter_k_score.shape, attention_mask.shape)
                # print(pooled_inter_k_score.shape)
                # print(neg_inter_cos_sim.shape)
                # print(inter_cos_sim.shape)
                # exit()

            inter_labels = torch.arange(inter_cos_sim.size(0)).long().to(q_i.device)

            if self.negative_prob > 0:
                weights = torch.tensor(
                    [
                        [0.0] * (inter_cos_sim.size(-1) - neg_inter_cos_sim.size(-1))
                        + [0.0] * i
                        + [self.hard_negative_weight]
                        + [0.0] * (neg_inter_cos_sim.size(-1) - i - 1)
                        for i in range(neg_inter_cos_sim.size(-1))
                    ]
                ).to(q_i.device)
                inter_cos_sim = inter_cos_sim + weights

            interaction_cl_loss = self.cl_loss_fn(inter_cos_sim, inter_labels)
        else:
            q = batch["skills"]  # augmented q_i, augmented q_j and original q
            r = batch["responses"]  # augmented r_i, augmented r_j and original r

            attention_mask = batch["attention_mask"]

        q_embed = self.question_embed(q)
        i_embed = self.get_interaction_embed(q, r)

        x, y = q_embed, i_embed
        for block in self.question_encoder:
            x, _ = block(mask=1, query=x, key=x, values=x, apply_pos=True)

        for block in self.interaction_encoder:
            y, _ = block(mask=1, query=y, key=y, values=y, apply_pos=True)

        for block in self.knoweldge_retriever:
            x, attn = block(mask=0, query=x, key=x, values=y, apply_pos=True)

        retrieved_knowledge = torch.cat([x, q_embed], dim=-1)

        output = torch.sigmoid(self.out(retrieved_knowledge)).squeeze()

        if self.training:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "cl_loss": question_cl_loss + interaction_cl_loss,
                "attn": attn,
                "question": q[:, 1:]
            }
        else:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "attn": attn,
                "question": q[:, 1:],
                "x": x,
            }

        return out_dict

    def alignment_and_uniformity(self, out_dict):
        return (
            out_dict["question_alignment"],
            out_dict["interaction_alignment"],
            out_dict["question_uniformity"],
            out_dict["interaction_uniformity"],
        )

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1

        loss = self.loss_fn(pred[mask], true[mask])
        if self.training:
            cl_loss = torch.mean(out_dict["cl_loss"])  # torch.mean() for multi-gpu FIXME
            loss = loss + self.reg_cl * cl_loss

        return loss, len(pred[mask]), true[mask].sum().item()

    def get_interaction_embed(self, skills, responses):
        masked_responses = responses * (responses > -1).long()
        interactions = skills + self.num_skills * masked_responses
        return self.interaction_embed(interactions)


# https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class CL4KTTransformerLayer(Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super(CL4KTTransformerLayer, self).__init__()
        """
            This is a Basic Block of Transformer paper.
            It contains one Multi-head attention object.
            Followed by layer norm and position-wise feed-forward net and dropotu layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttentionWithIndividualFeatures(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )

        # Two layer norm and two dropout layers
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        self.linear1 = Linear(d_model, d_ff)
        self.activation = GELU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)

        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block: object of type BasicBlock(nn.Module). It contains maksed_attn_head objects which is of type MultiHeadAttnetion(nn.Module).
            mask: 0 means that it can peek (엿보다) only past values. 1 means that block can peek only current and past values
            query: Queries. In Transformer paper it is the input for both encoder and decoder
            key: Keys. In transformer paper it is the input for both encoder and decoder
            values: Values. In transformer paper it is the input for encoder and encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the alyer andr returned
        """

        batch_size, seqlen = query.size(0), query.size(1)
        """
        when mask==1
        >>> nopeek_mask (for question encoder, knoweldge encoder)
            array([[[[0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0]]]], dtype=uint8)

         >>> src_mask
            tensor([[[[ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]]]])

        when mask==0 (for knowledge retriever)
        >>> nopeek_mask
            array([[[[1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1]]]], dtype=uint8)

        >>> src_mask
            tensor([[[[False, False, False, False, False],
                    [ True, False, False, False, False],
                    [ True,  True, False, False, False],
                    [ True,  True,  True, False, False],
                    [ True,  True,  True,  True, False]]]])

        row: target, col: source
        """
        device = query.get_device()
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")

        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        bert_mask = torch.ones_like(src_mask).bool()

        if mask == 0:
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask)
        elif mask == 1:
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask)
        else:  # mask == 2
            query2, attn = self.masked_attn_head(query, key, values, mask=bert_mask)

        query = query + self.dropout1((query2))  # residual connection
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)

        return query, attn


class MultiHeadAttentionWithIndividualFeatures(Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super(MultiHeadAttentionWithIndividualFeatures, self).__init__()
        """
        It has projection layer for getting keys, queries, and values. Followed by attention and a connected layer.
        """

        # d_feature=d_model // n_heads
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = Linear(d_model, d_model, bias=bias)
        self.k_linear = Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = Linear(d_model, d_model, bias=bias)
        self.dropout = Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = Linear(d_model, d_model, bias=bias)
        self.gammas = Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions batch_size * num_heads * seqlen * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        gammas = self.gammas
        scores, attn_scores = individual_attention(
            q, k, v, self.d_k, mask, self.dropout, gammas
        )

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        # concat torch.Size([24, 200, 256])   [batch_size, seqlen, d_model]
        # print('concat', concat.shape)
        output = self.out_proj(concat)

        return output, attn_scores


def individual_attention(q, k, v, d_k, mask, dropout, gamma=None):
    """
    This is called by MultiHeadAttention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(
        d_k
    )  # [batch_size, 8, seq_len, seq_len]
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float()

        distcum_scores = torch.cumsum(scores_, dim=-1)

        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)

        device = distcum_scores.get_device()
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor)
        position_effect = position_effect.to(device)

        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    m = Softplus()

    gamma = -1.0 * m(gamma).unsqueeze(0)

    total_effect = torch.clamp(
        torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
    )

    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    attn_scores = scores
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, attn_scores