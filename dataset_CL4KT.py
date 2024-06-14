import os
import json
import math
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class SimCLRDatasetWrapper(Dataset):
    def __init__(
            self,
            ds: Dataset,
            path: str,
            seq_len: int,
            mask_prob: float,
            crop_prob: float,
            permute_prob: float,
            replace_prob: float,
            negative_prob: float,
            eval_mode=False,
    ):
        super().__init__()
        self.ds = ds
        # load question data
        with open(os.path.join(path, "data", "question_data.json"), encoding="utf8") as q_data:
            question_data = json.load(q_data)
        self.correct_rate = torch.tensor([question["correct_rate"] for question in question_data])
        self.seq_len = 200
        self.mask_prob = mask_prob
        self.crop_prob = crop_prob
        self.permute_prob = permute_prob
        self.replace_prob = replace_prob
        self.negative_prob = negative_prob
        self.eval_mode = eval_mode

        self.num_questions = self.ds.num_questions
        self.num_skills = self.ds.num_skills
        self.q_mask_id = self.num_questions + 1
        self.s_mask_id = self.num_skills + 1
        self.easier_skills = self.ds.easier_skills
        self.harder_skills = self.ds.harder_skills

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        original_data = self.ds[index]
        q_seq = original_data["questions"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]
        attention_mask = original_data["attention_mask"]

        if self.eval_mode:
            return {
                "questions": q_seq,
                "skills": s_seq,
                "responses": r_seq,
                "attention_mask": attention_mask,
            }

        else:
            q_seq_list = original_data["questions"].tolist()
            s_seq_list = original_data["skills"].tolist()
            r_seq_list = original_data["responses"].tolist()

            t1 = augment_kt_seqs(
                q_seq_list,
                s_seq_list,
                r_seq_list,
                self.mask_prob,
                self.crop_prob,
                self.permute_prob,
                self.replace_prob,
                self.negative_prob,
                self.easier_skills,
                self.harder_skills,
                self.q_mask_id,
                self.s_mask_id,
                self.seq_len,
                seed=index,
            )

            t2 = augment_kt_seqs(
                q_seq_list,
                s_seq_list,
                r_seq_list,
                self.mask_prob,
                self.crop_prob,
                self.permute_prob,
                self.replace_prob,
                self.negative_prob,
                self.easier_skills,
                self.harder_skills,
                self.q_mask_id,
                self.s_mask_id,
                self.seq_len,
                seed=index + 1,
            )

            aug_q_seq_1, aug_s_seq_1, aug_r_seq_1, negative_r_seq, attention_mask_1 = t1
            aug_q_seq_2, aug_s_seq_2, aug_r_seq_2, _, attention_mask_2 = t2

            aug_q_seq_1 = torch.tensor(aug_q_seq_1, dtype=torch.long)
            aug_q_seq_2 = torch.tensor(aug_q_seq_2, dtype=torch.long)
            aug_s_seq_1 = torch.tensor(aug_s_seq_1, dtype=torch.long)
            aug_s_seq_2 = torch.tensor(aug_s_seq_2, dtype=torch.long)
            aug_r_seq_1 = torch.tensor(aug_r_seq_1, dtype=torch.long)
            aug_r_seq_2 = torch.tensor(aug_r_seq_2, dtype=torch.long)
            negative_r_seq = torch.tensor(negative_r_seq, dtype=torch.long)
            attention_mask_1 = torch.tensor(attention_mask_1, dtype=torch.long)
            attention_mask_2 = torch.tensor(attention_mask_2, dtype=torch.long)

            ret = {
                "questions": (aug_q_seq_1, aug_q_seq_2, q_seq),
                "skills": (aug_s_seq_1, aug_s_seq_2, s_seq),
                "responses": (aug_r_seq_1, aug_r_seq_2, r_seq, negative_r_seq),
                "attention_mask": (attention_mask_1, attention_mask_2, attention_mask),
            }
            return ret

    def __getitem__(self, index):
        return self.__getitem_internal__(index)


class MostRecentQuestionSkillDataset(Dataset):
    def __init__(self, path, fold_list, num_skills, num_questions):
        # load fold data
        logs_all = []
        for fold_data in fold_list:
            with open(os.path.join(path, "data", fold_data), encoding="utf8") as log_data:
                logs = json.load(log_data)
            logs_all += logs

        self.df = json_to_df(logs_all)
        self.seq_len = 200
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["item_id"].values[-self.seq_len:]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.skills = [
            u_df["skill_id"].values[-self.seq_len:]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values[-self.seq_len:]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        for s_list, r_list in zip(self.skills, self.responses):
            for s, r in zip(s_list, r_list):
                skill_correct[s] += r
                skill_count[s] += 1

        skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s]) for s in skill_correct
        }
        ordered_skills = [
            item[0] for item in sorted(skill_difficulty.items(), key=lambda x: x[1])
        ]
        self.easier_skills = {}
        self.harder_skills = {}
        for i, s in enumerate(ordered_skills):
            if i == 0:  # the hardest
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = s
            elif i == len(ordered_skills) - 1:  # the easiest
                self.easier_skills[s] = s
                self.harder_skills[s] = ordered_skills[i - 1]
            else:
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = ordered_skills[i - 1]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )

        for i, elem in enumerate(zip(self.questions, self.skills, self.responses)):
            q, s, r = elem
            self.padded_q[i, -len(q):] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, -len(s):] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, -len(r):] = torch.tensor(r, dtype=torch.long)
            self.attention_mask[i, -len(s):] = torch.ones(len(s), dtype=torch.long)

    def __getitem__(self, index):

        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return self.len


def json_to_df(logs):
    user_id_list, item_id_list, timestamp_list = [], [], []
    correct_list, option_list, skill_id_list = [], [], []

    # ["user_id", "item_id", "timestamp", "correct", "skill_id"]
    for student in logs:
        seq_len = len(student["question"])
        user_id_list += [student["user"]] * seq_len
        item_id_list += student["question"]
        timestamp_list += [t for t in range(seq_len)]
        correct_list += student["score"]
        skill_id_list += [skills[-1] for skills in student["concept"]]

    df = pd.DataFrame(list(zip(user_id_list, item_id_list, timestamp_list, correct_list, skill_id_list)),
                      columns=["user_id", "item_id", "timestamp", "correct", "skill_id"])

    return df


def augment_kt_seqs(
        q_seq,
        s_seq,
        r_seq,
        mask_prob,
        crop_prob,
        permute_prob,
        replace_prob,
        negative_prob,
        easier_skills,
        harder_skills,
        q_mask_id,
        s_mask_id,
        seq_len,
        seed=None,
        skill_rel=None,
):
    # masking (random or PMI 등을 활용해서)
    # 구글 논문의 Correlated Feature Masking 등...
    rng = random.Random(seed)
    np.random.seed(seed)

    masked_q_seq = []
    masked_s_seq = []
    masked_r_seq = []
    negative_r_seq = []

    if mask_prob > 0:
        for q, s, r in zip(q_seq, s_seq, r_seq):
            prob = rng.random()
            if prob < mask_prob and s != 0:
                prob /= mask_prob
                if prob < 0.8:
                    masked_q_seq.append(q_mask_id)
                    masked_s_seq.append(s_mask_id)
                elif prob < 0.9:
                    masked_q_seq.append(
                        rng.randint(1, q_mask_id - 1)
                    )  # original BERT처럼 random한 확률로 다른 token으로 대체해줌
                    masked_s_seq.append(
                        rng.randint(1, s_mask_id - 1)
                    )  # randint(start, end) [start, end] 둘다 포함
                else:
                    masked_q_seq.append(q)
                    masked_s_seq.append(s)
            else:
                masked_q_seq.append(q)
                masked_s_seq.append(s)
            masked_r_seq.append(r)  # response는 나중에 hard negatives로 활용 (0->1, 1->0)

            # reverse responses
            neg_prob = rng.random()
            if neg_prob < negative_prob and r != -1:  # padding
                negative_r_seq.append(1 - r)
            else:
                negative_r_seq.append(r)
    else:
        masked_q_seq = q_seq[:]
        masked_s_seq = s_seq[:]
        masked_r_seq = r_seq[:]

        for r in r_seq:
            # reverse responses
            neg_prob = rng.random()
            if neg_prob < negative_prob and r != -1:  # padding
                negative_r_seq.append(1 - r)
            else:
                negative_r_seq.append(r)

    """
    skill difficulty based replace
    """
    # print(harder_skills)
    if replace_prob > 0:
        for i, elem in enumerate(zip(masked_s_seq, masked_r_seq)):
            s, r = elem
            prob = rng.random()
            if prob < replace_prob and s != 0 and s != s_mask_id:
                if (
                        r == 0 and s in harder_skills
                ):  # if the response is wrong, then replace a skill with the harder one
                    masked_s_seq[i] = harder_skills[s]
                elif (
                        r == 1 and s in easier_skills
                ):  # if the response is correct, then replace a skill with the easier one
                    masked_s_seq[i] = easier_skills[s]

    true_seq_len = np.sum(np.asarray(q_seq) != 0)
    if permute_prob > 0:
        reorder_seq_len = math.floor(permute_prob * true_seq_len)
        start_idx = (np.asarray(q_seq) != 0).argmax()
        while True:
            start_pos = rng.randint(start_idx, seq_len - reorder_seq_len)
            if start_pos + reorder_seq_len < seq_len:
                break

        # reorder (permute)
        perm = np.random.permutation(reorder_seq_len)
        masked_q_seq = (
                masked_q_seq[:start_pos]
                + np.asarray(masked_q_seq[start_pos: start_pos + reorder_seq_len])[
                    perm
                ].tolist()
                + masked_q_seq[start_pos + reorder_seq_len:]
        )
        masked_s_seq = (
                masked_s_seq[:start_pos]
                + np.asarray(masked_s_seq[start_pos: start_pos + reorder_seq_len])[
                    perm
                ].tolist()
                + masked_s_seq[start_pos + reorder_seq_len:]
        )
        masked_r_seq = (
                masked_r_seq[:start_pos]
                + np.asarray(masked_r_seq[start_pos: start_pos + reorder_seq_len])[
                    perm
                ].tolist()
                + masked_r_seq[start_pos + reorder_seq_len:]
        )

    # To-Do: check this crop logic!
    if 0 < crop_prob < 1:
        crop_seq_len = math.floor(crop_prob * true_seq_len)
        if crop_seq_len == 0:
            crop_seq_len = 1
        start_idx = (np.asarray(q_seq) != 0).argmax()
        while True:
            start_pos = rng.randint(start_idx, seq_len - crop_seq_len)
            if start_pos + crop_seq_len < seq_len:
                break

        masked_q_seq = masked_q_seq[start_pos: start_pos + crop_seq_len]
        masked_s_seq = masked_s_seq[start_pos: start_pos + crop_seq_len]
        masked_r_seq = masked_r_seq[start_pos: start_pos + crop_seq_len]

    pad_len = seq_len - len(masked_q_seq)

    attention_mask = [0] * pad_len + [1] * len(masked_s_seq)
    masked_q_seq = [0] * pad_len + masked_q_seq
    masked_s_seq = [0] * pad_len + masked_s_seq
    masked_r_seq = [-1] * pad_len + masked_r_seq

    return masked_q_seq, masked_s_seq, masked_r_seq, negative_r_seq, attention_mask
