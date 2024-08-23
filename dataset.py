import os
import json
from random import randint, Random
from copy import deepcopy
import collections

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class KTDataset(Dataset):
    def __init__(self, path, fold_list, max_concept_len, flip_rate=0.8, seed=0):
        # load fold data
        logs_all = []
        for fold_data in fold_list:
            with open(os.path.join(path, "data", fold_data), encoding="utf8") as log_data:
                logs = json.load(log_data)
            logs_all += logs

        # load question data
        with open(os.path.join(path, "data", "question_data.json"), encoding="utf8") as q_data:
            question_data = json.load(q_data)
        max_option_list = [question["option_len"] for question in question_data]
        self.correct_rate = torch.tensor([question["correct_rate"] for question in question_data])

        data = self.generate_data(logs_all, max_option_list, max_concept_len)
        self.user = data["user"]
        self.question = data["question"]
        self.concept = data["concept"]
        self.score = data["score"]
        self.option = data["option"]
        self.answer = data["answer"]
        self.unchosen = data["unchosen"]

        cl_data = self.generate_cl_data(flip_rate, max_option_list, seed)
        self.pos_score = cl_data["pos_score"]
        self.pos_option = cl_data["pos_option"]
        self.neg_score = cl_data["neg_score"]
        self.neg_option = cl_data["neg_option"]

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        return (self.user[idx], self.question[idx], self.concept[idx], self.score[idx],
                self.option[idx], self.answer[idx], self.unchosen[idx],
                self.pos_score[idx], self.pos_option[idx], self.neg_score[idx], self.neg_option[idx])

    def generate_data(self, logs, max_option_list, max_concept_len):
        users, questions, concepts, scores = [], [], [], []
        options, answers, unchosen_options = [], [], []

        max_options = np.array(max_option_list)

        for log in logs:
            seq_len = len(log["question"])
            user = [log["user"]] * seq_len
            question = log["question"]
            concept = [con[::-1] + [-1] * (max_concept_len - len(con)) for con in log["concept"]]
            score = log["score"]
            option = log["option"]
            answer = log["answer"]
            max_option = max_options[question].tolist()
            unchosen_option = [(opt + randint(1, max_opt-1)) % max_opt
                               for opt, max_opt in zip(option, max_option)]
            users.append(user)
            questions.append(question)
            concepts.append(concept)
            scores.append(score)
            options.append(option)
            answers.append(answer)
            unchosen_options.append(unchosen_option)

        return {"user": users, "question": questions, "concept": concepts, "score": scores,
                "option": options, "answer": answers, "unchosen": unchosen_options}

    def generate_cl_data(self, flip_rate, max_option_list, seed):
        count = 0
        count_all, count_target_all, count_target, count_flip_pos, count_flip_neg = 0, 0, 0, 0, 0
        pos_scores, pos_options = [], []
        neg_scores, neg_options = [], []
        rand = Random(seed)
        if flip_rate > 0:
            for question, concept, score, option, answer in zip(self.question, self.concept, self.score, self.option, self.answer):
                concept = [c[0] for c in concept]
                concept_count = collections.Counter(concept)
                target_info = {}
                for t in range(len(question) - 1, 0, -1):
                    if 0.4 <= self.correct_rate[question[t]] <= 0.6:
                        if concept_count[concept[t]] >= 3 and concept[t] not in target_info:
                            target_info[concept[t]] = {"question": question[t], "concept": concept[t], "score": score[t],
                                                       "correct_rate": self.correct_rate[question[t]]}
                pos_score, pos_option = deepcopy(score), deepcopy(option)
                neg_score, neg_option = deepcopy(score), deepcopy(option)
                for t in range(len(question)):
                    count_all += 1
                    if concept[t] in target_info:
                        count_target_all += 1
                        if rand.random() < flip_rate:
                            count_target += 1
                            target_correct_rate = self.correct_rate[question[t]]
                            max_opt = max_option_list[question[t]]
                            # pos: change low correct rate question's response to 1
                            # neg: change high correct rate question's response to 0
                            if target_info[concept[t]]["score"] == 1:
                                if self.correct_rate[question[t]] <= target_correct_rate:
                                    if pos_score[t] != 1:
                                        count_flip_pos += 1
                                    pos_score[t] = 1
                                    pos_option[t] = answer[t]
                                else:
                                    if neg_score[t] != 0:
                                        count_flip_neg += 1
                                    neg_score[t] = 0
                                    neg_option[t] = (answer[t] + randint(1, max_opt - 1)) % max_opt
                            # pos: change high correct rate question's response to 0
                            # neg: change low correct rate question's response to 1
                            else:
                                if self.correct_rate[question[t]] > target_correct_rate:
                                    if pos_score[t] != 0:
                                        count_flip_pos += 1
                                    pos_score[t] = 0
                                    pos_option[t] = (answer[t] + randint(1, max_opt - 1)) % max_opt
                                else:
                                    if neg_score[t] != 1:
                                        count_flip_neg += 1
                                    neg_score[t] = 1
                                    neg_option[t] = answer[t]
                pos_scores.append(pos_score)
                pos_options.append(pos_option)
                neg_scores.append(neg_score)
                neg_options.append(neg_option)
            count_flip = count_flip_pos + count_flip_neg
            # if count_target > 0 and count_all > 0:
            #     print(count_all, count_target_all, count_target, count_flip_pos, count_flip_neg)
            #     print(f"{count_flip / count_target:.2f}, {count_flip / count_all:.2f}")
        else:
            pos_scores, pos_options = deepcopy(self.score), deepcopy(self.option)
            neg_scores, neg_options = deepcopy(self.score), deepcopy(self.option)

        return {"pos_score": pos_scores, "pos_option": pos_options, "neg_score": neg_scores, "neg_option": pos_options}


def pad_collate(batch):
    (user, question, concept, score, option, answer, unchosen,
     pos_score, pos_option, neg_score, neg_option) = zip(*batch)

    # convert into tensor
    user = [torch.LongTensor(u) for u in user]
    question = [torch.LongTensor(q) for q in question]
    concept = [torch.LongTensor(c) for c in concept]
    score = [torch.LongTensor(s) for s in score]
    option = [torch.LongTensor(o) for o in option]
    answer = [torch.LongTensor(a) for a in answer]
    unchosen = [torch.LongTensor(u) for u in unchosen]
    pos_score = [torch.LongTensor(s) for s in pos_score]
    pos_option = [torch.LongTensor(o) for o in pos_option]
    neg_score = [torch.LongTensor(s) for s in neg_score]
    neg_option = [torch.LongTensor(o) for o in neg_option]


    # apply pad_sequence
    user_pad = pad_sequence(user, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(question, batch_first=True, padding_value=-1)
    concept_pad = pad_sequence(concept, batch_first=True, padding_value=-1)
    score_pad = pad_sequence(score, batch_first=True, padding_value=-1)
    option_pad = pad_sequence(option, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answer, batch_first=True, padding_value=-1)
    unchosen_pad = pad_sequence(unchosen, batch_first=True, padding_value=-1)
    pos_score_pad = pad_sequence(pos_score, batch_first=True, padding_value=-1)
    pos_option_pad = pad_sequence(pos_option, batch_first=True, padding_value=-1)
    neg_score_pad = pad_sequence(neg_score, batch_first=True, padding_value=-1)
    neg_option_pad = pad_sequence(neg_option, batch_first=True, padding_value=-1)

    return (user_pad, question_pad, concept_pad, score_pad, option_pad, answer_pad, unchosen_pad,
            pos_score_pad, pos_option_pad, neg_score_pad, neg_option_pad)

