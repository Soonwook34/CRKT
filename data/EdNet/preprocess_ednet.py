import os
import json
from random import shuffle

import numpy as np
import pandas as pd
from tqdm import tqdm


# subject_ids, q_ids, correct_ans, ans, labels
DATA_JSON = "kt_ednet.json"

USER_NUM = 5000


def convert_column(source, target_list):
    if source not in target_list:
        print(source)
    return target_list.index(source) if source in target_list else None


def preprocess(path):
    # load json
    raw_path = os.path.join(path, "raw")
    data_path = os.path.join(raw_path, DATA_JSON)
    with open(data_path, encoding="utf8") as data:
        data_json = json.load(data)
    shuffle(data_json)

    user_id_data, question_data, is_correct_data = [], [], []
    answer_data, timestamp_data, concept_data, correct_answer_data = [], [], [], []

    time, user_id = 0, 0
    with tqdm(data_json, unit="stu") as data_bar:
        for logs in data_bar:
            if user_id == USER_NUM:
                break
            seq_len = len(logs["labels"])
            if seq_len < 10:
                continue
            user_id_data += [user_id] * seq_len
            question_data += logs["q_ids"]
            is_correct_data += logs["labels"]
            answer_data += logs["ans"]
            timestamp_data += [t for t in range(time, time + seq_len)]
            time += seq_len
            concept_data += [tuple(concept) for concept in logs["subject_ids"]]
            correct_answer_data += logs["correct_ans"]
            user_id += 1

    data_df = pd.DataFrame({"UserID": user_id_data,
                            "OriginalQuestionID": question_data,
                            "OriginalConceptID": concept_data,
                            "IsCorrect": is_correct_data,
                            "Answer": answer_data,
                            "CorrectAnswer": correct_answer_data,
                            "Timestamp": timestamp_data})
    data_df = data_df.drop_duplicates().dropna()
    data_df = data_df.astype({"UserID": int,
                              "OriginalQuestionID": int,
                              "OriginalConceptID": object,
                              "IsCorrect": int,
                              "Answer": int,
                              "CorrectAnswer": int,
                              "Timestamp": int})
    data_df = data_df.sort_values(by=["UserID", "Timestamp"], ignore_index=True)
    data_df["CorrectAnswer"] = data_df["CorrectAnswer"] - 1
    data_df["Answer"] = data_df["Answer"] - 1

    # metadata
    user_list = data_df["UserID"].unique().tolist()
    question_list = sorted(data_df["OriginalQuestionID"].unique().tolist())
    concept_tuple_list = data_df["OriginalConceptID"].unique().tolist()
    concept_list = sorted(list(set([c for cs in concept_tuple_list for c in cs])))

    # re-assign ID
    data_df["UserID"] = data_df.apply(lambda x: convert_column(x["UserID"], user_list), axis=1)
    data_df["QuestionID"] = data_df.apply(lambda x: convert_column(x["OriginalQuestionID"], question_list), axis=1)
    data_df["ConceptID"] = (data_df.apply(lambda x: [convert_column(c, concept_list) for c in x["OriginalConceptID"]], axis=1).apply(tuple))

    num_user, num_question, num_concept = len(user_list), len(question_list), len(concept_list)
    print(f"# of User: {num_user}\n"
          f"# of Question: {num_question}\n"
          f"# of Concept: {num_concept}\n"
          f"# of Interaction: {len(data_df)}\n")

    # get data
    relation_dict = get_concept_relation(data_df, len(concept_list))
    question_data = get_question_data(data_df)
    concept_map_vis_dict = get_concept_map_vis_data(data_df, relation_dict)

    return data_df, relation_dict, question_data, concept_map_vis_dict


def get_concept_relation(data_df, num_concept):
    # build correct matrix
    C = np.zeros((num_concept, num_concept))
    for user, logs in data_df.groupby(by=["UserID"]):
        score_list = logs["IsCorrect"].tolist()
        concept_list = logs["ConceptID"].tolist()
        for idx in range(len(score_list) - 1):
            if score_list[idx] == 1 and score_list[idx + 1] == 1:
                for concept_from in concept_list[idx]:
                    for concept_to in concept_list[idx + 1]:
                        C[concept_from][concept_to] += 1
    C_sum = np.sum(C, axis=1)
    C_sum = np.where(C_sum == 0, 1, C_sum)
    C /= C_sum[:, np.newaxis]

    # build transition matrix
    T = (C - np.min(C)) / (np.max(C) - np.min(C))
    threshold = np.mean(T) * 18  # ** 3
    T = T > threshold

    # concept relation (question as concept)
    source_list, target_list = T.nonzero()
    relation_source_list, relation_target_list = [], []
    for source, target in zip(source_list.tolist(), target_list.tolist()):
        if source != target:
            relation_source_list.append(source)
            relation_target_list.append(target)
    relation_list = list(zip(relation_source_list, relation_target_list))
    relation_directed_list, relation_undirected_list = [], []
    for (src, tar) in relation_list:
        if (tar, src) in relation_list:
            if src < tar:
                relation_undirected_list.append([src, tar])
        else:
            relation_directed_list.append([src, tar])

    # # concept relation (subject as concept)
    # relation_list = list(permutations(list(range(num_concept)), 2))  # dense graph
    # relation_directed_list, relation_undirected_list = [], []
    # for (src, tar) in relation_list:
    #     if (tar, src) in relation_list:
    #         if src < tar:
    #             relation_undirected_list.append([src, tar])
    #     else:
    #         relation_directed_list.append([src, tar])

    relation_dict = {"directed": relation_directed_list,
                     "undirected": relation_undirected_list}

    return relation_dict


def get_question_data(data_df):
    # get question dict
    correct_rate_dict = data_df.groupby(by=["QuestionID"])["IsCorrect"].mean().to_dict()
    question_data_df = data_df[["QuestionID", "OriginalQuestionID", "ConceptID", "OriginalConceptID", "CorrectAnswer"]].copy()
    question_data_df["CorrectRate"] = question_data_df.apply(lambda x: correct_rate_dict[x["QuestionID"]], axis=1)
    question_data_df = question_data_df.drop_duplicates(subset=["QuestionID"]).dropna()
    question_data_df = question_data_df.sort_values(by=["QuestionID"], ignore_index=True)
    question_dict = question_data_df.to_dict(orient="index")
    option_len = int(max(data_df["CorrectAnswer"].max(), data_df["Answer"].max())) + 1

    # generate question data
    question_data = []
    for q_data in question_dict.values():
        data = {
            "question": q_data["QuestionID"],
            "original_question": q_data["OriginalQuestionID"],
            "concept": q_data["ConceptID"],
            "original_concept": q_data["OriginalConceptID"],
            "correct_rate": q_data["CorrectRate"],
            "question_text": False,
            "question_image": False,
            "concept_text": False,
            "answer": q_data["CorrectAnswer"],
            "option_len": option_len,
            "option_text": False
        }
        question_data.append(data)

    return question_data


def get_concept_map_vis_data(data_df, relation_dict):
    # get concept dict
    concept_df = data_df[["ConceptID", "OriginalConceptID"]].copy()
    concept_df = concept_df.drop_duplicates().dropna()
    concept_id_list = concept_df["ConceptID"].tolist()
    original_concept_id_list = concept_df["OriginalConceptID"].tolist()
    concept_dict = {}
    for concept_ids, original_concept_ids in zip(concept_id_list, original_concept_id_list):
        for concept_id, original_concept_id in zip(concept_ids, original_concept_ids):
            concept_dict[concept_id] = {"ConceptID": concept_id,
                                        "OriginalConceptID": original_concept_id,
                                        "ConceptGroup": 0}

    # generate concept map visualization data
    concept_list = sorted(list(concept_dict.keys()))
    nodes = [{"id": concept_dict[concept]["ConceptID"],
              "name": False,
              "original_id": concept_dict[concept]["OriginalConceptID"],
              "group": concept_dict[concept]["ConceptGroup"]}
             for concept in concept_list]
    links = []
    for (source, target) in relation_dict["directed"]:
        links.append({"source": source, "target": target})
    for (source, target) in relation_dict["undirected"]:
        links.append({"source": source, "target": target})
        links.append({"source": target, "target": source})
    concept_graph_dict = {"nodes": nodes, "links": links}

    return concept_graph_dict


"""
========================================================
# of User: 5000
# of Question: 12080
# of Concept: 189
# of Option per Question: 1 ~ 4 (avg. 3.13)
# of Interaction: 599898
--------------------------------------------------------
# of Concept per Question: 1 ~ 7 (avg. 2.27)
# of Concept Relation: 225 (223 directed, 2 undirected)
--------------------------------------------------------
# of Interaction per Student: 10 ~ 200 (avg. 119.98)
Correct Response Rate: 65.75%
Sparsity: 99.01%
========================================================
"""
