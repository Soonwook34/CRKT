import os
from itertools import permutations

import numpy as np
import pandas as pd


OPTION_LIST = ["A", "B", "C", "D", "E"]
SCORE_LIST = [False, True]
# ID-STUDENT, ID-QUESTION, STUDENT-CHOICE, CORRECT-CHOICE, CORRECT?, SCORE1, SCORE2, SCORE3, SCORE4, ES
INTERACTION_CSV = "enem_dep.csv"


def convert_column(source, target_list):
    return target_list.index(source) if source in target_list else None


def convert_question(id):
    return int(id.split('-')[0]) * 1000 + int(id.split('-')[1])
    # return f"{id.split('-')[0]}{int(id.split('-')[1]):02d}"


def preprocess(path):
    # load csv
    raw_path = os.path.join(path, "raw")
    interaction_path = os.path.join(raw_path, INTERACTION_CSV)
    interaction_df = pd.read_csv(interaction_path)

    # preprocess column
    interaction_df["UserID"] = interaction_df["ID-STUDENT"]
    interaction_df["OriginalQuestionID"] = interaction_df.apply(lambda x: convert_question(x["ID-QUESTION"]), axis=1)
    interaction_df["IsCorrect"] = interaction_df.apply(lambda x: convert_column(x["CORRECT?"], SCORE_LIST), axis=1)
    interaction_df["Answer"] = interaction_df.apply(lambda x: convert_column(x["STUDENT-CHOICE"], OPTION_LIST), axis=1)
    interaction_df["Timestamp"] = list(range(len(interaction_df)))
    interaction_df["OriginalConceptID"] = interaction_df.apply(lambda x: [x["OriginalQuestionID"]], axis=1).apply(tuple)  # question as concept
    # interaction_df["OriginalConceptID"] = interaction_df.apply(lambda x: [int(x["ID-QUESTION"].split("-")[0])], axis=1).apply(tuple)  # subject as concept
    interaction_df["CorrectAnswer"] = interaction_df.apply(lambda x: convert_column(x["CORRECT-CHOICE"], OPTION_LIST), axis=1)
    interaction_df["ConceptGroup"] = interaction_df.apply(lambda x: [int(x["ID-QUESTION"].split("-")[0])], axis=1).apply(tuple)
    interaction_df = interaction_df[["UserID", "OriginalQuestionID", "OriginalConceptID", "IsCorrect", "Answer", "Timestamp", "CorrectAnswer", "ConceptGroup"]]

    # merge dataframe
    data_df = interaction_df
    data_df = data_df.drop_duplicates().dropna()
    data_df = data_df.astype({"UserID": int,
                              "OriginalQuestionID": int,
                              "OriginalConceptID": object,
                              "IsCorrect": int,
                              "Answer": int,
                              "CorrectAnswer": int,
                              "Timestamp": int,
                              "ConceptGroup": object})
    data_df = data_df.sort_values(by=["UserID", "Timestamp"], ignore_index=True)

    # metadata
    user_list = data_df["UserID"].unique().tolist()
    question_list = sorted(data_df["OriginalQuestionID"].unique().tolist())
    concept_tuple_list = data_df["OriginalConceptID"].unique().tolist()
    concept_list = sorted(list(set([c for cs in concept_tuple_list for c in cs])))
    concept_group_tuple_list = data_df["ConceptGroup"].unique().tolist()
    concept_group_list = sorted(list(set([c for cs in concept_group_tuple_list for c in cs])))

    # re-assign ID
    data_df["UserID"] = data_df.apply(lambda x: convert_column(x["UserID"], user_list), axis=1)
    data_df["QuestionID"] = data_df.apply(lambda x: convert_column(x["OriginalQuestionID"], question_list), axis=1)
    data_df["ConceptID"] = data_df.apply(lambda x: [convert_column(c, concept_list) for c in x["OriginalConceptID"]], axis=1).apply(tuple)
    data_df["ConceptGroup"] = data_df.apply(lambda x: [convert_column(g, concept_group_list) + 1 for g in x["ConceptGroup"]], axis=1).apply(tuple)

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
    threshold = np.mean(T)  # ** 3
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
    option_len = len(OPTION_LIST)

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
    concept_df = data_df[["ConceptID", "OriginalConceptID", "ConceptGroup"]].copy()
    concept_df = concept_df.drop_duplicates().dropna()
    concept_id_list = concept_df["ConceptID"].tolist()
    original_concept_id_list = concept_df["OriginalConceptID"].tolist()
    concept_group_list = concept_df["ConceptGroup"].tolist()
    concept_dict = {}
    for concept_ids, original_concept_ids, concept_groups in zip(concept_id_list, original_concept_id_list, concept_group_list):
        for concept_id, original_concept_id, concept_group in zip(concept_ids, original_concept_ids, concept_groups):
            concept_dict[concept_id] = {"ConceptID": concept_id,
                                        "OriginalConceptID": original_concept_id,
                                        "ConceptGroup": concept_group}

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
# of User: 10000
# of Question: 185
# of Concept: 185
# of Option per Question: 5 ~ 5 (avg. 5.00)
# of Interaction: 1793106
--------------------------------------------------------
# of Concept per Question: 1 ~ 1 (avg. 1.00)
# of Concept Relation: 211 (211 directed, 0 undirected)
--------------------------------------------------------
# of Interaction per Student: 80 ~ 180 (avg. 179.31)
Correct Response Rate: 33.73%
Sparsity: 3.08%
========================================================
"""
