import os
from ast import literal_eval

import pandas as pd


OPTION_LIST = [1, 2, 3, 4]
# QuestionId, SubjectId
QUESTION_CSV = "data/metadata/question_metadata_task_3_4.csv"
# QuestionId, UserId, AnswerId, IsCorrect, CorrectAnswer, AnswerValue
INTERACTION_CSV = "data/train_data/train_task_3_4.csv"
# SubjectId, Name, ParentId, Level
CONCEPT_CSV = "data/metadata/subject_metadata.csv"


def convert_column(source, target_list):
    return target_list.index(source) if source in target_list else None


def get_image_path(source, image_path, image_list):
    image_file = f"{source}.jpg"
    return os.path.join(image_path, image_file) if image_file in image_list else ""


def search_dict(id, target_dict, id_key, target_key):
    index = convert_column(id, target_dict[id_key])
    return target_dict[target_key][index] if index != None else None


def preprocess(path):
    # load csv
    raw_path = os.path.join(path, "raw")
    image_path = os.path.join(raw_path, "data", "images")
    question_path = os.path.join(raw_path, QUESTION_CSV)
    interaction_path = os.path.join(raw_path, INTERACTION_CSV)
    concept_path = os.path.join(raw_path, CONCEPT_CSV)
    question_df = pd.read_csv(question_path)
    interaction_df = pd.read_csv(interaction_path)
    concept_df = pd.read_csv(concept_path)
    relation_df = concept_df.copy()

    # preprocess column
    question_df["OriginalQuestionID"] = question_df["QuestionId"]
    question_df["OriginalConceptID"] = question_df.apply(lambda x: literal_eval(x["SubjectId"]), axis=1).apply(tuple)
    question_df = question_df[["OriginalQuestionID", "OriginalConceptID"]]

    concept_df["OriginalConceptID"] = concept_df["SubjectId"]
    concept_df["ConceptText"] = concept_df["Name"]
    concept_df["Group"] = concept_df["Level"]
    concept_df = concept_df[["OriginalConceptID", "ConceptText", "Group"]]
    concept_df = concept_df.sort_values(by=["OriginalConceptID"], ignore_index=True)
    concept_dict = concept_df.to_dict(orient="list")

    image_list = os.listdir(image_path)
    question_df["QuestionImage"] = question_df.apply(lambda x: get_image_path(x["OriginalQuestionID"],
                                                                              image_path,
                                                                              image_list), axis=1)
    question_df["ConceptText"] = question_df.apply(lambda x: [search_dict(c, concept_dict, "OriginalConceptID", "ConceptText")
                                                              for c in x["OriginalConceptID"]], axis=1).apply(tuple)
    question_df["ConceptGroup"] = question_df.apply(lambda x: [search_dict(c, concept_dict, "OriginalConceptID", "Group") + 1
                                                              for c in x["OriginalConceptID"]], axis=1).apply(tuple)
    question_df = question_df.drop_duplicates(subset=["OriginalQuestionID"]).dropna()
    question_df = question_df[["OriginalQuestionID", "OriginalConceptID", "QuestionImage", "ConceptText", "ConceptGroup"]]

    interaction_df["UserID"] = interaction_df["UserId"]
    interaction_df["OriginalQuestionID"] = interaction_df["QuestionId"]
    interaction_df["Answer"] = interaction_df.apply(lambda x: convert_column(x["AnswerValue"], OPTION_LIST), axis=1)
    interaction_df["CorrectAnswer"] = interaction_df.apply(lambda x: convert_column(x["CorrectAnswer"], OPTION_LIST), axis=1)
    interaction_df["Timestamp"] = list(range(len(interaction_df)))
    interaction_df = interaction_df[["UserID", "OriginalQuestionID", "IsCorrect", "Answer", "Timestamp", "CorrectAnswer"]]

    relation_df["Source"] = relation_df["ParentId"]
    relation_df["Target"] = relation_df["SubjectId"]
    relation_df = relation_df[["Source", "Target", "Level"]]

    # merge dataframe
    data_df = pd.merge(left=interaction_df, right=question_df, how="inner", on=["OriginalQuestionID"])
    data_df = data_df.drop_duplicates().dropna()
    data_df = data_df.astype({"UserID": int,
                              "OriginalQuestionID": int,
                              "OriginalConceptID": object,
                              "IsCorrect": int,
                              "Answer": int,
                              "CorrectAnswer": int,
                              "Timestamp": int,
                              "QuestionImage": str,
                              "ConceptText": object,
                              "ConceptGroup": object})
    data_df = data_df.sort_values(by=["UserID", "Timestamp"], ignore_index=True)

    # metadata
    user_list = data_df["UserID"].unique().tolist()
    question_list = sorted(data_df["OriginalQuestionID"].unique().tolist())
    concept_tuple_list = data_df["OriginalConceptID"].unique().tolist()
    concept_list = sorted(list(set([c for cs in concept_tuple_list for c in cs])))

    # re-assign ID
    data_df["UserID"] = data_df.apply(lambda x: convert_column(x["UserID"], user_list), axis=1)
    data_df["QuestionID"] = data_df.apply(lambda x: convert_column(x["OriginalQuestionID"], question_list), axis=1)
    data_df["ConceptID"] = data_df.apply(lambda x: [convert_column(c, concept_list) for c in x["OriginalConceptID"]], axis=1).apply(tuple)
    relation_df["Source"] = relation_df.apply(lambda x: convert_column(x["Source"], concept_list), axis=1)
    relation_df["Target"] = relation_df.apply(lambda x: convert_column(x["Target"], concept_list), axis=1)

    # get data
    relation_dict = get_concept_relation(relation_df)
    question_data = get_question_data(data_df)
    concept_map_vis_dict = get_concept_map_vis_data(data_df, relation_dict)

    return data_df, relation_dict, question_data, concept_map_vis_dict


def get_concept_relation(relation_df):
    relation_df = relation_df.drop_duplicates().dropna()
    relation_df = relation_df.astype({"Source": int, "Target": int})
    relation_df = relation_df.sort_values(by=["Source", "Target"], ignore_index=True)

    # concept relation
    relation_source_list = relation_df["Source"].tolist()
    relation_target_list = relation_df["Target"].tolist()
    relation_list = list(zip(relation_source_list, relation_target_list))
    relation_directed_list, relation_undirected_list = [], []
    for (src, tar) in relation_list:
        if (tar, src) in relation_list:
            if src < tar:
                relation_undirected_list.append([src, tar])
        else:
            relation_directed_list.append([src, tar])

    relation_dict = {"directed": relation_directed_list,
                     "undirected": relation_undirected_list}

    return relation_dict


def get_question_data(data_df):
    # get question dict
    correct_rate_dict = data_df.groupby(by=["QuestionID"])["IsCorrect"].mean().to_dict()
    question_data_df = data_df[["QuestionID", "OriginalQuestionID", "ConceptID", "OriginalConceptID",
                                "CorrectAnswer", "QuestionImage", "ConceptText"]].copy()
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
            "question_image": q_data["QuestionImage"],
            "concept_text": q_data["ConceptText"],
            "answer": q_data["CorrectAnswer"],
            "option_len": option_len
        }
        question_data.append(data)

    return question_data


def get_concept_map_vis_data(data_df, relation_dict):
    # get concept dict
    concept_df = data_df[["ConceptID", "OriginalConceptID", "ConceptText", "ConceptGroup"]].copy()
    concept_df = concept_df.drop_duplicates().dropna()
    concept_id_list = concept_df["ConceptID"].tolist()
    original_concept_id_list = concept_df["OriginalConceptID"].tolist()
    concept_text_list = concept_df["ConceptText"].tolist()
    concept_group_list = concept_df["ConceptGroup"].tolist()
    concept_dict = {}
    for concept_ids, original_concept_ids, concept_texts, concept_groups in zip(concept_id_list, original_concept_id_list,
                                                                                concept_text_list, concept_group_list):
        for concept_id, original_concept_id, concept_text, concept_group in zip(concept_ids, original_concept_ids,
                                                                                concept_texts, concept_groups):
            concept_dict[concept_id] = {"ConceptID": concept_id,
                                        "OriginalConceptID": original_concept_id,
                                        "ConceptText": concept_text,
                                        "ConceptGroup": concept_group}

    # generate concept map visualization data
    concept_list = sorted(list(concept_dict.keys()))
    nodes = [{"id": concept_dict[concept]["ConceptID"],
              "name": concept_dict[concept]["ConceptText"],
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


def get_concept_graph_data(path, concept_list, relation_dict):
    # load csv
    raw_path = os.path.join(path, "raw")
    concept_path = os.path.join(raw_path, CONCEPT_CSV)
    concept_df = pd.read_csv(concept_path)

    # SubjectId, Name, ParentId, Level
    # preprocess column
    concept_df = concept_df.drop_duplicates(subset=["SubjectId"]).fillna("")
    concept_df["OriginalConceptID"] = concept_df["SubjectId"]
    concept_df["ConceptID"] = concept_df.apply(lambda x: convert_column(x["OriginalConceptID"], concept_list), axis=1)
    concept_df = concept_df.drop_duplicates().dropna(subset=["ConceptID"])
    concept_df["ConceptName"] = concept_df.apply(lambda x: x["Name"].strip(), axis=1)
    concept_df["Group"] = concept_df["Level"]

    concept_df = concept_df[["OriginalConceptID", "ConceptID", "ConceptName", "Group"]]
    concept_df = concept_df.sort_values(by=["ConceptID"], ignore_index=True)

    concept_dict = concept_df.to_dict(orient="index")

    nodes = [{"id": concept_dict[concept]["ConceptName"],
              "original_id": concept_dict[concept]["OriginalConceptID"],
              "group": concept_dict[concept]["Group"]}
             for concept in concept_dict]
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
# of User: 4918
# of Question: 948
# of Concept: 86
# of Option per Question: 1 ~ 4 (avg. 3.95)
# of Interaction: 1382727
--------------------------------------------------------
# of Concept per Question: 4 ~ 6 (avg. 4.02)
# of Concept Relation: 85 (85 directed, 0 undirected)
--------------------------------------------------------
# of Interaction per Student: 50 ~ 827 (avg. 281.16)
Correct Response Rate: 53.73%
Sparsity: 70.34%
========================================================
"""
