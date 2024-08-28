import os

import pandas as pd


# id, question_rich_text, question_title, explanation, hint_text, question_text, difficulty
QUESTION_CSV = "Questions.csv"
# id, question_id, knowledgecomponent_id
QUESTION_CONCEPT_CSV = "Question_KC_Relationships.csv"
# id, name, description
CONCEPT_CSV = "KCs.csv"
# id, choice_text, is_correct, question_id
OPTION_CSV = "Question_Choices.csv"
# id, selection_change, start_time, end_time, difficulty_feedback, trust_feedback, answer_state,
# answer_text, student_id, hint_used, question_id, answer_choice_id, is_hidden
INTERACTION_CSV = "Transaction.csv"
# id, from_knowledgecomponent_id,to_knowledgecomponent_id
RELATION_CSV = "KC_Relationships.csv"


def convert_column(source, target_list):
    return target_list.index(source) if source in target_list else None


def combine_question(questions):
    return "\n".join([text.strip() for text in questions if len(text) > 0])


def preprocess(path):
    # load csv
    raw_path = os.path.join(path, "raw")
    question_path = os.path.join(raw_path, QUESTION_CSV)
    question_concept_path = os.path.join(raw_path, QUESTION_CONCEPT_CSV)
    concept_path = os.path.join(raw_path, CONCEPT_CSV)
    option_path = os.path.join(raw_path, OPTION_CSV)
    interaction_path = os.path.join(raw_path, INTERACTION_CSV)
    relation_path = os.path.join(raw_path, RELATION_CSV)
    question_df = pd.read_csv(question_path)
    question_concept_df = pd.read_csv(question_concept_path)
    concept_df = pd.read_csv(concept_path)
    option_df = pd.read_csv(option_path)
    interaction_df = pd.read_csv(interaction_path, parse_dates=["start_time", "end_time"])
    relation_df = pd.read_csv(relation_path)

    # preprocess column
    concept_df = concept_df.drop_duplicates(subset=["id"]).fillna("")
    concept_df["OriginalConceptID"] = concept_df["id"]
    concept_df["ConceptText"] = concept_df.apply(lambda x: x["name"].strip(), axis=1)
    concept_df = concept_df[["OriginalConceptID", "ConceptText"]]

    question_concept_df["OriginalQuestionID"] = question_concept_df["question_id"]
    question_concept_df["OriginalConceptID"] = question_concept_df["knowledgecomponent_id"]
    question_concept_df = pd.merge(left=question_concept_df, right=concept_df, how="inner", on=["OriginalConceptID"])
    question_concept_id_df = question_concept_df.groupby(by=["OriginalQuestionID"])["OriginalConceptID"].apply(tuple)
    question_concept_text_df = question_concept_df.groupby(by=["OriginalQuestionID"])["ConceptText"].apply(tuple)
    question_concept_df = pd.merge(left=question_concept_id_df, right=question_concept_text_df, how="inner", on=["OriginalQuestionID"])

    question_df = question_df.fillna("")
    question_df["OriginalQuestionID"] = question_df["id"]
    question_df["hint_text"] = question_df.apply(lambda x:
                                                 f"hint: {x['hint_text']}" if len(x["hint_text"]) > 0 else "", axis=1)
    question_df["QuestionText"] = question_df.apply(lambda x: combine_question([x["question_rich_text"],
                                                                                x["explanation"],
                                                                                x["hint_text"]]), axis=1)
    question_df = pd.merge(left=question_df, right=question_concept_df, how="inner", on=["OriginalQuestionID"])
    question_df["QuestionID"] = question_df["OriginalQuestionID"]
    question_df["ConceptID"] = question_df["OriginalConceptID"]
    question_df = question_df[["OriginalQuestionID", "OriginalConceptID", "ConceptText", "QuestionText"]]

    option_df = option_df.sort_values(by=["question_id", "id"], ignore_index=True)
    option_df["OriginalQuestionID"] = option_df["question_id"]
    option_df["OriginalOptionID"] = option_df["id"]
    option_df["CorrectAnswer"] = option_df["id"]
    option_df["OptionText"] = option_df["choice_text"]
    option_id_df = option_df.groupby(by=["OriginalQuestionID"])["OriginalOptionID"].apply(tuple)
    answer_df = option_df[option_df["is_correct"] == True][["OriginalQuestionID", "CorrectAnswer"]]
    option_text_df = option_df.groupby(by=["OriginalQuestionID"])["OptionText"].apply(tuple)
    option_df = pd.merge(left=option_id_df, right=answer_df, how="inner", on=["OriginalQuestionID"])
    option_df = pd.merge(left=option_df, right=option_text_df, how="inner", on=["OriginalQuestionID"])

    question_df = pd.merge(left=question_df, right=option_df, how="inner", on=["OriginalQuestionID"])
    question_dict_df = question_df.set_index(keys=["OriginalQuestionID"], inplace=False, drop=False)
    question_dict = question_dict_df["OriginalOptionID"].to_dict()
    question_df["CorrectAnswer"] = question_df.apply(lambda x: convert_column(x["CorrectAnswer"],
                                                                              question_dict[x["OriginalQuestionID"]]), axis=1)
    question_df = question_df.drop_duplicates(subset=["OriginalQuestionID"]).dropna()
    question_df = question_df[["OriginalQuestionID", "OriginalConceptID", "CorrectAnswer",
                               "QuestionText", "ConceptText", "OptionText", "OriginalOptionID"]]

    interaction_df = interaction_df.sort_values(by=["start_time"])
    interaction_df["UserID"] = interaction_df["student_id"]
    interaction_df["OriginalQuestionID"] = interaction_df["question_id"]
    interaction_df["IsCorrect"] = interaction_df.apply(lambda x: convert_column(x["answer_state"], [False, True]), axis=1)
    interaction_df["Answer"] = interaction_df.apply(lambda x: convert_column(x["answer_choice_id"],
                                                                             question_dict[x["OriginalQuestionID"]]), axis=1)
    interaction_df["Timestamp"] = list(range(len(interaction_df)))
    interaction_df = interaction_df[["UserID", "OriginalQuestionID", "IsCorrect", "Answer", "Timestamp"]]

    relation_df["Source"] = relation_df["from_knowledgecomponent_id"]
    relation_df["Target"] = relation_df["to_knowledgecomponent_id"]
    relation_df = relation_df[["Source", "Target"]]

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
                              "QuestionText": str,
                              "ConceptText": object,
                              "OptionText": object,
                              "OriginalOptionID": object})
    data_df = data_df.sort_values(by=["UserID", "Timestamp"], ignore_index=True)

    # metadata
    user_list = data_df["UserID"].unique().tolist()
    question_list = sorted(data_df["OriginalQuestionID"].unique().tolist())
    concept_tuple_list = data_df["OriginalConceptID"].unique().tolist()
    concept_list = list(set([c for cs in concept_tuple_list for c in cs]))

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

    concept_graph_dict = get_concept_graph_data(path, concept_list, relation_dict)

    return data_df, relation_dict, concept_graph_dict


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
                                "CorrectAnswer", "QuestionText", "ConceptText", "OptionText", "OriginalOptionID"]].copy()
    question_data_df["CorrectRate"] = question_data_df.apply(lambda x: correct_rate_dict[x["QuestionID"]], axis=1)
    question_data_df = question_data_df.drop_duplicates(subset=["QuestionID"]).dropna()
    question_data_df = question_data_df.sort_values(by=["QuestionID"], ignore_index=True)
    question_dict = question_data_df.to_dict(orient="index")

    # generate question data
    question_data = []
    for q_data in question_dict.values():
        data = {
            "question": q_data["QuestionID"],
            "original_question": q_data["OriginalQuestionID"],
            "concept": q_data["ConceptID"],
            "original_concept": q_data["OriginalConceptID"],
            "correct_rate": q_data["CorrectRate"],
            "question_text": q_data["QuestionText"],
            "question_image": False,
            "concept_text": q_data["ConceptText"],
            "answer": q_data["CorrectAnswer"],
            "option_len": len(q_data["OptionText"]),
            "original_option": q_data["OriginalOptionID"],
            "option_text": q_data["OptionText"]
        }
        question_data.append(data)

    return question_data


def get_concept_map_vis_data(data_df, relation_dict):
    # get concept dict
    concept_df = data_df[["ConceptID", "OriginalConceptID", "ConceptText"]].copy()
    concept_df = concept_df.drop_duplicates().dropna()
    concept_id_list = concept_df["ConceptID"].tolist()
    original_concept_id_list = concept_df["OriginalConceptID"].tolist()
    concept_text_list = concept_df["ConceptText"].tolist()
    concept_dict = {}
    for concept_ids, original_concept_ids, concept_texts in zip(concept_id_list, original_concept_id_list, concept_text_list):
        for concept_id, original_concept_id, concept_text in zip(concept_ids, original_concept_ids, concept_texts):
            concept_dict[concept_id] = {"ConceptID": concept_id,
                                        "OriginalConceptID": original_concept_id,
                                        "ConceptText": concept_text}

    # generate concept map visualization data
    concept_list = sorted(list(concept_dict.keys()))
    nodes = [{"id": concept_dict[concept]["ConceptID"],
              "name": concept_dict[concept]["ConceptText"],
              "original_id": concept_dict[concept]["OriginalConceptID"],
              "group": 1}
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
# of User: 1264
# of Question: 212
# of Concept: 93
# of Option per Question: 2 ~ 5 (avg. 3.54)
# of Interaction: 161952
--------------------------------------------------------
# of Concept per Question: 1 ~ 4 (avg. 1.90)
# of Concept Relation: 86 (0 directed, 86 undirected)
--------------------------------------------------------
# of Interaction per Student: 1 ~ 1171 (avg. 128.13)
Correct Response Rate: 76.45%
Sparsity: 39.56%
========================================================
"""
