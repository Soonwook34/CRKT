import os
import argparse
import json
import random
import itertools


class PreprocessArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--dataset", type=str, default="DBE_KT22")
        self.add_argument("--input_dir", type=str, default="./data")
        self.add_argument("--min_seq_len", type=int, default=10)
        self.add_argument("--max_seq_len", type=int, default=200)
        self.add_argument("--k_fold", type=int, default=5)
        self.add_argument("--test_ratio", type=float, default=0.2)
        self.add_argument("--seed", type=int, default=0)


def preprocess_raw(path):
    match dataset_name:
        case "Poly":
            from data.Poly.preprocess_poly import preprocess
        case "ENEM":
            from data.ENEM.preprocess_enem import preprocess
        case "NIPS34":
            from data.NIPS34.preprocess_nips34 import preprocess
        case "DBE_KT22":
            from data.DBE_KT22.preprocess_dbe_kt22 import preprocess
        case "EdNet":
            from data.EdNet.preprocess_ednet import preprocess
        case _:
            print(f"Preprocessing for dataset {dataset_name} is not implemented.")
            exit()

    return preprocess(path)


def get_status(df, relation):
    users = df["UserID"].unique().tolist()
    question_info_df = df.drop_duplicates(subset=["QuestionID"])
    questions = question_info_df["QuestionID"].tolist()
    concept_tuples = question_info_df["ConceptID"].tolist()
    concepts = list(set([c for cs in concept_tuples for c in cs]))
    concept_lens = [len(cs) for cs in concept_tuples]
    option_lens = df.drop_duplicates(subset=["QuestionID", "Answer"]).groupby(by=["QuestionID"])["Answer"].count().tolist()
    interaction_lens = df.groupby(by=["UserID"])["Timestamp"].count().tolist()
    scores = df["IsCorrect"].tolist()
    total_interaction = len(users) * len(questions)
    num_directed = len(relation["directed"])
    num_undirected = len(relation["undirected"])

    print(f"========================================================\n"
          f"# of User: {len(users)}\n"
          f"# of Question: {len(questions)}\n"
          f"# of Concept: {len(concepts)}\n"
          f"# of Option per Question: {min(option_lens)} ~ {max(option_lens)} "
          f"(avg. {sum(option_lens) / len(option_lens):.2f})\n"
          f"# of Interaction: {len(df)}\n"
          f"--------------------------------------------------------\n"
          f"# of Concept per Question: {min(concept_lens)} ~ {max(concept_lens)} "
          f"(avg. {sum(concept_lens) / len(concept_lens):.2f})\n"
          f"# of Concept Relation: {num_directed + num_undirected} ({num_directed} directed, {num_undirected} undirected)\n"
          f"--------------------------------------------------------\n"
          f"# of Interaction per Student: {min(interaction_lens)} ~ {max(interaction_lens)} "
          f"(avg. {sum(interaction_lens) / len(interaction_lens):.2f})\n"
          f"Correct Response Rate: {sum(scores) / len(scores) * 100:.2f}%\n"
          f"Sparsity: {(total_interaction - len(df)) / total_interaction * 100:.2f}%\n"
          f"========================================================")

    return


def make_dataset(df, min_seq_len, max_seq_len):
    dataset = []
    for user, logs in df.groupby(by=["UserID"]):
        user_id = user[0]
        question_list = logs["QuestionID"].tolist()
        score_list = logs["IsCorrect"].tolist()
        option_list = logs["Answer"].tolist()
        concept_list = logs["ConceptID"].tolist()
        answer_list = logs["CorrectAnswer"].tolist()
        # split data if data length exceed max_seq_len
        for idx in range(0, len(question_list), max_seq_len):
            data = {"user": user_id,
                    "question": question_list[idx:idx+max_seq_len],
                    "score": score_list[idx:idx+max_seq_len],
                    "option": option_list[idx:idx+max_seq_len],
                    "concept": concept_list[idx:idx+max_seq_len],
                    "answer": answer_list[idx:idx+max_seq_len]}
            dataset.append(data)
    # remove data if data length is less than max_seq_len
    dataset = [data for data in dataset if len(data["question"]) >= min_seq_len]
    print(f"Preprocessed dataset: {len(dataset)} sequences\n")

    return dataset


def split_train_test(dataset, test_ratio):
    random.shuffle(dataset)
    num_dataset = len(dataset)
    num_test = int(num_dataset * test_ratio)
    num_train = num_dataset - num_test
    train_set, test_set = dataset[:num_train], dataset[num_train:]
    print(f"Train set: {num_train}\nTest set: {num_test}\n")

    return train_set, test_set


def split_k_fold(dataset, k):
    num_dataset = len(dataset)
    fold_sets, fold_len = [], []
    fold_size = num_dataset / k
    fold_index = 0.0
    for _ in range(k):
        start = int(fold_index)
        fold_index += fold_size
        end = int(fold_index)
        fold_data = dataset[start:end]
        fold_sets.append(fold_data)
        fold_len.append(len(fold_data))
    print(f"{k}-fold: {fold_len}\n")

    return fold_sets


def save_json(save_path, save_file, data, indent=None):
    with open(os.path.join(save_path, save_file), "w", encoding="utf8") as output_file:
        json.dump(data, output_file, indent=indent, ensure_ascii=False)

    return


if __name__ == "__main__":
    args = PreprocessArgParser().parse_args()
    print(args)

    dataset_name = args.dataset
    path = os.path.join(args.input_dir, args.dataset)
    data_path = os.path.join(path, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    print("Generate raw dataset...")
    data_df, relation_dict, question_data, concept_map_vis_dict = preprocess_raw(path)
    get_status(data_df, relation_dict)

    save_file = os.path.join(data_path, "data.csv")
    data_df.to_csv(save_file, index=False)
    save_json(data_path, "relation.json", relation_dict, indent=4)
    save_json(data_path, "question_data.json", question_data, indent=4)
    print(f"Raw dataset saved. ({save_file})\n")
    save_json(data_path, "concept_map_vis.json", concept_map_vis_dict, indent=4)
    print(f"Concept map visualization Data saved. ({save_file})\n")

    print("Preprocess raw dataset...")
    total_dataset = make_dataset(data_df, args.min_seq_len, args.max_seq_len)
    save_json(data_path, "preprocessed.json", total_dataset)

    print("Split train-test...")
    random.seed(args.seed)
    train_dataset, test_dataset = split_train_test(total_dataset, args.test_ratio)
    save_json(data_path, "train.json", train_dataset)
    save_json(data_path, "test.json", test_dataset)

    print(f"Split train dataset into {args.k_fold}-fold...")
    random.seed(args.seed)
    fold_datasets = split_k_fold(train_dataset, args.k_fold)
    for fold, fold_dataset in enumerate(fold_datasets):
        save_json(data_path, f"fold{fold}.json", fold_dataset)


    data_config = {
        "num_user": int(data_df["UserID"].max() + 1),
        "num_question": int(data_df["QuestionID"].max() + 1),
        "num_concept": max(itertools.chain(*data_df["ConceptID"].tolist())) + 1,
        "num_option": int(data_df["Answer"].max() + 1),
        "num_interaction": len(data_df),
        "range_sequence": (args.min_seq_len, args.max_seq_len),
        "max_concept_len": max([len(concepts) for concepts in data_df["ConceptID"].tolist()]),
        "num_train": len(train_dataset),
        "num_test": len(test_dataset),
        "num_fold": [len(dataset) for dataset in fold_datasets]
    }
    config_file = os.path.join(args.input_dir, "data_config.json")
    if os.path.exists(config_file):
        with open(config_file, encoding="utf8") as config_data:
            config_json = json.load(config_data)
    else:
        config_json = {}
    config_json[args.dataset] = data_config
    save_json(args.input_dir, "data_config.json", config_json, indent=4)

    print("Done!")

