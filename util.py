import os
from time import time, localtime, strftime
import argparse
import json
import csv
import random

import numpy as np
import torch
from torch.backends import cudnn
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, remove_self_loops
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


class MainArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        # config
        self.add_argument("--dataset", type=str, default="DBE_KT22")
        self.add_argument("--k_fold", type=int, default=5)
        self.add_argument("--input_dir", type=str, default="./data")
        self.add_argument("--output_dir", type=str, default=".")
        self.add_argument("--seed", type=int, default=42)
        self.add_argument("--exp_name", type=str, default="no-name")
        self.add_argument("--gpu", type=int, default=0)
        # train
        self.add_argument("--model", type=str, default="CRKT")
        self.add_argument("--batch", type=int, default=128)
        self.add_argument("--epoch", type=int, default=200)
        self.add_argument("--lr", type=float, default=1e-3)
        # model - Common
        self.add_argument("--dim_c", type=int, default=32)
        self.add_argument("--dim_q", type=int, default=32)
        self.add_argument("--dropout", type=float, default=0.1)
        self.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)
        # model - CRKT
        self.add_argument("--lamb", type=float, default=0.2)    # lambda
        self.add_argument("--layer_g", type=int, default=2)     # L
        self.add_argument("--dim_g", type=int, default=32)      # d_g
        self.add_argument("--top_k", type=int, default=10)      # k
        self.add_argument("--alpha", type=float, default=0.1)   # coefficient for top-k loss
        self.add_argument("--beta", type=float, default=0.1)    # coefficient CL loss
        # model - AKT, CL4KT, DTransformer
        self.add_argument("--num_heads", type=int, default=8)
        self.add_argument("--d_ff", type=int, default=1024)
        self.add_argument("--n_blocks", type=int, default=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def write_log(file, str, console=True):
    with open(file, "a", encoding="utf8") as f:
        f.write(f"{str}\n")
    if console:
        print(str)
    return


def set_config(args):
    model_exp_str = f"{args.model}_{args.exp_name}"
    args.dataset_path = os.path.join(args.input_dir, args.dataset)
    args.log_path = os.path.join(args.output_dir, "logs", args.dataset, model_exp_str)
    args.log_file = os.path.join(args.log_path, f"{model_exp_str}.txt")
    args.result_file = os.path.join(args.log_path, f"{model_exp_str}_result.txt")
    args.result_csv_file = os.path.join(args.output_dir, "logs", args.dataset, "results.csv")
    args.save_path = os.path.join(args.log_path, "save")
    with open(os.path.join(args.input_dir, "data_config.json")) as config_data:
        config = json.load(config_data)[args.dataset]
    args.user = config["num_user"]
    args.question = config["num_question"]
    args.concept = config["num_concept"]
    args.option = config["num_option"]
    args.max_concept_len = config["max_concept_len"]

    return


def load_model(model_name, args):
    match model_name:
        case "CRKT":
            concept_map_adj = get_concept_graph(args.dataset_path, args.concept, self_loops=False)
            with open(os.path.join(args.dataset_path, "data", "question_data.json"), encoding="utf8") as q_data:
                question_data = json.load(q_data)
            option_list = [question["option_len"] for question in question_data]
            from models.CRKT import CRKT
            model = CRKT(num_c=args.concept,
                         num_q=args.question,
                         num_o=args.option,
                         dim_c=args.dim_c,
                         dim_q=args.dim_q,
                         dim_g=args.dim_g,
                         num_heads=args.num_heads,
                         layer_g=args.layer_g,
                         top_k=args.top_k,
                         lamb=args.lamb,
                         map=concept_map_adj,
                         option_list=option_list,
                         dropout=args.dropout,
                         bias=args.bias)
        case "DKT":
            from models.DKT import DKT
            model = DKT(num_c=args.concept,
                        emb_size=args.dim_c,
                        dropout=args.dropout)
        case "GKT":
            from models.GKT import GKT
            model = GKT(concept_num=args.concept,
                        hidden_dim=args.dim_c,
                        embedding_dim=args.dim_c,
                        graph_type="ori",  # args.type_g,
                        dropout=args.dropout,
                        bias=args.bias,
                        binary=True,
                        has_cuda=True,
                        args=args)
        case "SAKT":
            from models.SAKT import SAKT
            model = SAKT(num_c=args.question,
                         seq_len=200,
                         emb_size=args.dim_q,
                         num_attn_heads=args.num_heads,
                         dropout=args.dropout,
                         num_en=1)
        case "AKT":
            from models.AKT import AKT
            model = AKT(n_question=args.concept,
                        n_pid=args.question,
                        d_model=args.dim_c,
                        n_blocks=args.n_blocks,
                        dropout=args.dropout,
                        d_ff=args.d_ff,
                        num_attn_heads=args.num_heads)
        case "CL4KT":
            from models.CL4KT import CL4KT
            model = CL4KT(num_skills=args.concept,
                          num_questions=args.question,
                          seq_len=200,
                          args=args)
        case "DTransformer":
            from models.DTransformer import DTransformer
            model = DTransformer(n_questions=args.concept,
                                 n_pid=args.question,
                                 d_model=args.dim_q,
                                 n_heads=8,
                                 n_know=16,  # (1, 2, 4, 8, 16, 32)
                                 n_layers=args.n_blocks,
                                 dropout=args.dropout,
                                 lambda_cl=0.001,  # (0.0001, 0.001, 0.01, 0.1)
                                 proj=False,
                                 hard_neg=True,
                                 window=1,
                                 shortcut=False)
        case "DP_DKT":
            from models.DP_DKT import DP_DKT
            model = DP_DKT(args)
        case "AKT_plus":
            from models.prev.AKT_plus import AKT_plus
            model = AKT_plus(args)

    return model


def get_concept_graph(path, num_concept, type_g="ori", self_loops=True):
    # load concept map data
    match type_g:
        case "ori":
            relation_file = "relation.json"
        case _:
            relation_file = f"relation_{type_g}.json"
    with open(os.path.join(path, "data", relation_file), encoding="utf8") as dependency_data:
        dependency_dict = json.load(dependency_data)

    # build edge data
    direct_edges = dependency_dict["directed"]
    undirect_edges = dependency_dict["undirected"]
    undirect_edges += [[tar, src] for (src, tar) in undirect_edges]
    edges = torch.tensor(direct_edges + undirect_edges).t().contiguous()
    if self_loops:
        edges = add_remaining_self_loops(edges)[0]
    else:
        edges = remove_self_loops(edges)[0]

    # return concept map adj
    return to_dense_adj(edge_index=edges, max_num_nodes=num_concept).squeeze(0)


def timestamp():
    return strftime("%Y-%m-%d %H:%M:%S", localtime(time()))


def calc_metric(y_true, y_hat, y_pred, y_majority, y_minority, pred_loss, data_count):
    result_dict = {}

    # macro-{acc, auc)
    confusion_mat = confusion_matrix(y_true, y_pred)
    result_dict["Macro-ACC"] = np.mean(confusion_mat.diagonal() / confusion_mat.sum(axis=1))
    result_dict["Macro-AUC"] = roc_auc_score(y_true, y_hat, average="macro", multi_class="raise")
    # micro-{acc, auc)
    result_dict["Micro-ACC"] = accuracy_score(y_true, y_pred)
    result_dict["Micro-AUC"] = roc_auc_score(y_true, y_hat, average="micro", multi_class="raise")
    # overfit metric (not named yet)
    y_true_tensor = torch.tensor(y_true)
    y_hat_tensor = torch.tensor(y_hat)
    y_pred_tensor = torch.tensor(y_pred)
    y_majority_tensor = torch.tensor(y_majority)
    y_minority_tensor = torch.tensor(y_minority)
    majority_mask = y_true_tensor.int() == y_majority_tensor
    minority_mask = y_true_tensor.int() == y_minority_tensor
    y_true_majority = y_true_tensor[majority_mask].cpu()
    y_hat_majority = y_hat_tensor[majority_mask].cpu()
    y_pred_majority = y_pred_tensor[majority_mask].cpu()
    result_dict["Majority-ACC"] = accuracy_score(y_true_majority, y_pred_majority)
    result_dict["Majority-AUC"] = roc_auc_score(y_true_majority, y_hat_majority, average="micro", multi_class="raise")
    y_true_minority = y_true_tensor[minority_mask].cpu()
    y_hat_minority = y_hat_tensor[minority_mask].cpu()
    y_pred_minority = y_pred_tensor[minority_mask].cpu()
    result_dict["Minority-ACC"] = accuracy_score(y_true_minority, y_pred_minority)
    result_dict["Minority-AUC"] = roc_auc_score(y_true_minority, y_hat_minority, average="micro", multi_class="raise")
    # loss
    result_dict["loss"] = pred_loss / data_count

    result_str = make_result_str(result_dict)

    return result_dict, result_str


def make_result_str(result_dict):
    result_str = ""
    for metric in result_dict:
        result_str += f"{metric}={result_dict[metric]:.4f}, "
    return result_str[:-2]


def result_to_csv(csv_file, exp_name, results_keys, results_mean, results_std):
    result_columns = ["exp_name"]

    for key in results_keys:
        result_columns.append(f"{key} mean")
        result_columns.append(f"{key} std")

    result_values = [exp_name]
    for mean, std in zip(results_mean, results_std):
        result_values.append(mean)
        result_values.append(std)

    if not os.path.exists(csv_file):
        result_data = [result_columns, result_values]
    else:
        with open(csv_file, "r", encoding="utf8") as output_file:
            result_data = list(csv.reader(output_file))
        result_data.append(result_values)

    with open(csv_file, "w", encoding="utf8") as output_file:
        writer = csv.writer(output_file)
        writer.writerows(result_data)

    return
