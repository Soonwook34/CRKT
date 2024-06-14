import os
from ast import literal_eval
from time import time
from pprint import pformat

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import (BCELoss, CrossEntropyLoss, BCEWithLogitsLoss)
from torch.optim import Adam

from dataset import KTDataset
from dataset import pad_collate
from util import MainArgParser
from util import (set_seed, write_log, set_config, load_model, timestamp, calc_metric, result_to_csv)


def train(args, fold):
    device = args.device
    args.device = device
    max_patience = 10

    # load dataset
    train_fold_list = [f"fold{fold}.json" for fold in range(args.k_fold)]
    valid_fold = [train_fold_list.pop(fold)]
    train_set = KTDataset(args.dataset_path, train_fold_list, args.max_concept_len)
    valid_set = KTDataset(args.dataset_path, valid_fold, args.max_concept_len)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, collate_fn=pad_collate)
    valid_loader = DataLoader(valid_set, batch_size=args.batch, shuffle=True, collate_fn=pad_collate)

    # load model
    model = load_model(args.model, args)
    model = model.to(device)

    # optimizer and loss function
    optimizer = Adam(model.parameters(), lr=args.lr)
    # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # DTransformer
    kt_loss = BCELoss()
    ot_loss = CrossEntropyLoss(ignore_index=-1)
    pos_weight = torch.tensor((args.concept - args.top_k) / args.top_k, device=args.device)
    r_loss = BCEWithLogitsLoss(pos_weight=pos_weight)
    cl_loss = CrossEntropyLoss(ignore_index=-1)

    print_str = f"[{timestamp()}] fold {fold} start"
    write_log(args.log_file, print_str)

    # train start
    best_val_loss = 1e+32
    patience_count = 0
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.
        data_count = 0
        # train
        with (tqdm(train_loader, unit="it") as train_bar):
            for data in train_bar:
                optimizer.zero_grad()
                forward_dict = forward_model(model, data, device)
                output = forward_dict["output"]
                score = forward_dict["score"]
                seq_mask = forward_dict["seq_mask"]
                output = output[seq_mask].flatten()
                score_true = score[:, 1:][seq_mask].float().flatten()
                train_len = len(score_true)
                # backward
                match model.name:
                    case "CRKT":
                        loss = kt_loss(output, score_true)
                        if forward_dict["r_target"] is not None:
                            r_output = forward_dict["r_target"][seq_mask].flatten().unsqueeze(-1)
                            r_true = forward_dict["g_target"][seq_mask].float().flatten().unsqueeze(-1)
                            loss += args.alpha * r_loss(r_output, r_true)
                        if forward_dict["inter_label"] is not None:
                            loss += args.beta * cl_loss(forward_dict["inter_cossim"], forward_dict["inter_label"])
                    case "DTransformer":
                        loss = forward_dict["loss"]
                    case "DP_DKT":
                        ot_output = forward_dict["ot_output"][seq_mask].reshape(-1, args.option)
                        option_true = forward_dict["option"][:, 1:][seq_mask].flatten()
                        loss = args.lamb * kt_loss(output, score_true) + (1 - args.lamb) * ot_loss(ot_output, option_true)
                    case _:
                        loss = kt_loss(output, score_true)
                loss.backward()
                optimizer.step()
                # update console
                data_count += train_len
                train_loss += loss.item() * train_len
                train_bar.set_postfix(loss=f"{train_loss / data_count:.6f}")

        # validation
        result_dict, result_str = predict(model, valid_loader, valid_set.correct_rate, device)
        val_loss = result_dict["loss"]

        # write log
        print_str = f"[{timestamp()}] epoch={epoch + 1:2d} | {result_str}"
        write_log(args.log_file, print_str)

        # save best model and early stop
        if best_val_loss > val_loss:
            patience_count = 0
            best_model_file = os.path.join(args.save_path, f"fold{fold}_best.pt")
            torch.save(model.state_dict(), best_model_file, pickle_protocol=4)
            best_val_loss = val_loss
        else:
            patience_count += 1
            if patience_count >= max_patience:
                print(f"Patience count reached at {max_patience}. Early stopped.")
                break
            else:
                print(f"Patience count updated. ({patience_count}/{max_patience})")
                print(f"Best valid loss: {best_val_loss:.6f}, Now: {val_loss:.6f}")

    return best_val_loss


def test(args, fold):
    device = args.device

    # load dataset
    test_set = KTDataset(args.dataset_path, ["test.json"], args.max_concept_len)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, collate_fn=pad_collate)

    # load model
    model = load_model(args.model, args)
    model = model.to(device)
    model.load_state_dict(torch.load(f"{args.save_path}/fold{fold}_best.pt"))

    # predict
    result_dict, result_str = predict(model, test_loader, test_set.correct_rate, device)

    # write log
    print_str = f"[{timestamp()}] model={args.model} exp={args.exp_name} (fold {fold + 1}/{args.k_fold})\n{result_str}\n"
    write_log(args.log_file, print_str)
    write_log(args.result_file, print_str.replace("\n", " "), console=False)

    return result_dict


def predict(model, pred_loader, correct_rate, device):
    model.eval()
    pred_loss = 0.
    data_count = 0
    kt_loss = BCELoss()
    ot_loss = CrossEntropyLoss(ignore_index=-1)
    pos_weight = torch.tensor((args.concept - args.top_k) / args.top_k, device=args.device)
    r_loss = BCEWithLogitsLoss(pos_weight=pos_weight)

    y_true, y_hat, y_pred, y_question = [], [], [], []
    with tqdm(pred_loader, unit="it") as pred_bar, torch.no_grad():
        for data in pred_bar:
            forward_dict = forward_model(model, data, device)
            output = forward_dict["output"]
            score = forward_dict["score"]
            question = forward_dict["question"]
            seq_mask = forward_dict["seq_mask"]
            output = output[seq_mask].flatten()
            score_true = score[:, 1:][seq_mask].float().flatten()
            question_valid = question[:, 1:][seq_mask].float().flatten()
            pred_len = len(score_true)
            # pred
            output_pred = (output >= 0.5).int()
            y_true += score_true.cpu().tolist()
            y_hat += output.cpu().tolist()
            y_pred += output_pred.cpu().tolist()
            y_question += question_valid.cpu().tolist()
            # loss
            match model.name:
                case "CRKT":
                    if forward_dict["r_target"] is not None:
                        r_output = forward_dict["r_target"][seq_mask].flatten().unsqueeze(-1)
                        r_true = forward_dict["g_target"][seq_mask].float().flatten().unsqueeze(-1)
                        loss = kt_loss(output, score_true) + args.alpha * r_loss(r_output, r_true)
                    else:
                        loss = kt_loss(output, score_true)
                case "DTransformer":
                    loss = forward_dict["loss"]
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                case "DP_DKT":
                    ot_output = forward_dict["ot_output"][seq_mask].reshape(-1, args.option)
                    option_true = forward_dict["option"][:, 1:][seq_mask].flatten()
                    loss = args.lamb * kt_loss(output, score_true) + (1 - args.lamb) * ot_loss(ot_output, option_true)
                case _:
                    loss = kt_loss(output, score_true)
            # update console
            data_count += pred_len
            pred_loss += loss.item() * pred_len
            pred_bar.set_postfix(loss=f"{pred_loss / data_count:.6f}")

    y_majority = (correct_rate[y_question] >= 0.5).int().tolist()
    y_minority = (correct_rate[y_question] < 0.5).int().tolist()
    result_dict, result_str = calc_metric(y_true, y_hat, y_pred, y_majority, y_minority, pred_loss, data_count)

    return result_dict, result_str


def forward_model(model, data, device):
    (user, question, concept, score, option, answer, unchosen,
     pos_score, pos_option, neg_score, neg_option) = data
    user, question, concept, score = user.to(device), question.to(device), concept.to(device), score.to(device)
    option, answer, unchosen = option.to(device), answer.to(device), unchosen.to(device)
    pos_score, pos_option, neg_score, neg_option = pos_score.to(device), pos_option.to(device), neg_score.to(device), neg_option.to(device)

    seq_mask = torch.ne(score, -1)[:, 1:]
    forward_dict = {"score": score, "question": question, "seq_mask": seq_mask}
    match model.name:
        case "CRKT":
            output, r_target, g_target, inter_cossim, inter_label = model(question, concept, score, option, unchosen,
                                                                          pos_score, pos_option, neg_score, neg_option)
            forward_dict["r_target"] = r_target
            forward_dict["g_target"] = g_target
            forward_dict["inter_cossim"] = inter_cossim
            forward_dict["inter_label"] = inter_label
        case "DKT":
            concept = concept[:, :, 0]
            output = model(concept, score)
        case "GKT":
            concept = concept[:, :, 0]
            output = model(score, concept)
        case "SAKT":
            output = model(question[:, :-1], score[:, :-1], question[:, 1:])
        case "AKT":
            concept = concept[:, :, 0]
            output, _ = model(concept, score, question)
            output = output[:, 1:]
        case "DTransformer":
            concept = concept[:, :, 0]
            output, loss, pred_loss, cl_loss = model.get_cl_loss(q=concept, s=score, pid=question)
            # output, loss = model.get_loss(q=concept, s=score, pid=question)
            forward_dict["loss"] = loss
        case "DP_DKT":
            output, ot_output = model(question, concept, score, option, answer)
            forward_dict["ot_output"] = ot_output
            forward_dict["option"] = option
        case _:
            print(f"forward() for model {model.name} not found.")
            exit()
    forward_dict["output"] = output

    return forward_dict


if __name__ == "__main__":
    args = MainArgParser().parse_args()
    args.lr = float(args.lr)
    args.version = literal_eval(args.version)
    if not isinstance(args.version, list):
        raise Exception("Wrong type of args.version. ex) [0,1,0,0]")
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_list = ["DKT", "GKT", "SAKT", "AKT", "DTransformer", "DP_DKT", "CRKT",
                  "CRKT_no_unchosen", "CRKT_no_option", "CRKT_no_topk", "CRKT_no_map"]
    if args.model not in model_list:
        raise Exception(f"Model Not Found: {args.model} is not in {model_list}.")

    # set seed and config
    set_seed(args.seed)
    set_config(args)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    write_log(args.log_file, args)

    # cross validation
    start_total = time()
    val_loss_list = []
    results = []
    test_result_dict = {}
    for fold in range(args.k_fold):
        start_fold = time()

        val_loss = train(args, fold)
        val_loss_list.append(val_loss)
        test_result_dict = test(args, fold)
        results.append(list(test_result_dict.values()))

        fold_time = int(time() - start_fold)
        time_str = f"[{timestamp()}] fold {fold} done. ({fold_time // 3600:02d}:{(fold_time % 3600) // 60:02d}:{fold_time % 60:02d})\n"
        write_log(args.log_file, time_str)

    results_keys = list(test_result_dict.keys())
    results_mean = np.mean(np.array(results), axis=0).tolist()
    results_std = np.std(np.array(results), axis=0).tolist()
    results_dict = {key: f"{mean:.4f}±{std:.4f}" for key, mean, std in zip(results_keys, results_mean, results_std)}
    valid_loss_mean = np.mean(np.array(val_loss_list)).tolist()
    valid_loss_std = np.std(np.array(val_loss_list)).tolist()
    results_dict["valid_loss"] = f"{valid_loss_mean:.4f}±{valid_loss_std:.4f}"

    results_str = pformat(results_dict, sort_dicts=False)
    print_str_total = f"[{timestamp()}] model={args.model} exp={args.exp_name} ({args.k_fold}-Fold Cross Validation)\n{results_str}\n"
    write_log(args.log_file, print_str_total)
    write_log(args.result_file, print_str_total, console=False)

    total_time = int(time() - start_total)
    time_str = f"[{timestamp()}] All done. ({total_time // 3600:02d}:{(total_time % 3600) // 60:02d}:{total_time % 60:02d})"
    write_log(args.log_file, time_str)

    result_to_csv(args.result_csv_file, f"{args.model}_{args.exp_name}",
                  results_keys + ["Valid-loss"], results_mean + [valid_loss_mean], results_std + [valid_loss_std])
