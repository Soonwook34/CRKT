import os
from time import time
from pprint import pformat

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam
from accelerate import Accelerator

from dataset_CL4KT import SimCLRDatasetWrapper, MostRecentQuestionSkillDataset
from util import MainArgParser
from util import (set_seed, write_log, set_config, load_model, timestamp, calc_metric, result_to_csv)


def train(args, fold):
    accelerator = Accelerator()
    device = accelerator.device
    args.device = device
    max_patience = 10

    # load dataset
    train_fold_list = [f"fold{fold}.json" for fold in range(args.k_fold)]
    valid_fold = [train_fold_list.pop(fold)]
    train_set = SimCLRDatasetWrapper(
        MostRecentQuestionSkillDataset(args.dataset_path, train_fold_list, args.concept, args.question),
        args.dataset_path,
        seq_len=200,
        mask_prob=0.2,
        crop_prob=0.3,
        permute_prob=0.5,
        replace_prob=0.5,
        negative_prob=1.0,
        eval_mode=False)

    valid_set = SimCLRDatasetWrapper(
        MostRecentQuestionSkillDataset(args.dataset_path, valid_fold, args.concept, args.question),
        args.dataset_path,
        seq_len=200,
        mask_prob=0,
        crop_prob=0,
        permute_prob=0,
        replace_prob=0,
        negative_prob=0,
        eval_mode=True)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, generator=torch.Generator(device=device))
    valid_loader = DataLoader(valid_set, batch_size=args.batch, shuffle=True, generator=torch.Generator(device=device))

    # load model
    model = load_model(args.model, args)

    # optimizer and loss function
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_function = BCELoss()

    model, train_loader, valid_loader, optimizer = accelerator.prepare(model, train_loader, valid_loader, optimizer)

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
        with tqdm(train_loader, unit="it") as train_bar:
            for data in train_bar:
                optimizer.zero_grad()
                match model.name:
                    case "CL4KT":
                        output, loss, score, _, seq_mask = forward_model(model, data, device)
                output = output[seq_mask].flatten()
                score_true = score[seq_mask].float().flatten()
                train_len = len(score_true)
                # backward
                match model.name:
                    case "CL4KT":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    case _:
                        loss = loss_function(output, score_true)
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

    return


def test(args, fold):
    accelerator = Accelerator()
    device = args.device

    # load dataset
    test_set = SimCLRDatasetWrapper(
        MostRecentQuestionSkillDataset(args.dataset_path, ["test.json"], args.concept, args.question),
        args.dataset_path,
        seq_len=200,
        mask_prob=0,
        crop_prob=0,
        permute_prob=0,
        replace_prob=0,
        negative_prob=0,
        eval_mode=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, generator=torch.Generator(device=device))

    # load model
    model = load_model(args.model, args)
    model.load_state_dict(torch.load(f"{args.save_path}/fold{fold}_best.pt"))

    model, test_loader = accelerator.prepare(model, test_loader)

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
    loss_function = BCELoss()

    y_true, y_hat, y_pred, y_question = [], [], [], []
    with tqdm(pred_loader, unit="it") as pred_bar, torch.no_grad():
        for data in pred_bar:
            match model.name:
                case "CL4KT":
                    output, loss, score, question, seq_mask = forward_model(model, data, device)
            output = output[seq_mask].flatten()
            score_true = score[seq_mask].float().flatten()
            question_valid = question[seq_mask].float().flatten()
            pred_len = len(score_true)
            # pred
            output_pred = (output >= 0.5).int()
            y_true += score_true.cpu().tolist()
            y_hat += output.cpu().tolist()
            y_pred += output_pred.cpu().tolist()
            y_question += question_valid.cpu().tolist()
            # loss
            match model.name:
                case "CL4KT":
                    loss = loss
                case _:
                    loss = loss_function(output, score_true)
            # update console
            data_count += pred_len
            pred_loss += loss.item() * pred_len
            pred_bar.set_postfix(loss=f"{pred_loss / data_count:.6f}")

    y_majority = (correct_rate[y_question] >= 0.5).int().tolist()
    y_minority = (correct_rate[y_question] < 0.5).int().tolist()
    result_dict, result_str = calc_metric(y_true, y_hat, y_pred, y_majority, y_minority, pred_loss, data_count)

    return result_dict, result_str


def forward_model(model, data, device):
    match model.name:
        case "CL4KT":
            out_dict = model(data)
            output = out_dict["pred"]
            loss = model.loss(data, out_dict)[0]
            score = out_dict["true"]
            question = out_dict["question"]
            seq_mask = torch.ne(score, -1)
            return output, loss, score, question, seq_mask
        case _:
            print(f"forward() for model {model.name} not found.")
            exit()

    return output, score, question, seq_mask


if __name__ == "__main__":
    args = MainArgParser().parse_args()
    args.lr = float(args.lr)
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_list = ["CL4KT"]
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
    results = []
    test_result_dict = {}
    for fold in range(args.k_fold):
        start_fold = time()

        train(args, fold)
        test_result_dict = test(args, fold)
        results.append(list(test_result_dict.values()))

        fold_time = int(time() - start_fold)
        time_str = f"[{timestamp()}] fold {fold} done. ({fold_time // 3600:02d}:{(fold_time % 3600) // 60:02d}:{fold_time % 60:02d})\n"
        write_log(args.log_file, time_str)

    results_keys = list(test_result_dict.keys())
    results_mean = np.mean(np.array(results), axis=0).tolist()
    results_std = np.std(np.array(results), axis=0).tolist()
    results_dict = {key: f"{mean:.4f}±{std:.4f}" for key, mean, std in zip(results_keys, results_mean, results_std)}

    results_str = pformat(results_dict, sort_dicts=False)
    print_str_total = f"[{timestamp()}] model={args.model} exp={args.exp_name} ({args.k_fold}-Fold Cross Validation)\n{results_str}\n"
    write_log(args.log_file, print_str_total)
    write_log(args.result_file, print_str_total, console=False)

    total_time = int(time() - start_total)
    time_str = f"[{timestamp()}] All done. ({total_time // 3600:02d}:{(total_time % 3600) // 60:02d}:{total_time % 60:02d})"
    write_log(args.log_file, time_str)

    result_to_csv(args.result_csv_file, f"{args.model}_{args.exp_name}", results_keys, results_mean, results_std)
