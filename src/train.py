from __future__ import print_function

import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import snntorch.functional as SF
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, classification_report

from src.utils.metric import calc_score
from src.utils.supcontrast import AverageMeter
from src.utils.supcontrast import warmup_learning_rate


def train_supcon(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == "SupCon":
            loss = criterion(features, labels)
        elif opt.method == "SimCLR":
            loss = criterion(features)
        else:
            raise ValueError("contrastive method not supported: {}".
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if opt.verbose and (idx + 1) % opt.print_freq == 0:
            print("Train: [{0}][{1}/{2}]\t"
                  "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  "loss {loss.val:.3f} ({loss.avg:.3f})".format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
    return losses.avg


def valid_supcon(valid_loader, model, criterion, opt):
    """one epoch validating"""
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for idx, (images, labels, _) in enumerate(valid_loader):
            data_time.update(time.time() - end)

            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # compute loss
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if opt.method == "SupCon":
                loss = criterion(features, labels)
            elif opt.method == "SimCLR":
                loss = criterion(features)
            else:
                raise ValueError("contrastive method not supported: {}".
                                format(opt.method))
            # update metric
            losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return 1/losses.avg


def train_model(
    device,
    task,
    dataloader,
    model,
    criterion,
    optimizer,
    spike=False,
    n_epochs=10,
    print_every=1,
    verbose=True,
    plot_results=True,
    validation=True,
    save_ckpt=False,
    model_name=None,
    strategy=["score"],
):
    """Basic training flow for neural network.
    Args:
        device: either cpu or cuda for acceleration.
        task: full task for score calculation.
        dataloader: data loader containing training data.
        model: network to train.
        criterion: loss function.
        optimizer: optimizer for weights update.
        spike: True to initiate spike-train process (default is False).
        n_epochs: number of epochs (default is 10).
        print_every: int for number of epoch before printing (default is 1).
        verbose: True to enable verbosity (default is True).
        plot_result: True to plot results (default is True).
        validation: True to run validation as well (default is True).
        save_model: True to save checkpoint to ckpts folder (default is False).
        model_name: the name of model to be trained (default is None).
    Returns:
        model: trained model.
        best_epoch: epoch which yield the best score.
        best_score: best score obtained.
        train_loss[-1]: the latest training loss.
        val_loss[-1]: the latest validation loss.
        best_info: detailed information which yield the best score.
    """
    best_dict = {}
    best_result = {}
    for item in strategy:
        best_result[item] = 0
    losses = []
    start = time.time()
    print("\nTraining for {} epochs...".format(n_epochs))
    for epoch in range(n_epochs):
        if verbose == True and epoch % print_every == 0:
            print("\n\nEpoch {}/{}:".format(epoch + 1, n_epochs))

        if validation == True:
            evaluation = ["train", "val"]
        else:
            evaluation = ["train"]

        # Each epoch has a training and validation phase
        for phase in evaluation:
            if phase == "train":
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            for data, label, info in dataloader[phase]:
                data, label, info = data.to(device), label.to(device), info.to(device)

                # forward + backward + optimize
                x = data
                outputs = model(x)  # spk_rec, mem_rec if spike=True
                acc = calc_accuracy(outputs, label, spike)
                loss = calc_loss(criterion, outputs, label, spike)  # loss function
                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # record loss statistics
                running_loss += loss.item()

            losses.append(running_loss)

            if verbose == True and epoch % print_every == 0:
                print(
                    "{} loss: {:.4f} | acc: {:.4f}|".format(phase, running_loss, acc),
                    end=" ",
                )

        val_score, val_acc = valid_model(
            device=device,
            task=int(task[-2]),
            dataloader=dataloader[evaluation[-1]],
            trained_model=model,
            verbose=False,
            spike=spike,
        )

        val_results = {
        "score": val_score,
        "accuracy": val_acc,
        "loss": losses[-1],  # ✅ use direct loss, no inversion
    }
        for item in strategy:
            if val_results[item] > best_result[item]:
                best_result[item] = val_results[item]
                best_dict[item] = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                } | val_results

    if verbose == True:
        print("\nFinished Training  | Time:{}".format(time.time() - start))

    if plot_results == True:
        plt.figure(figsize=(10, 10))
        plt.plot(losses[0::2], label="train_loss")
        if validation == True:
            plt.plot(losses[1::2], label="validation_loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.draw()

    if save_ckpt:
        PATH = "ckpts/{}_CheckPoint_task{}.pt".format(model_name, task)
        torch.save(best_dict, PATH)

    return best_dict
def calc_accuracy(output, Y, spike=False):
    if isinstance(output, tuple):
        output, _ = output
    if spike:
        train_acc = SF.acc.accuracy_rate(output[0], Y)
    else:
        _, max_indices = torch.max(output, 1)
        train_acc = (max_indices == Y).sum().item() / max_indices.size(0)
    return train_acc


def calc_loss(criterion, output, Y, spike=False):
    if isinstance(output, tuple):
        output, _ = output
    if spike:
        return criterion(output[0], Y)
    else:
        return criterion(output, Y)
        
def valid_model(device, task, dataloader, trained_model, verbose=False, spike=False):
    """Post Evaluation Metric Platform. Feed in the trained model and train/validation data loader.
    Args:
        device: either cpu or cuda for acceleration.
        dataloader: dataloader containing data for evaluation.
        trained_model: model used for evaluation.
        verbose: True to enable verbosity (True as default).
    Returns:
        classification accuracy obtained from sklearn's accuracy score.
    """
    truth = []
    preds = []
    for data, label, info in dataloader:
        data, label, info = data.to(device), label.to(device), info.to(device)
        x = data
        outputs, _ = trained_model(x)  # <-- unpack outputs and ignore aux_loss
        if spike:
            _, idx = outputs[0].sum(dim=0).max(1)
            preds.append(idx.cpu().numpy().tolist())
        else:
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.cpu().numpy().tolist())
        truth.append(label.cpu().numpy().tolist())

    preds_flat = [item for sublist in preds for item in sublist]
    truth_flat = [item for sublist in truth for item in sublist]

    score, *_ = calc_score(truth_flat, preds_flat, verbose, task=task)
    accuracy = accuracy_score(truth_flat, preds_flat)
    if verbose:
        print("\nEvaluating....")
        print("Accuracy:", accuracy)
        print(classification_report(truth_flat, preds_flat))
    return score, accuracy



def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_mixup(
    device,
    task,
    dataloader,
    model,
    criterion,
    optimizer,
    spike=False,
    n_epochs=10,
    print_every=1,
    verbose=True,
    plot_results=True,
    validation=True,
    save_ckpt=False,
    model_name=None,
    strategy=["score"],
    aux_loss_scale=0.01,  # scale factor for MoE aux_loss
):
    best_dict = {}
    best_result = {}
    for item in strategy:
        best_result[item] = 0
    losses = []
    start = time.time()
    print("\nTraining for {} epochs...".format(n_epochs))
    for epoch in range(n_epochs):
        if verbose and epoch % print_every == 0:
            print("\n\nEpoch {}/{}:".format(epoch + 1, n_epochs))

        evaluation = ["train", "val"] if validation else ["train"]

        for phase in evaluation:
            total = 0
            correct = 0
            model.train(phase == "train")

            running_loss = 0.0
            for data, label, info in dataloader[phase]:
                data, label, info = data.to(device), label.to(device), info.to(device)

                # Mixup
                data, label_a, label_b, lam = mixup_data(data, label)
                data, label_a, label_b = map(Variable, (data, label_a, label_b))

                # Forward pass
                x = data
                outputs, aux_loss = model(x)  # <-- handle MoE aux_loss

                # Compute mixup loss + aux_loss
                loss = mixup_criterion(criterion, outputs, label_a, label_b, lam)
                loss += aux_loss_scale * aux_loss

                running_loss += loss.item()

                # Accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (lam * predicted.eq(label_a.data).cpu().sum().float() +
                            (1 - lam) * predicted.eq(label_b.data).cpu().sum().float())
                acc = correct / total

                # Backward + optimize
                optimizer.zero_grad()
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            losses.append(running_loss)

            if verbose and epoch % print_every == 0:
                print("{} loss: {:.4f} | acc: {:.4f}".format(phase, running_loss, acc), end=" ")

        # Validation metrics
        val_score, val_acc = valid_model(
            device=device,
            task=int(task[-2]),
            dataloader=dataloader[evaluation[-1]],
            trained_model=model,
            verbose=False,
            spike=spike,
        )

        val_results = {
            "score": val_score,
            "accuracy": val_acc,
            "loss": losses[-1],  # ✅ use direct loss, no inversion
        }

        for item in strategy:
            if val_results[item] > best_result[item]:
                best_result[item] = val_results[item]
                best_dict[item] = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                } | val_results

    if verbose:
        print("\nFinished Training  | Time:{}".format(time.time() - start))

    if plot_results:
        plt.figure(figsize=(10, 10))
        plt.plot(losses[0::2], label="train_loss")
        if validation:
            plt.plot(losses[1::2], label="validation_loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.draw()

    if save_ckpt:
        PATH = "ckpts/{}_CheckPoint_task{}.pt".format(model_name, task)
        torch.save(best_dict, PATH)

    return best_dict
def train_mixup_moe(
    device,
    task,
    dataloader,
    model,
    criterion,
    optimizer,
    spike=False,
    n_epochs=10,
    print_every=1,
    verbose=True,
    plot_results=True,
    validation=True,
    save_ckpt=False,
    model_name=None,
    strategy=["score"],
    use_moe=False,
    aux_loss_weight=0.01,  # Weight for MoE load-balancing loss
):
    """
    Training function with mixup augmentation and MoE support + expert usage & accuracy visualization.
    """
    best_dict = {}
    best_result = {item: 0 for item in strategy}
    losses = []
    start = time.time()
    print(f"\nTraining for {n_epochs} epochs...")

    # ✅ Initialize expert usage and accuracy counters if using MoE
    if use_moe:
        num_experts = model.classifier.num_experts
        total_expert_usage = torch.zeros(num_experts, device=device)
        expert_correct = torch.zeros(num_experts, device=device)
        expert_total = torch.zeros(num_experts, device=device)

    for epoch in range(n_epochs):
        if verbose and epoch % print_every == 0:
            print(f"\n\nEpoch {epoch + 1}/{n_epochs}:")

        evaluation = ["train", "val"] if validation else ["train"]

        for phase in evaluation:
            total = 0
            correct = 0
            running_loss = 0.0
            running_aux_loss = 0.0

            if phase == "train":
                model.train(True)
            else:
                model.eval()

            for data, label, info in dataloader[phase]:
                data, label, info = data.to(device), label.to(device), info.to(device)

                # Mix-up method
                data, label_a, label_b, lam = mixup_data(data, label)
                data, label_a, label_b = map(Variable, (data, label_a, label_b))

                # --------------------------
                # Forward pass
                # --------------------------
                if use_moe:
                    outputs, aux_loss, topk_idx = model(x=data, return_gates=True)

                    # ✅ Track expert usage and accuracy
                    with torch.no_grad():
                        _, predicted = torch.max(outputs, 1)
                        correct_a = predicted.eq(label_a)
                        correct_b = predicted.eq(label_b)
                        sample_correct = lam * correct_a.float() + (1 - lam) * correct_b.float()

                    for exp_id in range(model.classifier.num_experts):
                        # handle both top-1 and top-k gating shapes
                        mask = (topk_idx == exp_id)
                        if mask.dim() > 1:  # e.g., [batch_size, k]
                            mask = mask.any(dim=1)  # reduce to [batch_size]
                        count = mask.sum()
                        if count > 0:
                            total_expert_usage[exp_id] += count
                            expert_correct[exp_id] += sample_correct[mask].sum()
                            expert_total[exp_id] += count

                else:
                    outputs = model(data)
                    aux_loss = 0.0

                # --------------------------
                # Compute mixup loss + aux loss
                # --------------------------
                loss = mixup_criterion(criterion, outputs, label_a, label_b, lam)
                if use_moe:
                    total_loss = loss + aux_loss_weight * aux_loss
                    running_aux_loss += aux_loss.item()
                else:
                    total_loss = loss

                running_loss += loss.item()

                # --------------------------
                # Compute accuracy (mixup-style)
                # --------------------------
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (
                    lam * predicted.eq(label_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(label_b.data).cpu().sum().float()
                )
                acc = correct / total

                optimizer.zero_grad()
                if phase == "train":
                    total_loss.backward()
                    optimizer.step()

            losses.append(running_loss)

            if verbose and epoch % print_every == 0:
                if use_moe:
                    print(
                        f"{phase} loss: {running_loss:.4f} | aux_loss: {running_aux_loss:.4f} | acc: {acc:.4f} |",
                        end=" ",
                    )
                else:
                    print(
                        f"{phase} loss: {running_loss:.4f} | acc: {acc:.4f} |",
                        end=" ",
                    )

        # --------------------------
        # Validation phase
        # --------------------------
        val_score, val_acc = valid_model_moe(
            device=device,
            task=int(task[-2]),
            dataloader=dataloader[evaluation[-1]],
            trained_model=model,
            verbose=False,
            spike=spike,
            use_moe=use_moe,
        )

        val_results = {
            "score": val_score,
            "accuracy": val_acc,
            "loss": 1 / losses[-1],
        }

        # Track best model for each strategy
        for item in strategy:
            if val_results[item] > best_result[item]:
                best_result[item] = val_results[item]
                best_dict[item] = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                } | val_results

    # --------------------------
    # Finished Training
    # --------------------------
    if verbose:
        print(f"\nFinished Training  | Time: {time.time() - start:.2f}s")

    # --------------------------
    # Plot training losses
    # --------------------------
    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.plot(losses[0::2], label="train_loss")
        if validation:
            plt.plot(losses[1::2], label="val_loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.show()

    # --------------------------
    # ✅ Expert usage + accuracy visualization
    # --------------------------
    if use_moe:
        total_usage = total_expert_usage.cpu().numpy()
        usage_percent = total_usage / total_usage.sum() * 100

        expert_acc = (expert_correct / (expert_total + 1e-8) * 100).cpu().numpy()

        print("\n📊 Expert Metrics Summary")
        print("Expert | Usage(%) | Accuracy(%) | Total Samples")
        print("-" * 45)
        for i in range(model.classifier.num_experts):
            print(f"{i:6d} | {usage_percent[i]:8.2f} | {expert_acc[i]:10.2f} | {int(expert_total[i].item())}")

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(range(len(total_usage)), usage_percent, color='skyblue')
        plt.xlabel("Expert ID")
        plt.ylabel("Usage (%)")
        plt.title("Expert Usage Distribution")

        plt.subplot(1, 2, 2)
        plt.bar(range(len(expert_acc)), expert_acc, color='lightgreen')
        plt.xlabel("Expert ID")
        plt.ylabel("Accuracy (%)")
        plt.title("Expert Accuracy Distribution")
        plt.tight_layout()
        plt.show()

    # --------------------------
    # Save best checkpoint (optional)
    # --------------------------
    if save_ckpt:
        PATH = f"ckpts/{model_name}_CheckPoint_task{task}.pt"
        torch.save(best_dict, PATH)

    return best_dict

# def train_mixup_moe(
#     device,
#     task,
#     dataloader,
#     model,
#     criterion,
#     optimizer,
#     spike=False,
#     n_epochs=10,
#     print_every=1,
#     verbose=True,
#     plot_results=True,
#     validation=True,
#     save_ckpt=False,
#     model_name=None,
#     strategy=["score"],
#     use_moe=False,
#     aux_loss_weight=0.01,  # Weight for MoE load-balancing loss
# ):
#     """
#     Training function with mixup augmentation and MoE support.
    
#     Args:
#         device: either cpu or cuda for acceleration.
#         task: task identifier.
#         dataloader: dictionary with 'train' and optionally 'val' dataloaders.
#         model: model to train (can be PRS_classifier or PRS_classifier2 with MoE).
#         criterion: loss function for classification.
#         optimizer: optimizer for training.
#         spike: True to use spiking output (default: False).
#         n_epochs: number of training epochs.
#         print_every: print statistics every N epochs.
#         verbose: True to enable verbosity.
#         plot_results: True to plot training curves.
#         validation: True to perform validation.
#         save_ckpt: True to save checkpoints.
#         model_name: name for saving checkpoint.
#         strategy: list of metrics to track best model ('score', 'accuracy', 'loss').
#         use_moe: True if using MoE classifier (PRS_classifier2).
#         aux_loss_weight: weight for MoE load-balancing auxiliary loss.
    
#     Returns:
#         best_dict: dictionary containing best model checkpoints for each strategy.
#     """
#     best_dict = {}
#     best_result = {}
#     for item in strategy:
#         best_result[item] = 0
#     losses = []
#     start = time.time()
#     print("\nTraining for {} epochs...".format(n_epochs))
    
#     for epoch in range(n_epochs):
#         if verbose == True and epoch % print_every == 0:
#             print("\n\nEpoch {}/{}:".format(epoch + 1, n_epochs))

#         if validation == True:
#             evaluation = ["train", "val"]
#         else:
#             evaluation = ["train"]

#         # Each epoch has a training and validation phase
#         for phase in evaluation:
#             total = 0
#             correct = 0
#             running_loss = 0.0
#             running_aux_loss = 0.0  # Track auxiliary MoE loss separately
            
#             if phase == "train":
#                 model.train(True)  # Set model to training mode
#             else:
#                 model.train(False)  # Set model to evaluate mode
            
#             for data, label, info in dataloader[phase]:
#                 data, label, info = data.to(device), label.to(device), info.to(device)

#                 # Mix-up method
#                 data, label_a, label_b, lam = mixup_data(data, label)
#                 data, label_a, label_b = map(Variable, (data, label_a, label_b))
                
#                 # Forward pass
#                 x = data
#                 if use_moe:
#                     # MoE model returns outputs and auxiliary loss
#                     outputs, aux_loss = model(x)
#                 else:
#                     # Standard model returns only outputs
#                     outputs = model(x)
#                     aux_loss = 0.0
                
#                 # Compute main classification loss with mixup
#                 loss = mixup_criterion(criterion, outputs, label_a, label_b, lam)
                
#                 # Add auxiliary load-balancing loss for MoE
#                 if use_moe:
#                     total_loss = loss + aux_loss_weight * aux_loss
#                     running_aux_loss += aux_loss.item()
#                 else:
#                     total_loss = loss
                
#                 # Record loss statistics (moved here, only once)
#                 running_loss += loss.item()
                
#                 # Calculate accuracy with mixup
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += label.size(0)
#                 correct += (lam * predicted.eq(label_a.data).cpu().sum().float()
#                     + (1 - lam) * predicted.eq(label_b.data).cpu().sum().float())
#                 acc = 1.0 * correct / total

#                 # Zero the parameter gradients
#                 optimizer.zero_grad()

#                 # Backward + optimize only if in training phase
#                 if phase == "train":
#                     total_loss.backward()
#                     # Update the weights
#                     optimizer.step()

#             losses.append(running_loss)

#             if verbose == True and epoch % print_every == 0:
#                 if use_moe:
#                     print(
#                         "{} loss: {:.4f} | aux_loss: {:.4f} | acc: {:.4f} |".format(
#                             phase, running_loss, running_aux_loss, acc
#                         ),
#                         end=" ",
#                     )
#                 else:
#                     print(
#                         "{} loss: {:.4f} | acc: {:.4f} |".format(phase, running_loss, acc),
#                         end=" ",
#                     )

#         # Validation evaluation
#         val_score, val_acc = valid_model_moe(
#             device=device,
#             task=int(task[-2]),
#             dataloader=dataloader[evaluation[-1]],
#             trained_model=model,
#             verbose=False,
#             spike=spike,
#             use_moe=use_moe,
#         )

#         val_results = {
#             "score": val_score,
#             "accuracy": val_acc,
#             "loss": 1 / losses[-1],
#         }

#         # Track best models for each strategy
#         for item in strategy:
#             if val_results[item] > best_result[item]:
#                 best_result[item] = val_results[item]
#                 best_dict[item] = {
#                     "epoch": epoch + 1,
#                     "model_state_dict": model.state_dict(),
#                     "optimizer_state_dict": optimizer.state_dict(),
#                 } | val_results

#     if verbose == True:
#         print("\nFinished Training  | Time:{}".format(time.time() - start))

#     if plot_results == True:
#         plt.figure(figsize=(10, 10))
#         plt.plot(losses[0::2], label="train_loss")
#         if validation == True:
#             plt.plot(losses[1::2], label="validation_loss")
#         plt.legend()
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.draw()

#     if save_ckpt:
#         PATH = "ckpts/{}_CheckPoint_task{}.pt".format(model_name, task)
#         torch.save(best_dict, PATH)

#     return best_dict


def valid_model_moe(device, task, dataloader, trained_model, verbose=False, spike=False, use_moe=False):
    """
    Post Evaluation Metric Platform with MoE support.
    
    Args:
        device: either cpu or cuda for acceleration.
        task: task identifier.
        dataloader: dataloader containing data for evaluation.
        trained_model: model used for evaluation.
        verbose: True to enable verbosity.
        spike: True to use spiking output.
        use_moe: True if using MoE classifier.
    
    Returns:
        score: evaluation score from calc_score.
        accuracy: classification accuracy.
    """
    truth = []
    preds = []
    
    for data, label, info in dataloader:
        data, label, info = data.to(device), label.to(device), info.to(device)
        x = data
        
        if use_moe:
            # MoE model returns outputs and auxiliary loss
            outputs, _ = trained_model(x)
        else:
            outputs = trained_model(x)
        
        if spike:
            _, idx = outputs[0].sum(dim=0).max(1)
            preds.append(idx.cpu().numpy().tolist())
        else:
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.cpu().numpy().tolist())
        truth.append(label.cpu().numpy().tolist())

    preds_flat = [item for sublist in preds for item in sublist]
    truth_flat = [item for sublist in truth for item in sublist]

    score, *_ = calc_score(truth_flat, preds_flat, verbose, task=task)
    accuracy = accuracy_score(truth_flat, preds_flat)
    
    if verbose == True:
        print("\nEvaluating....")
        print("Accuracy:", accuracy)
        print(classification_report(truth_flat, preds_flat))
    
    return score, accuracy
