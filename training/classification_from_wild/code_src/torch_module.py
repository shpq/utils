import time
from tqdm import tqdm
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import timm
import os
from code_src.models import get_model
from code_src.utils import ordinal_cost_fn, compute_mae_and_mse


torch.manual_seed(0)


def train_torch(cfg, kwargs):
    def train(
        model,
        model_name,
        criterion,
        optimizer,
        scheduler,
        batch_size,
        num_epochs=25,
        device=None,
        quantize=False,
        mode="classification",
        weights=None
    ):
        since = time.time()
        model = model.to(device)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            model = model.to(device)
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "valid"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode
                print(phase)
                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                pbar = tqdm(
                    enumerate(dataloaders[phase]),
                    total=len(dataloaders[phase]),
                )

                if quantize and phase == "valid":
                    model = model.to("cpu")
                for index, chunk in pbar:
                    torch.cuda.empty_cache()
                    if mode in ["classification", "regression"]:
                        inputs, labels = chunk
                    elif mode == "ordinal_regression":
                        inputs, labels, levels = chunk
                    torch.cuda.empty_cache()
                    scale_value = 1 / batch_size / max(index, 1)
                    if mode == "classification":
                        pbar.set_description(
                            "{} Loss: {:.4f} Acc: {:.4f}".format(
                                phase,
                                running_loss * scale_value,
                                running_corrects * scale_value,
                            )
                        )
                    elif mode == "regression":
                        pbar.set_description(
                            "{} Loss: {:.4f} ".format(
                                phase,
                                running_loss * scale_value,
                            )
                        )
                    elif mode == "ordinal_regression":
                        pbar.set_description(
                            (
                                "Epoch: %03d/%03d | Loss: %.4f"
                                % (
                                    epoch + 1,
                                    cfg.training.epochs,
                                    running_loss * scale_value,
                                )
                            )
                        )
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if mode == "ordinal_regression":
                        levels = levels.to(device)
                    torch.cuda.empty_cache()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        if quantize and phase == "valid":
                            model = model.to("cpu")
                            quantized_model = torch.quantization.convert(
                                model.eval(), inplace=False)
                            quantized_model.eval()
                            quantized_model = quantized_model.to("cpu")
                            inputs = inputs.to("cpu")
                            if mode in ["classification", "regression"]:
                                outputs = quantized_model(inputs)
                                outputs = outputs.to(device)
                            elif mode == "ordinal_regression":
                                outputs, probas = quantized_model(inputs)
                                outputs = outputs.to(device)
                                probas = probas.to(device)
                        else:
                            if mode in ["classification", "regression"]:
                                outputs = model(inputs)
                            elif mode == "ordinal_regression":
                                outputs, probas = model(inputs)
                        if mode == "classification":
                            _, preds = torch.max(outputs, 1)
                        if mode in ["classification", "regression"]:
                            loss = criterion(outputs, labels)
                        elif mode == "ordinal_regression":
                            loss = criterion(outputs, levels, weights)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    if mode == "classification":
                        running_corrects += (preds ==
                                             labels.data).float().sum()

                if phase == "train":
                    train_loss = running_loss / dataset_sizes[phase]
                    if mode == "classification":
                        train_acc = running_corrects / dataset_sizes[phase]

                if phase == "valid":
                    if mode == "classification":
                        scheduler.step(train_acc)
                    elif mode == ["regression", "ordinal_regression"]:
                        scheduler.step(-train_loss)

                if mode == "classification":
                    print(running_corrects)
                    epoch_acc = running_corrects / dataset_sizes[phase]
                print(dataset_sizes[phase])
                epoch_loss = running_loss / dataset_sizes[phase]
                if mode == "classification":
                    print(
                        "{} Loss: {:.4f} Acc: {:.4f}".format(
                            phase, epoch_loss, epoch_acc
                        )
                    )
                elif mode in ["regression", "ordinal_regression"]:
                    print(
                        "{} Loss: {:.4f} ".format(
                            phase, epoch_loss
                        )
                    )
                # deep copy the model
                if phase == "valid":
                    if mode == "classification":
                        name = os.path.join(
                            cfg.training.checkpoints_path,
                            "{}_epoch_{}_val_acc_{:.2f}_vs_{:.2f}_val_loss_{:.2f}_vs_{:.2f}.pth".format(
                                cfg.dataset.csv_name,
                                epoch,
                                epoch_acc,
                                train_acc,
                                epoch_loss,
                                train_loss,
                            ))
                    elif mode == "regression":
                        name = os.path.join(
                            cfg.training.checkpoints_path,
                            "{}_epoch_{}_val_loss_{:.2f}_vs_{:.2f}.pth".format(
                                cfg.dataset.csv_name,
                                epoch,
                                epoch_loss,
                                train_loss,
                            ))
                    elif mode == "ordinal_regression":
                        name = os.path.join(
                            cfg.training.checkpoints_path,
                            "{}_epoch_{}_val_loss_{:.2f}_vs_{:.2f}.pth".format(
                                cfg.dataset.csv_name,
                                epoch,
                                epoch_loss,
                                train_loss,
                            ))
                    os.makedirs(cfg.training.checkpoints_path, exist_ok=True)
                    if quantize:
                        torch.save(quantized_model.state_dict(), name)
                    else:
                        torch.save(model.state_dict(), name)

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        return model

    dataloaders = kwargs["dataloaders"]
    dataset_sizes = {x: len(kwargs[x]) for x in ["train", "valid"]}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.training.mode in ["classification", "ordinal_regression"]:
        num_classes = len(np.unique(kwargs["train"].label))
    elif cfg.training.mode == "regression":
        num_classes = 1

    model_ft = get_model(cfg, num_classes)

    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    if cfg.training.depth_trainable > 0:
        layers_to_choose = []
        for name, param in model_ft.named_parameters():
            if "conv" in name:
                layers_to_choose.append(name)
        layers_to_choose += [name]
        layer_train = layers_to_choose[
            -min(cfg.training.depth_trainable, len(layers_to_choose))
        ]
        r = 0
        for name, param in model_ft.named_parameters():

            if layer_train != name and r == 0:
                param.requires_grad = False
            else:
                r = 1
                param.requires_grad = True
        params_to_update = model_ft.parameters()
    weights = [
        v
        for k, v in sorted(kwargs["class_weight"].items(), key=lambda x: x[0])
    ]
    weights = torch.FloatTensor(weights).cuda()
    if cfg.training.mode == "classification":
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif cfg.training.mode == "regression":
        criterion = nn.MSELoss()
    elif cfg.training.mode == "ordinal_regression":
        criterion = ordinal_cost_fn

    optimizer_ft = optim.Adam(params_to_update, lr=cfg.training.lr)

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, factor=cfg.training.gamma, patience=cfg.training.epoch_reduce, mode="max")
    if cfg.training.mode in ["classification"]:  # , "ordinal_regression"]:
        print(f"weights: {weights}")

    model_ft = train(
        model_ft,
        cfg.model.name,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        batch_size=cfg.training.batch_size,
        num_epochs=cfg.training.epochs,
        device=device,
        quantize=cfg.model.quantize,
        mode=cfg.training.mode,
        weights=weights
    )
