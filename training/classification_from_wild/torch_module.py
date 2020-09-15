import time
from tqdm import tqdm
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import timm
# from torchsummary import summary
from utils import create_folder, TrainConfig
import os
from custom_models.mobilenet_v2_quantization import MobileNetV2Q


def train_torch(FLAGS, kwargs):

    dataloaders = kwargs["dataloaders"]
    dataset_sizes = {x: len(kwargs[x]) for x in ["train", "valid"]}

    def train(
        model,
        model_name,
        criterion,
        optimizer,
        scheduler,
        batch_size,
        num_epochs=25,
        device=None,
        quantize=False
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
                    inputs, labels = chunk
                    torch.cuda.empty_cache()
                    scale_value = 1 / batch_size / max(index, 1)
                    pbar.set_description(
                        "{} Loss: {:.4f} Acc: {:.4f}".format(
                            phase,
                            running_loss * scale_value,
                            running_corrects * scale_value,
                        )
                    )
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    torch.cuda.empty_cache()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        if quantize and phase == "valid":
                            quantized_model = torch.quantization.convert(
                                model.eval(), inplace=False)
                            quantized_model.eval()
                            inputs = inputs.to("cpu")
                            outputs = quantized_model(inputs)
                            outputs = outputs.to(device)
                        else:
                            outputs = model(inputs)

                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (preds == labels.data).float().sum()

                if phase == "train":
                    train_loss = running_loss / dataset_sizes[phase]
                    train_acc = running_corrects / dataset_sizes[phase]
                    scheduler.step()
                if quantize and phase == "valid":
                    if epoch > 2:
                        # Freeze quantizer parameters
                        model.apply(torch.quantization.disable_observer)
                    if epoch > 1:
                        # Freeze batch norm mean and variance estimates
                        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                print(running_corrects)
                print(dataset_sizes[phase])
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(
                        phase, epoch_loss, epoch_acc
                    )
                )

                # deep copy the model
                if phase == "valid":
                    name = os.path.join(
                        TrainConfig.checkpoints_folder,
                        FLAGS.csv,
                        "{}__{}__epoch_{}__val_acc_{:.3f}_vs_{:.3f}_val_loss_{:.2f}_vs_{:.2f}.pth".format(
                            FLAGS.csv,
                            FLAGS.pretrained,
                            epoch,
                            epoch_acc,
                            train_acc,
                            epoch_loss,
                            train_loss,
                        ))
                    create_folder(name)
                    if quantize:
                        torch.save(quantized_model.state_dict(), name)
                    else:
                        torch.save(model.state_dict(), name)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model, name)
        return model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    depth_trainable = FLAGS.depth_trainable
    num_classes = len(np.unique(kwargs['train'].label))
    if FLAGS.pretrained.startswith("hub_"):
        model_name = FLAGS.pretrained[4:]
        model_ft = torch.hub.load(
            'pytorch/vision:v0.6.0', model_name, pretrained=True)
        model_ft.classifier[1] = nn.Linear(
            model_ft.classifier[1].in_features, num_classes)
    elif FLAGS.pretrained == "custom_mobilenetv2":
        model_ft = MobileNetV2Q()
        pretrained_model = torch.hub.load(
            'pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True,
            num_classes=num_classes)
        pretrained_model.classifier[1] = nn.Linear(
            pretrained_model.classifier[1].in_features, num_classes)
        model_ft.load_state_dict(pretrained_model.state_dict())
    else:
        model_ft = timm.create_model(
            FLAGS.pretrained, pretrained=True, num_classes=num_classes
        )

    if FLAGS.quantize:
        model_ft = model_ft.to("cpu")
        model_ft.eval()
        model_ft.fuse_model()
        qconfig = FLAGS.qconfig if FLAGS.qconfig else "qnnpack"
        model_ft.qconfig = torch.quantization.get_default_qat_qconfig(qconfig)
        torch.quantization.prepare_qat(model_ft, inplace=True)

    if FLAGS.saved != '-':
        model_ft.load_state_dict(torch.load(os.path.join(
            TrainConfig.checkpoints_folder, FLAGS.csv, FLAGS.saved)))
        print(FLAGS.saved + ' loaded')

    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    if depth_trainable > 0:
        layers_to_choose = []
        for name, param in model_ft.named_parameters():
            if "conv" in name:
                layers_to_choose.append(name)
        layers_to_choose += [name]
        layer_train = layers_to_choose[
            -min(depth_trainable, len(layers_to_choose))
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
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer_ft = optim.SGD(params_to_update, lr=FLAGS.lr, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=FLAGS.epoch_reduce, gamma=FLAGS.gamma
    )

    input_size = (3, FLAGS.img_size, FLAGS.img_size)
    # print(summary(model_ft, input_size=input_size))
    print(f"weights: {weights}")

    model_ft = train(
        model_ft,
        FLAGS.pretrained,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        batch_size=FLAGS.batch_size,
        num_epochs=30,
        device=device,
        quantize=FLAGS.quantize,
    )
