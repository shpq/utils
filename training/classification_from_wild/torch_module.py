import time
from tqdm import tqdm
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import timm
from torchsummary import summary
from utils import create_folder, TrainConfig


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
    ):
        since = time.time()
        model = model.to(device)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
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
                    name = "models_checkpoints/{}/{}__torch_{}__epoch_{}_val_acc_{:.3f}_vs_{:.3f}_val_loss_{:.2f}_vs_{:.3f}.h5".format(
                        FLAGS.csv,
                        FLAGS.csv,
                        FLAGS.pretrained,
                        epoch,
                        epoch_acc,
                        train_acc,
                        epoch_loss,
                        train_loss,
                    )
                    create_folder(name)
                    torch.save(model, name)
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
    if FLAGS.saved == '-':
        model_ft = timm.create_model(
            FLAGS.pretrained, pretrained=True, num_classes=num_classes
        )
    else:
        model_ft = timm.create_model(
            FLAGS.pretrained, pretrained=False, num_classes=num_classes
        )
        
        model_ft = torch.load(TrainConfig.checkpoints_folder + FLAGS.csv + '/' + FLAGS.saved)
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
    print(summary(model_ft, input_size=input_size))
    print(f"weights : {weights}")
    model_ft = train(
        model_ft,
        FLAGS.pretrained,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        batch_size=FLAGS.batch_size,
        num_epochs=30,
        device=device,
    )
