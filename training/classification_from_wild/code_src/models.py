import torch.nn as nn
import timm
import torch
from code_src.custom_mobilenet_v2 import MobileNetV2
from code_src.custom_mobilenet_v2_ordinal import MobileNetV2 as MobileNetV2O
from torchsummary import summary
from torch.quantization import QuantStub, DeQuantStub


def get_model_uni(cfg, num_classes, ordinal=False):
    if cfg.model.src == "hub":
        model_ft = torch.hub.load(
            cfg.model.hub_repo, cfg.model.name, pretrained=True)
        if "classifier" in dir(model_ft):
            model_ft.classifier[1] = nn.Linear(
                model_ft.classifier[1].in_features, num_classes)
        elif "head" in dir(model_ft):
            model_ft.head = nn.Linear(
                model_ft.head.in_features, num_classes)
        else:
            raise NotImplementedError
    elif cfg.model.src == "timm":
        model_ft = timm.create_model(
            cfg.model.name, pretrained=True, num_classes=num_classes,
            drop_rate=float(cfg.model.dropout)
        )
    elif cfg.model.src == "custom":
        if cfg.model.name == "mobilenet_v2":
            if ordinal:
                model_ft = MobileNetV2O(num_classes=num_classes)
            else:
                model_ft = MobileNetV2(num_classes=num_classes)
            model = torch.hub.load(
                'pytorch/vision:v0.6.0', cfg.model.name, pretrained=True)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, num_classes)
            state_dict = model.state_dict()
            model_ft.load_state_dict(model.state_dict())
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return model_ft


class OrdinalRegression(nn.Module):
    def __init__(self, cfg, num_classes):
        super(OrdinalRegression, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        back_model = get_model_uni(cfg, num_classes - 1, ordinal=True)

        back_model.fuse_model()
        self.num_classes = num_classes

        self.fc = list(back_model.modules())[-1]
        back_model.classifier[1] = nn.Identity()
        self.back = back_model

        self.linear_1_bias = nn.Parameter(
            torch.zeros(self.num_classes - 1).float()
        )

    def forward(self, x):

        x = self.quant(x)

        x = self.back(x)

        x = x.view(x.size(0), -1)

        logits = self.fc(x)
        logits = self.dequant(logits)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)

        return logits, probas



def get_model(cfg, num_classes):
    if cfg.training.mode in ["classification", "regression"]:
        model = get_model_uni(cfg, num_classes)


        if cfg.model.quantize:
            # model = nn.Sequential(torch.quantization.QuantStub(),
            #                          model, torch.quantization.DeQuantStub())
            summary(model.cuda(), (3, *cfg.training.img_size))
            model.train()
            model.qconfig = torch.quantization.get_default_qat_qconfig(
                cfg.model.qat_qconfig)
            # make smart fuse
            model.fuse_model()
            model = torch.quantization.prepare_qat(model)

        if cfg.model.pretrained_path:
            model.load_state_dict(torch.load(cfg.model.pretrained_path))
            print(cfg.model.pretrained_path + ' loaded')
        return model

    elif cfg.training.mode == "ordinal_regression":
        model = OrdinalRegression(cfg, num_classes)
        if cfg.model.quantize:
            model.train()
            model.qconfig = torch.quantization.get_default_qat_qconfig(
                cfg.model.qat_qconfig)

            model = torch.quantization.prepare_qat(model)
        return model