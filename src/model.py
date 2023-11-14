import torch.nn as nn

from torchvision import models

def build_model(pretrained=True, fine_tune=False, num_classes=10):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.mobilenet_v3_large(pretrained=pretrained)

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    # model.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes)
    # Change the final classification head and add dropout
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.5),  # Add dropout layer with a dropout probability of 0.5
        nn.Linear(in_features=1280, out_features=num_classes)
    )

    return model