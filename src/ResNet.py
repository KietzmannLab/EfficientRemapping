import torch
import torchvision

class ResNet(torch.nn.Module):
    """
    Loads pre-trained ResNet 18, returns activations of a hidden layer (depending on what is commented in forward, not changable in runtime)
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
        torchvision.disable_beta_transforms_warning()
        self.preprocess = v2.Compose([
            v2.Resize(224, antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)

        # x = self.model.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.model.fc(x)
        return x
