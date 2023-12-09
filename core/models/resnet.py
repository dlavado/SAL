
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchmetrics.functional import accuracy



from core.models.lit_modules.lit_wrapper import LitWrapperModel



def create_model(in_channels, num_classes, pretrained=False):
    if pretrained:
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.maxpool = nn.Identity()
        return model
    

    model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
    model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


from torch.optim.lr_scheduler import ReduceLROnPlateau
class LitResnet(LitWrapperModel):

    def __init__(self, 
                 num_classes: int, 
                 in_channels: int,
                 optimizer_name: str,
                 pretrained=False,
                 learning_rate=0.01, 
                 metric_initializer=None):

        model = create_model(in_channels, num_classes, pretrained=pretrained)
        criterion = nn.CrossEntropyLoss()
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, num_classes=num_classes)


    def forward(self, x):
        return self.model(x)
    
    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output, dim=1)
    

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(
    #         self.model.parameters(),
    #         lr=self.hparams.learning_rate,
    #         momentum=0.9,
    #         weight_decay=5e-4,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": ReduceLROnPlateau(optimizer, mode="max", factor=0.01, patience=5, verbose=False),
    #             "monitor": "val_Accuracy",
    #             "frequency": 1,
    #             # If "monitor" references validation metrics, then "frequency" should be set to a
    #             # multiple of "trainer.check_val_every_n_epoch".
    #             },
    #         }
    

    

