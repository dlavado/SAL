
import torch
import torch.nn as nn

from core.models.FC_Classifier import Classifier_OutLayer
from core.models.lit_modules.lit_wrapper import LitWrapperModel
from torchmetrics.functional import accuracy



class Conv_Feature_Extractor(nn.Module):

    def __init__(self, in_channels=1, hidden_dim=128, kernel_size=3):
        super(Conv_Feature_Extractor, self).__init__()

        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size, 1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)


    def conv_block(self, in_channels, out_channels, kernel_size):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    


class CNN_Classifier(nn.Module):
    

    def __init__(self, in_channels=1, hidden_dim=128, ghost_sample:torch.Tensor = None, kernel_size=3, num_classes=10):
        """
        Convolutional Neural Network Classifier

        Parameters
        ----------

        in_channels: int
            Number of input channels

        hidden_dim: int
            Number of hidden dimensions

        ghost_sample: torch.Tensor
            Ghost sample to be used to produce feature dimensions for the FC layer

        kernel_size: int
            Kernel size for the convolutional layer

        num_classes: int
            Number of classes for the output layer
        """
        super().__init__()
        self.cnn = Conv_Feature_Extractor(in_channels, hidden_dim, kernel_size)
        ghost_shape = self.cnn(ghost_sample).shape
        # print(ghost_shape)
        # input("Press Enter to continue...")
        self.classifier = Classifier_OutLayer(torch.prod(torch.tensor(ghost_shape[1:])), num_classes)


    def forward(self, x):
        x = self.cnn(x)
        #print(x.shape) # batch_size, hidden_dim, new_height, new_width
        x = self.classifier(x)
        return x


class Lit_CNN_Classifier(LitWrapperModel):

    def __init__(self, 
                 in_channels=1, 
                 hidden_dim=128, 
                 kernel_size=3,
                 ghost_sample:torch.Tensor = None,
                 num_classes=10,
                 optimizer_name = 'adam', 
                 learning_rate=0.01, 
                 metric_initializer=None):

        model = CNN_Classifier(in_channels, hidden_dim, ghost_sample, kernel_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer)


    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output, dim=1)



