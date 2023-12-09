
import torch
import torch.nn as nn

class ElasticNetRegularization(nn.Module):
    def __init__(self, alpha=0.5, l1_ratio=0.5):
        """
        Initialize the Elastic Net regularization module.

        Args:
            alpha (float): The overall regularization strength. A higher value emphasizes
                regularization.
            l1_ratio (float): The ratio between L1 (Lasso) and L2 (Ridge) regularization.
                For l1_ratio = 0, only L2 regularization is applied. For l1_ratio = 1, only
                L1 regularization is applied.
        """
        super(ElasticNetRegularization, self).__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def forward(self, parameters):
        """
        Compute Elastic Net regularization loss for a set of parameters.

        Args:
            parameters (iterable): Iterator containing model parameters.

        Returns:
            torch.Tensor: Elastic Net regularization loss.
        """
        l1_reg = 0.0
        l2_reg = 0.0

        for param in parameters:
            l1_reg += torch.norm(param, p=1)
            l2_reg += torch.norm(param, p=2)

        total_reg = self.alpha * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg)
        return total_reg
    

if __name__  == '__main__':

    # Example usage:
    # Create an instance of ElasticNetRegularization and add it to your model's optimizer
    # elastic_net_reg = ElasticNetRegularization(alpha=0.01, l1_ratio=0.5)

    # Assuming 'my_optimizer' is your optimizer and 'my_parameters' is an iterator of parameters
    # my_optimizer = torch.optim.SGD(my_parameters, lr=0.01, momentum=0.9)
    # my_optimizer.add_param_group({'params': elastic_net_reg.parameters()})

    # # During training, you can compute and backpropagate the regularization loss along with your main loss
    # for inputs, targets in dataloader:
    #     my_optimizer.zero_grad()
    #     outputs = my_model(inputs)
    #     loss = criterion(outputs, targets)
        
    #     # Compute the Elastic Net regularization loss and add it to the main loss
    #     reg_loss = elastic_net_reg(my_parameters)
    #     total_loss = loss + reg_loss

    #     total_loss.backward()
    #     my_optimizer.step()
    pass