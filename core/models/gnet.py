

from typing import Any
import torch 
from torch import nn
from torch.nn import functional as F

from core.models.FC_Classifier import Classifier_OutLayer



class Gaussian_Kernel(nn.Module):

    def __init__(self, kernel_size=(3,3), factor=1, stddev=1, mean=0) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.factor = factor
        self.stddev = stddev
        self.mean = mean
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.kernel = self.compute_kernel()

    def gaussian(self, x:torch.Tensor, epsilon=1e-8) -> torch.Tensor:
        center = torch.tensor([(self.kernel_size[0]-1)/2, (self.kernel_size[1]-1)/2], dtype=torch.float, requires_grad=True, device=self.device)

        x_c = x - center # Nx2
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        gauss_dist = x_c_norm**2 - (self.mean + epsilon)**2 

        return self.factor*torch.exp((gauss_dist**2) * (-1 / (2*(self.stddev + epsilon)**2)))
    
    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor - torch.sum(tensor) / torch.prod(torch.tensor(self.kernel_size)) 
    
    def compute_kernel(self):

        floor_idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[0], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True))
            ).T.reshape(-1, 2) # Nx2 vector form of the indices
       
       
        kernel = self.gaussian(floor_idxs)
        kernel = self.sum_zero(kernel)
        kernel = torch.t(kernel).view(1, *self.kernel_size) # CxHxW

        #assert kernel.requires_grad

        return kernel


class Gaussian_Kernels(nn.Module):

    def __init__(self, kernel_size, factors:torch.Tensor, stds:torch.Tensor, means:torch.Tensor) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.factors = factors
        self.stds = stds
        self.means = means # shape = (num_ops, num_gaussians) e.g. (128, 10)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.kernel = self.compute_kernel()

    def gaussians(self, x:torch.Tensor, epsilon=1e-6) -> torch.Tensor:
        center = torch.tensor([(self.kernel_size[0]-1)/2, (self.kernel_size[1]-1)/2], dtype=torch.float, requires_grad=True, device=self.device)
        try:
            x_c = x - center # Nx2
            x_c_norm = torch.linalg.norm(x_c, dim=1).to(self.device) # N
            gauss_dist = x_c_norm[None, None, :]**2 - (self.means[:, :, None]**2).to(self.device) # 128x10xN 
            gauss = torch.normal(gauss_dist, torch.relu(self.stds[:, :, None]) + epsilon)
            # gauss = torch.exp((gauss_dist**2) * (-1 / (2*(self.stds[:, :, None] + epsilon)**2))) # 128x10xN
            # gauss = torch.nan_to_num(gauss, nan=0.0, posinf=0.0, neginf=0.0)
            #return self.factors[:, :, None]*gauss
        
        except RuntimeError as e:
            print(e)
            print(f"stds: {self.stds.shape}\n {self.stds}")
            print(f"means: {self.means.shape}\n {self.means}")
            print(f"gauss dist: {gauss_dist.shape}\n {gauss_dist}")
            # print((gauss_dist**2) * (-1 / (2*(self.stds[:, :, None] + epsilon)**2)))
            gauss = torch.normal(gauss_dist.cpu(), torch.relu(self.stds[:, :, None].cpu()) + epsilon).to(self.device)

        return gauss

    
    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        # tensor shape = (num_ops, num_gaussians, N)
        # apply zero sum to last dimension
        return tensor - torch.sum(tensor, dim=-1, keepdim=True) / torch.prod(torch.tensor(self.kernel_size))
    
    def compute_kernels(self):

        floor_idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[0], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True))
            ).T.reshape(-1, 2) # Nx2 vector form of the indices
       
       
        kernel = self.gaussians(floor_idxs) # num_opsxnum_gaussiansxN
        kernel = self.sum_zero(kernel)
        kernel = kernel.unsqueeze(-1) # num_opsxnum_gaussiansxNx1
        num_ops, num_gaussians, _, _ = kernel.shape
        kernel = torch.transpose(kernel, -2, -1).view(num_ops*num_gaussians, 1, *self.kernel_size) # CxHxW

        return kernel # shape = (num_ops*num_gaussians, 1, H, W)




class IENEO_Fam(nn.Module):

    def __init__(self, num_operators:int, num_gaussians:int, kernel_size:tuple) -> None:
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        self.num_operators = num_operators
        self.num_gaussians = num_gaussians
        self.kernel_size = kernel_size

        # centers are the means of the gaussians
        self.mus = nn.Parameter(torch.rand((num_operators, num_gaussians), requires_grad=True, device=self.device))

        # sigmas are the standard deviations of the gaussians, so they must be positive
        self.sigmas = torch.randn((num_operators, num_gaussians), requires_grad=True, device=self.device)
        self.sigmas = torch.abs(self.sigmas) + 1e-8 # make sure they are positive
        self.sigmas = nn.Parameter(self.sigmas).to(self.device)

        # convex combination weights of the gaussians
        self.lambdas = torch.rand((num_operators, num_gaussians), requires_grad=True, device=self.device)
        self.lambdas = torch.softmax(self.lambdas, dim=1) # make sure they sum to 1
        self.lambdas = nn.Parameter(self.lambdas, requires_grad=True)   

    def maintain_convexity(self):
        with torch.no_grad():
            self.lambdas = nn.Parameter(torch.relu(torch.tanh(self.lambdas)), requires_grad=True).to(self.device)
            self.lambdas[:, -1] = 1 - torch.sum(self.lambdas[:, :-1], dim=1)
            # self.lambdas = nn.Parameter(lambdas, requires_grad=True)
        # self.lambdas = nn.Parameter(torch.relu(torch.tanh(self.lambdas)), requires_grad=True)
        # self.lambdas[:, -1] = 1 - torch.sum(self.lambdas[:, :-1], dim=1)
        # self.lambdas = nn.Parameter(torch.softmax(self.lambdas, dim=1)) # make sure they sum to 1


    def compute_kernels(self):
        return Gaussian_Kernels(self.kernel_size, torch.ones_like(self.sigmas), self.sigmas, self.mus).compute_kernels()

    def forward(self, x):

        self.kernels = self.compute_kernels()
        # kernels.shape = (num_operators*num_gaussians, 1, kernel_size[0], kernel_size[1])
        
        # print(kernels.shape, kernels.device)

        # apply the kernels to the input
        self.kernels = torch.cat([self.kernels for _ in range(x.shape[1])], dim=1)
        conv = F.conv2d(x, self.kernels) # shape = (B, num_operators*num_gaussians, H, W)
        conv_view = conv.view(conv.shape[0], self.num_operators, self.num_gaussians, *conv.shape[2:]) # shape = (B, num_operators, num_gaussians, H, W)

        # print(conv.shape)
        # apply the convex combination weights
        #cvx_comb = torch.sum(conv_view*self.lambdas[None, :, :, None, None], dim=2) # shape = (B, num_operators, H, W)
        cvx_comb = torch.mul(conv_view, self.lambdas[None, :, :, None, None]) # multiply the gaussians by the weights
        cvx_comb = torch.sum(cvx_comb, dim=2) # shape = (B, num_operators, H, W)

        return cvx_comb





class IENEO_Layer(nn.Module):


    def __init__(self, hidden_dim=128, kernel_size=(3, 3), gauss_hull_size=10) -> None:
        """
        IENEO Layer is a layer that applies a gaussian hull to the input and then max pools it

        A gaussian hull is a convex combination of gaussian kernels, where the weights of the convex combination are learned
        This grants the layer equivariance with respect to isomorphisms, so Euclidean transformations (i.e., translations, rotations, reflections, etc.)

        Parameters
        ----------

        hidden_dim: int
            The number of gaussians hulls to  instantiate
        
        kernel_size: tuple
            The size of the gaussian kernels to use

        gauss_hull_num: int
            The number of gaussians to use in each gaussian hull
        """
        super().__init__()
    
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ieneo = IENEO_Fam(hidden_dim, gauss_hull_size, kernel_size).to(self.device)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def get_cvx_coeffs(self):
        return self.ieneo.lambdas

    
    def maintain_convexity(self):
        self.ieneo.maintain_convexity()
    
    def forward(self, x):
        x = self.ieneo(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
        




class IENEONet(nn.Module):


    def __init__(self, in_channels=1, hidden_dim=128,ghost_sample:torch.Tensor = None, gauss_hull_size=5, kernel_size=(3,3), num_classes=10):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feature_extractor = IENEO_Layer(hidden_dim, kernel_size, gauss_hull_size).to(device)
        ghost_shape = self.feature_extractor(ghost_sample.to(device)).shape
        self.classifier = Classifier_OutLayer(torch.prod(torch.tensor(ghost_shape[1:])), num_classes).to(device)

    def print_cvx_combination(self) -> str:
        coeffs = self.feature_extractor.get_cvx_coeffs()
        return f"mean cvx combination: {torch.sum(coeffs, dim=1).mean():.3f} | mean lambda: {coeffs.mean():.3f} | max lambda: {coeffs.max():.3f} | min lambda: {coeffs.min():.3f}"

    def maintain_convexity(self):
        self.feature_extractor.maintain_convexity()

    def nonneg_loss(self):
        """
        TODO: NOT WORKING
        Returns the sum of the negative values of the sigmas and lambdas
        """
        return torch.sum(torch.relu(-self.feature_extractor.ieneo.sigmas)) + torch.sum(torch.relu(-self.feature_extractor.ieneo.lambdas))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

  

    
        


