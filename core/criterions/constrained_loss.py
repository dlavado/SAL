# %% 
import os
from typing import Iterator, Mapping, Tuple
import numpy as np
import torch
import torch.nn as nn
import sys


sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from core.constraints.constraint import Constraint




class Constrained_Loss(nn.Module):

    def __init__(self,
                 model_params:Iterator[Tuple[str, nn.Parameter]],
                 objective_function:nn.Module,
                 constraints:Mapping[str, Constraint]=None) -> None:
        
        """
        Constrained Loss.
        The objective function represents the primal problem, i.e., the data fidelity term.
        The constraints represent penalty terms that are added to the objective function and enforce 
        that the model parameters remain within the feasible set.


        Parameters
        ----------
        `objective_function` - torch.nn.Module:
            The objective function to be minimized. Models data fidelity.

        `constraints` - Dict[str, Constraint] or List[Constraint]:
            A dictionary of Constraint objects. Each constraint object contains a constraint of the optimization problem.

        """
        super().__init__()

        self.objective_function = objective_function

        self.model_params = {p_name: p for p_name, p in model_params}

        self.constraints = constraints

    
    def forward(self, y_pred, y_gt) -> torch.Tensor:

        data_fidelity = self.objective_function(y_pred, y_gt)

        if self.constraints is not None:            
            return data_fidelity + self._compute_constraint_norm()
        
        return data_fidelity
        
    def _compute_constraint_norm(self) -> float:
        """
        Computes the constraint norm w.r.t. theta_n.
        """

        constraint_norm = 0.0

        for constraint in self.constraints.values():
            # print(f"Constraint {constraint}: {constraint.evaluate_constraint(self.model_params.items())}")
            constraint_norm += constraint.evaluate_constraint(self.model_params.items())

        if constraint_norm > 100.0:
            return torch.log(torch.tensor(constraint_norm, device='cuda:0')) # log is used to stabilize the training process
        
        return constraint_norm



    def get_constraint_violation(self, model_params:Iterator[Tuple[str, nn.Parameter]]=None) -> float:
        """
        Computes the constraint violation w.r.t. theta_n.
        """
        if model_params is None:
            model_params = self.model_params
        else:
            model_params = {p_name: p for p_name, p in model_params}

        constraint_eval = torch.tensor(0.0, device='cuda:0')    

        for constraint in self.constraints.values():
            weight = constraint.weight if constraint.weight > 0 else 1.0
            constraint_eval += constraint.evaluate_constraint(model_params.items())/weight

        return constraint_eval