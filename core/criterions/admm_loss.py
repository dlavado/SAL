

from typing import Iterable, Iterator, Mapping, Tuple
import numpy as np
import torch
import torch.nn as nn
import sys


sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.constraints.constraint import Constraint



class ADMM_Loss(nn.Module):

    def __init__(self, objective_function:nn.Module, 
                 constraints:Mapping[str, Constraint], 
                 model_params:Iterator[Tuple[str, nn.Parameter]], 
                 init_penalty = 1.0, 
                 penalty_update_factor = 1.1,
                 max_penalty = 10,
                 stepsize = 10,
                 stepsize_update_factor = 0.5) -> None:
        """

        Alternating Direction Method of Multipliers (ADMM) Loss.
        This loss is utilized to solve the following optimization problem:\n
  

           \minimize_{\.theta} `objective_function`(\.theta) + `constraint_function`(\.theta)

        = \minimize_{\.theta, \psi} `objective_function`(\.theta) + `constraint_function`(\psi)
                s.t. \psi = \.theta    

        In this setting, the ADMM algorithm is of the following form:

        Repeat until convergence:

            1. \.theta^{k+1} = argmin_{\.theta} `objective_function`(\.theta) + \.rho/2 ||\.theta - \psi^k + \.lambda^k||^2

            2. \psi^{k+1} = argmin_{\psi} `constraint_function`(\psi) + \.rho/2 ||\.theta^{k +1} - \psi + \.lambda^k||^2

            3. \.lambda^{k+1} = \.lambda^k + \.rho(\.theta^{k+1} - \psi^{k+1})

        Parameters
        ----------

        `objective_function` - torch.nn.Module:
            The objective function to be minimized. Models data fidelity.

        `constraints` - List[Constraint]:
            A list of Constraint objects. Each constraint object contains a constraint os the optimization problem.

        `model` - nn.Module:
            The model that is being optimized.

        `penalty_factor` - float:
            (a.k.a., \.rho.)
            The penalty factor used to enforce the \.theta = \psi constraint. The higher the penalty, the stiffer the opt problem becomes.

        """
        super().__init__()

        self.objective_function = objective_function
        self.data_fid_weight = 1.5 # weight of the data fidelity term in the loss function.

        self.constraints:Mapping[str, Constraint] = constraints

        self.constraint_weight = max([c.weight for c in constraints.values()])

        self.model_params = {}

        self.psi, self.lag_multipliers = {}, {} # \psi and \lambda
        self.theta_k = {} # \.theta^{k} 
        for key, theta_p in model_params:
            # \psi are the primal optimization variables where the constraints are enforced.
            self.psi[key] = torch.tensor(theta_p.clone(), requires_grad=False, device='cuda:0')
            # Lagrangian multipliers of the optimization variables. a.k.a., \lambda.
            # Intuitively, these can be thought as an offset between \.theta and \psi.
            self.lag_multipliers[key] = torch.zeros_like(theta_p, requires_grad=False, device='cuda:0')

            # \.theta^{k} are the optimization variables of the ADMM algorithm.
            self.theta_k[key] = torch.tensor(theta_p.clone(), requires_grad=False, device='cuda:0')

            self.model_params[key] = theta_p


        self.current_constraint_norm  = self.get_constraint_violation()
        self.best_constraint_norm = self.current_constraint_norm

        self.penalty_factor = init_penalty  # a.k.a., \.rho
        self.max_penalty = max_penalty

        self.constraint_tolerance = 5
        self.constraint_tolerance_counter = 0

        self.stepsize = stepsize
        self.penalty_update_factor = penalty_update_factor
        self.stepsize_update_factor = stepsize_update_factor


    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor):
        """
        Computes the forward pass of the ADMM loss.
        The consists in an update of the main optimization variables \.theta (that is, step 1. in the ADMM algorithm).~

        Parameters
        ----------

        `y_pred` - torch.Tensor:
            The predicted output of the model.

        `y_gt` - torch.Tensor:
            The ground truth output.

        """
        
        return  self.data_fid_weight   * self.objective_function(y_pred, y_gt) + \
                self.constraint_weight * ( self.ADMM_regularizer(self.model_params.items()) + self.Stochastic_ADMM_regularizer(self.model_params.items()) )
    
    
    def Stochastic_ADMM_regularizer(self, theta_n:Iterator[Tuple[str, nn.Parameter]]):
        """
        || \.theta_k+1 - \.theta_k ||_2^2 / (2 * stepsize)
        """

        diff_norm = [torch.norm(p_value - self.theta_k[p_name], p=2) for p_name, p_value in theta_n]

        return sum(diff_norm) / (2*self.stepsize)
    
    
    def ADMM_regularizer(self, theta_n:Iterator[Tuple[str, nn.Parameter]]):
        """
        || \.theta - \psi + \.lambda ||_2^2
        """
        pows = [torch.norm(p_value - self.psi[p_name] + self.lag_multipliers[p_name], 2) for p_name, p_value in theta_n] # tradtional ADMM formulation


        return (self.penalty_factor/2) * sum(pows) # L2 norm

    
    
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


    def update_psi(self):
        """

        Updates the optimization variables \psi (that is, step 2. in the ADMM algorithm).

        Since the constraints we are considering are linear, this problem has a closed-form solution.
        Specifically, the solution is given by:

            \psi_{k+1} = \.theta_{k+1} + \.lambda_{k+1} if `constraints`(\.theta_{k+1}) = 0 else project onto the feasible set.
        """
        #print(f"psi: {np.array(list(self.psi.values()))}")
        for constraint in self.constraints.values():
            updated_psi = constraint.update_psi(self.psi, self.theta_k, self.lag_multipliers)

            for key in updated_psi:
                self.psi[key] = updated_psi[key]

                   

    def update_lag_multipliers(self):
        """
        Updates the Lagrangian multipliers \.lambda (that is, step 3. in the ADMM algorithm).
        """

        for key in self.lag_multipliers:
            # Lagrangian multipliers are updated by adding the offset between \.theta and \psi. These should be non-negative.
            self.lag_multipliers[key] = self.penalty_factor * (self.theta_k[key] - self.psi[key])
            # self.lag_multipliers[key] = self.lag_multipliers[key] + self.penalty_factor * (self.theta_k[key] - self.psi[key])
        

    def update_stepsize(self):
        self.stepsize = self.stepsize * self.stepsize_update_factor
        
    def update_penalty(self, factor=None):
        if factor is None:
            factor = self.penalty_update_factor
        else:
            self.penalty_factor = min(self.penalty_factor * factor, self.max_penalty)

        if self.constraint_tolerance_counter > self.constraint_tolerance:
            self.penalty_factor = min(self.penalty_factor * factor, self.max_penalty)
            self.constraint_tolerance_counter = 0
        else:
            self.constraint_tolerance_counter += 1

    def get_lag_multipliers(self):
        return self.lag_multipliers

    def get_best_constraint_norm(self):
        return self.best_constraint_norm
    
    def update_best_constraint_norm(self):
        """
        Updates the best constraint norm
        """
        self.best_constraint_norm = min(self.best_constraint_norm, self.get_constraint_violation())

    
    def update_theta_k(self):
        self.theta_k = {key: torch.tensor(value.clone(), requires_grad=False, device='cuda:0') for key, value in self.model_params.items()}

    def get_theta_k(self):
        return self.theta_k
    
    def get_psi(self):
        return self.psi

