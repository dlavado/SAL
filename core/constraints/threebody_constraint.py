

from .constraint import Constraint
from typing import Iterator, Mapping, Tuple, Union
import torch.nn as nn
import torch
from torch.autograd import Variable


def grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input."""
    return torch.autograd.grad(
        outputs, 
        inputs, 
        grad_outputs=torch.ones_like(outputs), 
        create_graph=True
    )


class ThreeBodyConstraint(Constraint):

    def __init__(self, weight: float) -> None:
        super().__init__(weight, f"Three Body Constraint")

    
    def three_body_gravity(self, model_out, t):
        """
        This functions calculates the gravity governing equations of three bodies in space.
        Specifically, it calculates the acceleration of each body in the x and y directions w.r.t. time.

        Parameters
        ----------

        `model_pred` - torch.Tensor:
            The model predictions, which are the positions of the three bodies.


        `t` - torch.Tensor:
            The time tensor.

        Returns
        -------

        `f` - torch.Tensor:
            The acceleration of each body in the x and y directions w.r.t. time.
        """

        # print(f"--Loc: three_body_gravity()--")

        # print(f"model_out shape: {model_out.shape} req_grad: {model_out.requires_grad}")


        u_x1 = model_out[:, 0:1]
        u_y1 = model_out[:, 1:2]
        u_x2 = model_out[:, 2:3]
        u_y2 = model_out[:, 3:4]
        u_x3 = model_out[:, 4:5]
        u_y3 = model_out[:, 5:6]

        # print(f"shapes: {u_x1.shape, u_y1.shape, u_x2.shape, u_y2.shape, u_x3.shape, u_y3.shape}")
        # print(f"req_grad: {u_x1.requires_grad, u_y1.requires_grad, u_x2.requires_grad, u_y2.requires_grad, u_x3.requires_grad, u_y3.requires_grad}")

        # First Derivative
        u_x1_t = grad(u_x1, t)[0]
        u_y1_t = grad(u_y1, t)[0]
        u_x2_t = grad(u_x2, t)[0]
        u_y2_t = grad(u_y2, t)[0]
        u_x3_t = grad(u_x3, t)[0]
        u_y3_t = grad(u_y3, t)[0]

        # Second Derivative
        u_x1_tt = grad(u_x1_t, t)[0]
        u_y1_tt = grad(u_y1_t, t)[0]
        u_x2_tt = grad(u_x2_t, t)[0]
        u_y2_tt = grad(u_y2_t, t)[0]
        u_x3_tt = grad(u_x3_t, t)[0]
        u_y3_tt = grad(u_y3_t, t)[0]

        u_r1_tt = torch.cat([u_x1_tt, u_y1_tt], dim=1)
        u_r2_tt = torch.cat([u_x2_tt, u_y2_tt], dim=1)
        u_r3_tt = torch.cat([u_x3_tt, u_y3_tt], dim=1)

        u_r1 = model_out[:, 0:2]
        u_r2 = model_out[:, 2:4]
        u_r3 = model_out[:, 4:6]

        # Functions 
        # input(f"shapes: {u_r1_tt.shape}, {u_r1.shape}, {u_r2.shape}, {u_r3.shape}")
        f1 = torch.sum(u_r1_tt, dim=-1) + torch.sum(u_r1 - u_r2) / torch.norm((u_r1 - u_r2), p=2) + torch.sum(u_r1 - u_r3) / torch.norm((u_r1 - u_r3), p=2)
        f2 = torch.sum(u_r2_tt, dim=-1) + torch.sum(u_r2 - u_r3) / torch.norm((u_r2 - u_r3), p=2) + torch.sum(u_r2 - u_r1) / torch.norm((u_r2 - u_r1), p=2)
        f3 = torch.sum(u_r3_tt, dim=-1) + torch.sum(u_r3 - u_r2) / torch.norm((u_r3 - u_r2), p=2) + torch.sum(u_r3 - u_r1) / torch.norm((u_r3 - u_r1), p=2)

        f = f1 + f2 + f3
        return f

    def constraint_function(self, model_out, time) -> float:
        """
        The constraint function.
        """
        return torch.sum(self.three_body_gravity(model_out, time)**2) # sum of squares error

    
    def evaluate_constraint(self, model_out, time) -> float:
        """
        Evaluates the constraint function.

        If the constraint defined by `constraint function` is satisfied, then its evaluation is 0. 
        Else, it is a positive number (follows the flag logic in `_flag_constraint()`)
        """
        return self.weight*self.constraint_function(model_out, time)
    

    def _constraint_on_params(self, model_parameters: Iterator[Tuple[str, nn.Parameter]]) -> dict:
        """
        Evaluates the constraint function on the model parameters.
        """
        # no constraint is directly imposed on the model parameters
        return {param_name: 0 for param_name, param in model_parameters}
    
    def update_psi(self, psi:dict, 
                         theta: dict, 
                         lag_multiplier: dict) -> dict:
        """
        Updates the \psi values of the ADMM algothm.
        """
        eval = {}
        for p_name, p_value in psi.items():
                # the constraint is always satisfied w.r.t. the model parameters
                eval[p_name] = theta[p_name] + lag_multiplier[p_name]

        return eval



    def auglag_forward(self, model, model_out, time) -> dict:
        """
        Forward pass for the augmented lagrangian method.
        """

        constraint_loss = self.evaluate_constraint(model_out, time)

        # zero the gradients
        model.zero_grad()

        constraint_loss.backward(retain_graph=True) # compute gradients

        # with torch.no_grad(): # regular gradient descent
        #     for param in model.parameters():
        #         param -= param.grad * lr

        model_update = {param_name: - param.grad for param_name, param in model.named_parameters()}

        return model_update, constraint_loss.item()
    

    def admm_forward(self, model, psi, theta, lag_multiplier, model_out, time) -> dict:

        constraint_loss = self.evaluate_constraint(model_out, time)

        constraint_loss.backward()

        param_update = {param_name: - param.grad for param_name, param in model.named_parameters()}

        eval = {}
        for p_name, _ in psi.items():
                # if the constraint it satisfied, then the evaluation is 0
                if torch.sum(param_update.get(p_name, torch.zeros((1,)))) == 0:
                    eval[p_name] = theta[p_name] + lag_multiplier[p_name]
                else:
                    eval[p_name] = param_update[p_name]

        return eval, constraint_loss.item()