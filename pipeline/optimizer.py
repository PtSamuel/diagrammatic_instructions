import pytorch_kinematics as pk
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import logit, read_yaml, draw_points_2D, draw_trajectory
from pipeline.ebm_trainer import SurfaceClassifier, PlanarClassifier

class Optimizer:

    def __init__(self, working_directory, config, chain, constraints=False):

        self.config = config

        if constraints:
            
            constraint_classifier_path = os.path.join(working_directory, config['io']['constraint_classifier_name'])
        
            try:
                constraint_classifier = PlanarClassifier().to('cuda')
                constraint_classifier.load_state_dict(torch.load(constraint_classifier_path))

                constraint_points_path = os.path.join(working_directory, config['io']['constraint_name'])
                self.constraint_points = np.load(constraint_points_path)

                self.constraint_classifier = constraint_classifier
                self.sigmoidless_constraint_classifier = nn.Sequential(*list(constraint_classifier.children())[0][:-1])

            except:
                print("constraints are turned on but no classifier is found at", constraint_classifier_path)
                self.constraint_points = None
                self.constraint_classifier = None
                self.sigmoidless_constraint_classifier = None
            
        else:
            self.constraint_points = None
            self.constraint_classifier = None
            self.sigmoidless_constraint_classifier = None

        roi_points_path = os.path.join(working_directory, config['io']['pointcloud_name'])

        # ROI
        classifier_path = os.path.join(working_directory, config['io']['classifier_name'])
        classifier = SurfaceClassifier().to('cuda')
        classifier.load_state_dict(torch.load(classifier_path))
        self.roi_points = np.load(roi_points_path)

        # Constraint
        self.classifier = classifier
        self.sigmoidless_classifier = nn.Sequential(*list(classifier.children())[0][:-1])

        self.chain = chain
        self.joint_limits = Optimizer.find_joint_angles_from_chain(chain)
        self.N = 1024

    @staticmethod
    def find_joint_angles_from_chain(chain):
        if chain is None:
            return None
        joint_limits_lo, joint_limits_hi = chain.get_joint_limits()
        joint_limits = list(zip(joint_limits_lo, joint_limits_hi))
        joint_names = [joint.name for joint in chain.get_joints()]
        return dict(zip(joint_names, joint_limits))

    def find_gradient(self, model, x):
        model.train()
        x = torch.clone(x.detach()).cuda()
        x.requires_grad = True
        output = model(x)
        output.backward()
        return x.grad

    def newtons(self, sigmoidless_classifier, start, threshold_prob=0.9707, eps=0.001, max_its=20):

        start = torch.clone(start)
        threshold = logit(threshold_prob)

        var = start.cuda()
        for i in range(max_its):
            grad = self.find_gradient(sigmoidless_classifier, var)
            energy = sigmoidless_classifier(var)
            
            mag = grad.norm()
            assert mag != 0, "gradient is zero"
            residual = energy - threshold
            step = residual / mag
            var -= step / mag * grad

            var = var.detach()

            if torch.abs(residual) <= eps:
                return var

        return None

    def compute_energy(self, sigmoidless_classifier, base_config, joint_configs):
        combined_configs = torch.cat((base_config.repeat(self.N, 1), joint_configs), 1)
        end_effectors = self.chain.forward_kinematics(combined_configs)
        end_positions = end_effectors.get_matrix()[:,0:3,3]
        sigmoidless_classifier.eval()
        return torch.mean(sigmoidless_classifier(end_positions))

    def compute_base_gradient(self, sigmoidless_classifier, base_config, joint_configs):
        copied = base_config.detach().clone().cuda()
        copied.requires_grad = True
        combined_configs = torch.cat((copied.repeat(self.N, 1), joint_configs), 1)
        end_effectors = self.chain.forward_kinematics(combined_configs)
        end_positions = end_effectors.get_matrix()[:,0:3,3]
        
        sigmoidless_classifier.train()
        energy = torch.mean(sigmoidless_classifier(end_positions))
        energy.backward()
        grad = copied.grad
        return energy.detach().clone(), grad.detach().clone()

    def line_search(self, sigmoidless_classifier, line, joint_configs):

        best_index = None
        highest_energy = None
        for i in range(len(line)):
            base_cur = line[i]
            energy = self.compute_energy(sigmoidless_classifier, base_cur, joint_configs)
            if highest_energy is None or energy > highest_energy:
                highest_energy = energy
                best_index = i

        return best_index, highest_energy
    
    def optimize(self, ic=None):

        joint_low = torch.tensor([self.joint_limits[i][0] for i in self.joint_limits.keys() if i.find("mobile") == -1], device='cuda')
        joint_high = torch.tensor([self.joint_limits[i][1] for i in self.joint_limits.keys() if i.find("mobile") == -1], device='cuda')
        print(joint_low, joint_high)

        torch.manual_seed(1984)
        rand = torch.rand((self.N, len(joint_low)), device='cuda')
        joint_configs = rand * (joint_high - joint_low) + joint_low
        joint_configs.requires_grad = False

        num_segs = 64
        line_length = 0.5

        if ic is None:
            base_var = torch.tensor([0.0, 0, 0, 0]).cuda()
            base_var[0:3] = torch.tensor(self.roi_points.mean(1))
        else:
            base_var = ic.detach().clone().flatten().cuda()

        base_var.requires_grad = False

        bases = [(None, base_var.clone())]
        energies = []

        for i in range(40):

            energy, grad = self.compute_base_gradient(self.sigmoidless_classifier, base_var, joint_configs)
            if grad.norm() < 0.1:
                break
            energies += [energy]
            
            base_var += grad * 0.005

            z_lo, z_hi = self.joint_limits['mobile_joint_z']
            base_var[2] = torch.clip(base_var[2], z_lo, z_hi)
            base_var_unprojected = base_var.clone()

            threshold_prob = 0.95
            threshold_energy = logit(threshold_prob)

            if self.sigmoidless_constraint_classifier is not None and \
                    self.sigmoidless_constraint_classifier(base_var[0:2]) < threshold_energy:
                projected_xy = self.newtons(self.sigmoidless_constraint_classifier, base_var[0:2], threshold_prob=threshold_prob)
                base_var[0:2] = projected_xy
                bases += [(base_var_unprojected, base_var.clone())]
            else:
                bases += [(None, base_var.clone())]

            if (i + 1) % 10 == 0:
                print(base_var)
        
        return bases, energies