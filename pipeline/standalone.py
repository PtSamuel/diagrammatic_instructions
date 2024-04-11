from pipeline.ebm_trainer import PointDataset, EBMTrainer, SurfaceClassifier
from optimizer import Optimizer
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import get_sampling_space

def get_sigmoidless(classifier):
    return nn.Sequential(*list(classifier.children())[0][:-1])

class StandaloneTrainer(EBMTrainer):

    def __init__(self, pointcloud, num_samples=262144, batch_size=1024, epochs=1, sampling_space=None):

        self.pointcloud = pointcloud
        if sampling_space is None:
            sampling_space = get_sampling_space(pointcloud, 1)
        self.dataset = PointDataset(
            self.pointcloud,
            num_samples=num_samples, 
            sampling_space=sampling_space
        )
        
        self.classifier = SurfaceClassifier().to('cuda')
        self.train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.AdamW(self.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        self.Loss = nn.BCELoss()

        self.epochs = epochs

    def get_sigmoidless(self):
        return get_sigmoidless(self.classifier)

    def save_classifier(self, classifier_save_path):
        print("saving model at:", classifier_save_path)
        torch.save(self.classifier.state_dict(), classifier_save_path)

class StandaloneOptimizer(Optimizer): 

    def __init__(self, classifier, pointcloud, chain, config, constraint_classifier=None):

        self.config = config
        self.classifier = classifier
        self.sigmoidless_classifier = get_sigmoidless(classifier)
        self.roi_points = pointcloud

        if constraint_classifier is not None:
            self.constraint_classifier = constraint_classifier
            self.sigmoidless_constraint_classifier = get_sigmoidless(constraint_classifier)
        else:
            self.constraint_classifier = None
            self.sigmoidless_constraint_classifier = None

        self.joint_limits = Optimizer.find_joint_angles_from_chain(chain)
        self.chain = chain
        self.N = 1024