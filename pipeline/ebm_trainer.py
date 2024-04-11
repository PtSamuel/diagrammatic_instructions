import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import read_yaml, draw_pointcloud, draw_points_2D, visualize_classifier, inflate_box, visualize_planar_classifier, find_bounding_box, get_sampling_space

class SurfaceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PointDataset(Dataset):
    
    def __init__(self, pointcloud, num_samples=1024, positive_prob=0.5, sampling_space=None):
        
        self.num_points = pointcloud.shape[1]
        assert len(pointcloud.shape) == 2 and pointcloud.shape[0] == 3, "pointcloud must be shape [N, 3]"
        assert self.num_points > 0, "pointcloud is empty"

        pointcloud = pointcloud.astype(np.float32)
        
        self.pointcloud = torch.tensor(pointcloud)
        self.num_samples = num_samples
        self.positive_prob = positive_prob
        
        xmax = pointcloud[0, :].max()
        xmin = pointcloud[0, :].min()
        ymax = pointcloud[1, :].max()
        ymin = pointcloud[1, :].min()
        zmax = pointcloud[2, :].max()
        zmin = pointcloud[2, :].min()
        
        self.bounding_box = torch.tensor([
            [xmin, ymin, zmin],
            [xmax, ymax, zmax]
        ])
        
        if sampling_space is None:
            self.sampling_space = inflate_box(self.bounding_box, factor=5)
        else:
            assert sampling_space.shape == (2, 3), "shape of sampling space must be [2, 3]"
            self.sampling_space = sampling_space
    
    def draw_sample(self):
        rand_index = np.random.randint(self.num_points)
        return self.pointcloud[:, rand_index]
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        if torch.rand(1) < self.positive_prob:
            return self.draw_sample(), torch.tensor(1)
        else:
            sample = self.sampling_space[0] + torch.rand(3) * (self.sampling_space[1] - self.sampling_space[0])
            return sample, torch.tensor(0)

class PlanarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PlanarDataset(Dataset):
    
    def __init__(self, pointcloud, num_samples=1024, positive_prob=0.5, sampling_space=None):
        
        self.num_points = pointcloud.shape[1]
        assert len(pointcloud.shape) == 2, "pointcloud must be shape [3, N] or [2, N]"
        assert pointcloud.shape[0] == 2 or pointcloud.shape[0] == 3, "pointcloud must be shape [3, N] or [2, N]"
        assert self.num_points > 0, "pointcloud is empty"

        pointcloud = pointcloud.astype(np.float32)
        pointcloud = torch.tensor(pointcloud)
        pointcloud = pointcloud[0:2, :]
        
        self.pointcloud = pointcloud
        self.num_samples = num_samples
        self.positive_prob = positive_prob
        
        xmax = pointcloud[0, :].max()
        xmin = pointcloud[0, :].min()
        ymax = pointcloud[1, :].max()
        ymin = pointcloud[1, :].min()
        
        self.bounding_box = torch.tensor([
            [xmin, ymin],
            [xmax, ymax]
        ])
        
        if sampling_space is None:
            self.sampling_space = inflate_box(self.bounding_box, factor=5)
        else: 
            assert sampling_space.shape == (2, 2), "shape of sampling space must be [2, 2]"
            self.sampling_space = sampling_space
    
    def draw_sample(self):
        rand_index = np.random.randint(self.num_points)
        return self.pointcloud[:, rand_index]
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        if torch.rand(1) < self.positive_prob:
            return self.draw_sample(), torch.tensor(1)
        else:
            sample = self.sampling_space[0] + torch.rand(2) * (self.sampling_space[1] - self.sampling_space[0])
            return sample, torch.tensor(0)

class EBMTrainer:

    def __init__(self, working_directory, config):

        self.config = config

        pointcloud_path = os.path.join(working_directory, config['io']['pointcloud_name'])
        self.pointcloud = np.load(pointcloud_path)
        self.dataset = PointDataset(self.pointcloud, num_samples=config['train']['num_samples'], 
                                    sampling_space=inflate_box(find_bounding_box(self.pointcloud), factor=0.3))
                                    # sampling_space=torch.tensor([[-5.0, -5, -5], [5, 5, 5]]))
        
        self.classifier = SurfaceClassifier().to('cuda')
        self.train_loader = DataLoader(self.dataset, batch_size=self.config['train']['batch_size'], shuffle=True)
        self.optimizer = optim.AdamW(self.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        self.Loss = nn.BCELoss()

        self.epochs = config['train']['epochs']

        self.classifier_save_path = os.path.join(working_directory, config['io']['classifier_name'])

    def draw_pointcloud(self):
        draw_pointcloud(self.pointcloud, cube=True)

    def visualize_classifier(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        bounding_box = inflate_box(self.dataset.bounding_box, 0.5)
        visualize_classifier(ax, self.classifier, self.dataset, num_points=10000, box=bounding_box, cube=True)
        plt.show()

    def train(self):
       
        for epoch in range(self.epochs):

            progress = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="train")
            self.classifier.train()
            
            running_loss = None

            for i, (points, labels) in progress:
                self.optimizer.zero_grad()
            
                points = points.float().cuda()
                labels = labels[:,None].float().cuda()
                
                preds = self.classifier.forward(points)
                loss = self.Loss(preds, labels)
                loss.backward()
                self.optimizer.step()
                
                if running_loss == None:
                     running_loss = float(loss)
                else:
                    running_loss = running_loss * 0.9 + float(loss) * 0.1
                
                progress.set_description(f"it: {i}, loss: {running_loss}")

            self.visualize_classifier()

    def save_classifier(self):
        print("saving model at:", self.classifier_save_path)
        torch.save(self.classifier.state_dict(), self.classifier_save_path)

class PlanarEBMTrainer(EBMTrainer):

    def __init__(self, working_directory, config):

        self.config = config

        pointcloud_path = os.path.join(working_directory, config['io']['constraint_name'])
        self.pointcloud = np.load(pointcloud_path)
        self.classifier = PlanarClassifier().to('cuda')
        self.dataset = PlanarDataset(self.pointcloud, num_samples=config['train']['num_samples'],   
                                     sampling_space=get_sampling_space(self.pointcloud, 3)[:, 0:2])
        
        self.train_loader = DataLoader(self.dataset, batch_size=self.config['train']['batch_size'], shuffle=True)
        self.optimizer = optim.AdamW(self.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        self.Loss = nn.BCELoss()

        self.epochs = config['train']['epochs']

        self.classifier_save_path = os.path.join(working_directory, config['io']['constraint_classifier_name'])

    def draw_pointcloud(self):
        # draw_pointcloud(self.pointcloud, cube=True)
        fig, ax = plt.subplots(1, 1)
        draw_points_2D(ax, self.pointcloud, cube=True)
        plt.show()

    def visualize_classifier(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        bounding_box = inflate_box(self.dataset.bounding_box, 0.5)
        visualize_planar_classifier(ax, self.classifier, self.dataset, num_points=10000, box=bounding_box, cube=True)
        plt.show()