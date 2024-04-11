import sys
sys.path.append('../pipeline')

import os
import torch
from torch import nn
import numpy as np
import plotly.graph_objects as go
from utils import read_yaml, logit, get_sampling_space
from general import Pose
from plotly.subplots import make_subplots
from optimizer import Optimizer
from pointcloud import PointCloud
from ebm_trainer import SurfaceClassifier, PlanarClassifier

class SceneVisualizer:

    def __init__(self, config, working_directory, tag_to_world=None):

        classifier_path = os.path.join(working_directory, config['io']['classifier_name'])
        classifier = SurfaceClassifier().to('cuda')
        classifier.load_state_dict(torch.load(classifier_path))

        self.classifier = classifier
        self.ebm = nn.Sequential(*list(classifier.children())[0][:-1])

        pc = PointCloud(images_path=working_directory, high_res_depth=False, config=config) 
        pc.find_camera_pose()
        mask_all = np.ones((480, 640))

        self.roi_mask = pc.get_roi_mask()
        self.constraint_mask = pc.get_constraint_mask()
        self.spectator_mask = np.logical_and(np.logical_and(mask_all, np.logical_not(self.roi_mask)), np.logical_not(self.constraint_mask)).astype(np.uint8)

        self.spectator_points, self.spectator_pixels = pc.compute_points_in_world(self.spectator_mask, tag_to_world, pixels=True)
        self.roi_points, self.roi_pixels = pc.compute_points_in_world(self.roi_mask, tag_to_world, pixels=True)
        self.constraint_points, self.constraint_pixels = pc.compute_points_in_world(self.constraint_mask, tag_to_world, pixels=True)

    def visualize_scene(self, traces=None, heatmap=False, limits=None):
        
        if limits is None:
            limits = get_sampling_space(self.roi_points, 0.4)

        plots = [
            colorize(self.roi_points, self.roi_pixels, lambda x: (232, 16, 67)),
            colorize(self.constraint_points, self.constraint_pixels, lambda x: (201, 247, 148)),
            colorize(self.spectator_points, self.spectator_pixels, lambda x: x),
        ]
        if traces is not None:
            plots += traces
        if heatmap:
            heapmap = generate_energies_heatmap(self.ebm, limits=limits, lo=-20, hi=6)
            plots += [heapmap]
        
        fig = go.Figure(data=plots)
        fig.update_layout(
            scene=dict(
                aspectmode="data"
            )
        )
        fig.show()

working_directory = "../pipeline/saves/drawer"

config_path = "../pipeline/config.yaml"
config = read_yaml(config_path)

def generate_2d_heatmap(energy, limits, lo=None, hi=None, color_scale="Oranges", opacity=0.05):

    ranges = limits[1] - limits[0]
    np.random.seed(0)
    l = 30
    X, Y = np.mgrid[:l+1, :l+1] / l
    X = limits[0, 0] + ranges[0] * X 
    Y = limits[0, 1] + ranges[1] * Y 
    
    stacked = np.stack([X.flatten(), Y.flatten()], axis=1).astype(np.float32)
    values = energy(torch.tensor(stacked).cuda()).detach().cpu().numpy().reshape((l + 1, l + 1)).transpose()
    return go.Contour(
        z=values,
        line_smoothing=0.85,
        x=limits[0, 0]+np.mgrid[:l+1]/l*ranges[0],
        y=limits[0, 1]+np.mgrid[:l+1]/l*ranges[1],
        contours=dict(
            coloring='heatmap',
            start=-2.007607936859131,
            end=4.595120906829834,
            size=1.6506822109222412
        ),
        line_width=1,
    )

def generate_projections(opt, energy_model, fig, num_points=10, limits=np.array([[-1, -1], [1, 1]]), threshold_prob=0.95):
    
    np.random.seed(1929)
    points = np.zeros((2, num_points))
    ranges = limits[1] - limits[0]
    points[0, :] = limits[0, 0] + ranges[0] * np.random.sample(num_points)
    points[1, :] = limits[0, 1] + ranges[1] * np.random.sample(num_points)

    r = 0.02

    results_x, results_y = [], []
    threshold_energy = logit(threshold_prob)
    for point in points.transpose():
        t = torch.tensor(point).float()
        if energy_model(t.cuda()) < threshold_energy:
            res = opt.newtons(energy_model, t, threshold_prob)
            if res is not None:
                results_x += [None, point[0]]
                results_y += [None, point[1]]
                results_x.append(res[0].item())
                results_y.append(res[1].item())
                fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=point[0]-r, y0=point[1]-r,
                    x1=point[0]+r, y1=point[1]+r,
                    opacity=0.8,
                    line_color='rgb(255,255,255)',
                    fillcolor="orange",
                    row=1, col=3
                )
                fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=results_x[-1]-r, y0=results_y[-1]-r,
                    x1=results_x[-1]+r, y1=results_y[-1]+r,
                    opacity=0.8,
                    line_color='rgb(255,255,255)',
                    fillcolor="blue",
                    row=1, col=3
                )

    return go.Scatter(
        x=results_x, 
        y=results_y, 
        marker=dict(
            size=7,
            symbol='circle',
            color="rgb(0,68,140)"
        ),
        line=dict(
            color='rgb(39,85,107)',
            width=3,
        ),
    )

opt = Optimizer(working_directory, config, None, constraints=True)
energy_model = opt.sigmoidless_constraint_classifier

sv = SceneVisualizer(config, working_directory, tag_to_world=Pose(np.array([
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1]
]), np.array([
    [0], [-1], [0]
])))
constraint_points = sv.constraint_points[0:2, :]

fig = make_subplots(rows=1, cols=3)
limits = np.array([constraint_points.min(axis=1), constraint_points.max(axis=1)])
limits[0, :] -= (limits[1] - limits[0]) * 0.06
limits[1, :] += (limits[1] - limits[0]) * 0.06

fig.add_trace(generate_2d_heatmap(energy_model, limits=limits), 1, 1)
fig.add_trace(go.Scatter(x=constraint_points[0], y=constraint_points[1],
    mode='markers',
    name='markers',
    marker=dict(
        size=16,
        symbol='square'
    )
), 1, 2)
fig.add_trace(go.Scatter(x=constraint_points[0], y=constraint_points[1],
    mode='markers',
    name='markers',
    marker=dict(
        size=16,
        symbol='square'
    )
), 1, 3)
fig.add_trace(generate_projections(opt, energy_model, fig, num_points=40, limits=np.array([[-1, -2], [0.5, 0]]), threshold_prob=0.95), 1, 3)

fig.update_layout(
    scene=dict(
        aspectmode="data"
    )
)
fig.show()