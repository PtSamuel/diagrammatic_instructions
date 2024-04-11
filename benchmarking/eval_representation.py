from IPython.display import display
import numpy as np
import torch
from torch import nn
import plotly.graph_objects as go

import sys
sys.path.append('../')
sys.path.append('../pipeline')
from pipeline.standalone import StandaloneTrainer
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from utils import get_sampling_space, find_bounding_box
from visual_utils import generate_densities, generate_scatter_3d, generate_energies_heatmap

def generate_plane(num_points, x, y, z, lx, ly):
    points = np.zeros((3, num_points))
    points[0, :] = x
    points[1, :] = y
    points[2, :] = z
    
    points[0, :] += np.random.sample(num_points) * lx
    points[1, :] += np.random.sample(num_points) * ly

    return points

def generate_planes(num_points, x, y, z, lx, ly, lz):
    points = np.zeros((3, num_points * 3))
    points[0, :] = x
    points[1, :] = y
    points[2, :] = z
    
    points[1, 0:num_points] += np.random.sample(num_points) * ly
    points[2, 0:num_points] += np.random.sample(num_points) * lz

    points[0, num_points:2*num_points] += np.random.sample(num_points) * lx
    points[2, num_points:2*num_points] += np.random.sample(num_points) * lz

    points[0, 2*num_points:] += np.random.sample(num_points) * lx
    points[1, 2*num_points:] += np.random.sample(num_points) * ly

    return points

def generate_circle(points_num, radius, x, y, z):
    circle_points = np.zeros((3, points_num))
    circle_points[0, :] = x
    circle_points[1, :] = y
    circle_points[2, :] = z

    circle_points_raw = np.zeros((2, points_num * 2))
    circle_points_raw[0] = radius * 2 * (np.random.sample(points_num * 2) - 0.5)
    circle_points_raw[1] = radius * 2 * (np.random.sample(points_num * 2) - 0.5)

    good_indices = np.where((circle_points_raw ** 2).sum(axis=0) < radius ** 2)[0]
    assert len(good_indices) > points_num, "unenough good points!"
    circle_points[0:2, :] += circle_points_raw[:, good_indices][:, 0:points_num]

    return circle_points

def generate_mix():

    box_pos = np.array([-1, 0, 0.5])
    box_lengths = np.array([1, 0.5, 1])
    circle_pos = np.array([1, 0, 0.5])
    radius = 0.5

    points_num = 1024
    box_points = np.zeros((3, points_num))
    circle_points = np.zeros((3, points_num))

    box_points[0] = box_pos[0] + box_lengths[0] * (np.random.sample(points_num) - 0.5)
    box_points[1] = box_pos[1] + box_lengths[1] * (np.random.sample(points_num) - 0.5)
    box_points[2] = box_pos[2] + box_lengths[2] * 0.5

    circle_points = np.zeros((3, points_num))
    circle_points[0, :] = circle_pos[0]
    circle_points[1, :] = circle_pos[1]
    circle_points[2, :] = circle_pos[2]

    circle_points_raw = np.zeros((2, points_num * 2))
    circle_points_raw[0] = radius * 2 * (np.random.sample(points_num * 2) - 0.5)
    circle_points_raw[1] = radius * 2 * (np.random.sample(points_num * 2) - 0.5)

    good_indices = np.where((circle_points_raw ** 2).sum(axis=0) < radius ** 2)[0]
    assert len(good_indices) > points_num, "unenough good points!"
    circle_points[0:2, :] += circle_points_raw[:, good_indices][:, 0:points_num]

    return np.concatenate([box_points, circle_points], axis=1)

def generate_star(num_points):
    
    def generate_vertices(num_points, radius_outer, radius_inner):
        angles = np.linspace(0, 2*np.pi, 2*num_points, endpoint=False)
        vertices = []
        for i, angle in enumerate(angles):
            radius = radius_outer if i % 2 == 0 else radius_inner
            vertices.append((radius*np.cos(angle), radius*np.sin(angle)))
        return vertices

    def is_inside_star(point, star_vertices):
        # Ray casting algorithm to check if a point is inside the star
        x, y = point
        num_intersections = 0
        for i in range(len(star_vertices)):
            x1, y1 = star_vertices[i]
            x2, y2 = star_vertices[(i + 1) % len(star_vertices)]
            if (y1 > y) != (y2 > y) and x < ((x2 - x1) * (y - y1) / (y2 - y1) + x1):
                num_intersections += 1
        return num_intersections % 2 == 1

    star_vertices = generate_vertices(5, 0.5, 0.2)

    num_samples = num_points * 5
    xmin = min(point[0] for point in star_vertices) - 0.1
    xmax = max(point[0] for point in star_vertices) + 0.1
    ymin = min(point[1] for point in star_vertices) - 0.1
    ymax = max(point[1] for point in star_vertices) + 0.1
    sample_points = np.random.uniform(xmin, xmax, (num_samples, 2))
    sample_points[:, 1] = np.random.uniform(ymin, ymax, num_samples)

    filtered_points = [point for point in sample_points if is_inside_star(point, star_vertices)]

    x_filtered = [point[0] for point in filtered_points]
    y_filtered = [point[1] for point in filtered_points]

    points = np.zeros((3, num_points))
    points[0, :] = x_filtered[:num_points]
    points[1, :] = y_filtered[:num_points]

    return points


layout = go.Layout(
    scene=dict(
        aspectmode="data"
    )
)

def eval_gmm(points, test_points=None, n_components=1, visualize=False):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(points.transpose())

    bbox = find_bounding_box(points)
    bbox[0] -= 0.1
    bbox[1] += 0.1

    if visualize:
        densities = generate_densities(gmm, bbox)
        fig = go.Figure(data=[
            densities,
            generate_scatter_3d(points[0], points[1], points[2])
        ], layout=layout)
        fig.update_layout(legend_title_text="Contestant")

        fig.show()

    if test_points is not None:
        return gmm.score(test_points.transpose())
    return gmm.score(points.transpose())

def eval_kde(points, test_points=None, bandwidth=0.01, visualize=False):
    kde = KernelDensity(kernel='tophat', bandwidth=bandwidth)
    kde.fit(points.transpose())

    bbox = find_bounding_box(points)
    bbox[0] -= 0.1
    bbox[1] += 0.1

    if visualize:
        densities = generate_densities(kde, bbox)

        fig = go.Figure(data=[
            densities,
            generate_scatter_3d(points[0], points[1], points[2])
        ], layout=layout)
        fig.update_layout(legend_title_text="Contestant")

        fig.show()

    if test_points is not None:
        scores = kde.score_samples(test_points.transpose())
        scores = scores[scores != float('-inf')]
        return scores.mean()
    return kde.score(points.transpose()) / points.shape[1]

def evaluate_EBM_score(points, test_points=None, visualize=True):
    
    sampling_space = get_sampling_space(points, 0.3)
    if sampling_space[0][0] == sampling_space[1][0]:
        sampling_space[0][0] -= 0.1
        sampling_space[1][0] += 0.1
    if sampling_space[0][1] == sampling_space[1][1]:
        sampling_space[0][1] -= 0.1
        sampling_space[1][1] += 0.1
    if sampling_space[0][2] == sampling_space[1][2]:
        sampling_space[0][2] -= 0.1
        sampling_space[1][2] += 0.1
    print(sampling_space)
    # return
    trainer = StandaloneTrainer(points, epochs=2, sampling_space=sampling_space)
    trainer.train()

    sigmoidless = trainer.get_sigmoidless()
    sigmoidless.eval()

    if visualize:
        bbox = find_bounding_box(test_points)
        center = bbox.numpy().mean(axis=0)
        bbox[0] -= 0.2
        bbox[1] += 0.2
        fig = go.Figure(data=[
            generate_scatter_3d(test_points[0], test_points[1], test_points[2]),
            generate_energies_heatmap(sigmoidless, limits=bbox, lo=0, hi=10, opacity=1.0)
        ], layout=layout)

        fig.show()

    step_size = 0.01
    steps = 400
    
    X, Y, Z = np.mgrid[:steps, :steps, :steps]
    
    queries = np.stack([
        ((X - steps * 0.5) * 0.05 + center[0]).flatten(),
        ((Y - steps * 0.5) * 0.05 + center[1]).flatten(),
        ((Z - steps * 0.5) * 0.05 + center[2]).flatten()
    ], axis=1).astype(np.float32)

    num_samples = queries.shape[0]
    energies = torch.zeros(num_samples)
    batch_size = 1000
    
    sigmoidless.eval()

    for i in range(num_samples // batch_size):
        with torch.no_grad():
            batch = torch.tensor(queries[i*batch_size:(i+1)*batch_size, :], device='cuda')
            energies[i*batch_size:(i+1)*batch_size] = sigmoidless(batch)[:, 0]
        del batch
        torch.cuda.empty_cache()

    normalizer = torch.exp(energies).sum() * (step_size ** 3)
    print("normalizer:", normalizer)
    predicted_energies = sigmoidless(torch.tensor(test_points.transpose().astype(np.float32)).cuda())

    score = (predicted_energies - torch.log(normalizer)).mean()

    return score

def run_test(points, test_points):
    gmm10_score = eval_gmm(points, test_points, 10)
    gmm50_score = eval_gmm(points, test_points, 50)
    gmm100_score = eval_gmm(points, test_points, 100, visualize=True)
    print("GMM10, GMM50, GMM100 scores:", gmm10_score, gmm50_score, gmm100_score)

    kde_score = eval_kde(points, test_points=test_points, visualize=True, bandwidth=0.03)
    print("KDE score:", kde_score)

    ebm_score = evaluate_EBM_score(points, test_points=test_points, visualize=True)
    print("EBM score:", ebm_score)

if __name__ == '__main__':

    np.random.seed(1981)
    planes_points = generate_planes(2048, 0, 0, 0, 0.5, 0.5, 0.5)
    planes_test_points = generate_planes(2048, 0, 0, 0, 0.5, 0.5, 0.5)
    run_test(planes_points, planes_test_points)
    # 6.034110410939731 6.18155530966279 6.177272246934915
    # 3.8507599527552046
    # 8.0386

    plane_points = generate_plane(2048, 0, 0, 0, 0.5, 0.5)
    np.random.seed(1948)
    plane_test_points = generate_plane(2048, 0, 0, 0, 0.5, 0.5)
    run_test(plane_points, plane_test_points)
    # 7.213460529250171 7.217162708034762 7.118826029947771
    # 4.909026402115548
    # 9.1938

    circle_points = generate_circle(2048, 0.25, 0, 0, 0)
    circle_test_points = generate_circle(2048, 0.25, 0, 0, 0)
    run_test(circle_points, circle_test_points)
    # 7.4903772715101855 7.506047399200328 7.388923228657831
    # 5.169293501301827
    # 9.3810

    np.random.seed(2024)
    mixed_points = generate_mix()
    mixed_test_points = generate_mix()
    run_test(mixed_points, mixed_test_points)
    # 5.5964111900721 5.611942460636847 5.464596947727539
    # 2.8587408704112467
    # 6.9786

    np.random.seed(1981)
    star_points = generate_star(4096)
    star_test_points = generate_star(4096)
    run_test(star_points, star_test_points)
    # 7.023321276175753 7.095423767593136 7.080894574979694
    # 4.753248128000735
    # 8.6557

    pass
    
