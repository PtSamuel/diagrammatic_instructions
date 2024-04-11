import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

def gaussian_plot():

    limits = np.array([[-2.0, -2, 0], [2, 2, 2]])
    ranges = limits[1] - limits[0]
    
    # Generate nicely looking random 3D-field
    np.random.seed(0)
    l = 30
    X, Y, Z = np.mgrid[:l+1, :l+1, :l+1] / l
    X = limits[0, 0] + ranges[0] * X 
    Y = limits[0, 1] + ranges[1] * Y 
    Z = limits[0, 2] + ranges[2] * Z 
    vol = np.zeros((l + 1, l + 1, l + 1))
    
    pts = (l * np.random.rand(3, 15)).astype(np.int32)
    vol[tuple(indices for indices in pts)] = 1
    
    from scipy import ndimage
    vol = ndimage.gaussian_filter(vol, 4)
    vol /= vol.max()
    
    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=vol.flatten(),
            isomin=0.2,
            isomax=0.7,
            opacity=0.1,
            surface_count=25,
        )
    )
    fig.update_layout(scene_xaxis_showticklabels=False,
                      scene_yaxis_showticklabels=False,
                      scene_zaxis_showticklabels=False)
    fig.show()

def generate_scatter_3d(x, y, z):
    return go.Scatter3d(
        x=x, 
        y=y, 
        z=z, 
        mode='markers', 
        marker=dict(
            size=1
        )
    )

def generate_energies_heatmap(sigmoidless_classifier, limits, lo=None, hi=None):

    ranges = limits[1] - limits[0]
    np.random.seed(0)
    l = 30
    X, Y, Z = np.mgrid[:l+1, :l+1, :l+1] / l
    X = limits[0, 0] + ranges[0] * X 
    Y = limits[0, 1] + ranges[1] * Y 
    Z = limits[0, 2] + ranges[2] * Z 
    
    stacked = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)
    values = sigmoidless_classifier(torch.tensor(stacked).cuda()).detach().cpu().numpy()
    return go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=values.flatten(),
        opacity=0.1,
        isomin=lo,
        isomax=hi,
        surface_count=25,
        colorscale='RdBu'
    )

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

def generate_densities(gmm, limits):

    ranges = limits[1] - limits[0]
    np.random.seed(0)
    l = 30
    X, Y, Z = np.mgrid[:l+1, :l+1, :l+1] / l
    X = limits[0, 0] + ranges[0] * X 
    Y = limits[0, 1] + ranges[1] * Y 
    Z = limits[0, 2] + ranges[2] * Z 
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    pos = np.stack([X, Y, Z], axis=1)
    print(pos.shape)
    
    densities = np.exp(gmm.score_samples(pos))
    print(densities)
    
    return go.Volume(
        x=X, y=Y, z=Z,
        value=densities,
        opacity=0.1,
        surface_count=25,
        colorscale='RdBu'
    )

def logit(p):
    p = torch.tensor(p)
    return torch.log(p / (1 - p))

def get_sampling_space(pointcloud, factor=1):
    return inflate_box(find_bounding_box(pointcloud), factor=factor)

def inflate_box(box, factor=5):
    ranges = box[1] - box[0]
    inflated = torch.stack([
        box[0] - ranges * factor,
        box[1] + ranges * factor,
    ], axis=0)
    return inflated

def find_bounding_box(pointcloud):

    return torch.tensor(np.array([
        pointcloud.min(axis=1),
        pointcloud.max(axis=1)
    ]))

def box_to_cube(box):
    
    lengths = box[1] - box[0]
    cube_length = lengths.max()
    
    center = box.mean(axis=0)
    cube = torch.stack([
        center - cube_length * 0.5,
        center + cube_length * 0.5
    ], axis=0)
    return cube

def draw_pointcloud(points, limits=None, cube=False):

    if points is None:
        print("cannot draw if pointcloud is None")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(points[0], points[1], points[2], c=points[2], cmap='jet', marker='o', s=5)

    canon = np.eye(3)
    ax.quiver(0, 0, 0, canon[0, 0], canon[1, 0], canon[2, 0], color='r', label='X-axis')
    ax.quiver(0, 0, 0, canon[0, 1], canon[1, 1], canon[2, 1], color='g', label='Y-axis')
    ax.quiver(0, 0, 0, canon[0, 2], canon[1, 2], canon[2, 2], color='b', label='Z-axis')
    
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
    cbar.set_label('Values')
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    if limits is None:
        limits = torch.tensor([
            [
                points[0].min() - 0.1, 
                points[1].min() - 0.1, 
                points[2].min() - 0.1
            ], [
                points[0].max() + 0.1, 
                points[1].max() + 0.1, 
                points[2].max() + 0.1
            ]
        ])

    if cube:
        limits = box_to_cube(limits)

    ax.set_xlim(limits[:, 0])
    ax.set_ylim(limits[:, 1])
    ax.set_zlim(limits[:, 2])
    
    ax.set_title('3D Scatter Plot with Color Mapping')
    ax.view_init(20, -10, 0)
    plt.show()

def draw_trajectory(ax, bases, energies):
    if len(bases) == 0:
        print("no base configs")
        return
    
    base_prev = bases[0][1].cpu()
    for i in range(1, len(bases)):
        intermediate, result = bases[i]
        result = result.cpu()
        if intermediate is None:
            ax.plot([base_prev[0], result[0]], [base_prev[1], result[1]], marker='o', color='blue')
        else:
            intermediate = intermediate.cpu()
            ax.plot([base_prev[0], intermediate[0]], [base_prev[1], intermediate[1]], marker='o', color='blue')
            ax.plot([intermediate[0], result[0]], [intermediate[1], result[1]], marker='o', color='red')
        base_prev = result

def draw_points_2D(ax, points, limits=None, cube=False):

    if points is None:
        print("cannot draw if pointcloud is None")
        return

    ax.scatter(points[0], points[1], marker='o', s=5)
    
    if limits is None:
        limits = torch.tensor([
            [
                points[0].min() - 0.1, 
                points[1].min() - 0.1
            ], [
                points[0].max() + 0.1, 
                points[1].max() + 0.1
            ]
        ])

    if cube:
        limits = box_to_cube(limits)

    ax.set_xlim(limits[:, 0])
    ax.set_ylim(limits[:, 1])
    
    ax.set_title('draw_points_2D')

def visualize_classifier(ax, classifier, dataset, num_points=1000, positive_rate=0.2, box=torch.tensor([[-1.0, -1, -1], [3.0, 3, 3]]), cube=False):

    if cube:
        box = box_to_cube(box)

    torch.manual_seed(1984)   
    start = box[0, :]
    end = box[1, :]
    samples = start[None,:].repeat(num_points, 1) + \
        torch.rand(num_points, 3) * (end - start)[None,:].repeat(num_points, 1)
    
    num_positive_samples = int(positive_rate * num_points)
    for i in range(num_positive_samples):
        samples[i] = dataset.draw_sample().float()
        
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]
    
    classifier.eval()
    infered = classifier(samples.cuda()).cpu()
    
    # Filter points based on the function
    positive_points_mask = (infered[:,0] > 0.9)
    print(num_points, positive_points_mask.sum())
    positive_x = x[positive_points_mask]
    positive_y = y[positive_points_mask]
    positive_z = z[positive_points_mask]
    
    negative_points_mask = torch.logical_not(positive_points_mask)
    negative_x = x[negative_points_mask]
    negative_y = y[negative_points_mask]
    negative_z = z[negative_points_mask]

    canon = np.eye(3)
    ax.quiver(0, 0, 0, canon[0, 0], canon[1, 0], canon[2, 0], color='r', label='X-axis')
    ax.quiver(0, 0, 0, canon[0, 1], canon[1, 1], canon[2, 1], color='g', label='Y-axis')
    ax.quiver(0, 0, 0, canon[0, 2], canon[1, 2], canon[2, 2], color='b', label='Z-axis')
    
    # Plot all points in light gray
    ax.scatter(negative_x, negative_y, negative_z, color='lightsteelblue', alpha=0.3, label='All Points', s=1)
    
    # Plot filtered points in red
    ax.scatter(positive_x, positive_y, positive_z, color='red', label='Filtered Points', s=1)

    ax.set_xlim(box[:, 0])
    ax.set_ylim(box[:, 1])
    ax.set_zlim(box[:, 2])
    
    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set plot title
    ax.set_title('Filtered 3D Points')
    
    # Add a legend
    ax.legend()

def visualize_planar_classifier(ax, classifier, dataset, num_points=1000, positive_rate=0.2, box=torch.tensor([[-1.0, -1], [1.0, 1]]), cube=False):

    if cube:
        box = box_to_cube(box)

    torch.manual_seed(1984)   
    start = box[0, :]
    end = box[1, :]
    samples = start[None,:].repeat(num_points, 1) + \
        torch.rand(num_points, 2) * (end - start)[None,:].repeat(num_points, 1)
    
    num_positive_samples = int(positive_rate * num_points)
    for i in range(num_positive_samples):
        samples[i] = dataset.draw_sample().float()
        
    x = samples[:, 0]
    y = samples[:, 1]
    
    classifier.eval()
    infered = classifier(samples.cuda()).cpu()
    
    positive_points_mask = (infered[:,0] > 0.9)
    positive_x = x[positive_points_mask]
    positive_y = y[positive_points_mask]
    
    negative_points_mask = torch.logical_not(positive_points_mask)
    negative_x = x[negative_points_mask]
    negative_y = y[negative_points_mask]
    
    ax.scatter(negative_x, negative_y, color='lightsteelblue', alpha=0.3, label='All Points', s=1)
    ax.scatter(positive_x, positive_y, color='red', label='Filtered Points', s=1)

    ax.set_xlim(box[:, 0])
    ax.set_ylim(box[:, 1])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    ax.set_title('Filtered 3D Points')
    ax.legend()

def visualize_distance(sigmoidless_classifier, num_points=10000, box=torch.tensor([[-1.0, -1, -1], [3.0, 3, 3]])):

    torch.manual_seed(1984)
    start = box[0, :]
    end = box[1, :]
    samples = start[None,:].repeat(num_points, 1) + \
        torch.rand(num_points, 3) * (end - start)[None,:].repeat(num_points, 1)

    sigmoidless_classifier.eval()
    results = sigmoidless_classifier(samples.cuda())

    x = samples[:,0]
    y = samples[:,1]
    z = samples[:,2]
    values = results[:,0].cpu().detach().numpy()  # Values for color mapping
    
    # Create a 3D scatter plot with color mapping
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    canon = np.eye(3)
    ax.quiver(0, 0, 0, canon[0, 0], canon[1, 0], canon[2, 0], color='r', label='X-axis')
    ax.quiver(0, 0, 0, canon[0, 1], canon[1, 1], canon[2, 1], color='g', label='Y-axis')
    ax.quiver(0, 0, 0, canon[0, 2], canon[1, 2], canon[2, 2], color='b', label='Z-axis')
    
    # Use the 'viridis' colormap for color mapping
    scatter = ax.scatter(x, y, z, c=values, cmap='jet', marker='o', s=1)
    
    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
    cbar.set_label('Values')
    
    # Set labels for the axes
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    # Set title
    ax.set_title('3D Scatter Plot with Color Mapping')
    
    # Show the plot
    plt.show()
    
def visualize_gradient(sigmoidless_classifier, num_points=1000, box=torch.tensor([[-1.0, -1, -1], [3.0, 3, 3]])):

    torch.manual_seed(1984)
    start = box[0, :]
    end = box[1, :]
    samples = start[None,:].repeat(num_points, 1) + \
        torch.rand(num_points, 3) * (end - start)[None,:].repeat(num_points, 1)
    
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]

    sigmoidless_classifier.train()
    sigmoidless_classifier.zero_grad()
    samples.requires_grad = True
    output = sigmoidless_classifier(samples.cuda())
    sum_for_grad = torch.sum(output)
    sum_for_grad.backward()
    
    grad = samples.grad / torch.norm(samples.grad, dim=1)[:,None].repeat(1, 3)
    grad.requires_grad = False
    samples.requires_grad = False
    x_grad = grad[:,0]
    y_grad = grad[:,1]
    z_grad = grad[:,2]
    u = x + x_grad
    v = y + y_grad
    w = z + z_grad
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    canon = np.eye(3)
    ax.quiver(0, 0, 0, canon[0, 0], canon[1, 0], canon[2, 0], color='r', label='X-axis')
    ax.quiver(0, 0, 0, canon[0, 1], canon[1, 1], canon[2, 1], color='g', label='Y-axis')
    ax.quiver(0, 0, 0, canon[0, 2], canon[1, 2], canon[2, 2], color='b', label='Z-axis')
    
    # Plot arrows using quiver
    ax.quiver(x, y, z, x_grad, y_grad, z_grad, length=0.2, normalize=True, color='r')
    
    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Arrows in 3D with Matplotlib')
    
    
    # Show the plot
    plt.show()

def draw_SO3(R):

    axes = np.eye(3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(0, 0, 0, axes[0, 0], axes[1, 0], axes[2, 0], color='r', label='X-axis')
    ax.quiver(0, 0, 0, axes[0, 1], axes[1, 1], axes[2, 1], color='g', label='Y-axis')
    ax.quiver(0, 0, 0, axes[0, 2], axes[1, 2], axes[2, 2], color='b', label='Z-axis')

    ax.quiver(0, 0, 0, R[0, 0], R[1, 0], R[2, 0], color='r', linestyle='dashed')
    ax.quiver(0, 0, 0, R[0, 1], R[1, 1], R[2, 1], color='g', linestyle='dashed')
    ax.quiver(0, 0, 0, R[0, 2], R[1, 2], R[2, 2], color='b', linestyle='dashed')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_box_aspect([np.max(axes[0]), np.max(axes[1]), np.max(axes[2])])

    ax.set_aspect('equal')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.legend()
    ax.view_init(20, 20, 0)

    plt.show()

def read_yaml(config_path):

    from yaml import load, dump
    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper

    stream = open(config_path, 'r')
    config = load(stream, Loader=Loader)

    return config