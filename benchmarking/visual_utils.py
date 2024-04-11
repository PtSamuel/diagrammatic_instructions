import numpy as np
import plotly.graph_objects as go
import pybullet as p
import torch

def generate_ebm_samples(ebm, count, limits=np.array([[-1, -1, 0], [1, 1, 1]]), scaler=160):
    ebm.eval()

    points_buf = np.zeros((3, count))
    
    batch_size = 4096
    cur = 0
    np.random.seed(985211)
    while cur < count:
        points = limits[0] + (limits[1] - limits[0]) * np.random.random_sample((batch_size, 3))
        with torch.no_grad():
            energy = torch.exp(ebm(torch.tensor(points).float().cuda())).cpu().numpy().flatten() / scaler
        selected_indices = np.random.rand(batch_size) <= energy
        num_selected = selected_indices.sum()
        num_selected = min(num_selected, count - cur)
        points = points.T
        points_buf[:, cur:cur+num_selected] = points[:, selected_indices][:, 0:num_selected]
        cur += num_selected
    return points_buf

def get_joint_type(t):
    if t == 0:
        return "JOINT_REVOLUTE"
    elif t == 1:
        return "JOINT_PRISMATIC"
    elif t == 2:
        return "JOINT_SPHERICAL"
    elif t == 3:
        return "JOINT_PLANAR"
    else:
        return "JOINT_FIXED"

def urdf_setup(urdf_path, gui=True):
    if p.isConnected():
        p.resetSimulation()
    else:
        p.connect(p.GUI if gui else p.DIRECT)
        p.setRealTimeSimulation(0)

    robot = p.loadURDF(urdf_path)
    num_joints = p.getNumJoints(robot)

    valid_joints = {}
    valid_indices = []

    for joint_index in range(num_joints):

        joint_info = p.getJointInfo(robot, joint_index)

        joint_name = joint_info[1].decode('utf-8')
        joint_type = get_joint_type(joint_info[2])
        
        # print(f"Joint {joint_index}:")
        # print(f"  Name: {joint_name}")  # Joint name
        # print(f"  Type: {joint_type}")  # Joint type (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, etc.)
        # print(f"  First Position index: {joint_info[3]}")
        # print(f"  First Velocity index: {joint_info[4]}")
        # print(f"  Joint Lower: {joint_info[8]}")
        # print(f"  Joint Higher: {joint_info[9]}")
        # print()

        if joint_type == "JOINT_FIXED":
            continue

        valid_joints[joint_name] = (joint_index, joint_type, joint_info[8:10])
        valid_indices.append(joint_index)

    ll = np.array([i[1][2][0] for i in valid_joints.items()])
    ul = np.array([i[1][2][1] for i in valid_joints.items()])
    jr = ul - ll

    return robot, valid_indices, ul, ll, jr

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

def draw_cube(x, y, z, lx, ly, lz):
    lx2 = lx * 0.5
    ly2 = ly * 0.5
    lz2 = lz * 0.5
    x = x + np.array([-lx2, lx2, -lx2, lx2, -lx2, lx2, -lx2, lx2])
    y = y + np.array([-ly2, -ly2, ly2, ly2, -ly2, -ly2, ly2, ly2])
    z = z + np.array([lz2, lz2, lz2, lz2, -lz2, -lz2, -lz2, -lz2])
    i = [0, 3, 6, 5, 4, 1, 6, 3, 0, 2, 5, 3]
    j = [1, 2, 5, 6, 1, 4, 3, 6, 2, 4, 3, 5]
    k = [2, 1, 4, 7, 0, 5, 2, 7, 4, 6, 1, 7]
    
    cube_trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
        text=["Face 1", "Face 2", "Face 3", "Face 4", "Face 5", "Face 6"],
        hoverinfo="text",
        opacity=0.8,
        color='#9658ed',
        flatshading=True
    )

    return cube_trace

def draw_circle(x, y, z, r):
    num_segs = 128
    ths = np.linspace(0, 2 * np.pi, num_segs + 1)
    x = np.concatenate([[x], x + r * np.cos(ths)])
    y = np.concatenate([[y], y + r * np.sin(ths)])
    z = np.concatenate([[z], np.ones(num_segs + 1) * r])

    i = [0] * num_segs
    j = [i for i in range(1, num_segs + 1)]
    k = [i + 1 for i in range(1, num_segs + 1)]
    
    cube_trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
        text=["Face 1", "Face 2", "Face 3", "Face 4", "Face 5", "Face 6"],
        hoverinfo="text",
        opacity=0.8,
        color='#58ede6',
        flatshading=True
    )

    return cube_trace

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

def generate_energies_heatmap(sigmoidless_classifier, limits, lo=None, hi=None, color_scale="Oranges", opacity=0.05):

    ranges = limits[1] - limits[0]
    np.random.seed(0)
    l = 30
    X, Y, Z = np.mgrid[:l+1, :l+1, :l+1] / l
    X = limits[0, 0] + ranges[0] * X 
    Y = limits[0, 1] + ranges[1] * Y 
    Z = limits[0, 2] + ranges[2] * Z 
    print(limits[0], limits[1])
    
    stacked = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)
    values = sigmoidless_classifier(torch.tensor(stacked).cuda()).detach().cpu().numpy()
    return go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=values.flatten(),
        opacity=opacity,
        isomin=lo,
        isomax=hi,
        surface_count=25,
        colorscale=color_scale
    )

def generate_densities(gmm, limits):

    ranges = limits[1] - limits[0]
    np.random.seed(0)
    l = 50
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