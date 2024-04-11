import time
import math
import pybullet as p
import pytorch_kinematics as pk
import numpy as np
import torch
import pybullet_data
import random
from torch import nn
from torch.func import hessian
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from visualizer import Visualizer
from tqdm import tqdm
from visualize_optimization import SceneVisualizer
import os

import sys
sys.path.append('../pipeline')

from utils import read_yaml, draw_pointcloud, get_sampling_space
from optimizer import Optimizer
from visual_utils import generate_ebm_samples, generate_scatter_3d
from general import Pose

class Evaluator:

    def __init__(self, working_directory, man, config):
        self.man = man
        self.opt = Optimizer(working_directory, config, man.chain, constraints=True)
        self.constraint_classifier = self.opt.constraint_classifier
        self.sigmoidless_constraint_classifier = nn.Sequential(*list(self.constraint_classifier.children())[0][:-1])
    
    @staticmethod
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

    @staticmethod   
    def random_config(ll, ul):
        temp = ll + (ul - ll) * np.random.random_sample(len(ul))
        return temp
    
    def try_reach_pos_simple(self, robot, ee_id, pos, valid_indices, ul, ll, jr, tolerance=0.001, halt=False):

        num_trials = 8
        
        for i in range(num_trials):
            
            joint_angles = p.calculateInverseKinematics(
                robot,
                ee_id,
                pos,
                solver=0
            )
            
            p.resetJointStatesMultiDof(robot, valid_indices, [[i] for i in joint_angles])
            
            joint_angles = p.calculateInverseKinematics(
                robot,
                ee_id,
                pos,
                lowerLimits=list(ll),
                upperLimits=list(ul),
                jointRanges=list(jr),
                restPoses=list(Evaluator.random_config(ll, ul))
            )
            
            p.resetJointStatesMultiDof(robot, valid_indices, [[i] for i in joint_angles])
            
            angles = np.array(joint_angles)
            if halt:
                print(np.logical_and(ll <= joint_angles, joint_angles <= ul).all())
            if np.logical_and(ll <= joint_angles, joint_angles <= ul).all():
                actual_pos = p.getLinkState(robot, ee_id)[0]
                dist2 = ((np.array(pos) - np.array(actual_pos)) ** 2).sum()
                if halt:
                    print(angles)
                    print(actual_pos, pos, dist2)
                    input('')
                if dist2 <= tolerance ** 2:
                    if halt:
                        input("success")
                    return angles
               
        return None
    
    @staticmethod
    def sample_from_unit_circle():
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.sqrt(np.random.uniform(0, 1))
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        return x, y
    
    def try_reach_pos(self, robot, ee_id, pos, valid_indices, ul, ll, jr, tolerance=0.001):

        num_trials = 16
        
        for i in range(num_trials):
            
            joint_angles = p.calculateInverseKinematics(
                robot,
                ee_id,
                pos,
                solver=0
            )
            
            p.resetJointStatesMultiDof(robot, valid_indices, [[i] for i in joint_angles])
            
            joint_angles = p.calculateInverseKinematics(
                robot,
                ee_id,
                pos,
                lowerLimits=list(ll),
                upperLimits=list(ul),
                jointRanges=list(jr),
                restPoses=list(Evaluator.random_config(ll, ul))
            )
            
            p.resetJointStatesMultiDof(robot, valid_indices, [[i] for i in joint_angles])
            
            actual_pos = p.getLinkState(robot, ee_id)[0]
            dist2 = ((np.array(pos) - np.array(actual_pos)) ** 2).sum()
            
            angles = np.array(joint_angles)
            if dist2 <= tolerance ** 2 and np.logical_and(ll <= joint_angles, joint_angles <= ul).all():

                base_x, base_y = angles[0:2]
                base_xy = torch.tensor([base_x, base_y]).float().cuda()
                if self.constraint_classifier == None or self.constraint_classifier(base_xy) > 0.95:
                    return angles
                else:
                    
                    for splash in range(4):

                        base_xy = self.opt.newtons(self.sigmoidless_constraint_classifier, base_xy)
                        angles[0] = float(base_xy[0])
                        angles[1] = float(base_xy[1])
                        p.resetJointStatesMultiDof(robot, valid_indices, [[i] for i in angles])
                        new_attempt = self.try_reach_pos_simple(robot, ee_id, pos, valid_indices, ul, ll, jr, tolerance=tolerance)
                        if new_attempt is not None:
                            return new_attempt

        return None
    
    @staticmethod
    def setup(urdf_path, gui):
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
            joint_type = Evaluator.get_joint_type(joint_info[2])
            
            if joint_type == "JOINT_FIXED":
                continue

            valid_joints[joint_name] = (joint_index, joint_type, joint_info[8:10])
            valid_indices.append(joint_index)

        ll = np.array([i[1][2][0] for i in valid_joints.items()])
        ul = np.array([i[1][2][1] for i in valid_joints.items()])
        jr = ul - ll

        return robot, valid_indices, ul, ll, jr
    
    def randomize_base(self, gui=True):
        
        robot, valid_indices, ul, ll, jr = Evaluator.setup(self.man.urdf_path, gui)
        base_h = np.array(ul[0:4])
        base_l = np.array(ll[0:4])

        while True:
            rand_base = base_l + (base_h - base_l) * np.random.sample(4)
            base_xy = torch.tensor([rand_base[0], rand_base[1]]).float().cuda()
            if self.constraint_classifier == None or self.constraint_classifier(base_xy) > 0.95:
                return rand_base


    def optimize_base_naive(self, pointcloud, tolerance=0.001, gui=True):

        time_start = time.time()
        
        robot, valid_indices, ul, ll, jr = Evaluator.setup(self.man.urdf_path, gui)
        
        num_samples = 512
        base_poses = np.zeros((4, num_samples))
        count = 0

        pbar = tqdm(range(num_samples))
        while count < num_samples:
            rand_index = np.random.randint(pointcloud.shape[1])
            point = pointcloud[:, rand_index]

            angles = self.try_reach_pos(robot, self.man.ee_id, point, valid_indices, ul, ll, jr, tolerance=tolerance)

            if len(p.getContactPoints(robot, robot)) == 0:
                if angles is not None:
                    
                    base_poses[:, count] = angles[0:4]
                    count += 1
                    pbar.update(1)
                    
            else:
                print("collision")

        print("naive takes time:", time.time() - time_start)

        draw_pointcloud(base_poses[0:3, :])
        base_placement = base_poses.mean(axis=1)

        return base_placement
    
    def optimize_base(self, ics):

        time_start = time.time()
        
        bases_sols, energies_sols = list(zip(
            *list(self.opt.optimize(ic=ic) for ic in ics)
        ))

        print("optimizer takes time:", (time.time() - time_start) / len(bases_sols))

        final_bases = [bases[-1][1].cpu() for bases in bases_sols]
        return final_bases

    def check_reachable(self, pos, ee_id, robot, valid_indices, ul, ll, jr, tolerance=0.001):

        joint_angles = self.try_reach_pos_simple(robot, ee_id, pos, valid_indices, ul, ll, jr, tolerance=tolerance, halt=False)
        return joint_angles is not None

    def compute_reachability(self, pointcloud, base_placement, tolerance=0.001, gui=True):
        robot, valid_indices, ul, ll, jr = Evaluator.setup(self.man.urdf_path_fixed, gui=gui)
        print(ul, ll)
        p.resetBasePositionAndOrientation(robot, base_placement[:-1], p.getQuaternionFromEuler([0, 0, base_placement[3]]))
        num_points = pointcloud.shape[1]
        reachable_count = 0
        for i in range(num_points):
            point = pointcloud[:, i]
            if self.check_reachable(list(point), self.man.ee_id_fixed, robot, valid_indices, ul, ll, jr, tolerance=tolerance):
                reachable_count += 1
        return reachable_count / num_points
    
class Manipulator:
    def __init__(self, urdf_path, ee_name, ee_id, urdf_path_fixed, ee_name_fixed, ee_id_fixed):
        self.urdf_path = urdf_path
        urdf = open(urdf_path).read()
        self.chain = pk.build_serial_chain_from_urdf(urdf, end_link_name=ee_name).to(device='cuda')
        self.ee_id = ee_id

        self.urdf_path_fixed = urdf_path_fixed
        urdf_fixed = open(urdf_path_fixed).read()
        self.chain_fixed = pk.build_serial_chain_from_urdf(urdf_fixed, end_link_name=ee_name_fixed).to(device='cuda')
        self.ee_id_fixed = ee_id_fixed

def randomize_placement(config, working_directory, man, gui=True, tag_to_world=None):

    tolerance = 0.0006 ** 0.5
    evaluator = Evaluator(working_directory, man, config)

    points_from_dist = generate_ebm_samples(evaluator.opt.sigmoidless_classifier, 2048, limits=get_sampling_space(evaluator.opt.roi_points, factor=0.2))
    np.random.seed(192)
    rand_base = evaluator.randomize_base(gui=gui)
    reachability_rate = evaluator.compute_reachability(points_from_dist, rand_base, tolerance=tolerance)
    print(f"randomized base placement:", rand_base, "reachability:", reachability_rate)

def eval_scene(config, working_directory, man, gui=True, ics=[None], tag_to_world=None):

    tolerance = 0.0006 ** 0.5

    evaluator = Evaluator(working_directory, man, config)

    points_from_dist = generate_ebm_samples(evaluator.opt.sigmoidless_classifier, 2048, limits=get_sampling_space(evaluator.opt.roi_points, factor=0.2))

    naive_base_placement = evaluator.optimize_base_naive(points_from_dist, tolerance=tolerance, gui=gui)
    print(naive_base_placement)

    base_placements = evaluator.optimize_base(ics)
    print(base_placements)

    sv = SceneVisualizer(config, working_directory, tag_to_world)
    traces = [generate_scatter_3d(*points_from_dist)]
    traces += [
        go.Scatter3d(
            x=[naive_base_placement[0]], 
            y=[naive_base_placement[1]], 
            z=[naive_base_placement[2]], 
            mode='markers', 
            marker=dict(
                size=10,
                symbol='circle',
                color="rgb(255,0,0)"
            )
        )
    ] + [
        go.Scatter3d(
            x=[base_placement[0]], 
            y=[base_placement[1]], 
            z=[base_placement[2]], 
            mode='markers', 
            marker=dict(
                size=10,
                symbol='circle',
                color="rgb(0,255,0)"
            )
        )
    for base_placement in base_placements]
    sv.visualize_scene(traces=traces, heatmap=True)
    print(points_from_dist)
    input("")

    reachability_rate = evaluator.compute_reachability(points_from_dist, naive_base_placement, tolerance=tolerance)
    print("naive placement success rate:", reachability_rate)

    reachability_rates = [evaluator.compute_reachability(points_from_dist, i, tolerance=tolerance) for i in base_placements]
    
    print("success rates:", reachability_rates)

if __name__ == '__main__':

    urdf_path_panda = "../urdf/franka_mobile_panda/mobile_panda_with_gripper.urdf"
    urdf_path_panda_fixed = "../urdf/franka_mobile_panda/mobile_panda_with_gripper_fixed.urdf"
    urdf_path_z1 = "../urdf/z1_description/xacro/z1.urdf"
    urdf_path_z1_fixed = "../urdf/z1_description/xacro/z1_fixed.urdf"

    config_path = "../pipeline/config.yaml"
    config = read_yaml(config_path)

    man_panda = Manipulator(
        urdf_path_panda, 'panda_grasptarget', 16,
        urdf_path_panda_fixed, 'panda_grasptarget', 12
    )

    man_z1 = Manipulator(
        urdf_path_z1, 'gripper', 10,
        urdf_path_z1_fixed, 'gripper', 6
    )

    # Tables A
    working_directory = "../pipeline/saves/tables_a"
    
    # Different every time
    # randomize_placement(config, working_directory, man_z1, gui=True, tag_to_world=None)
    # randomized base placement: [ 0.2717363  -1.06787796  0.35710591  0.06649968] reachability: 0.01123046875 192
    
    eval_scene(config, working_directory, man_z1, gui=True, ics=[
        None,
        torch.tensor([1.5, 0, 0.2, 0]),
        torch.tensor([-0.5, -1.5, 0.3, 0]),
    ])

    # Naive reachability: 0.0166015625
    # Runtime: 2.233849048614502
    # Optimal reachabilities: [0.3046875, 0.3076171875, 0.30810546875]
    # Runtime: 0.35022513071695965 s

    # Tables B
    working_directory = "../pipeline/saves/tables_b"

    # Different every time
    # randomize_placement(config, working_directory, man_z1, gui=True, tag_to_world=None)
    # randomized base placement: [ 0.2717363  -1.06787796  0.35710591  0.06649968] reachability: 0.02587890625 seed 192

    eval_scene(config, working_directory, man_z1, gui=True, ics = [
        torch.tensor([0.6, -1.2, 0.4, 0]),
        torch.tensor([-0.5, -1.5, 0.3, 0]),
        torch.tensor([0.3, -1, 0.3, 0]),
    ])

    # Naive reachability: 0.0283203125
    # Runtime: 2.417327404022217 s
    # Optimal reachabilities: [0.04443359375, 0.5, 0.0458984375]
    # Runtime:  0.42438411712646484 s
    
    # Drawer
    working_directory = "../pipeline/saves/drawer"

    # Different every time
    # randomize_placement(config, working_directory, man_z1, gui=True, tag_to_world=Pose(np.array([
    #     [0, -1, 0],
    #     [-1, 0, 0],
    #     [0, 0, -1]
    # ]), np.array([
    #     [0], [-1], [0]
    # ])))
    # randomized base placement: [-0.13704905 -1.1283503   0.20075546  2.87722077] reachability: 0.5966796875 seed 1984

    eval_scene(config, working_directory, man_z1, gui=True, ics=[
        torch.tensor([0.6, -1.2, 0.4, 0]),
        torch.tensor([-0.5, -1.5, 0.3, 0]),
        torch.tensor([0.3, -1, 0.3, 0]),
    ], tag_to_world=Pose(np.array([
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, -1]
    ]), np.array([
        [0], [-1], [0]
    ])))

    # Naive reachability: 0.70263671875
    # Runtime: 2.3063149452209473 s
    # Optimal reachabilities: [0.7763671875, 0.7783203125, 0.77783203125]
    # Runtime: 0.44042491912841797 s
    
    # Mixed
    working_directory = "../pipeline/saves/mixed"

    # Different every time
    # randomize_placement(config, working_directory, man_z1, gui=True, tag_to_world=Pose(np.array([
    #     [0, -1, 0],
    #     [-1, 0, 0],
    #     [0, 0, -1]
    # ]), np.array([
    #     [0], [-1], [0]
    # ])))
    # randomized base placement: [-0.28231832 -1.18392419  0.36786509 -0.78814523] reachability: 0.07177734375 seed 1092

    eval_scene(config, working_directory, man_z1, gui=True, ics=[
        None,
        torch.tensor([-0.25, -0.93, 0.1, 0])
    ], tag_to_world=Pose(np.array([
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, -1]
    ]), np.array([
        [0], [-1], [0]
    ])))

    # Naive reachability: 0.150390625
    # Runtime: 2.1132431030273438 s
    # Optimal reachabilities: [0.61865234375, 0.611328125]
    # Runtime: 0.4508085250854492 s

    pass