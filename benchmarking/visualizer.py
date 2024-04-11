import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Visualizer:
    
    def __init__(self, working_directory, config, high_res_depth=False, constraint=True):
        
        self.config = config
       
        rgb_path = os.path.join(working_directory, config['io']['high_res_rgb_name'])
        low_res_size = (config['data']['low_res_rbg']['width'], config['data']['low_res_rbg']['height'])

        self.high_res_depth = high_res_depth
        depth_path = os.path.join(working_directory, 
            config['io']['high_res_depth_name'] if high_res_depth else \
            config['io']['low_res_depth_name']
        )
        
        try:
            self.rgb_image = np.asarray(Image.open(rgb_path))
            self.low_res_rgb_image = cv2.resize(self.rgb_image, low_res_size)
            self.depth_image = np.load(depth_path)
        except:
            print(f"load image from {working_directory} failed")
            exit(1)
        
        self.depth_image[self.depth_image == 65535] = 0
        
        pointcloud_pixels_path = os.path.join(working_directory, config['io']['pointcloud_pixels_name'])
        self.region = np.load(pointcloud_pixels_path)

        if constraint:
            constraint_pixels_path = os.path.join(working_directory, config['io']['constraint_pixels_name'])    
            try:
                self.region_constraint = np.load(constraint_pixels_path)
            except:
                print("cannot open", constraint_pixels_path)

    def label_rgb_region_inplace(self, rgb_region, region, color, opacity=0.5):
        mask = cv2.resize(region, rgb_region.shape[1::-1])
        rgb_region[mask > 0] = rgb_region[mask > 0] * (1 - opacity) + np.array(color) * opacity

    def show_rgb(self, ax, roi=True, constraints=True):
        
        rgb_region = np.array(self.rgb_image)
        if roi:
            self.label_rgb_region_inplace(rgb_region, self.region, [255, 142, 140], opacity=0.75)
        if constraints:
            self.label_rgb_region_inplace(rgb_region, self.region_constraint, [201, 247, 148])

        ax.imshow(rgb_region)
        return rgb_region

    def show_depth(self, ax, roi=True, constraints=True):

        depth_uint8 = (self.depth_image / self.depth_image.max() * 255).astype(np.uint8)
        depth_region = depth_uint8[:,:,np.newaxis].repeat(3, axis=2)

        if roi:
            self.label_rgb_region_inplace(depth_region, self.region, [232, 16, 67])
        if constraints:
            self.label_rgb_region_inplace(depth_region, self.region_constraint, [201, 247, 148])

        depth_region[self.depth_image == 0] = 0
        ax.imshow(depth_region)
        return depth_region
    
    def visualize(self, roi=True, constraints=True):
            
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.axis("off")
        ax2.axis("off")
        self.show_rgb(ax1, roi=roi, constraints=constraints)
        self.show_depth(ax2, roi=roi, constraints=constraints)
        plt.show()