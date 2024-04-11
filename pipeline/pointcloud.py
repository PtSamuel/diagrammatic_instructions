
import numpy as np
import cv2
from dt_apriltags import Detector
import matplotlib.pyplot as plt
import os
from datetime import datetime
from utils import draw_SO3, read_yaml, find_bounding_box
from general import Pose

class PointCloud:
    def __init__(self, images_path, high_res_depth, config):

        self.config = config

        self.camera_params = list(config['camera_intrinsics']['high_res_depth'].values())

        self.apriltag_family = config['apriltag']['family']
        self.tag_size = config['apriltag']['tag_size']
        
        rgb_path = os.path.join(images_path, config['io']['high_res_rgb_name'])
        print("loading images from:", rgb_path)
        low_res_size = (config['data']['low_res_rbg']['width'], config['data']['low_res_rbg']['height'])

        self.high_res_depth = high_res_depth
        depth_path = os.path.join(images_path, 
            config['io']['high_res_depth_name'] if high_res_depth else \
            config['io']['low_res_depth_name']
        )

        from PIL import Image

        try:
            self.rgb_image = np.asarray(Image.open(rgb_path))
            # self.rgb_image = plt.imread(rgb_path).astype(np.uint8)
            # self.low_res_rgb_image = np.array(self.rgb_image) 
            # self.low_res_rgb_image.resize((640, 480))
            self.low_res_rgb_image = cv2.resize(self.rgb_image, low_res_size)
            self.depth_image = np.load(depth_path)
        except:
            print(f"load image from {images_path} failed")
            exit(1)
        
        self.depth_image[self.depth_image == 65535] = 0
        self.region = None
        self.region_constraint = None

        self.ui_width = config['ui']['width']
        self.ui_height = config['ui']['height']

        self.pointcloud_save_path = os.path.join(images_path, config['io']['pointcloud_name'])
        self.pointcloud_constraint_save_path = os.path.join(images_path, config['io']['constraint_name'])
        self.pointcloud_pixels_save_path = os.path.join(images_path, config['io']['pointcloud_pixels_name'])
        self.constraint_pixels_save_path = os.path.join(images_path, config['io']['constraint_pixels_name'])

    def get_roi_mask(self):
        try:
            mask = np.load(self.pointcloud_pixels_save_path)
            return mask
        except:
            print("cannot open", self.pointcloud_pixels_save_path)

    def get_constraint_mask(self):
        try:
            mask = np.load(self.constraint_pixels_save_path)
            return mask
        except:
            print("cannot open", self.constraint_pixels_save_path)

    def convert_pixel_to_xyz(self, u, v, depth_raw):
        if self.high_res_depth:
            fx, fy, cx, cy = self.camera_params
        else:
            fx, fy, cx, cy = list(self.config['camera_intrinsics']['low_res_depth'].values())

        Z = depth_raw[v, u] * 0.001
        X = Z * (u - cx) / fx
        Y = Z * (v - cy) / fy
        return np.array([X, Y, Z])
    
    def convert_camera_xyz_to_tag(self, Pcam, tag):
        return tag.pose_R.T @ (Pcam - tag.pose_t)
    
    def convert_uv_to_tag(self, u, v, depth_raw, tag):
        return self.convert_camera_xyz_to_tag(self.convert_pixel_to_xyz(u, v, depth_raw), tag)
    
    @staticmethod
    def draw_boxes(image, tags):
        image_copy = np.array(image)
        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(image_copy, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))
            cv2.putText(
                image_copy, str(tag.tag_id),
                org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255)
            )
        return image_copy

    def find_camera_pose(self, visualize=False):

        at_detector = Detector(
            families=self.apriltag_family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        self.tags = at_detector.detect(
            cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2GRAY), 
            estimate_tag_pose=True, 
            camera_params=self.camera_params, 
            tag_size=self.tag_size
        )

        if self.tags != []:
            print(self.tags[0])
            if visualize:
                plt.imshow(PointCloud.draw_boxes(self.rgb_image, self.tags))
                plt.show()
                draw_SO3(self.tags[0].pose_R)
            if len(self.tags) > 1:
                self.tags = self.tags[:1]
        else:
            print("no tag is detected")
            exit(1)
    
    # image must be None or a [H, W, 3] np array
    def annotate(self):
        
        import tkinter as tk
        from PIL import Image, ImageTk
        from skimage.draw import polygon

        def on_click(event, canvas):
            canvas.drag_start = (event.x, event.y)
            if canvas.samples is None:
                canvas.samples = [[canvas.drag_start]]
            else: canvas.samples.append([canvas.drag_start])
            canvas.bind("<B1-Motion>", lambda e: on_drag(e, canvas))

        def on_drag(event, canvas):
            x, y = event.x, event.y
            canvas.samples[-1].append((x, y))
            canvas.create_line(canvas.samples[-1][-2], canvas.samples[-1][-1], fill='red', width=2)

        def on_release(event, canvas):
            canvas.unbind("<B1-Motion>")

            if canvas.drag_start != canvas.samples[-1][-1]:
                canvas.samples[-1].append(canvas.drag_start)
            canvas.create_line(canvas.samples[-1][-2], canvas.samples[-1][-1], fill='red', width=2)

        def on_click_constraint(event, canvas):
            canvas.drag_start_constraint = (event.x, event.y)
            if canvas.samples_constraint is None:
                canvas.samples_constraint = [[canvas.drag_start_constraint]]
            else: canvas.samples_constraint.append([canvas.drag_start_constraint])
            canvas.bind("<B1-Motion>", lambda e: on_drag_constraint(e, canvas))

        def on_drag_constraint(event, canvas):
            x, y = event.x, event.y
            canvas.samples_constraint[-1].append((x, y))
            canvas.create_line(canvas.samples_constraint[-1][-2], canvas.samples_constraint[-1][-1], fill='green', width=2)

        def on_release_constraint(event, canvas):
            canvas.unbind("<B1-Motion>")

            if canvas.drag_start_constraint != canvas.samples_constraint[-1][-1]:
                canvas.samples_constraint[-1].append(canvas.drag_start_constraint)
            canvas.create_line(canvas.samples_constraint[-1][-2], canvas.samples_constraint[-1][-1], fill='green', width=2)

        def start_drawing(canvas):
            canvas.bind("<Button-1>", lambda e: on_click(e, canvas))
            canvas.bind("<ButtonRelease-1>", lambda e: on_release(e, canvas))

        def start_drawing_constraint(canvas):
            canvas.bind("<Button-1>", lambda e: on_click_constraint(e, canvas))
            canvas.bind("<ButtonRelease-1>", lambda e: on_release_constraint(e, canvas))

        def add_region():
            start_drawing(canvas)

        def add_constraint():
            start_drawing_constraint(canvas)

        def reset_selection(canvas):
            canvas.create_image(0, 0, image=canvas.photo, anchor='nw')
            canvas.samples = None
            if canvas.samples_constraint is not None:
                for constraint in canvas.samples_constraint:
                    for i in range(len(constraint) - 1):
                        canvas.create_line(constraint[i], constraint[i + 1], fill='green', width=2)
                        if i == len(constraint) - 2:
                            canvas.create_line(constraint[-1], constraint[0], fill='green', width=2)

        def reset_constraint_selection(canvas):
            canvas.create_image(0, 0, image=canvas.photo, anchor='nw')
            canvas.samples_constraint = None
            if canvas.samples is not None:
                for area in canvas.samples:
                    for i in range(len(area) - 1):
                        canvas.create_line(area[i], area[i + 1], fill='red', width=2)
                        if i == len(area) - 2:
                            canvas.create_line(area[-1], area[0], fill='red', width=2)

        def compute_region(canvas, samples, target_shape):
            
            expand_ratio = target_shape[1] / canvas.photo.width()

            mask = np.zeros(target_shape, dtype=np.uint8)

            for i in samples:

                x, y = zip(*i)
                
                x = list(map(lambda a : a * expand_ratio, x))
                y = list(map(lambda a : a * expand_ratio, y))

                rr, cc = polygon(y, x)
                rr = np.clip(rr, 0, target_shape[0])
                cc = np.clip(cc, 0, target_shape[1])

                mask[rr, cc] = 1

            return mask

        def compute_regions(canvas, target_shape):
            
            if canvas.samples is None:
                print("nothing is drawn")
            else:
                try:
                    self.region = compute_region(canvas, canvas.samples, target_shape)
                except:
                    print("compute region failure")
            
            if canvas.samples_constraint is None:
                print("no constraint is drawn")
            else:
                try:
                    self.region_constraint = compute_region(canvas, canvas.samples_constraint, target_shape)
                except:
                    print("compute constraint region failure")

        root = tk.Tk()
        root.title("Image Loader and Editor")

        canvas = tk.Canvas(root, width=1600, height=1200)
        canvas.pack(padx=10, pady=10)

        canvas.samples = None
        canvas.samples_constraint = None

        status_label = tk.Label(root, text="Select an action", relief=tk.SUNKEN, anchor='w')
        status_label.pack(side=tk.BOTTOM, fill=tk.X)

        canvas.config(width=self.ui_width, height=self.ui_height)
        photo = ImageTk.PhotoImage(Image.fromarray(self.low_res_rgb_image))
        canvas.photo = photo 
        canvas.create_image(0, 0, image=photo, anchor='nw')
        status_label.config(text="Image loaded")

        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.TOP, fill=tk.X)

        add_region_button = tk.Button(button_frame, text="Add Region", command=add_region)
        add_region_button.pack(side="left", padx=10, pady=10)

        add_constraint_button = tk.Button(button_frame, text="Add Constraint", command=add_constraint)
        add_constraint_button.pack(side="left", padx=10, pady=10)

        button_reset = tk.Button(button_frame, text="Reset Selection", command=lambda: reset_selection(canvas))
        button_reset.pack(side="left", padx=10, pady=10)

        button_reset_constraint = tk.Button(button_frame, text="Reset Constraint", command=lambda: reset_constraint_selection(canvas))
        button_reset_constraint.pack(side="left", padx=10, pady=10)

        compute_region_button = tk.Button(button_frame, text="Compute Region", command=lambda: compute_regions(canvas, self.depth_image.shape))
        compute_region_button.pack(side="left", padx=10, pady=10)

        root.mainloop()
        return self.region, self.region_constraint

    def compute_points_in_world(self, region, tag_to_world: Pose=None, pixels=False, visualize=False):

        if region is None:
            print("cannot compute points in world frame if no region is selected")
            return None    
        
        if np.all(region == 0):
            print("cannot compute points in world frame if region is empty")
            return None
        
        if len(self.tags) > 1:
            print("cannot compute points in world frame if no april tag detected")
            return None
        
        if len(self.tags) == 0:
            print("cannot compute points in world frame if april tag is not unique")
            return None
        
        region[self.depth_image == 0] = 0
        
        ys, xs = np.where(region)
        print("number of points in pointcloud:", len(xs))

        if visualize:
            resized_region = cv2.resize(region, self.rgb_image.shape[1::-1])

            rgb_region = np.array(self.rgb_image)
            rgb_region[:, :, 0][resized_region > 0] = 255
            rgb_region[:, :, 1][resized_region > 0] = 0
            rgb_region[:, :, 2][resized_region > 0] = 0

            depth_uint8 = (self.depth_image / self.depth_image.max() * 255).astype(np.uint8)
            depth_region = depth_uint8[:,:,np.newaxis].repeat(3, axis=2)
            depth_region[ys, xs] = (255, 0, 0)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(rgb_region)
            ax2.imshow(depth_region)
            fig.show()
        
        if tag_to_world is None:
            tag_to_world = Pose(
                np.array(self.config['apriltag']['R']), 
                np.array(self.config['apriltag']['t'])
            )
        
        points_in_tag = self.convert_uv_to_tag(xs, ys, self.depth_image, self.tags[0])
        points_in_world = tag_to_world.transform(points_in_tag)

        if pixels:
            return points_in_world, self.low_res_rgb_image[ys, xs]

        return points_in_world

    def save_pointcloud(self, pointcloud):

        if pointcloud is None:
            print("cannot save if pointcloud is None")
            return

        save_dir = os.path.dirname(self.pointcloud_save_path)
        assert save_dir, f"{save_dir} does not exist"

        with open(self.pointcloud_save_path, 'wb') as f:
            np.save(f, pointcloud, allow_pickle=False)

        print("saved pointcloud at:", save_dir)

    def save_constraint(self, pointcloud_constraint):

        if pointcloud_constraint is None:
            print("cannot save if constraint pointcloud is None")
            return

        save_dir = os.path.dirname(self.pointcloud_constraint_save_path)
        assert save_dir, f"{save_dir} does not exist"

        with open(self.pointcloud_constraint_save_path, 'wb') as f:
            np.save(f, pointcloud_constraint, allow_pickle=False)

        print("saved constraint at:", save_dir)

    def save_pointcloud_pixels(self, region):
        if region is None:
            print("cannot save if region is None")
            return

        save_dir = os.path.dirname(self.pointcloud_pixels_save_path)
        assert save_dir, f"{save_dir} does not exist"

        with open(self.pointcloud_pixels_save_path, 'wb') as f:
            np.save(f, region, allow_pickle=False)

        print("saved pointcloud pixels at:", save_dir)

    def save_constraint_pixels(self, region_constraint):
        if region_constraint is None:
            print("cannot save if region is None")
            return

        save_dir = os.path.dirname(self.constraint_pixels_save_path)
        assert save_dir, f"{save_dir} does not exist"

        with open(self.constraint_pixels_save_path, 'wb') as f:
            np.save(f, region_constraint, allow_pickle=False)

        print("saved constraint pixels at:", save_dir)
