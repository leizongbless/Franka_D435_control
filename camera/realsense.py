import os,sys,time,copy
import open3d
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle as pkl
import datetime
import threading
from collections import deque
from typing import Tuple
import requests
import open3d as o3d
import numpy as np
import cv2
import pyspacemouse
import pyrealsense2 as rs
from termcolor import cprint
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from ..geometry import transform_camera_to_marker
"""
Reinforcement Learning with Prior Reference Policy
"""
class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]
    def d435i_intrinsics(self):
        intrinsics = rs.intrinsics()
        intrinsics.width = 640
        intrinsics.height = 480
        intrinsics.ppx = 321.472
        intrinsics.ppy = 238.99
        intrinsics.fx = 608.25  
        intrinsics.fy = 606.932  
        intrinsics.model = rs.distortion.none
        intrinsics.coeffs = [0, 0, 0, 0, 0]
        return intrinsics

    def L515_intrinsics(self):
        intrinsics = rs.intrinsics()
        intrinsics.width = 640
        intrinsics.height = 480
        intrinsics.ppx = 329.2
        intrinsics.ppy = 249.603
        intrinsics.fx = 597.574 
        intrinsics.fy = 597.479 

        intrinsics.model = rs.distortion.brown_conrady  

        intrinsics.coeffs = [0.138903, -0.469718, -0.000189729, 0.000617071, 0.432173]

        return intrinsics



    def __init__(self, name, serial_number, dim=(640, 480), fps=30, depth=False):
        self.name = name
        self.depth_scale = 0.001  
        

        # The currently deployed device is a D435I.  Use its intrinsics rather
        # than the old L515 calibration that was previously hard-coded here.
        self.intrinsics = self.d435i_intrinsics()

        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.depth = depth

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            # pc = rs.pointcloud()  # 默认初始化
            self.pc = rs.pointcloud() 
            self.p = rs.points()
            # self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
            self.cfg.enable_stream(rs.stream.depth, dim[0]//2, dim[1]//2, rs.format.z16, fps)
        
        
        self.profile = self.pipe.start(self.cfg)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.frame_queue = deque(maxlen=2)
        self.enable = True
        self.t = threading.Thread(target=self._read_img, daemon = True)
        self.t.start()

    def _read_img(self):
        while self.enable:
            
            frames = self.pipe.wait_for_frames()
            
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            
            if self.depth:
                depth_frame = aligned_frames.get_depth_frame()
                
                self.p = self.pc.calculate(depth_frame)
                self.pc.map_to(color_frame)
                # # space point
                vertex_data = self.p.get_vertices()
                vertices = np.asanyarray(vertex_data).copy()
                vtx = vertices.view(np.float32).reshape(-1,3)
                # # pointcloud color
                texture_data = self.p.get_texture_coordinates()
                # texture_coords = np.asanyarray(texture_data)
                texture_co = np.asarray(color_frame.get_data()).reshape(-1,3).copy()
                texture_co = texture_co/255.
                texture_co = np.clip(texture_co, 0., 1,)
                

                

                marker2camera = [
                    0.2364458334575025,
                    0.18099756287414623,
                    0.7879964396312072, 
                    -0.002088750550759319, 
                    0.9247641972605546, 
                    -0.3783259216928415, 
                    -0.040942808421516035]

                t = np.array(marker2camera[0:3])  # translation
                q = np.array(marker2camera[3:])   # [x, y, z, w]
                r = R.from_quat(q)
                R_cam2marker = r.as_matrix()  # R

                vtx_marker = transform_camera_to_marker(vtx, R_cam2marker, t)
            
                            
                x = vtx_marker[:, 0]
                y = vtx_marker[:, 1]
                z = vtx_marker[:, 2]

                # mask = (y >= -0.010) & (z >= 0.0277) & (z <= 0.5) 
                mask = (z>0.021)&(y >= -0.3)&(x>-0.11)#(z>0.012) (y >= -0.25) &(x >-0.03)
                vtx_clipped = vtx_marker[mask]
                # tex_clipped = texture_co[mask]
                # print(vtx_clipped.shape)
                # print(vtx_clipped.shape)
                # vtx_down = furthest_point_sampling(vtx_clipped, 2048)
                # vtx_down = fast_furthest_point_sampling(points=vtx_clipped, n_samples=2048, batch_size=256)
                vtx_down = voxel_downsampling(vtx_clipped, n_samples=2048)

                # # 可视化点云
                # points = o3d.geometry.PointCloud()
                # points.points = o3d.utility.Vector3dVector(vtx_clipped)
                # points.colors = o3d.utility.Vector3dVector(tex_clipped)

                # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # o3d.visualization.draw_geometries([points,axis], window_name="PointCloud in Marker Frame")
                

                    
            
            if color_frame.is_video_frame():
                image = np.asarray(color_frame.get_data())
                
                if self.depth and depth_frame.is_depth_frame():
                    # depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                    # output = np.concatenate((image, depth), axis=-1)

                    # points_np = np.asarray(points.points)
                    # points_np = np.clip(points_np, -100.0, 100.0)
                    output = {
                        'color_frame':color_frame,
                        'depth_frame':depth_frame,
                        "vtx_down":vtx_down
                        # 'pointcloud':points_np.copy(),
                        # "vis_points": points.copy()
                    }
                    # print(f'image : {image.shape}')
                    # print(f'output : {output.shape}')
                  

                    self.frame_queue.append(output)
                    # return True, output
                else:
                    # When depth=False, still create a dictionary with color_frame
                    output = {
                        'color_frame': color_frame
                    }
                    self.frame_queue.append(output)
                    
                    # return True, image
            else:
                raise NotImplementedError("error happen when reading frames")
                return False, None

    def get_latest_frame(self):
        if self.frame_queue:
            # print(self.frame_queue[-1].copy())
            # print(f'frame_queue : {len(self.frame_queue)}')
            return self.frame_queue[-1].copy()
        else:
            return None
 
    def close(self):
        self.enable = False
        self.pipe.stop()
        self.cfg.disable_all_streams()

    
        
class PointNetEncoderXYZ(nn.Module):
    def __init__(
            self,
            in_channels: int = 3, 
            out_channels: int = 1024, 
            use_layer_norm: bool = False,
            final_norm: str = "none",
            use_projection: bool = True,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs):
        self.device = device
        super(PointNetEncoderXYZ, self).__init__()
        block_channels = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use layer norm: {}".format(use_layer_norm), "cyan")
        cprint("[PointNetEncoderXYZ] final norm: {}".format(final_norm), "cyan")

        assert in_channels == 3, cprint(
            f"[PointNetEncoderXYZ] in_channels must be 3, but got {in_channels}", "red"
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channels[0]),
            nn.LayerNorm(block_channels[0]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channels[0], block_channels[1]),
            nn.LayerNorm(block_channels[1]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channels[1], block_channels[2]), 
            nn.LayerNorm(block_channels[2]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),   
        )
        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channels[2], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == "norm":
            self.final_projection = nn.Linear(block_channels[-1], out_channels)
        else:
            raise ValueError(f"Unsupported final norm: {final_norm}")
        
        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] Not using projection, final projection is Identity", "cyan")

        self.to(self.device)

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]  # Global max pooling
        x = self.final_projection(x)
        return x

def furthest_point_sampling(points, n_samples=2048, colors=None):
    """
    points: [N, 3] tensor containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically &lt;&lt; N 
    """
    # Convert points to PyTorch tensor if not already and move to GPU
    points = torch.Tensor(points).cuda()  # [N, 3]
    if colors is not None:
        colors = torch.Tensor(colors).cuda()

    # Number of points
    num_points = points.size(0)  # N

    # Initialize an array for the sampled indices
    sample_inds = torch.zeros(n_samples, dtype=torch.long).cuda()  # [S]

    # Initialize distances to inf
    dists = torch.ones(num_points).cuda() * float('inf')  # [N]

    # Select the first point randomly
    selected = torch.randint(num_points, (1,), dtype=torch.long).cuda()  # [1]
    sample_inds[0] = selected

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        last_added = sample_inds[i - 1]  # Scalar
        dist_to_last_added_point = torch.sum((points[last_added] - points) ** 2, dim=-1)  # [N]

        # If closer, update distances
        dists = torch.min(dist_to_last_added_point, dists)  # [N]

        # Pick the one that has the largest distance to its nearest neighbor in the sampled set
        selected = torch.argmax(dists)  # Scalar
        sample_inds[i] = selected
    if colors is None:
        return points[sample_inds].cpu().numpy()
    else:
        return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()  # [S, 3]

def fast_furthest_point_sampling(points, n_samples=2048, colors=None, batch_size=32):
    """
    
    参数:
        points: [N, 3] 
        n_samples: target number of pointcloud
        colors: optional
        batch_size: batch to calculate distance
    """
    # torch tensor
    points = torch.tensor(points, dtype=torch.float32).cuda()
    N = points.shape[0]
    
    # init index
    sample_inds = torch.zeros(n_samples, dtype=torch.long).cuda()
    
    # random first point
    first_idx = torch.randint(N, (1,)).item()
    sample_inds[0] = first_idx
    
    #
    dists = torch.full((N,), float('inf'), dtype=torch.float32).cuda()
    
    # batch distance
    for i in range(1, n_samples, batch_size):
        # 
        batch_end = min(i + batch_size, n_samples)
        batch_size_current = batch_end - i
        
        # 
        selected_points = points[sample_inds[:i]]
        
        
        dist_matrix = torch.cdist(points, selected_points, p=2)  # [N, i]
        min_dists = dist_matrix.min(dim=1).values  # [N]
        
        
        dists = torch.min(dists, min_dists)
        
        
        mask = torch.ones(N, dtype=torch.bool).cuda()
        mask[sample_inds[:i]] = False
        
        _, topk_inds = torch.topk(dists[mask], batch_size_current, largest=True)
        valid_inds = torch.where(mask)[0]
        new_inds = valid_inds[topk_inds]
        
        sample_inds[i:batch_end] = new_inds
    
    if colors is None:
        return points[sample_inds].cpu().numpy()
    else:
        colors_tensor = torch.tensor(colors).cuda()
        return points[sample_inds].cpu().numpy(), colors_tensor[sample_inds].cpu().numpy()

def voxel_downsampling(points, n_samples=2048, mode='centroid', max_iter=10):
    """
    voxel point sample
    """
    points_tensor = torch.tensor(points, dtype=torch.float32)
    points_tensor = points_tensor.to('cuda')
    # boarder info
    bbox_min = torch.min(points_tensor, dim=0)[0]
    bbox_max = torch.max(points_tensor, dim=0)[0]
    volume = torch.prod(bbox_max - bbox_min).item()

    # voxel_size 
    voxel_min = ((volume / n_samples) ** (1/3)) * 0.1
    voxel_max = ((volume / n_samples) ** (1/3)) * 2.0

    best_sampled = None
    best_diff = float('inf')

    for _ in range(max_iter):
        voxel_size = (voxel_min + voxel_max) / 2

        voxel_indices = torch.floor((points_tensor - bbox_min) / voxel_size).long()
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        )

        if mode == 'centroid':
            sum_points = torch.zeros((len(unique_voxels), 3), dtype=torch.float32, device='cuda')
            sum_points.index_add_(0, inverse_indices, points_tensor)
            sampled_points = sum_points / counts.unsqueeze(1)

        elif mode == 'random':
            rand_idx = torch.randint(0, counts.max().item(), (len(unique_voxels),), device='cuda')
            selected_indices = []
            for i in range(len(unique_voxels)):
                mask = (inverse_indices == i).nonzero(as_tuple=True)[0]
                if len(mask) > 0:
                    selected = mask[rand_idx[i] % len(mask)]
                    selected_indices.append(selected)
            sampled_points = points_tensor[torch.stack(selected_indices)]

        diff = abs(sampled_points.shape[0] - n_samples)
        if diff < best_diff:
            best_sampled = sampled_points
            best_diff = diff

        if sampled_points.shape[0] > n_samples:
            voxel_min = voxel_size
        else:
            voxel_max = voxel_size

    # compensation
    num_points = best_sampled.shape[0]
    if num_points > n_samples:
        indices = torch.randperm(num_points, device='cuda')[:n_samples]
        final_sampled = best_sampled[indices]
    elif num_points < n_samples:
        repeat_count = n_samples - num_points
        indices = torch.randint(0, num_points, (repeat_count,), device='cuda')
        final_sampled = torch.cat([best_sampled, best_sampled[indices]], dim=0)
    else:
        final_sampled = best_sampled

    return final_sampled.cpu().numpy()
