import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
import os.path as osp

import numpy as np
from PIL import Image
from tqdm import tqdm
from scene.cameras import Camera
from typing import NamedTuple
from utils.graphics_utils import focal2fov, fov2focal
import glob
from torchvision import transforms as T
import open3d as o3d
from tqdm import trange
import imageio.v2 as iio
import cv2
import copy
import torch
import torch.nn.functional as F
from utils.general_utils import inpaint_depth, inpaint_rgb
import torch
from pytorch3d.ops import sample_farthest_points

def generate_se3_matrix(translation, rotation_rad):


    # Create rotation matrices around x, y, and z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_rad[0]), -np.sin(rotation_rad[0])],
                   [0, np.sin(rotation_rad[0]), np.cos(rotation_rad[0])]])

    Ry = np.array([[np.cos(rotation_rad[1]), 0, np.sin(rotation_rad[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_rad[1]), 0, np.cos(rotation_rad[1])]])

    Rz = np.array([[np.cos(rotation_rad[2]), -np.sin(rotation_rad[2]), 0],
                   [np.sin(rotation_rad[2]), np.cos(rotation_rad[2]), 0],
                   [0, 0, 1]])

    # Combine rotations
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Create S(3) matrix
    se3_matrix = np.eye(4)


    se3_matrix[:3, :3] = R
    se3_matrix[:3, 3] = translation

    return se3_matrix

def update_extr(c2w, rotation_deg, radii_mm):
        rotation_rad = np.radians(rotation_deg)
        translation = np.array([-radii_mm * np.sin(rotation_rad) , 0, radii_mm * (1 - np.cos(rotation_rad))])
        # translation = np.array([0, 0, 10])
        se3_matrix = generate_se3_matrix(translation, (0,rotation_rad,0)) # transform_C_C'
        extr = np.linalg.inv(se3_matrix) @ np.linalg.inv(c2w) # transform_C'_W = transform_C'_C @ (transform_W_C)^-1
        
        return np.linalg.inv(extr) # c2w
    
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
    Zfar: float
    Znear: float

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

class EndoNeRF_Dataset(object):
    def __init__(
        self,
        datadir,
        downsample=1.0,
        test_every=8
    ):
        self.img_wh = (
            int(640 / downsample),
            int(512 / downsample),
        )
        self.root_dir = datadir
        self.downsample = downsample 
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()
        self.white_bg = False

        self.load_meta()
        print(f"meta data loaded, total image:{len(self.image_paths)}")
        
        n_frames = len(self.image_paths)
        self.train_idxs = [i for i in range(n_frames) if (i-1) % test_every != 0]
        self.test_idxs = [i for i in range(n_frames) if (i-1) % test_every == 0]
        self.video_idxs = [i for i in range(n_frames)]
        # self.train_idxs = [142, 143]
        # self.test_idxs = [143, 144]
        # self.video_idxs = [142, 143, 144]
        
        self.maxtime = 1.0
        
    def load_meta(self):
        """
        Load meta data from the dataset.
        """

        print(f"{self.root_dir = }")
        
        # coordinate transformation 
        if 'stereomis' in self.root_dir:
            print("Loading meta data from StereoMIS dataset...")
            # poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
            # try:
            #     poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
            # except: 
            #     # No far and near
            #     poses = poses_arr.reshape([-1, 3, 5])  # (N_cams, 3, 5)
            # # StereoMIS has well calibrated intrinsics predict using DL model
            # # which resize the image before predict so H,W is 320*250 instead of the size of training images which is640*512
            # old_H, old_W, focal = poses[0, :, -1]
            # # focal = focal / self.downsample
            # cx = 640//2
            # scale_w = 640 / (old_W)
            # f_x = focal * scale_w

            # cy = 512//2
            # scale_h = 512 / (old_H)
            # f_y = focal * scale_h
            
            # self.focal = (f_x, f_y)
            # self.K = np.array([[f_x, 0 , cx],
            #                    [0, f_y, cy],
            #                    [0, 0, 1]]).astype(np.float32)
            # poses = np.concatenate([poses[..., :1], poses[..., 1:2], poses[..., 2:3], poses[..., 3:4]], -1)

            poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
            try:
                poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
            except: 
                # No far and near
                poses = poses_arr.reshape([-1, 3, 5])  # (N_cams, 3, 5)
            # StereoMIS has well calibrated intrinsics
            cy, cx, focal =  poses[0, :, -1]
            cy = 512//2
            cx = 640//2
            focal = focal / self.downsample
            self.focal = (focal, focal)
            self.K = np.array([[focal, 0 , cx],
                               [0, focal, cy],
                               [0, 0, 1]]).astype(np.float32)
            poses = np.concatenate([poses[..., :1], poses[..., 1:2], poses[..., 2:3], poses[..., 3:4]], -1)
        else: 
            print("Loading meta data from EndoNeRF dataset...")
            # load poses
            poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
            poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
            H, W, focal = poses[0, :, -1]
            focal = focal / self.downsample
            self.focal = (focal, focal)
            self.K = np.array([[focal, 0 , W//2],
                                [0, focal, H//2],
                                [0, 0, 1]]).astype(np.float32)
            poses = np.concatenate([poses[..., :1], -poses[..., 1:2], -poses[..., 2:3], poses[..., 3:4]], -1)

            # with open(os.path.join(self.root_dir, "K.npy"), "wb") as f:
            #     np.save(f, self.K)
            #     np.save(f, 4.274914654808847)

        with open(os.path.join(self.root_dir, "K.npy"), "rb") as f:
            _ = np.load(f) # K
            self.baseline = np.load(f) # baseline

        # prepare poses
        self.image_poses = []
        self.image_times = []
        for idx in range(poses.shape[0]):
            pose = poses[idx]
            c2w = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0) # 4x4
            
            # # ======================Generate the novel view for infer (StereoMIS)==========================
            # thetas = np.linspace(0, 30, poses.shape[0], endpoint=False)
            # c2w = update_extr(c2w, rotation_deg=thetas[idx], radii_mm=30)
            # # =================================================================================
            
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, -1] #w2c
            R = np.transpose(R) #c2w
            self.image_poses.append((R, T))
            # change time=idx/cnt_images to idx, process when deform since each set need different offset
            self.image_times.append(idx/poses.shape[0])
            # self.image_times.append(idx)
        
        # get paths of images, depths, masks, etc.
        agg_fn = lambda filetype, ext: sorted(glob.glob(os.path.join(self.root_dir, filetype, f"*.{ext}")))
        self.image_paths = agg_fn("images", "png")
        self.image_right_paths = agg_fn("images_right", "png")
        self.depth_paths = agg_fn("depth", "png")
        self.fs_depth_paths = agg_fn("fs_depth", "npy")
        self.masks_paths = agg_fn("masks", "png")

        assert len(self.image_paths) == poses.shape[0], "the number of images should equal to the number of poses"
        assert len(self.image_right_paths) == poses.shape[0], "the number of right images should equal to the number of poses"
        assert len(self.depth_paths) == poses.shape[0], "the number of depth images should equal to number of poses"
        assert len(self.fs_depth_paths) == poses.shape[0], "the number of fs_depth images should equal to number of poses"
        assert len(self.masks_paths) == poses.shape[0], "the number of masks should equal to the number of poses"
        
    def format_infos(self, split):
        cameras = []

        # Remove last training for optical flow t+1
        if split == 'train': idxs = self.train_idxs
        elif split == 'test': idxs = self.test_idxs
        else: idxs = self.video_idxs

        
        
        for idx in tqdm(idxs):
            # mask / depth
            mask_path = self.masks_paths[idx]
            mask = Image.open(mask_path).convert('L')
            # StereoMIS 
            if 'stereomis' in self.root_dir:
                mask = np.array(mask)
            else:
                mask = 1 - np.array(mask) / 255.0
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path))
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 99.8)
            depth = np.clip(depth, close_depth, inf_depth) 
            depth = torch.from_numpy(depth)
            mask = self.transform(mask).bool()
            # color
            color = np.array(Image.open(self.image_paths[idx]))/255.0
            image = self.transform(color)
            # times           
            time = self.image_times[idx]
            # poses
            R, T = self.image_poses[idx]
            # fov
            FovX = focal2fov(self.focal[0], self.img_wh[0])
            FovY = focal2fov(self.focal[1], self.img_wh[1])

            image_right = self.transform(np.array(Image.open(self.image_right_paths[idx]))/255.0)

            # fs_depth = np.load(self.fs_depth_paths[idx])
            # fs_depth = torch.from_numpy(fs_depth)
            fs_depth = depth

            # depth = fs_depth
            
            cameras.append(Camera(colmap_id=idx, R=R, T=T, FoVx=FovX, FoVy=FovY,image=image, image_right=image_right, baseline=self.baseline, depth=depth, fs_depth=fs_depth, mask=mask, gt_alpha_mask=None,
                          image_name=f"{idx}", uid=idx, data_device=torch.device("cuda"), time=time,
                          Znear=None, Zfar=None, K=self.K, h=self.img_wh[1], w=self.img_wh[0]))
        return cameras
    
    def filling_pts_colors(self, filling_mask, ref_depth, ref_image):
         # bool
        refined_depth = inpaint_depth(ref_depth, filling_mask)
        refined_rgb = inpaint_rgb(ref_image, filling_mask)
        return refined_rgb, refined_depth

    def get_sparse_pts(self, st, cen, ed, sample=True):
        R, T = self.image_poses[cen]
        depth = np.array(Image.open(self.depth_paths[cen]))
        depth_mask = np.ones(depth.shape).astype(np.float32)
        close_depth = np.percentile(depth[depth!=0], 0.1)
        inf_depth = np.percentile(depth[depth!=0], 99.9)
        depth_mask[depth>inf_depth] = 0
        depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
        depth_mask[depth==0] = 0
        depth[depth_mask==0] = 0
        if 'stereomis' in self.root_dir:
            mask = np.array(Image.open(self.masks_paths[cen]).convert('L'))
        else:
            mask = 1 - np.array(Image.open(self.masks_paths[cen]))/255.0
        mask = np.logical_and(depth_mask, mask)   
        color = np.array(Image.open(self.image_paths[cen]))/255.0
        # color_uint8 = np.array(Image.open(self.image_paths[0]), dtype=np.uint8)
        # filling_mask = np.logical_not(mask)
        # color, depth = self.filling_pts_colors(ref_image=color_uint8, ref_depth=depth, filling_mask=filling_mask)
        # color = color/255.0
        
        # st, cen, ed
        pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
        c2w = self.get_camera_poses((R, T))
        pts = self.transform_cam2cam(pts, c2w)
        
        pts, colors = self.search_pts_colors_with_motion(st, cen, ed, pts, colors, mask, c2w)
        
        normals = np.zeros((pts.shape[0], 3))

        if sample:
            num_sample = int(0.1 * pts.shape[0])
            sel_idxs = np.random.choice(pts.shape[0], num_sample, replace=False)
            # sampled_points, sel_idxs = sample_farthest_points(
            #     torch.from_numpy(pts).float().unsqueeze(0), 
            #     lengths=None,          
            #     K=int(0.1 * pts.shape[0]), 
            #     # random_start_point=True
            # )
            # sel_idxs  = sel_idxs[0].cpu().numpy()  
            # print("Shape of sampled compl_pts points:", sampled_points.shape)   
            # print("Shape of sampled compl_pts indices:", sel_idxs.shape) 
            pts = pts[sel_idxs, :]
            colors = colors[sel_idxs, :]
            normals = normals[sel_idxs, :]
        print("pt cloud shape: ", pts.shape)
        return pts, colors, normals

    # def calculate_motion_masks(self, st, cen, ed):
    #     images = []
    #     for j in range(0, len(self.image_poses)):
    #         color = np.array(Image.open(self.image_paths[j]))/255.0
    #         images.append(color)
    #     images = np.asarray(images).mean(axis=-1)
    #     diff_map = np.abs(images - images.mean(axis=0))
    #     diff_thrshold = np.percentile(diff_map[diff_map!=0], 95)
    #     return diff_map > diff_thrshold

    def calculate_motion_masks(self, st, cen, ed):
        images = []
        # 0, 261
        for j in range(st, ed):
            color = np.array(Image.open(self.image_paths[j])) / 255.0
            images.append(color)
    
        # Convert images to grayscale
        images = np.asarray(images).mean(axis=-1)
    
        # Initialize MOG2 background subtractor
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=ed-st, varThreshold=4, detectShadows=False
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        motion_masks = []
        # TODO: check center?
        for i, frame in enumerate(images[::-1]):
            # Scale to uint8 for MOG2
            # if i == cen: continue
            frame_uint8 = (frame * 255).astype(np.uint8)
            
            # Apply MOG2 to get foreground mask
            fg_mask = mog2.apply(frame_uint8)
            

            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Removes small noise
            #fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fills small holes
    
            # Append the motion mask
            motion_masks.append(fg_mask)
        # Stack motion masks to match diff_map shape
        motion_masks = np.stack(motion_masks[::-1], axis=0)
        return motion_masks > 0
    def calculate_motion_masks_invert(self, st, cen, ed):
        images = []
        # 0, 261
        for j in range(st, ed):
            color = np.array(Image.open(self.image_paths[j])) / 255.0
            images.append(color)
        
        # Convert images to grayscale
        images = np.asarray(images).mean(axis=-1)
        
        # Initialize MOG2 background subtractor
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=ed-st, varThreshold=4, detectShadows=False
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        motion_masks = []
        # TODO: check center?
        for i, frame in enumerate(images):
            # Scale to uint8 for MOG2
            # if i == cen: continue
            frame_uint8 = (frame * 255).astype(np.uint8)
            
            # Apply MOG2 to get foreground mask
            fg_mask = mog2.apply(frame_uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Removes small noise
            #fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel) 
    
            # Append the motion mask
            motion_masks.append(fg_mask)
        
        # Stack motion masks to match diff_map shape
        motion_masks = np.stack(motion_masks[::-1], axis=0)
        return motion_masks > 0   
        
    def search_pts_colors_with_motion(self, st, cen, ed, ref_pts, ref_color, ref_mask, ref_c2w):
        # calculating the motion mask
        motion_mask = self.calculate_motion_masks(st, cen, ed)
        interval = 1
        if len(self.image_poses) > 150: # in case long sequence
            interval = 2
        if cen == 0:
            for j in range(1,  len(self.image_poses), interval):
                # if j > len(self.image_poses)//2 : 
                #     motion_mask[0] = motion_mask[0]*False
                    
                ref_mask_not = np.logical_not(ref_mask)
                ref_mask_not = np.logical_or(ref_mask_not, motion_mask[0])
                R, T = self.image_poses[j]
                c2w = self.get_camera_poses((R, T))
                c2ref = np.linalg.inv(ref_c2w) @ c2w
                depth = np.array(Image.open(self.depth_paths[j]))
                color = np.array(Image.open(self.image_paths[j]))/255.0
                # mask = 1 - np.array(Image.open(self.masks_paths[0]))/255.0   
                if 'stereomis' in self.root_dir:
                    mask = np.array(Image.open(self.masks_paths[j]).convert('L'))
                else:
                    mask = 1 - np.array(Image.open(self.masks_paths[j]))/255.0       
                depth_mask = np.ones(depth.shape).astype(np.float32)
                close_depth = np.percentile(depth[depth!=0], 3.0)
                inf_depth = np.percentile(depth[depth!=0], 99.8)
                depth_mask[depth>inf_depth] = 0
                depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
                depth_mask[depth==0] = 0
                mask = np.logical_and(depth_mask, mask)
                depth[mask==0] = 0
                
                pts, colors, mask_refine = self.get_pts_cam(depth, mask, color)
                pts = self.transform_cam2cam(pts, c2ref) # Nx3
                X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
                X, Y, Z = X[Z!=0], Y[Z!=0], Z[Z!=0]
                X_Z, Y_Z = X / Z, Y / Z
                X_Z = (X_Z * self.focal[0] + self.K[0,-1]).astype(np.int32)
                Y_Z = (Y_Z * self.focal[1] + self.K[1,-1]).astype(np.int32)
                # Out of the visibility
                out_vis_mask = ((X_Z > (self.img_wh[0]-1)) + (X_Z < 0) +\
                        (Y_Z > (self.img_wh[1]-1)) + (Y_Z < 0))>0
                out_vis_pt_idx = np.where(out_vis_mask)[0]
                visible_mask = (1 - out_vis_mask)>0
                X_Z = np.clip(X_Z, 0, self.img_wh[0]-1)
                Y_Z = np.clip(Y_Z, 0, self.img_wh[1]-1)
                coords = np.stack((Y_Z, X_Z), axis=-1)
                proj_mask = np.zeros((self.img_wh[1], self.img_wh[0])).astype(np.float32)
                proj_mask[coords[:, 0], coords[:, 1]] = 1
                compl_mask = (ref_mask_not * proj_mask)
                index_mask = compl_mask.reshape(-1)[mask_refine]
                compl_idxs = np.nonzero(index_mask.reshape(-1))[0]
                if compl_idxs.shape[0] <= 50:
                    continue
                compl_pts = pts[compl_idxs, :]
                compl_pts = self.transform_cam2cam(compl_pts, ref_c2w)
                compl_colors = colors[compl_idxs, :]
                sel_idxs = np.random.choice(compl_pts.shape[0], int(0.1*compl_pts.shape[0]), replace=True)
    
                
                # sampled_points, sel_idxs = sample_farthest_points(
                #     torch.from_numpy(compl_pts).float().unsqueeze(0), 
                #     lengths=None,          
                #     K=int(0.1 * compl_pts.shape[0]), 
                #     # random_start_point=True
                # )
    
                # sel_idxs  = sel_idxs[0].cpu().numpy()           # shape (474,)
                # selected_pts = compl_pts[sel_idxs_np]              # shape (K, 3)
    
                # print("Shape of sampled compl_pts points:", sampled_points.shape)   
                # print("Shape of sampled compl_pts indices:", sel_idxs.shape) 
    
    
                ref_pts = np.concatenate((ref_pts, compl_pts[sel_idxs]), axis=0)
                ref_color = np.concatenate((ref_color, compl_colors[sel_idxs]), axis=0)
                ref_mask = np.logical_or(ref_mask, compl_mask)
    
                # ref_pts = np.concatenate((ref_pts, compl_pts), axis=0)
                # ref_color = np.concatenate((ref_color, compl_colors), axis=0)
                # ref_mask = np.logical_or(ref_mask, compl_mask)
    
            
            if ref_pts.shape[0] > 600000:
                sel_idxs = np.random.choice(ref_pts.shape[0], 500000, replace=True) 
                # sampled_points, sel_idxs = sample_farthest_points(
                #     torch.from_numpy(ref_pts).float().unsqueeze(0), 
                #     lengths=None,          
                #     K=500000, 
                #     # random_start_point=True
                # )
                # sel_idxs  = sel_idxs[0].cpu().numpy()  
                # print("Shape of sampled ref_pts points:", sampled_points.shape)   
                # print("Shape of sampled ref_pts indices:", sel_idxs.shape) 
                ref_pts = ref_pts[sel_idxs]         
                ref_color = ref_color[sel_idxs] 
        else:
            motion_mask = self.calculate_motion_masks_invert(st, cen, ed)
            for j in range(len(self.image_poses)-1, 0,  -interval):
                if j < len(self.image_poses)//2 : 
                    motion_mask[0] = motion_mask[0]*False
                ref_mask_not = np.logical_not(ref_mask)
                ref_mask_not = np.logical_or(ref_mask_not, motion_mask[0])
                R, T = self.image_poses[j]
                c2w = self.get_camera_poses((R, T))
                c2ref = np.linalg.inv(ref_c2w) @ c2w
                depth = np.array(Image.open(self.depth_paths[j]))
                color = np.array(Image.open(self.image_paths[j]))/255.0
                # mask = 1 - np.array(Image.open(self.masks_paths[0]))/255.0   
                if 'stereomis' in self.root_dir:
                    mask = np.array(Image.open(self.masks_paths[j]).convert('L'))
                else:
                    mask = 1 - np.array(Image.open(self.masks_paths[j]))/255.0       
                depth_mask = np.ones(depth.shape).astype(np.float32)
                close_depth = np.percentile(depth[depth!=0], 3.0)
                inf_depth = np.percentile(depth[depth!=0], 99.8)
                depth_mask[depth>inf_depth] = 0
                depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
                depth_mask[depth==0] = 0
                mask = np.logical_and(depth_mask, mask)
                depth[mask==0] = 0
                
                pts, colors, mask_refine = self.get_pts_cam(depth, mask, color)
                pts = self.transform_cam2cam(pts, c2ref) # Nx3
                X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
                X, Y, Z = X[Z!=0], Y[Z!=0], Z[Z!=0]
                X_Z, Y_Z = X / Z, Y / Z
                X_Z = (X_Z * self.focal[0] + self.K[0,-1]).astype(np.int32)
                Y_Z = (Y_Z * self.focal[1] + self.K[1,-1]).astype(np.int32)
                # Out of the visibility
                out_vis_mask = ((X_Z > (self.img_wh[0]-1)) + (X_Z < 0) +\
                        (Y_Z > (self.img_wh[1]-1)) + (Y_Z < 0))>0
                out_vis_pt_idx = np.where(out_vis_mask)[0]
                visible_mask = (1 - out_vis_mask)>0
                X_Z = np.clip(X_Z, 0, self.img_wh[0]-1)
                Y_Z = np.clip(Y_Z, 0, self.img_wh[1]-1)
                coords = np.stack((Y_Z, X_Z), axis=-1)
                proj_mask = np.zeros((self.img_wh[1], self.img_wh[0])).astype(np.float32)
                proj_mask[coords[:, 0], coords[:, 1]] = 1
                compl_mask = (ref_mask_not * proj_mask)
                index_mask = compl_mask.reshape(-1)[mask_refine]
                compl_idxs = np.nonzero(index_mask.reshape(-1))[0]
                if compl_idxs.shape[0] <= 50:
                    continue
                compl_pts = pts[compl_idxs, :]
                compl_pts = self.transform_cam2cam(compl_pts, ref_c2w)
                compl_colors = colors[compl_idxs, :]
                sel_idxs = np.random.choice(compl_pts.shape[0], int(0.1*compl_pts.shape[0]), replace=True)
    
                
                # sampled_points, sel_idxs = sample_farthest_points(
                #     torch.from_numpy(compl_pts).float().unsqueeze(0), 
                #     lengths=None,          
                #     K=int(0.1 * compl_pts.shape[0]), 
                #     # random_start_point=True
                # )
    
                # sel_idxs  = sel_idxs[0].cpu().numpy()           # shape (474,)
                # selected_pts = compl_pts[sel_idxs_np]              # shape (K, 3)
    
                # print("Shape of sampled compl_pts points:", sampled_points.shape)   
                # print("Shape of sampled compl_pts indices:", sel_idxs.shape) 
    
    
                ref_pts = np.concatenate((ref_pts, compl_pts[sel_idxs]), axis=0)
                ref_color = np.concatenate((ref_color, compl_colors[sel_idxs]), axis=0)
                ref_mask = np.logical_or(ref_mask, compl_mask)
    
                # ref_pts = np.concatenate((ref_pts, compl_pts), axis=0)
                # ref_color = np.concatenate((ref_color, compl_colors), axis=0)
                # ref_mask = np.logical_or(ref_mask, compl_mask)
    
            
            if ref_pts.shape[0] > 600000:
                sel_idxs = np.random.choice(ref_pts.shape[0], 500000, replace=True) 
                # sampled_points, sel_idxs = sample_farthest_points(
                #     torch.from_numpy(ref_pts).float().unsqueeze(0), 
                #     lengths=None,          
                #     K=500000, 
                #     # random_start_point=True
                # )
                # sel_idxs  = sel_idxs[0].cpu().numpy()  
                # print("Shape of sampled ref_pts points:", sampled_points.shape)   
                # print("Shape of sampled ref_pts indices:", sel_idxs.shape) 
                ref_pts = ref_pts[sel_idxs]         
                ref_color = ref_color[sel_idxs] 
        return ref_pts, ref_color
    
         
    def get_camera_poses(self, pose_tuple):
        R, T = pose_tuple
        R = np.transpose(R)
        w2c = np.concatenate((R, T[...,None]), axis=-1)
        w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        return c2w
    
    def get_pts_cam(self, depth, mask, color, disable_mask=False):
        W, H = self.img_wh
        i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
        cx = self.K[0,-1]
        cy = self.K[1,-1]
        X_Z = (i-cx) / self.focal[0]
        Y_Z = (j-cy) / self.focal[1]
        Z = depth
        X, Y = X_Z * Z, Y_Z * Z
        pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        color = color.reshape(-1, 3)
        
        if not disable_mask:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam[mask, :]
            color_valid = color[mask, :]
        else:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam
            color_valid = color
            color_valid[mask==0, :] = np.ones(3)
                    
        return pts_valid, color_valid, mask
        
    def get_maxtime(self):
        return self.maxtime
    
    def transform_cam2cam(self, pts_cam, pose):
        pts_cam_homo = np.concatenate((pts_cam, np.ones((pts_cam.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(pose @ np.transpose(pts_cam_homo))
        xyz = pts_wld[:, :3]
        return xyz

class SCARED_Dataset(EndoNeRF_Dataset):
    def __init__(
        self,
        datadir,
        downsample=1.0,
        test_every=8,
        skip_every=None,
        mode="binocular",
        depth_far_thresh=300.0,
        depth_near_thresh=0.03,
    ):
        # do NOT call super().__init__ because EndoNeRF hardcodes 640x512 and loads poses_bounds.npy
        if skip_every is None:
            if "dataset_1" in datadir:
                skip_every = 2
            elif "dataset_2" in datadir:
                skip_every = 1
            elif "dataset_3" in datadir:
                skip_every = 4
            elif "dataset_6" in datadir:
                skip_every = 8
            elif "dataset_7" in datadir:
                skip_every = 8
            else:
                skip_every = 1

        self.img_wh = (
            int(1280 / downsample),
            int(1024 / downsample),
        )
        self.root_dir = datadir
        self.downsample = downsample
        self.skip_every = skip_every
        self.mode = mode
        self.depth_far_thresh = depth_far_thresh
        self.depth_near_thresh = depth_near_thresh

        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()
        self.white_bg = False

        self.load_meta()
        print(f"meta data loaded, total image:{len(self.image_paths)}")

        n_frames = len(self.image_paths)
        self.train_idxs = [i for i in range(n_frames) if (i - 1) % test_every != 0]
        self.test_idxs = [i for i in range(n_frames) if (i - 1) % test_every == 0]
        self.video_idxs = [i for i in range(n_frames)]

        self.maxtime = 1.0

    def _resize_rgb(self, rgb):
        # print(f"Original RGB shape: {rgb.shape}, target shape: {self.img_wh}")
        h, w = rgb.shape[:2]
        if (w, h) == self.img_wh:
            return rgb
        # print(f"Resizing RGB from {(w, h)} to {self.img_wh}")
        return cv2.resize(rgb, self.img_wh, interpolation=cv2.INTER_AREA)

    def _resize_depth_or_mask(self, arr):
        # print(f"Original shape: {arr.shape}, target shape: {self.img_wh}")
        h, w = arr.shape[:2]
        if (w, h) == self.img_wh:
            return arr
        # print(f"Resizing from {(w, h)} to {self.img_wh}")
        return cv2.resize(arr, self.img_wh, interpolation=cv2.INTER_NEAREST)

    def _scale_intrinsics(self, K, orig_w, orig_h):
        K = K.copy().astype(np.float32)
        sx = self.img_wh[0] / float(orig_w)
        sy = self.img_wh[1] / float(orig_h)
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        return K

    def load_meta(self):
        """
        Load SCARED data, but store it in EndoNeRF-compatible fields.
        """
        print(f"{self.root_dir = }")
        print("Loading meta data from SCARED dataset...")

        calibs_dir = osp.join(self.root_dir, "data", "frame_data")
        rgbs_dir = osp.join(self.root_dir, "data", "left_finalpass")
        disps_dir = osp.join(self.root_dir, "data", "disparity")
        monodisps_dir = osp.join(self.root_dir, "data", "left_monodam")
        reproj_dir = osp.join(self.root_dir, "data", "reprojection_data")

        frame_ids = sorted([x[:-5] for x in os.listdir(calibs_dir) if x.endswith(".json")])
        frame_ids = frame_ids[::self.skip_every]
        n_frames = len(frame_ids)

        self.frame_ids = frame_ids

        # EndoNeRF-compatible fields
        self.image_paths = []
        self.depth_paths = []
        self.masks_paths = []
        self.image_poses = []
        self.image_times = []

        # in-memory data for SCARED
        self.images = []
        self.depths = []
        self.masks = []
        self.camera_mats = []
        self.pose_mats = []
        self.bds = []

        c2w0 = None

        for idx in trange(n_frames, desc="Process frames"):
            frame_id = frame_ids[idx]

            # ---- calibration ----
            with open(osp.join(calibs_dir, f"{frame_id}.json"), "r") as f:
                calib_dict = json.load(f)

            K = np.array(calib_dict["camera-calibration"]["KL"], dtype=np.float32)

            rgb_path = osp.join(rgbs_dir, f"{frame_id}.png")
            rgb = iio.imread(rgb_path)
            orig_h, orig_w = rgb.shape[:2]

            K = self._scale_intrinsics(K, orig_w, orig_h)
            self.camera_mats.append(K)

            # SCARED convention from your other project
            c2w = np.linalg.inv(np.array(calib_dict["camera-pose"], dtype=np.float32))
            if c2w0 is None:
                c2w0 = c2w.copy()
            c2w = np.linalg.inv(c2w0) @ c2w
            self.pose_mats.append(c2w.astype(np.float32))

            # convert to EndoNeRF pose storage
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, -1]
            R = np.transpose(R)
            self.image_poses.append((R.astype(np.float32), T.astype(np.float32)))
            self.image_times.append(idx / n_frames)

            # ---- rgb ----
            rgb = rgb.astype(np.float32) / 255.0
            rgb = self._resize_rgb(rgb)
            self.images.append(rgb)
            self.image_paths.append(rgb_path)

            # ---- depth ----
            if self.mode == "binocular":
                disp_path = osp.join(disps_dir, f"{frame_id}.tiff")
                disp = iio.imread(disp_path).astype(np.float32)

                with open(osp.join(reproj_dir, f"{frame_id}.json"), "r") as f:
                    reproj_dict = json.load(f)
                Q = np.array(reproj_dict["reprojection-matrix"], dtype=np.float32)

                fl = Q[2, 3]
                bl = 1.0 / Q[3, 2]
                disp_const = fl * bl

                depth = np.zeros_like(disp, dtype=np.float32)
                valid_disp = disp != 0
                depth[valid_disp] = disp_const / disp[valid_disp]
                depth[depth > self.depth_far_thresh] = 0.0
                depth[depth < self.depth_near_thresh] = 0.0

                self.depth_paths.append(disp_path)

            elif self.mode == "monocular":
                depth_path = osp.join(monodisps_dir, f"{frame_id}.png")
                depth = iio.imread(depth_path).astype(np.float32) / 255.0
                depth = self.depth_near_thresh + (self.depth_far_thresh - self.depth_near_thresh) * depth
                self.depth_paths.append(depth_path)

            else:
                raise ValueError(f"{self.mode} is not implemented!")

            depth = self._resize_depth_or_mask(depth).astype(np.float32)
            self.depths.append(depth)

            # ---- mask ----
            depth_mask = (depth != 0).astype(np.float32)
            k = max(1, int(self.img_wh[0] / 128))
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel).astype(np.float32)
            self.masks.append(mask)
            self.masks_paths.append(None)

            # ---- bounds ----
            if np.any(depth != 0):
                bds = np.array([depth[depth != 0].min(), depth[depth != 0].max()], dtype=np.float32)
            else:
                bds = np.array([self.depth_near_thresh, self.depth_far_thresh], dtype=np.float32)
            self.bds.append(bds)

        self.images = np.stack(self.images, axis=0).astype(np.float32)
        self.depths = np.stack(self.depths, axis=0).astype(np.float32)
        self.masks = np.stack(self.masks, axis=0).astype(np.float32)
        self.camera_mats = np.stack(self.camera_mats, axis=0).astype(np.float32)
        self.pose_mats = np.stack(self.pose_mats, axis=0).astype(np.float32)
        self.bds = np.stack(self.bds, axis=0).astype(np.float32)

        # keep default intrinsics for inherited helpers
        # self.K = self.camera_mats[0]
        # self.focal = (self.K[0, 0], self.K[1, 1])

    def format_infos(self, split):
        cameras = []

        if split == "train":
            idxs = self.train_idxs
        elif split == "test":
            idxs = self.test_idxs
        else:
            idxs = self.video_idxs

        for idx in tqdm(idxs):
            image = self.transform(self.images[idx])
            mask = self.transform(self.masks[idx]).bool()
            depth = torch.from_numpy(self.depths[idx])
            time = self.image_times[idx]
            R, T = self.image_poses[idx]

            K = self.camera_mats[idx]
            focal_x, focal_y = K[0, 0], K[1, 1]
            FovX = focal2fov(focal_x, self.img_wh[0])
            FovY = focal2fov(focal_y, self.img_wh[1])

            cameras.append(
                Camera(
                    colmap_id=idx,
                    R=R,
                    T=T,
                    FoVx=FovX,
                    FoVy=FovY,
                    image=image,
                    depth=depth,
                    mask=mask,
                    gt_alpha_mask=None,
                    image_name=f"{idx}",
                    uid=idx,
                    data_device=torch.device("cuda"),
                    time=time,
                    Znear=self.depth_near_thresh,
                    Zfar=self.depth_far_thresh,
                    K=K,
                    h=self.img_wh[1],
                    w=self.img_wh[0],
                )
            )
        return cameras

    def calculate_motion_masks(self, st, cen, ed):
        images = []
        for j in range(st, ed):
            images.append(self.images[j])

        images = np.asarray(images).mean(axis=-1)

        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=max(1, ed - st), varThreshold=4, detectShadows=False
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        motion_masks = []

        for frame in images[::-1]:
            frame_uint8 = (frame * 255).astype(np.uint8)
            fg_mask = mog2.apply(frame_uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            motion_masks.append(fg_mask)

        motion_masks = np.stack(motion_masks[::-1], axis=0)
        return motion_masks > 0

    def calculate_motion_masks_invert(self, st, cen, ed):
        images = []
        for j in range(st, ed):
            images.append(self.images[j])

        images = np.asarray(images).mean(axis=-1)

        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=max(1, ed - st), varThreshold=4, detectShadows=False
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        motion_masks = []

        for frame in images:
            frame_uint8 = (frame * 255).astype(np.uint8)
            fg_mask = mog2.apply(frame_uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            motion_masks.append(fg_mask)

        motion_masks = np.stack(motion_masks[::-1], axis=0)
        return motion_masks > 0

    def search_pts_colors_with_motion(self, st, cen, ed, ref_pts, ref_color, ref_mask, ref_c2w):
        motion_mask = self.calculate_motion_masks(st, cen, ed)

        interval = 1
        if len(self.image_poses) > 150:
            interval = 2

        if cen == 0:
            search_range = range(1, len(self.image_poses), interval)
        else:
            motion_mask = self.calculate_motion_masks_invert(st, cen, ed)
            search_range = range(len(self.image_poses) - 1, 0, -interval)

        ref_K = self.camera_mats[cen]
        ref_focal = (ref_K[0, 0], ref_K[1, 1])

        for j in search_range:
            if cen != 0 and j < len(self.image_poses) // 2:
                motion_mask[0] = motion_mask[0] * False

            ref_mask_not = np.logical_not(ref_mask)
            ref_mask_not = np.logical_or(ref_mask_not, motion_mask[0])

            R, T = self.image_poses[j]
            c2w = self.get_camera_poses((R, T))
            c2ref = np.linalg.inv(ref_c2w) @ c2w

            depth = self.depths[j].copy()
            color = self.images[j].copy()
            mask = self.masks[j].copy()

            depth_mask = np.ones(depth.shape, dtype=np.float32)
            valid = depth != 0
            if not np.any(valid):
                continue

            close_depth = np.percentile(depth[valid], 3.0)
            inf_depth = np.percentile(depth[valid], 99.8)
            depth_mask[depth > inf_depth] = 0
            depth_mask[np.bitwise_and(depth < close_depth, depth != 0)] = 0
            depth_mask[depth == 0] = 0
            mask = np.logical_and(depth_mask, mask)
            depth[mask == 0] = 0

            K_j = self.camera_mats[j]
            self.K = K_j
            self.focal = (K_j[0, 0], K_j[1, 1])

            pts, colors, mask_refine = self.get_pts_cam(depth, mask, color)
            pts = self.transform_cam2cam(pts, c2ref)

            X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
            valid_z = Z != 0
            X, Y, Z = X[valid_z], Y[valid_z], Z[valid_z]

            X_Z, Y_Z = X / Z, Y / Z
            X_Z = (X_Z * ref_focal[0] + ref_K[0, -1]).astype(np.int32)
            Y_Z = (Y_Z * ref_focal[1] + ref_K[1, -1]).astype(np.int32)

            out_vis_mask = (
                ((X_Z > (self.img_wh[0] - 1)) + (X_Z < 0) +
                 (Y_Z > (self.img_wh[1] - 1)) + (Y_Z < 0)) > 0
            )

            X_Z = np.clip(X_Z, 0, self.img_wh[0] - 1)
            Y_Z = np.clip(Y_Z, 0, self.img_wh[1] - 1)

            coords = np.stack((Y_Z, X_Z), axis=-1)
            proj_mask = np.zeros((self.img_wh[1], self.img_wh[0]), dtype=np.float32)
            proj_mask[coords[:, 0], coords[:, 1]] = 1

            compl_mask = (ref_mask_not * proj_mask)
            index_mask = compl_mask.reshape(-1)[mask_refine]
            compl_idxs = np.nonzero(index_mask.reshape(-1))[0]

            if compl_idxs.shape[0] <= 50:
                continue

            compl_pts = pts[compl_idxs, :]
            compl_pts = self.transform_cam2cam(compl_pts, ref_c2w)
            compl_colors = colors[compl_idxs, :]

            num_sel = max(1, int(0.1 * compl_pts.shape[0]))
            sel_idxs = np.random.choice(compl_pts.shape[0], num_sel, replace=True)

            ref_pts = np.concatenate((ref_pts, compl_pts[sel_idxs]), axis=0)
            ref_color = np.concatenate((ref_color, compl_colors[sel_idxs]), axis=0)
            ref_mask = np.logical_or(ref_mask, compl_mask)

        if ref_pts.shape[0] > 600000:
            sel_idxs = np.random.choice(ref_pts.shape[0], 500000, replace=True)
            ref_pts = ref_pts[sel_idxs]
            ref_color = ref_color[sel_idxs]

        # restore default K/focal
        self.K = self.camera_mats[0]
        self.focal = (self.K[0, 0], self.K[1, 1])

        return ref_pts, ref_color

    def get_sparse_pts(self, st, cen, ed, sample=True):
        R, T = self.image_poses[cen]
        depth = self.depths[cen].copy()
        color = self.images[cen].copy()
        mask = self.masks[cen].copy()

        depth_mask = np.ones(depth.shape, dtype=np.float32)
        valid = depth != 0
        if np.any(valid):
            close_depth = np.percentile(depth[valid], 0.1)
            inf_depth = np.percentile(depth[valid], 99.9)
            depth_mask[depth > inf_depth] = 0
            depth_mask[np.bitwise_and(depth < close_depth, depth != 0)] = 0
            depth_mask[depth == 0] = 0
            depth[depth_mask == 0] = 0

        mask = np.logical_and(depth_mask, mask)

        K_cen = self.camera_mats[cen]
        self.K = K_cen
        self.focal = (K_cen[0, 0], K_cen[1, 1])

        pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
        c2w = self.get_camera_poses((R, T))
        pts = self.transform_cam2cam(pts, c2w)

        pts, colors = self.search_pts_colors_with_motion(st, cen, ed, pts, colors, mask, c2w)
        normals = np.zeros((pts.shape[0], 3), dtype=np.float32)

        if sample and pts.shape[0] > 0:
            num_sample = max(1, int(0.1 * pts.shape[0]))
            sel_idxs = np.random.choice(pts.shape[0], num_sample, replace=False)
            pts = pts[sel_idxs, :]
            colors = colors[sel_idxs, :]
            normals = normals[sel_idxs, :]

        # restore default K/focal
        self.K = self.camera_mats[0]
        self.focal = (self.K[0, 0], self.K[1, 1])

        print("pt cloud shape: ", pts.shape)
        return pts, colors, normals