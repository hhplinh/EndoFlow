import os
import torch
import numpy as np
import cv2
import imageio
from tqdm import tqdm
from argparse import ArgumentParser
from scene.flexible_deform_model import GaussianModel
from arguments import ModelParams, FDMHiddenParams, PipelineParams, get_combined_args
from scene import Scene
from utils.sh_utils import SH2RGB

def render_centers_dual(view, gaussian_model, dot_size=2):
    """
    Renders 3D Gaussian centers into two formats:
    1. Black dots on White background
    2. Colored dots on Black background
    """
    time = torch.tensor(view.time).to("cuda").float()
    
    # 1. Deform positions
    means3D = gaussian_model.get_xyz.detach().clone()
    scales = gaussian_model.get_scaling.detach().clone()
    rotations = gaussian_model.get_rotation.detach().clone()
    means3D, _, _ = gaussian_model.deformation(means3D, scales, rotations, time)

    # 2. Get Colors (Convert SH DC component to RGB)
    sh_dc = gaussian_model._features_dc.detach()
    colors = SH2RGB(sh_dc).squeeze(1) 
    colors = torch.clamp(colors, 0.0, 1.0)

    # 3. Project 3D -> NDC
    p_homo = torch.cat([means3D, torch.ones_like(means3D[..., :1])], dim=-1)
    p_view = p_homo @ view.full_proj_transform.to("cuda")
    
    w = p_view[..., 3:4]
    p_screen = p_view[..., :3] / (w + 1e-7)

    mask = (w.squeeze() > 0) & \
           (p_screen[:, 0] > -1) & (p_screen[:, 0] < 1) & \
           (p_screen[:, 1] > -1) & (p_screen[:, 1] < 1)

    p_screen = p_screen[mask]
    colors = colors[mask]

    # 4. Map NDC to Pixel Space
    img_w, img_h = view.image_width, view.image_height
    u = ((p_screen[:, 0] + 1.0) * 0.5 * img_w).long()
    v = ((p_screen[:, 1] + 1.0) * 0.5 * img_h).long()

    # 5. Draw to Buffers
    # Buffer A: Black and White (White Background)
    img_bw = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    # Buffer B: Color (Black Background)
    img_col = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    u_cpu = u.cpu().numpy()
    v_cpu = v.cpu().numpy()
    colors_cpu = (colors.cpu().numpy() * 255).astype(np.uint8)

    valid_idx = (u_cpu >= 0) & (u_cpu < img_w) & (v_cpu >= 0) & (v_cpu < img_h)
    
    v_valid = v_cpu[valid_idx]
    u_valid = u_cpu[valid_idx]

    # Draw BW
    img_bw[v_valid, u_valid] = [0, 0, 0]
    # Draw Color
    img_col[v_valid, u_valid] = colors_cpu[valid_idx]

    # 6. Dilation for visibility
    if dot_size > 1:
        kernel = np.ones((dot_size, dot_size), np.uint8)
        img_bw = cv2.erode(img_bw, kernel, iterations=1) # Erode because points are dark
        img_col = cv2.dilate(img_col, kernel, iterations=1) # Dilate because points are light

    return img_bw, img_col

def run_render(dataset, hyperparam, iteration, dot_size):
    with torch.no_grad():
        gaussians = [GaussianModel(dataset.sh_degree, hyperparam)]
        class Config: bidirectional = True
        scene = Scene(dataset, gaussians, 260, Config, load_iteration=iteration)
        
        model_path = dataset.model_path
        bw_path = os.path.join(model_path, "centers_bw", f"iter_{scene.loaded_iter}")
        col_path = os.path.join(model_path, "centers_color", f"iter_{scene.loaded_iter}")
        os.makedirs(bw_path, exist_ok=True)
        os.makedirs(col_path, exist_ok=True)

        views = scene.getVideoCameras() 
        frames_bw = []
        frames_col = []
        
        print(f"Rendering {len(views)} frames...")
        for idx, view in enumerate(tqdm(views)):
            img_bw, img_col = render_centers_dual(view, gaussians[0], dot_size=dot_size)
            
            file_name = f"{idx:05d}.png"
            cv2.imwrite(os.path.join(bw_path, file_name), img_bw)
            cv2.imwrite(os.path.join(col_path, file_name), cv2.cvtColor(img_col, cv2.COLOR_RGB2BGR))
            
            frames_bw.append(img_bw)
            frames_col.append(img_col)

        high_res_frames_bw = [cv2.resize(f, (img_bw.shape[1]*4, img_bw.shape[0]*4), interpolation=cv2.INTER_NEAREST) for f in frames_bw]
        high_res_frames_col = [cv2.resize(f, (img_col.shape[1]*4, img_col.shape[0]*4), interpolation=cv2.INTER_NEAREST) for f in frames_col]

        print("Compiling videos...")
        imageio.mimwrite(os.path.join(model_path, f"centers_bw_{scene.loaded_iter}.mp4"), frames_bw, fps=15, quality=9)
        imageio.mimwrite(os.path.join(model_path, f"centers_col_{scene.loaded_iter}.mp4"), frames_col, fps=15, quality=9)
        imageio.mimwrite(os.path.join(model_path, f"centers_bw_{scene.loaded_iter}_4x.mp4"), high_res_frames_bw, fps=15, quality=9)
        imageio.mimwrite(os.path.join(model_path, f"centers_col_{scene.loaded_iter}_4x.mp4"), high_res_frames_col, fps=15, quality=9)
        print("Done!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Dual Render: BW and Color Centers")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    hp = FDMHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--dot_size", default=1, type=int)
    args = get_combined_args(parser)
    
    run_render(lp.extract(args), hp.extract(args), args.iteration, args.dot_size)