from datetime import datetime
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Util function for loading point clouds
import numpy as np
from scipy.optimize import minimize
import cv2
import ffmpeg

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointsRasterizationSettings,
    # PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

import open3d as o3d

class PatchedPointsRenderer(nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    This renderer version has been fixed to support individual radii for each point
    """

    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        if not type(r) == float:
            idxs = fragments.idx.permute(0, 3, 1, 2).reshape(-1).long()
            r = r.flatten()[idxs]
            r = r.reshape(*dists2.size())
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images


# Setup
FORCE_CPU = False
if not FORCE_CPU and torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def numpy_to_float_device(array):
    return torch.from_numpy(array).float().to(device)

def float_device_to_numpy(array):
    return array.clone().detach().cpu().numpy()

def get_mask_from_pcl(pcl, color):
    t = 5
    return ((torch.abs(pcl[:, 0] - color[0] / 255) < t / 255) & (torch.abs(pcl[:, 1] - color[1] / 255) < t / 255) & (torch.abs(pcl[:, 2] - color[2] / 255) < t / 255))

def get_mask_from_image(image, color):
    t = 5
    return ((torch.abs(image[:, :, :, 0] - color[0] / 255) < t / 255) & (torch.abs(image[:, :, :, 1] - color[1] / 255) < t / 255) & (torch.abs(image[:, :, :, 2] - color[2] / 255) < t / 255))

def pitch_yaw_to_dir(pitch, yaw):
    x = np.cos(yaw / 180 * np.pi) * np.cos(pitch / 180 * np.pi)
    y = np.sin(yaw / 180 * np.pi) * np.cos(pitch / 180 * np.pi)
    z = np.sin(pitch / 180 * np.pi)
    return np.array([[x, y, z]])

def at_to_az_el(eye, at):
    at = (at - eye).flatten()
    r = np.linalg.norm(at)
    el = np.arcsin(at[2] / r)
    az = np.arctan2(at[1], at[0])
    print(f"yaw: {az / np.pi * 180:6.4f}, pitch: {el / np.pi * 180:6.4f}")
    return az, el

def to_str_list(np_arr):
    np_arr = np_arr.flatten().tolist()
    return[f"{i:6.3f}" for i in np_arr]

def get_pitch_mat(p):
    p = p / 180 * np.pi
    return np.array([[np.cos(p), 0 , np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])

def get_yaw_mat(yaw):
    y = yaw / 180 * np.pi
    return np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])

TAG_MAP = {
    0: np.array([0, 0, 0]),
    1: np.array([70, 70, 70]),
    2: np.array([100, 40, 40]),
    3: np.array([55, 90, 80]), # banner
    4: np.array([220, 20, 60]),
    5: np.array([153, 153, 153]),# pole
    6: np.array([157, 234, 50]),
    7: np.array([128, 64, 128]),
    8: np.array([244, 35, 232]),
    9: np.array([107, 142, 35]),
    10: np.array([0, 0, 142]), # car
    11: np.array([102, 102, 156]),
    12: np.array([220, 220, 0]), # sign
    13: np.array([70, 130, 180]),
    14: np.array([81, 0, 81]),
    15: np.array([150, 100, 100]),
    16: np.array([230, 150, 140]),
    17: np.array([180, 165, 180]),
    18: np.array([250, 170, 30]),# traffic light
    19: np.array([110, 190, 160]),
    20: np.array([170, 120, 50]),
    21: np.array([45, 60, 150]),
    22: np.array([145, 170, 100])
}

IGNORE_LIST = []#[3, 5, 10, 12, 18]

SKY_COLOR = (70/255, 130/255, 180/255)

SCALEDOWN_FACTOR = 2
LOOKAT_SCALE = 10
RADIUS_BASE_MULTIPLIER = 0.15 * 1.3

WIDTH = 960 // SCALEDOWN_FACTOR
HEIGHT = 600 // SCALEDOWN_FACTOR

USE_DIRECT_LOOKAT = True

output_folder = Path(__file__).absolute().parent.parent / "output_carla"
output_folder.mkdir(exist_ok=True)
output_folder = str(output_folder)

print("args:", sys.argv[1:])
num_in = sys.argv[1]
reps_in = int(sys.argv[2])
pct_in = float(sys.argv[3])
tag_in = str(sys.argv[4])
eval_in = str(sys.argv[5]) == "eval"

USE_PCT = pct_in#0.5 # 0.2 for others
USE_L1 = False
USE_HUBER = True

num = num_in#3
num_reps = reps_in#30
tag = tag_in#str(sys.argv[1]) if len(sys.argv) > 1 else "_tag"

pcl_filename = f"{output_folder}/{num}_pcl.ply"
segmentation_filename = f"{output_folder}/{num}_seg.png"

print(f"Input file: {segmentation_filename}")
filename_output = f"{output_folder}/optim_result_{datetime.now().strftime('%y%m%d%H%M%S')}.mp4"

coordinate_file = segmentation_filename.replace("_seg.png", ".txt")
with open(coordinate_file, "r") as f:
    lines  = f.readlines()
line_arr = lines[0].strip().split(" ")
line_arr = [float(a.split("=")[1]) for a in line_arr]

x_ff = line_arr[0]
y_ff = line_arr[1]
z_ff = line_arr[2]
p_ff = line_arr[3]
yaw_ff = line_arr[4]

print(f"Read from file: x={x_ff} y={y_ff} z={z_ff} p={p_ff} yaw={yaw_ff}")

# Load pcl data
pointcloud = o3d.io.read_point_cloud(pcl_filename).voxel_down_sample(0.05)
vertices = numpy_to_float_device(np.asarray(pointcloud.points))
print(f"Loaded {vertices.shape[0]} points")
colors = numpy_to_float_device(np.asarray(pointcloud.colors))

# Flip from UE4 coordinates to cartesian coordinates
vertices[:, 1] *= -1
y_ff = -y_ff
yaw_ff = -yaw_ff

# aabb_pcl = o3d.geometry.AxisAlignedBoundingBox(np.array([-5, -5, -1]).reshape((3, 1)), np.array([50, 80, 20]).reshape((3, 1))).get_oriented_bounding_box()
# aabb_pcl.rotate(get_yaw_mat(yaw_ff))
# aabb_pcl.translate(np.array([x_ff, y_ff, z_ff]))

# new_pcl = o3d.geometry.PointCloud()
# new_pcl.points = o3d.utility.Vector3dVector(float_device_to_numpy(vertices))
# new_pcl.colors = o3d.utility.Vector3dVector(float_device_to_numpy(colors))
# new_pcl.crop(aabb_pcl)
# vertices = numpy_to_float_device(np.asarray(new_pcl.points))
# colors = numpy_to_float_device(np.asarray(new_pcl.colors))
# print(f"Cropped to {vertices.shape[0]} points")

# Filter unused classes in point cloud
class_mask = torch.zeros((vertices.shape[0]), dtype=torch.float32, device=device) > 1
for i in IGNORE_LIST:
    class_mask |= get_mask_from_pcl(colors, TAG_MAP[i])
vertices = vertices[~class_mask]
colors = colors[~class_mask]
colors = torch.cat((colors, torch.ones((colors.shape[0], 1), dtype=torch.float32, device=device)), dim=1)

# Load image data
gt = cv2.imread(segmentation_filename)
gt = cv2.resize(gt, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR).astype(float) / 255
gt = numpy_to_float_device(gt)[:, :, [2, 1, 0]].unsqueeze(0)

# Filter unused classes in ground truth
class_mask = torch.zeros((gt.shape[0], gt.shape[1], gt.shape[2]), dtype=torch.float32, device=device) > 1
for i in IGNORE_LIST:
    class_mask |= get_mask_from_image(gt, TAG_MAP[i])
gt = gt * (~class_mask).unsqueeze(-1).expand_as(gt)

process = (
    ffmpeg
    .input('pipe:', format='rawvideo', pixel_format='rgb24', video_size=(WIDTH, HEIGHT), framerate=10)
    .output(filename_output, **{
        "framerate": 10,
        "video_size": (WIDTH, HEIGHT),
        "pix_fmt": 'yuv420p',
        "profile:v": "baseline",
        "level": 3.0
    })
    .run_async(pipe_stdin=True, quiet=True, overwrite_output=True)
)

class Model(nn.Module):

    def __init__(self, vertices, colors, gt, eye, lookat, fov):
        super().__init__()

        self.set_params(eye, lookat, initialize=True)

        UP_VEC = np.array([[0, 0, 1]])

        self.vertices = vertices
        self.colors = colors
        self.gt = gt
        self.gt_bg_mask = get_mask_from_image(self.gt, TAG_MAP[13]).unsqueeze(-1).expand_as(self.gt).clone()
        self.gt_ignore_mask = get_mask_from_image(self.gt, TAG_MAP[0]).unsqueeze(-1).expand_as(self.gt).clone()
        self.up_vec = numpy_to_float_device(UP_VEC)
        self.fov = numpy_to_float_device(fov)

        R, T = look_at_view_transform(eye=self.eye, at=self.lookat, up=self.up_vec)
        self.cameras = FoVPerspectiveCameras(fov=self.fov, znear=0.1, zfar=200, device=device, R=R, T=T)

        self.dists = torch.linalg.norm(self.vertices - self.eye.data, dim=1)
        self.radii = (RADIUS_BASE_MULTIPLIER / self.dists).unsqueeze(0)
        self.raster_settings = PointsRasterizationSettings(
            image_size=(HEIGHT, WIDTH), 
            # radius = 0.003,
            # radius = 0.0075,
            # radius = 0.02,
            radius = self.radii,
            points_per_pixel = 16
        )
        self.renderer = PatchedPointsRenderer(
            rasterizer=PointsRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            compositor=NormWeightedCompositor(background_color=SKY_COLOR)
        )
        torch.cuda.empty_cache()

    def set_params(self, eye, lookat, initialize=False):
        if initialize:
            self.eye = nn.Parameter(numpy_to_float_device(eye))
            self.lookat = nn.Parameter(numpy_to_float_device(lookat))
        else:
            self.eye.data = numpy_to_float_device(eye)
            self.lookat.data = numpy_to_float_device(lookat)
        if USE_DIRECT_LOOKAT:
            temp_vec = self.lookat.data - self.eye.data
            temp_vec = temp_vec / torch.linalg.norm(temp_vec) * LOOKAT_SCALE
            self.lookat.data = temp_vec
        else:
            temp_vec = self.lookat.data - self.eye.data
            temp_vec = temp_vec / torch.linalg.norm(temp_vec) * LOOKAT_SCALE
            self.lookat.data = self.eye.data + temp_vec

    def get_params(self):
        if USE_DIRECT_LOOKAT:
            return float_device_to_numpy(self.eye.data), float_device_to_numpy(self.lookat.data + self.eye.data)
        else:
            return float_device_to_numpy(self.eye.data), float_device_to_numpy(self.lookat.data)

    def forward(self):
        if USE_DIRECT_LOOKAT:
            R, T = look_at_view_transform(eye=self.eye, at=self.lookat + self.eye, up=self.up_vec)
        else:
            R, T = look_at_view_transform(eye=self.eye, at=self.lookat, up=self.up_vec)
        self.dists = torch.linalg.norm(self.vertices - self.eye, dim=1)
        self.radii = (RADIUS_BASE_MULTIPLIER / self.dists).unsqueeze(0)
        images = self.renderer(Pointclouds(points=[self.vertices], features=[self.colors]), R=R.to(device), T=T.to(device), fov=self.fov, eps=1e-6, radius=self.radii)[:, :, :, :3]
        bg_mask = get_mask_from_image(images, TAG_MAP[13]).unsqueeze(-1).expand_as(images).contiguous()
        and_mask = bg_mask & self.gt_bg_mask
        final_mask = (~(bg_mask & ~and_mask)) * (~self.gt_ignore_mask)
        # print(torch.numel(final_mask), final_mask.shape)
        final_mask = final_mask[:, int(HEIGHT * USE_PCT):]
        # print(torch.numel(final_mask), final_mask.shape)
        norm_factor = torch.numel(final_mask) / final_mask.sum()
        source = images[:, int(HEIGHT * USE_PCT):, :, :3] * final_mask
        target = self.gt[:, int(HEIGHT * USE_PCT):] * final_mask
        if USE_L1:
            loss = F.l1_loss(source, target) * norm_factor
        elif USE_HUBER:
            loss = F.huber_loss(source, target, delta=0.3) * norm_factor
        else:
            loss = F.mse_loss(source, target) * norm_factor
        images[:, int(HEIGHT * USE_PCT):] *= final_mask
        return loss, images

pitch, yaw = p_ff, yaw_ff
eye_in = np.array([[x_ff, y_ff, z_ff]])
lookat_in = eye_in + pitch_yaw_to_dir(pitch, yaw)
to_hor_fov = np.arctan(np.tan(45/180*np.pi)*5/8)*2/np.pi*180
fov_in = np.array([to_hor_fov])

iterations = 1000
lr = 0.05

model = Model(vertices=vertices, colors=colors, gt=gt, eye=eye_in, lookat=lookat_in, fov=fov_in).to(device)
model.train()

PRINT_EVAL = eval_in

if PRINT_EVAL:

    def target_fn(x):
        eye_in_temp = x[:3].reshape((1, 3))
        lookat_in = eye_in_temp + pitch_yaw_to_dir(x[3], x[4])
        model.set_params(eye_in_temp, lookat_in)
        # print(eye_in_temp, lookat_in)
        loss, _ = model()
        return loss.item()

    res_array = np.load(f"{output_folder}/result_raw.npy")
    print(f"Loading {output_folder}/result_raw.npy")
    res_array2 = res_array[:, 1:]
    res_array3 = np.zeros((res_array2.shape[0], 5))
    res_losses = np.zeros((res_array2.shape[0]))
    for i in range(res_array2.shape[0]):
        # print(f"Iter {i}")
        # print(res_array2[i])
        l1 = target_fn(res_array2[i, 0]+ np.array([x_ff, y_ff, z_ff, p_ff, yaw_ff]))
        l2 = target_fn(res_array2[i, 1]+ np.array([x_ff, y_ff, z_ff, p_ff, yaw_ff]))
        l3 = target_fn(res_array2[i, 2]+ np.array([x_ff, y_ff, z_ff, p_ff, yaw_ff]))
        # print(l1, l2, l3)

        res_loss = l3
        res_vals = res_array2[i, 2]
        if l2 < res_loss:
            # print("val2 > val3")
            # print(res_array2[i, 1], res_array2[i, 2])
            res_loss = l2
            res_vals = res_array2[i, 1]
        if l1 < res_loss:
            # print("val1 > val23")
            # print(res_array2[i, 0], res_vals)
            res_loss = l2
            res_vals = res_array2[i, 0]
        res_array3[i] = res_vals
        res_losses[i] = res_loss
    min_loss = res_losses.min()
    min_params = res_array3[res_losses.argmin()]
    indices = np.argsort(res_losses)
    # print("all vals:", res_array3)
    res_array4 = res_array3[indices[:10]]
    new_losses = res_losses[indices[:10]]
    # print("new losses:", new_losses, min_loss)
    res_array4 = res_array4[new_losses < min_loss * 1.3]
    # print("filtered vals:", res_array4)
    print(f"After outlier removal: {res_array4.shape[0]}")
    print(f"Mean offset:{np.linalg.norm(res_array3[:, :3], axis=1).mean():.4f}, Mean angle error: {np.abs(res_array3[:, 3:]).mean():.4f}")
    print(f"No outliers mean offset:{np.linalg.norm(res_array4[:, :3], axis=1).mean():.4f}, Mean angle error: {np.abs(res_array4[:, 3:]).mean():.4f}")
    print(f"Best result: {np.linalg.norm(min_params[:3]):.4f}, {np.abs(min_params[3:]).mean():.4f}")
    sys.exit(0)

lookat_in = eye_in + pitch_yaw_to_dir(pitch, yaw)
loss, images = model()
cv2.imwrite(f"{output_folder}/test_gt_out.png", (images.detach().squeeze().cpu().numpy() * 255)[:, :, ::-1].astype(np.uint8))
cv2.imwrite(f"{output_folder}/test_gt_ref.png", (gt.clone().detach().squeeze().cpu().numpy() * 255)[:, :, ::-1].astype(np.uint8))

offsets = []

num_samples = num_reps

res_array = np.zeros((num_samples, 4, 5))
res_array2 = np.zeros((num_samples))

for num_sample in range(num_reps):
    print(f"Sample {num_sample}")
    po = np.random.rand() * 10 - 5
    yawo = np.random.rand() * 10 - 5
    xo = np.random.rand() * 5 - 2.5
    yo = np.random.rand() * 5 - 2.5
    zo = np.random.rand() * 5 - 2.5
    print(f"Initial offset: {xo, yo, zo, po, yawo}")
    pitch, yaw = p_ff, yaw_ff
    eye_in = np.array([[x_ff, y_ff, z_ff]])
    pitch_n, yaw_n = pitch + po, yaw + yawo#-10 - 1.8, -6 + 1.3
    eye_in_n = eye_in + np.array([[xo, yo, zo]])#np.array([[-22, -26, 6.7]])+ np.array([[0.56, -0.24, 0.61]])
    pitch_range = 5
    yaw_range = 5
    x_range = 1.5
    y_range = 1.5
    z_range = 1.5
    num_subdiv = 15
    interval_mult = 0.98
    noise_mult = 0.05
    noise_offset = 0.025

    def target_fn(x):
        eye_in_temp = x[:3].reshape((1, 3))
        lookat_in = eye_in_temp + pitch_yaw_to_dir(x[3], x[4])
        model.set_params(eye_in_temp, lookat_in)
        loss, _ = model()
        return loss.item()

    def render_fn(x):
        eye_in_temp = x[:3].reshape((1, 3))
        lookat_in = eye_in_temp + pitch_yaw_to_dir(x[3], x[4])
        model.set_params(eye_in_temp, lookat_in)
        _, images = model()
        cv2.imwrite(f"{output_folder}/test_gt_ref.png", (images.detach().squeeze().cpu().numpy() * 255)[:, :, ::-1].astype(np.uint8))

    simplex_mult = 1
    def get_init_simplex(guess):
        init_simplex = np.stack([guess for _ in range(6)], axis=0)
        init_simplex[0, 0] += x_range * simplex_mult
        init_simplex[1, 0] += -x_range * simplex_mult
        init_simplex[1, 1] += y_range * simplex_mult
        init_simplex[2, 1] += -y_range * simplex_mult
        init_simplex[2, 2] += z_range * simplex_mult
        init_simplex[3, 2] += -z_range * simplex_mult
        init_simplex[3, 3] += pitch_range * simplex_mult
        init_simplex[4, 3] += -pitch_range * simplex_mult
        init_simplex[4, 4] += yaw_range * simplex_mult
        init_simplex[5, 4] += -yaw_range * simplex_mult
        return init_simplex

    res_array[num_sample, 0] = np.array([eye_in_n[0, 0], eye_in_n[0, 1], eye_in_n[0, 2], pitch_n, yaw_n])

    res = minimize(target_fn, np.array([eye_in_n[0, 0], eye_in_n[0, 1], eye_in_n[0, 2], pitch_n, yaw_n]), method='Nelder-Mead', options={'fatol': 1e-4, 'initial_simplex': get_init_simplex(np.array([eye_in_n[0, 0], eye_in_n[0, 1], eye_in_n[0, 2], pitch_n, yaw_n]))})#'maxiter': 3000
    print(np.array(res.x) - np.array([x_ff, y_ff, z_ff, p_ff, yaw_ff]), res.success)
    res_array[num_sample, 1] = np.array(res.x) - np.array([x_ff, y_ff, z_ff, p_ff, yaw_ff])
    res = minimize(target_fn, np.array(res.x), method='Nelder-Mead', options={'fatol': 1e-4, 'initial_simplex': get_init_simplex(np.array(res.x))})#'maxiter': 3000
    print(np.array(res.x) - np.array([x_ff, y_ff, z_ff, p_ff, yaw_ff]), res.success)
    res_array[num_sample, 2] = np.array(res.x) - np.array([x_ff, y_ff, z_ff, p_ff, yaw_ff])
    res = minimize(target_fn, np.array(res.x), method='Nelder-Mead', options={'fatol': 1e-6, 'initial_simplex': get_init_simplex(np.array(res.x))})#'maxiter': 3000
    print(np.array(res.x) - np.array([x_ff, y_ff, z_ff, p_ff, yaw_ff]), res.success)
    # render_fn(np.array(res.x))
    offsets.append(np.array(res.x) - np.array([x_ff, y_ff, z_ff, p_ff, yaw_ff]))
    print(f"Result: {res.x}")

    ret = np.array(res.x)
    eye_in_temp = ret[:3].reshape((1, 3))
    lookat_in = eye_in_temp + pitch_yaw_to_dir(ret[3], ret[4])
    model.set_params(eye_in_temp, lookat_in)
    loss, _ = model()
    res_array2[num_sample] = loss
    print(f"Result: {ret}, loss:{loss}")
    res_array[num_sample, 3] = np.array(res.x) - np.array([x_ff, y_ff, z_ff, p_ff, yaw_ff])

np.save(f"{output_folder}/result_raw.npy", res_array)
np.save(f"{output_folder}/result_loss_raw.npy", res_array2)


