from glob import glob
import pickle
import numpy as np

import open3d as o3d

color_map_original = {
  0 : [0, 0, 0],
  1 : [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0]
}

color_map = {
  0 : [0, 0, 0],
  1 : [0, 0, 0],
  10: [0, 0, 142],
  11: [119, 11, 32],
  13: [0, 60, 100],
  15: [0, 0, 230],
  16: [0, 80, 100],
  18: [0, 0, 70],
  20: [0, 0, 142],
  30: [220, 20, 60],
  31: [255, 0, 0],
  32: [255, 0, 0],
  40: [128, 64, 128],
  44: [244, 35, 232],
  48: [244, 35, 232],
  49: [152, 251, 152],
  50: [70, 70, 70],
  51: [190, 153, 153],
  52: [70, 70, 70],
  60: [128, 64, 128],
  70: [107, 142, 35],
  71: [107, 142, 35],
  72: [152, 251, 152],
  80: [153, 153, 153],
  81: [220, 220, 0],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0]
}

ignore_labels = [0, 1, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259]
remap = {49: 40, 48:40, 44:40, 72:40}

def parse_calibration(filename):
  """ read calibration file with given filename
      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib


def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename
      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  Tr_inv = np.linalg.inv(Tr)

  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

  return poses

# KITTI sequence ID
id = 7
# Number of reference frame we want to calibrate
num = 53
# First frame ID which is used for the lidar environment reconstruction
start = 0
# Number of lidar frames to accumulate
lidar_frames = 250
# Toggle 
FILTER_VEHICLES = False

if FILTER_VEHICLES:
  ignore_labels += [10, 15, 18, 20]

dataset_root = "datasets/KITTI"

velodyne_bin_dir = f"{dataset_root}/data_odometry_velodyne/dataset/sequences/{id:02d}/velodyne/"
# label_bin_dir = f"{dataset_root}/data_odometry_labels/dataset/sequences/00/labels/"
label_bin_dir = f"/{dataset_root}/label_infer/{id:02d}/"
calib_txt = f"{dataset_root}/data_odometry_calib/dataset/sequences/{id:02d}/calib.txt"
poses_txt = f"{dataset_root}/data_odometry_labels/dataset/sequences/{id:02d}/poses.txt"
image_path = f"{dataset_root}/data_odometry_color/dataset/sequences/{id:02d}/image_2/{num:06d}.png"
image_calib = f"{dataset_root}/data_odometry_color/dataset/sequences/{id:02d}/calib.txt"

calib = parse_calibration(calib_txt)
tr = calib["Tr"]
poses = parse_poses(poses_txt, calib)

velodyne_bins = sorted(glob(velodyne_bin_dir + "*.bin"))[start:start+lidar_frames]
label_bins = sorted(glob(label_bin_dir + "*.label"))[start:start+lidar_frames]

accum_pcl = o3d.geometry.PointCloud()

pcls = []
pcl_tfs = []

boxes = []
meshes = []
meshes_orig = []

ref_pose = poses[num]
ref_pose_inv = np.linalg.inv(ref_pose)
for i, velodyne_bin, label_bin in zip(range(len(velodyne_bins)), velodyne_bins, label_bins):
    i += start
    print(f"Loading {velodyne_bin}")
    scan = np.fromfile(velodyne_bin, dtype=np.float32)
    scan = scan.reshape((-1, 4))[:, :3]
    # labels = np.fromfile(label_bin, dtype=np.uint32)
    # labels = labels.reshape((-1)) & 0xFFFF
    labels = np.load(label_bin)
    local_pose = poses[i]
    combined_tf = ref_pose_inv @ local_pose

    color_arr = np.zeros((scan.shape[0], 3))
    for _id, color in color_map.items():
        color_arr[labels == _id] = np.array(color)
    color_arr = color_arr / 255

    mask = np.zeros((scan.shape[0])) > 0
    for j in ignore_labels:
      mask[labels == j] = True
    scan = scan[~mask]
    color_arr = color_arr[~mask]

    # Save ego vehicle bounding box to remove associated points
    add_right = 0.0
    half_width = 1.65 if id == 7 else 2.0
    # 1.7m for 01 and 05, 2 for 10
    aabb = o3d.geometry.AxisAlignedBoundingBox(np.array([-1.5, -half_width-add_right, -1.6]).reshape((3, 1)), np.array([2, half_width + add_right, 1.0]).reshape((3, 1))).get_oriented_bounding_box()
    mesh = o3d.geometry.TriangleMesh.create_box(width=3.5, height=2*half_width+2*add_right, depth=2.6).translate(np.array([-1.5, -half_width-add_right, -1.6]))
    mesh.rotate(combined_tf[:3, :3])
    mesh.translate(combined_tf[:3, 3])
    meshes.append((np.asarray(mesh.vertices), np.asarray(mesh.triangles)))
    meshes_orig.append(mesh)
    
    aabb.rotate(combined_tf[:3, :3])
    aabb.translate(combined_tf[:3, 3])
    boxes.append(aabb)

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(scan)
    pcl.colors = o3d.utility.Vector3dVector(color_arr)

    pcl.transform(combined_tf)
    accum_pcl += pcl

with open("meshes.pkl", "wb") as f:
  pickle.dump(meshes, f)

o3d.visualization.draw_geometries([accum_pcl] + meshes_orig)

mask = np.zeros(len(accum_pcl.points)) > 1

for aabb in boxes:
  indices = aabb.get_point_indices_within_bounding_box(accum_pcl.points)
  mask[indices] = True

indices_to_keep = np.nonzero(~mask)[0].tolist()
accum_pcl = accum_pcl.select_by_index(indices_to_keep)
o3d.visualization.draw_geometries([accum_pcl])

camera_calib = parse_calibration(image_calib)["P2"]

np.save(f"kitti_ideal_calib_{id:02d}.npy", camera_calib)
accum_pcl.transform(tr)
o3d.io.write_point_cloud(f"kitti_{id:02d}_{lidar_frames}.ply", accum_pcl)