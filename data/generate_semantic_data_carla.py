#!/usr/bin/env python

# This generation script is based on the CARLA API example for projecting lidar point clouds into RGB images

from datetime import datetime
import os
import sys

import carla

import argparse
from queue import Queue
from queue import Empty
from matplotlib import cm

import open3d as o3d

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure Pillow package is installed')

WAYPOINTS_T10HD = [(13, 15, 3), (-43, 15, 3), (-43, -75, 3)]
PERSPECTIVE_T10HD = (-43, 15, 7, -22, -51)
MAP_T10HD = "Town10HD_Opt"

WAYPOINTS_T02 = [(43, 187, 3), (43, 238, 3), (134, 238, 3)]
PERSPECTIVE_T02 = (40, 243, 6, -14, -45)
MAP_T02 = "Town02_Opt"

WAYPOINTS = WAYPOINTS_T02
PERSPECTIVE = PERSPECTIVE_T02
MAP = MAP_T02

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

TAG_MAP = {
    0: np.array([0, 0, 0]),
    1: np.array([70, 70, 70]),
    2: np.array([100, 40, 40]),
    3: np.array([55, 90, 80]),
    4: np.array([220, 20, 60]),
    5: np.array([153, 153, 153]),
    6: np.array([157, 234, 50]),
    7: np.array([128, 64, 128]),
    8: np.array([244, 35, 232]),
    9: np.array([107, 142, 35]),
    10: np.array([0, 0, 142]),
    11: np.array([102, 102, 156]),
    12: np.array([220, 220, 0]),
    13: np.array([70, 130, 180]),
    14: np.array([81, 0, 81]),
    15: np.array([150, 100, 100]),
    16: np.array([230, 150, 140]),
    17: np.array([180, 165, 180]),
    18: np.array([250, 170, 30]),
    19: np.array([110, 190, 160]),
    20: np.array([170, 120, 50]),
    21: np.array([45, 60, 150]),
    22: np.array([145, 170, 100])
}

def tag_array_to_colors(tags):
    colors = np.zeros((len(tags), 3))
    for tag, color in TAG_MAP.items():
        colors[tags == tag] = color / 255.0
    return colors

def sensor_callback(data, queue):
    queue.put(data)


def tutorial(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    print(client.get_available_maps())
    # world = client.get_world()
    # world = client.load_world("Town11")
    world = client.load_world(MAP, carla.MapLayer.Buildings | carla.MapLayer.Foliage | carla.MapLayer.Ground | carla.MapLayer.Walls)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    # Doing this sometimes crashes the server
    # world.unload_map_layer(carla.MapLayer.Decals)
    # world.unload_map_layer(carla.MapLayer.Particles)
    # world.unload_map_layer(carla.MapLayer.Props)
    world.unload_map_layer(carla.MapLayer.StreetLights)
    map = world.get_map()
    bp_lib = world.get_blueprint_library()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 3.0
    world.apply_settings(settings)

    camera = None
    lidar = None

    try:
        if not os.path.isdir('_out'):
            os.mkdir('_out')
        # Search the desired blueprints
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        segmentation_bp = bp_lib.filter("sensor.camera.semantic_segmentation")[0]
        lidar_bp = bp_lib.filter("sensor.lidar.ray_cast_semantic")[0]        start_position = start_transform.location
        start_rotaion = start_transform.rotation

        segmentation_bp.set_attribute("image_size_x", str(args.width))
        segmentation_bp.set_attribute("image_size_y", str(args.height))

        # if args.no_noise:
        #     lidar_bp.set_attribute('dropoff_general_rate', '0.0')
        #     lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
        #     lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        lidar_bp.set_attribute('upper_fov', str(args.upper_fov))
        lidar_bp.set_attribute('lower_fov', str(args.lower_fov))
        lidar_bp.set_attribute('channels', str(args.channels))
        lidar_bp.set_attribute('range', str(args.range))
        lidar_bp.set_attribute('points_per_second', str(args.points_per_second))

        # Spawn the blueprints
        start_transform = map.get_spawn_points()[0]
        x, y, z = PERSPECTIVE[:3]
        pitch, yaw = PERSPECTIVE[3:]
        x2, y2, z2 = WAYPOINTS[0]
        camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw)))
        segmentation = world.spawn_actor(
            blueprint=segmentation_bp,
            transform=carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw)))
        lidar = world.spawn_actor(
            blueprint=lidar_bp,
            transform=carla.Transform(carla.Location(x=x2, y=y2, z=z2)))

        # The sensor data will be saved in thread-safe Queues
        image_queue = Queue()
        segmentation_queue = Queue()
        lidar_queue = Queue()

        camera.listen(lambda data: sensor_callback(data, image_queue))
        lidar.listen(lambda data: sensor_callback(data, lidar_queue))
        segmentation.listen(lambda data: sensor_callback(data, segmentation_queue))

        pcl_all = o3d.geometry.PointCloud()

        wp_id = 0

        for frame in range(args.frames):
            world.tick()
            world_frame = world.get_snapshot().frame

            ds = datetime.now().strftime('%y%m%d%H%M%S') + f"_{frame:03d}"

            try:
                # Get the data once it's received.
                image_data = image_queue.get(True, 1.0)
                segmentation_data = segmentation_queue.get(True, 1.0)
                lidar_data = lidar_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            assert image_data.frame == segmentation_data.frame == lidar_data.frame == world_frame
            # At this point, we have the synchronized information from the 2 sensors.
            sys.stdout.write("\r(%d/%d) Simulation: %d Camera: %d Segmentation: %d Lidar: %d" %
                (frame, args.frames, world_frame, image_data.frame, segmentation_data.frame, lidar_data.frame) + ' ')
            sys.stdout.flush()

            # Get the raw BGRA buffer and convert it to an array of RGB of
            # shape (image_data.height, image_data.width, 3).
            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            # Get the lidar data and convert it to a numpy array.
            # p_cloud_size = len(lidar_data)
            # p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            # p_cloud = np.reshape(p_cloud, (p_cloud_size, 6))
            data = np.frombuffer(lidar_data.raw_data, dtype=np.dtype([
                                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                                ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

            # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
            # focus on the 3D points.
            tags = np.array(data['ObjTag'])#np.array(p_cloud[:, -1], dtype=int)
            obj_ids = np.array(data['ObjIdx'])

            # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
            local_lidar_points = np.array([data['x'], data['y'], data['z']])#np.array(p_cloud[:, :3]).T
            closest_idx = obj_ids[np.linalg.norm(local_lidar_points, axis=0).argmin()]
            print(f"id:{closest_idx}")
            mask = ~((tags == 10) & (obj_ids == closest_idx))
            local_lidar_points = local_lidar_points[:, mask]
            tags = tags[mask]

            # Add an extra 1.0 at the end of each 3d point so it becomes of
            # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
            local_lidar_points = np.r_[
                local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

            # This (4, 4) matrix transforms the points from lidar space to world space.
            lidar_2_world = lidar.get_transform().get_matrix()

            # Transform the points from lidar space to world space.
            world_points = np.dot(lidar_2_world, local_lidar_points)

            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(world_points[:3].T)
            colors = tag_array_to_colors(tags)
            pcl.colors = o3d.utility.Vector3dVector(colors)
            if wp_id < len(WAYPOINTS) - 1:
                pcl_all += pcl
            if frame == args.frames - 1:
                o3d.io.write_point_cloud(f"_out/{ds}_pcl.ply", pcl_all)

            # Save the image using Pillow module.
            if frame == args.frames - 1:
                image = Image.fromarray(im_array)
                image.save(f"_out/{ds}.png")

                segmentation_data.save_to_disk(f"_out/{ds}_seg.png", carla.ColorConverter.CityScapesPalette)

                current_transform = segmentation.get_transform()
                lx, ly, lz = current_transform.location.x, current_transform.location.y, current_transform.location.z
                pit, yaw, roll = current_transform.rotation.pitch, current_transform.rotation.yaw, current_transform.rotation.roll

                with open(f"_out/{ds}.txt", "w") as f:
                    f.write(f"x={lx} y={ly} z={lz} p={pit} y={yaw}\n")
                # camera.set_transform(carla.Transform(carla.Location(x=88 + (frame // 20) * 10, y=58, z=6), carla.Rotation(pitch=-10, yaw=0 + (frame // 20) * 4)))
                # segmentation.set_transform(carla.Transform(carla.Location(x=88 + (frame // 20) * 10, y=58, z=6), carla.Rotation(pitch=-10, yaw=0 + (frame // 20) * 4)))
            # waypoint = waypoint.next(0.75)[0]
            # vehicle.set_transform(waypoint.transform)
            current_transform = lidar.get_transform()
            lx, ly, lz = current_transform.location.x, current_transform.location.y, current_transform.location.z
            if wp_id < len(WAYPOINTS) - 1:
                new_dir = np.array(list(WAYPOINTS[wp_id + 1])) - np.array(list(WAYPOINTS[wp_id]))
                xn, yn, zn = new_dir / np.linalg.norm(new_dir) * 0.5
                lidar.set_transform(carla.Transform(carla.Location(lx+xn, ly+yn, lz+zn), current_transform.rotation))
                print(f"nx={lx+xn} ny={ly+yn} nz={lz+zn}")
                if np.linalg.norm(np.array(list(WAYPOINTS[wp_id + 1])) - (np.array([lx, ly, lz]) + new_dir / np.linalg.norm(new_dir) * 0.5)) <= 0.25:
                    wp_id += 1


    finally:
        # Apply the original settings when exiting.
        world.apply_settings(original_settings)

        # Destroy the actors in the scene.
        if camera:
            camera.destroy()
        if segmentation:
            segmentation.destroy()
        if lidar:
            lidar.destroy()


def main():
    """Start function"""
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor sync and projection tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='960x600',
        help='window resolution (default: 960x600)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=2,
        type=int,
        help='number of frames to record (default: 2)')
    # argparser.add_argument(
    #     '--no-noise',
    #     action='store_true',
    #     help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--upper-fov',
        metavar='F',
        default=45.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 45.0)')
    argparser.add_argument(
        '--lower-fov',
        metavar='F',
        default=-45.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -45.0)')
    argparser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=128.0,
        type=float,
        help='lidar\'s channel count (default: 128)')
    argparser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        metavar='N',
        default='100000',
        type=int,
        help='lidar points per second (default: 100000)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        tutorial(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()