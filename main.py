import numpy as np
import open3d as o3d
from CSF2 import CSF2

def point2point_mean_and_std_deviation(pcd1, pcd2):
    """
    返回两个点云的平均值差和方差
    :param path1: 点云1路径
    :param path2: 点云2路径
    :return: mean_distance: 平均值距离
    :return: std_distance: 标准差距离
    """
    # 计算点到点距离
    distances = pcd1.compute_point_cloud_distance(pcd2)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    print(f"Mean distance: {mean_distance}")
    print(f"Standard deviation of distances: {std_distance}")
    return mean_distance, std_distance

if __name__ == '__main__':
    #标定物icp
    file1_path = 'data/066_box.pcd'
    file2_path = 'data/079_box.pcd'
    pcd1 = o3d.io.read_point_cloud(file1_path)
    pcd2 = o3d.io.read_point_cloud(file2_path)

    o3d.visualization.draw_geometries([pcd1, pcd2])
    point2point_mean_and_std_deviation(pcd1,pcd2)
    points1 = np.array(pcd1.points)
    points2 = np.array(pcd2.points)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points1)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(points2)

    icp = o3d.pipelines.registration.registration_icp(source=source, target=target, max_correspondence_distance=0.1)
    source.transform(icp.transformation)
    point2point_mean_and_std_deviation(source,target)
    big_file_path1 = 'data/066.pcd'
    big_file_path2 = 'data/079.pcd'

    big_pcd1 = o3d.io.read_point_cloud(big_file_path1)
    big_pcd2 = o3d.io.read_point_cloud(big_file_path2)

    points1 = big_pcd1.points
    points2 = big_pcd2.points
    source.points = o3d.utility.Vector3dVector(points1)
    target.points = o3d.utility.Vector3dVector(points2)
    big_pcd1.transform(icp.transformation)
    print('标定物对全局：')
    print(icp.transformation)
    point2point_mean_and_std_deviation(big_pcd1, big_pcd2)

    # big_pcd1 = o3d.io.read_point_cloud(big_file_path1)
    # big_pcd2 = o3d.io.read_point_cloud(big_file_path2)
    # points1 = big_pcd1.points
    # points2 = big_pcd2.points
    # source.points = o3d.utility.Vector3dVector(points1)
    # target.points = o3d.utility.Vector3dVector(points2)
    # icp = o3d.pipelines.registration.registration_icp(source=source, target=target, max_correspondence_distance=0.1)
    # big_pcd1.transform(icp.transformation)
    # print("全局匹配")
    # point2point_mean_and_std_deviation(big_pcd1, big_pcd2)
    # o3d.visualization.draw_geometries([big_pcd1, big_pcd2])
    #
    # file1_ground_path = 'data/066_ground.ply'
    # file2_ground_path = 'data/079_ground.ply'
    #
    # ground_pcd1 = o3d.io.read_point_cloud(file1_ground_path)
    # ground_pcd2 = o3d.io.read_point_cloud(file2_ground_path)
    #
    # points1 = np.array(ground_pcd1.points)
    # points2 = np.array(ground_pcd2.points)
    # source = o3d.geometry.PointCloud()
    # source.points = o3d.utility.Vector3dVector(points1)
    # target = o3d.geometry.PointCloud()
    # source.points = o3d.utility.Vector3dVector(points2)
    #
    # big_pcd1 = o3d.io.read_point_cloud(big_file_path1)
    # big_pcd2 = o3d.io.read_point_cloud(big_file_path2)
    # icp = o3d.pipelines.registration.registration_icp(source=source, target=target, max_correspondence_distance=0.1)
    # big_pcd1.transform(icp.transformation)
    # big_pcd1.paint_uniform_color([1, 0, 0])
    # big_pcd2.paint_uniform_color([0, 1, 0])
    # print('地面点')
    # print(icp.transformation)
    # point2point_mean_and_std_deviation(big_pcd1, big_pcd2)
    # o3d.visualization.draw_geometries([big_pcd1, big_pcd2])



