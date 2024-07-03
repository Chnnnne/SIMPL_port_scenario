import bezier
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import comb
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.spatial import KDTree
import torch
# from shapely.geometry import LineString, Point


def get_bezier_curve_length(curve):
    def speed(t):
        dx, dy = curve.evaluate_hodograph(t)
        return np.sqrt(dx**2 + dy**2)
    
    curve_length,error = quad(speed,0,1)
    # print(res, error)
    return curve_length

def get_control_points(start, direction_start, end, direction_end, scale):
    direction_start_normalized = direction_start / np.linalg.norm(direction_start)
    direction_end_normalized = direction_end / np.linalg.norm(direction_end)
    
    # 控制点基于起始点和朝向
    control1 = start + direction_start_normalized * scale
    
    # 控制点基于终点和朝向
    control2 = end - direction_end_normalized * scale
    
    return control1, control2

def get_bezier_function(start_point, start_direction, end_point, end_direction, scale = 1.0):
    # rad-> unit vec
    start_point = np.asarray(start_point)
    end_point = np.asarray(end_point)
    start_direction = np.asarray([math.cos(start_direction), math.sin(start_direction)])
    end_direction = np.asarray([math.cos(end_direction), math.sin(end_direction)])
    scale = np.linalg.norm(end_point - start_point)*2.7/8
    control1, control2 = get_control_points(start_point, start_direction, end_point, end_direction, scale)
    nodes = np.asfortranarray([
        start_point,
        control1,
        control2,
        end_point
    ]).T
    curve = bezier.Curve(nodes, degree = 3)
    return curve, nodes
    # t_values = np.linspace(0, 1, 15)
    # sampled_points = curve.evaluate_multi(t_values)



def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """计算点(px, py)到由(x1, y1)和(x2, y2)定义的线段的最近距离"""
    line_vec = np.array([x2 - x1, y2 - y1])
    point_vec = np.array([px - x1, py - y1])
    line_len = line_vec.dot(line_vec)
    if line_len == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)  # 线段退化为点
    t = point_vec.dot(line_vec) / line_len
    if t < 0:
        nearest = np.array([x1, y1])
    elif t > 1:
        nearest = np.array([x2, y2])
    else:
        nearest = np.array([x1, y1]) + t * line_vec
    return np.sqrt((px - nearest[0]) ** 2 + (py - nearest[1]) ** 2)

def closest_distance_to_path(point, path):
    """点到路径的最近距离"""
    min_distance = float('inf')
    px, py = point
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        dist = point_to_segment_distance(px, py, x1, y1, x2, y2)
        if dist < min_distance:
            min_distance = dist
    return min_distance

def calculate_trajectory_max_projection_distance_use_KDTree(traj1, traj2, traj2_KDTree):
    # 求轨迹i到轨迹j的距离， 距离的定义是：traj1中所有点到traj2投影的最大值
    max_distance = - float('inf')
    for traj1_cord in traj1:# 计算traj1中每个点到traj2的距离（也即投影）
        _, nearest_idx = traj2_KDTree.query(traj1_cord) # 找到traj2上距离query_point最近的点的idx
        nearest_point = traj2[nearest_idx]
        line_segments = [] # list[items]  item = (start_point, end_point)

        # 构建最近点前后的线段
        if nearest_idx > 0:
            line_segments.append((traj2[nearest_idx - 1], nearest_point))
        if nearest_idx < len(traj2) - 1:
            line_segments.append((nearest_point, traj2[nearest_idx + 1]))
        # 计算查询点到最近点相关线段的最小距离
        min_distance = float('inf')
        for p1, p2 in line_segments:
            distance = point_to_segment_distance(traj1_cord[0], traj1_cord[1], p1[0], p1[1], p2[0], p2[1])
            if distance < min_distance:
                min_distance = distance
        if min_distance > max_distance:# 如果该点到轨迹的距离大于最大距离，则记录之
            max_distance = min_distance
        
    return max_distance

def calculate_trajectory_sum_projection_distance_use_KDTree(traj1, traj2, traj2_KDTree):
    # 求轨迹i到轨迹j的距离， 距离的定义是：traj1中所有点到traj2投影的最大值
    sum_distance = 0
    for traj1_cord in traj1:# 计算traj1中每个点到traj2的距离（也即投影）
        _, nearest_idx = traj2_KDTree.query(traj1_cord) # 找到traj2上距离query_point最近的点的idx
        nearest_point = traj2[nearest_idx]
        line_segments = [] # list[items]  item = (start_point, end_point)

        # 构建最近点前后的线段
        if nearest_idx > 0:
            line_segments.append((traj2[nearest_idx - 1], nearest_point))
        if nearest_idx < len(traj2) - 1:
            line_segments.append((nearest_point, traj2[nearest_idx + 1]))
        # 计算查询点到最近点相关线段的最小距离
        min_distance = float('inf')
        for p1, p2 in line_segments:
            distance = point_to_segment_distance(traj1_cord[0], traj1_cord[1], p1[0], p1[1], p2[0], p2[1])
            if distance < min_distance:
                min_distance = distance
        sum_distance += min_distance
        
    return sum_distance

def calculate_trajectory_max_projection_distance(traj1, traj2):
    # 求两条轨迹之间的距离
    # limit1 = int(len(traj1))
    # limit2 = int(len(traj2))
    # traj1 = traj1[:limit1]
    # traj2 = traj2[:limit2]
    # # 确保两轨迹长度相同
    # min_len = min(len(traj1), len(traj2))
    # traj1, traj2 = traj1[:min_len], traj2[:min_len]
    # distance = sum(np.linalg.norm(p1 - p2) for p1, p2 in zip(traj1, traj2))
    
    # 求traj1上每个点到traj2的最小距离，累加
    sum = 0
    max_dist = -1e9
    # for i in range(min_len):
    for traj_cord in traj1:
        max_dist = max(max_dist, closest_distance_to_path((traj_cord[0], traj_cord[1]), traj2))
    return max_dist

def create_KDTrees(trajectories):
    '''
    计算每条轨迹上的所有点所对应的KDTree
    '''
    kd_trees = []
    for traj in trajectories:
        kd_trees.append(KDTree(traj))
    return kd_trees


def calculate_distance_matrix(trajectories):
    # 计算所有轨迹间的距离矩阵
    n = len(trajectories)
    distance_matrix = np.zeros((n, n))
    kd_trees = create_KDTrees(trajectories)

    for i in range(n):
        for j in range(i + 1, n):
            # 求轨迹i到轨迹j的距离， 距离的定义是：traj1中所有点到traj2投影的最大值
            dist = calculate_trajectory_max_projection_distance_use_KDTree(trajectories[i], trajectories[j], kd_trees[j])
            # print("--"*100)
            # print(i,j,dist)
            # dist = calculate_trajectory_distance(trajectories[i], trajectories[j])
            # print(i,j,dist)
            # print("--"*100)
            distance_matrix[i, j] = distance_matrix[j, i] = dist
    return distance_matrix, kd_trees
    
def get_KDTree(traj_cords):
    return KDTree(traj_cords)

def cluster_trajs(traj_cords):
    '''
        - 过程:
            1. 计算距离矩阵（每两条轨迹之间的距离）
            2. 根据距离矩阵进行层次聚类（clutser之间用average方式）,将距离信息转换为一个树形结构(也即链接矩阵Z)
            3. 从链接矩阵Z中提取试剂的聚类结果，certerion设为distance，意味着簇之间的最大距离不超过0.6
    '''
    # 计算距离矩阵(完整) 也即计算轨迹i到轨迹j距离（方式：求点到轨迹的距离的最大值）
    distance_matrix, kd_trees = calculate_distance_matrix(traj_cords)

    # 使用层次聚类
    Z = linkage(squareform(distance_matrix), method='average') # 计算两个簇之间所有成对数据点之间的平均距离作为簇间距离，生成
    clusters = fcluster(Z, t=0.6, criterion='distance')
    # clusters:每条轨迹属于哪一个cluster， kd_trees每条轨迹对应的kd_tree   
    return clusters, kd_trees 



def bernstein_poly(i, n, t):
    """计算伯恩斯坦多项式的值"""
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_derivative(control_points):
    """
    计算贝塞尔曲线的一阶导数控制点
    input:
        - control_points   B,(n_order+1), 2
        - control_points   B,3,(n_order+1), 2
        - control_points   B,N,3m,(n_order+1), 2
    output:
        - derivative_points: B,3,n_order, 2
        - derivative_points   B,N,3m,n_order, 2
    """
    squeeze_one_flag = False
    squeeze_two_flag = False
    if len(control_points.shape) == 3:
        squeeze_two_flag = True
        control_points = control_points.unsqueeze(1).unsqueeze(1) # B,1,1,n+1,2
    if len(control_points.shape) == 4:
        squeeze_one_flag = True
        control_points = control_points.unsqueeze(1) # B,1,3,n+1,2
    n = control_points.shape[-2] - 1 # n阶贝塞尔曲线
    derivative = [n * (control_points[:,:,:,i + 1] - control_points[:,:,:,i]) for i in range(n)] # n个[B,N,M,2]
    derivative = torch.stack(derivative, dim=-2) # B,N,3m,n,2
    if squeeze_two_flag:
        derivative = derivative.squeeze(1).squeeze(1)# B,n,2
    if squeeze_one_flag:
        derivative = derivative.squeeze(1) # B,3,n,2
    return derivative

def bezier_curve(control_points, t_values):
    '''
    input:
        - control_points   B,(n_order+1), 2
        - control_points   B,3,(n_order+1), 2
        - control_points   B,N,3m,(n_order+1), 2
    return: 
        - curve_points  B, 50, 2
        - curve_points  B,3,50, 2
        - curve_points  B,N,3m,50, 2
    '''
    squeeze_one_flag = False
    squeeze_two_flag = False
    if len(control_points.shape) == 3:
        squeeze_two_flag = True
        control_points = control_points.unsqueeze(1).unsqueeze(1) # B,1,1,n+1,2
    if len(control_points.shape) == 4:
        squeeze_one_flag = True
        control_points = control_points.unsqueeze(1) # B,1,3,n+1,2
    B,N,M,n,_ = control_points.shape
    n -= 1 # n代表贝塞尔曲线的阶数
    curve_points = []
    for t_index, t in enumerate(t_values): # 采样个数
        point = torch.zeros(B,N,M,2).cuda()
        for i in range(n + 1): # 计算n+1次
            bernstein = bernstein_poly(i, n, t)
            point += bernstein * control_points[:,:,:,i,:] # B,N,M,2
        curve_points.append(point)
    curve_points = torch.stack(curve_points, dim=-2) # B,N,M,50,2
    if squeeze_two_flag:
        curve_points = curve_points.squeeze(1).squeeze(1) # B,50,2
    if squeeze_one_flag:
        curve_points = curve_points.squeeze(1) # B,M,50,2
    return curve_points



if __name__ == "__main__":
    # get_bezier_curve_length()
    curve, nodes = get_bezier_function([0,0],0, [8,8],math.pi/2)
    t_values = np.linspace(0, 1, 50)
    sampled_points = curve.evaluate_multi(t_values)
    # 绘制曲线和控制点
    fig, ax = plt.subplots()
    curve.plot(100, ax=ax)
    ax.plot(nodes[0, :], nodes[1, :], "ro--", label="Control points")
    ax.legend()

    fig.savefig("tmp.jpg")

