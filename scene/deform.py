import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[..., 0:1], -quaternion[..., 1:]), -1)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )


    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))



    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    
def quaternion_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Compute the quaternion representing the rotation from v1 to v2.
    v1 and v2 are normalized vectors of shape (..., 3).
    Returns a quaternion of shape (..., 4).
    """
    cross_prod = torch.cross(v1, v2, dim=-1)

    dot_prod = (v1 * v2).sum(dim=-1, keepdim=True)
    q = torch.cat([dot_prod, cross_prod], dim=-1)

    q = standardize_quaternion(q)

    q = q / (q.norm(dim=-1, keepdim=True)+1e-8)
    return q

def get_transform(start_points, end_points):
    # calculate the transformation matrix between end_points and start_points
    transformation_matrix = end_points - start_points # N x 3
    N = start_points.shape[0]
    start_vectors = start_points.unsqueeze(1) - start_points.unsqueeze(0) # N x N x 3
    end_vectors = end_points.unsqueeze(1) - end_points.unsqueeze(0) # N x N x 3

    l2_dis_start = torch.norm(start_vectors, dim=-1)
    l2_dis_start[torch.eye(l2_dis_start.shape[0]).bool()] = float('inf')

    start_vectors_norm = start_vectors / (start_vectors.norm(dim=-1, keepdim=True) + 1e-8)
    end_vectors_norm = end_vectors / (end_vectors.norm(dim=-1, keepdim=True) + 1e-8)

    w = -l2_dis_start   
    if start_points.shape[0] > 2:
        w = torch.where(w >= torch.topk(w, 2).values[:, -1].unsqueeze(-1), w, -float('inf'))
    w[w<-10000] =  0
    weight = w

    weight = 1 - weight/weight.sum(dim=-1, keepdim=True)

    q_diff = quaternion_between_vectors(start_vectors_norm, end_vectors_norm)  # N x N x 4

    q = (q_diff * weight.unsqueeze(-1)).sum(dim=1)  # N x 4
    q = standardize_quaternion(q)
    q = q / q.norm(dim=-1, keepdim=True)  # Normalize the quaternion

    return transformation_matrix, q
    
    
def deform_Gaussians(xyz, quant, start_points, end_points):
    if start_points.shape[0] == 1:
        new_xyz = xyz + (end_points - start_points)
        new_quant = quant
        return new_xyz, new_quant
    

    transform, q = get_transform(start_points, end_points) # N x 3, N x 4

    xyz_dis = xyz.unsqueeze(1) - start_points.unsqueeze(0) # n x N x 3

    if start_points.shape[0] > 2:

        w = -torch.norm(xyz_dis, dim=-1)
        w = torch.where(w >= torch.topk(w, 2).values[:, -1].unsqueeze(-1), w, -float('inf'))
    else:
        w = -torch.norm(xyz_dis, dim=-1)
    w[w<-10000] =  0

    weight=w
    weight = weight/weight.sum(dim=-1, keepdim=True)

    q = q.unsqueeze(0) # n x N x 4
    q = (q * weight.unsqueeze(-1)).sum(dim=1) # n x 4
    q = standardize_quaternion(q)

    transform = transform.unsqueeze(0).expand(xyz.shape[0], *transform.shape) # n, N, 3

    transform = (transform * weight.unsqueeze(-1)).sum(dim=1) # n x 3

    new_xyz = xyz + transform
    new_q = quaternion_multiply(q, quant) 
    # new_q = quant
    return new_xyz, new_q


def test_deform_Gaussians():

    start_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    end_points = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.7071, 0.7071, 0.0],
        [1.4142, 0.0, 0.0]
    ])

    xyz = torch.tensor([
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0]
    ])
    quant = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ])
    new_xyz, new_quant = deform_Gaussians(xyz, quant, start_points, end_points)

def quaternion_to_rotation_matrix(q):

    w, x, y, z = q
    R = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                  [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                  [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]])
    return R

# Helper function to plot a disk
def plot_disk(ax, center, radius=0.5, color='b', normal=np.array([0, 0, 1]), resolution=30):
    u = np.linspace(0, 2 * np.pi, resolution)
    x = radius * np.cos(u)
    y = radius * np.sin(u)
    z = np.zeros_like(x)
    

    rotated_disk = np.dot(normal, np.array([x, y, z]))
    x, y, z = rotated_disk
    
    x += center[0]
    y += center[1]
    z += center[2]
    
    ax.plot(x, y, z, color=color)

# Visualize disks at xyz points and control points
def visualize_disks_and_control_points(xyz, new_xyz, start_quats, new_quats, start_points, end_points):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    

    for i in range(xyz.shape[0]):
        center = xyz[i].numpy()  
        quat = start_quats[i].numpy()    
        normal = quaternion_to_rotation_matrix(quat) 
        plot_disk(ax, center, radius=0.5, color='b', normal=normal)  
    
    for i in range(new_xyz.shape[0]):
        center = new_xyz[i].numpy()  
        quat = new_quats[i].numpy()  
        normal = quaternion_to_rotation_matrix(quat) 
        plot_disk(ax, center, radius=0.5, color='r', normal=normal)  
    
    ax.scatter(start_points[:, 0], start_points[:, 1], start_points[:, 2], color='g', s=50, label='Start Points')
    
    ax.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2], color='m', s=50, label='End Points')

    # draw lines between start_points and end_points
    for i in range(start_points.shape[0]):
        ax.plot([start_points[i, 0], end_points[i, 0]], [start_points[i, 1], end_points[i, 1]], [start_points[i, 2], end_points[i, 2]], color='k', linestyle='--')
    
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    
    ax.legend()

    plt.show()

def test_visualize_disks_and_control_points():
    start_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    end_points = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.7071, 0.7071, 0.0],
        [1.4142, 0.0, 1]
    ])
    
    xyz = torch.tensor([
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0]
    ])
    quant = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ])

    new_xyz, new_quats = deform_Gaussians(xyz, quant, start_points, end_points)


    visualize_disks_and_control_points(xyz, new_xyz, quant, new_quats, start_points, end_points)

# test_visualize_disks_and_control_points()