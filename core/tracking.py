import numpy as np
from numpy.linalg import pinv, norm
from core.map_initialization import map_init_from_path
from liegroups.numpy import SE3

######################################################################
# Main Purpose of "tracking.py"                                      #
# Estimate mu_i+1(or R_i, t_i) by iteration of Gauss-Newton method   #
# iteration 10 times (s = 0, ... , 9)                                #
######################################################################


def trackPose(X_2_cur, X_3_prev, X_3_map, C):
    # init_pose, mu = get_initial_pose()
    init_pose = C.pose
    mu = np.zeros(6)
    pose = init_pose
    for s in range(10):
        X_3_cur = get_camera_coordinate(pose, X_3_map)
        J = get_Jacobian(X_3_cur, C.K)

        error = compute_error(X_3_cur, X_2_cur, C.K)
        delta = - (pinv(J.T@J)@J.T @ error).squeeze()
        mu = mu + delta
        motion = SE3.exp(mu).as_matrix()
        pose = motion @ init_pose
    return pose


def get_inputs():
    K = np.array([[3.10593801e+03, 0.00000000e+00, 1.53552466e+03],
                  [0.00000000e+00, 3.08841292e+03, 2.03002207e+03],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    _, X_2_prev, X_2_cur, X_3_map = map_init_from_path(
        "./data/all1.jpeg", "./data/all2.jpeg", 0.7, K)

    X_3_prev = X_3_map

    return X_3_map, X_2_prev, X_3_prev, X_2_cur, K


# nonhomogeneous (n, dim) -> homogeneous (n, dim+1)
def get_homogeneous(X):
    n = X.shape[0]
    ones = np.ones((n, 1))
    unproj_X = np.concatenate((X, ones), axis=1)

    return unproj_X


# homogeneous (n, dim) -> nonhomogeneous (n, dim-1)
def get_nonhomogeneous(X):
    dim = X.shape[-1]
    proj_X = X[:, :dim-1]/X[:, [dim-1]]

    return proj_X

### Temporary ver. ###
# set R_i+1, t_i+1 at s=0 as R_i, t_i


def get_initial_pose():
    mu = np.zeros(6)
    pose = np.eye(4)
    pose[0, 3] = -30
    pose[3, 3] = 1
    return pose, mu


# (n,3) nonhomogeneous world coordinate -> (n,3) homogeneous camera coordinate
def get_camera_coordinate(pose, X_3_map):
    X_3_map_h = get_homogeneous(X_3_map)

    X_3_h = X_3_map_h @ pose.T

    X_3 = get_nonhomogeneous(X_3_h)

    return X_3


# (n,3) -> (n,2)
# get normal plane(z=1) coordinate from 3D nonhomogeneous coordinate
def get_normal_coordinate(X_3):
    X_normal = X_3[:, :2]/X_3[:, [2]]
    return X_normal


# (n,2) -> (n,2)
# get image coordinate in pixel units from normal plane coordinate
def get_img_coordinate(X_normal, K):

    X_img = X_normal @ np.array([[K[0, 0], 0], [0, K[1, 1]]]
                                ) + np.array([K[0, 2], K[1, 2]])

    return X_img


# Compute LMS error
def compute_error(X_3_cur, X_2_cur, K):
    X_normal = get_normal_coordinate(X_3_cur)
    X_2_cur_estimation = get_img_coordinate(X_normal, K)
    error = X_2_cur_estimation-X_2_cur  # (n,2)

    error = error.reshape(-1, 1)

    return error


# (2n,6)
# d(image plane coordinate) / d(motion)
def get_Jacobian(X_3_cur, K):

    J_in = get_img_normal_Jacobian(K)  # (1,2,2)
    J_nm = get_normal_motion_Jacobian(X_3_cur)  # (n,2,6)

    J = np.matmul(J_in, J_nm)

    J = J.reshape(-1, 6)

    return J


# (n,2,6)
# d(normal plane coordinate(z=1)) / d(motion)
def get_normal_motion_Jacobian(X_3_cur):
    n = X_3_cur.shape[0]
    A = np.zeros((n, 2, 6))
    B = np.zeros((n, 1, 6))

    A[:, 0, 0] = 1
    A[:, 1, 1] = 1
    A[:, 0, 4] = X_3_cur[:, 2]  # z_i
    A[:, 1, 3] = -X_3_cur[:, 2]  # -z_i
    A[:, 0, 5] = -X_3_cur[:, 1]  # -y_i
    A[:, 1, 5] = X_3_cur[:, 0]  # x_i

    A = A / X_3_cur[:, [2]][:, None]

    B[:, 0, 2] = 1
    B[:, 0, 3] = X_3_cur[:, 1]  # y_i
    B[:, 0, 4] = -X_3_cur[:, 0]  # -x_i

    B = np.matmul(X_3_cur[:, :2, None], B) / \
        np.power(X_3_cur[:, [2]][:, None], 2)

    J_nm = A-B

    return J_nm


# (2,2)
# d(img plane coordinate) / d(normal plane coordinate(z=1))
def get_img_normal_Jacobian(K):

    J_in = np.array([[K[0, 0], 0], [0, K[1, 1]]])

    return J_in[None]


def tracking(FP, C):
    X_3_map = FP.X_3D_0
    X_2D_prev = FP.X_2D_prev
    X_3D_prev = FP.X_3D_prev
    X_2D_cur = FP.X_2D_cur

    C.pose = trackPose(X_2_cur=X_2D_cur, X_3_prev=X_3D_prev,
                       X_3_map=X_3_map, C=C)
    C.R = C.pose[:3, :3]
    C.t = C.pose[:3, 3]

    return C
