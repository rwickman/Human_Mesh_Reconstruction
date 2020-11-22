import numpy as np
import h5py
import tensorflow as tf

def load_mean_theta(args, num_params=85):
    mean = np.zeros((1, num_params))
    # Initialize scale at 0.9
    mean[0, 0] = 0.9
    # with h5py.File(args.smpl_mean_theta_file, "r") as f:
    #     print(f)
    #     print(f.keys())

    with h5py.File(args.smpl_mean_theta_file, "r") as f:
        mean_pose = f.get("pose")[()]
        # Ignore global rotation
        mean_pose[:3] = 0
        
        mean_shape = f.get("shape")[()]
        # This initializes the global pose to be up-right when projected
        mean_pose[0] = np.pi
        
        mean[0, args.num_cam_param:] = np.hstack((mean_pose, mean_shape))

        return tf.constant(mean, tf.float32)


def batch_rodrigues(theta, num_joints=24):
    """Perform Rodrigues' rotation formula"""
    batch_size = theta.shape[0]

    theta = tf.reshape(theta, [batch_size, num_joints, 3])
    batch_identity = tf.eye(3, 3, batch_shape=(batch_size, num_joints,))

    batch_theta_norm = tf.expand_dims(tf.norm(theta + 1e-8, axis=2), -1)
    batch_unit_norm_axis = tf.math.truediv(theta, batch_theta_norm)
    batch_skew_symm = batch_skew_symmetric(batch_unit_norm_axis)

    batch_cos = tf.expand_dims(tf.cos(batch_theta_norm), -1)
    batch_sin = tf.expand_dims(tf.sin(batch_theta_norm), -1)

    batch_unit_norm_axis = tf.expand_dims(batch_unit_norm_axis, -1)
    batch_outer = tf.matmul(batch_unit_norm_axis, batch_unit_norm_axis, transpose_b=True)
    rot_mat = batch_identity * batch_cos + (1 - batch_cos) * batch_outer + batch_sin * batch_skew_symm
    rot_mat = tf.reshape(rot_mat, [batch_size, num_joints, -1])
    return rot_mat


def batch_skew_symmetric(vector, num_joints=24):
    """computes skew symmetric matrix given vector
    Args:
        vector: [batch x (K + 1) x 3]
    Returns:
        skew_symm: [batch x (K + 1) x 3 x 3]
    """
    batch_size = vector.shape[0]

    zeros = tf.zeros([batch_size, num_joints, 3])

    # //@formatter:off
    skew_sym = tf.stack(
        [zeros[:, :, 0], -vector[:, :, 2], vector[:, :, 1],
         vector[:, :, 2], zeros[:, :, 1], -vector[:, :, 0],
         -vector[:, :, 1], vector[:, :, 0], zeros[:, :, 2]]
        , -1)
    # //@formatter:on

    return tf.reshape(skew_sym, [batch_size, num_joints, 3, 3])
# mean = load_mean_params("../models/neutral_smpl_mean_params.h5")
# print(mean)
# print(mean.shape)

def batch_global_rigid_transformation(rot_mat, joints, ancestors, rotate_base=False):
    """Computes absolute joint locations given pose.
        see equation 3 & 4 of SPML (http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf)
    Args:
        rot_mat     : [batch x (K + r) x 3 x 3] rotation matrix of K + r
                      with 'r' = 1 (global root rotation)
        joints      : [batch x (K + r) x 3] joint locations before posing
        ancestors   : K + r holding the ancestor id for every joint by index
        rotate_base : if True, rotates the global rotation by 90 deg in x axis,
                      else this is the original SMPL coordinate.
    Returns
        new_joints  : [batch x (K + 1) x 3] location of absolute joints
        rel_joints  : [batch x (K + 1) x 4 x 4] relative joint transformations for LBS.
    """
    batch_size = rot_mat.shape[0]
    num_joints = 24

    if rotate_base:
        # //@formatter:off
        rot_x = tf.constant([[1, 0, 0],
                             [0, -1, 0],
                             [0, 0, -1]], dtype=tf.float32)
        # //@formatter:on
        rot_x = tf.reshape(tf.tile(rot_x, [batch_size, 1]), [batch_size, 3, 3])
        root_rotation = tf.matmul(rot_mat[:, 0, :, :], rot_x)
    else:
        # global root rotation
        root_rotation = rot_mat[:, 0, :, :]

    def create_global_rot_for(_rotation, _joint):
        """creates the world transformation in homogeneous coordinates of joint
            see equation 4
        Args:
            _rotation: [batch x 3 x 3] rotation matrix of j's angles
            _joint: [batch x 3 x 1] single joint center of j
        Returns:
            _joint_world_trans: [batch x 4 x 4] world transformation in homogeneous
                                coordinates of joint j
        """
        _rot_homo = tf.pad(_rotation, [[0, 0], [0, 1], [0, 0]])
        _joint_homo = tf.concat([_joint, tf.ones([batch_size, 1, 1])], 1)
        _joint_world_trans = tf.concat([_rot_homo, _joint_homo], 2)
        return _joint_world_trans

    joints = tf.expand_dims(joints, -1)
    root_trans = create_global_rot_for(root_rotation, joints[:, 0])

    results = [root_trans]
    # compute global transformation for ordered set of joint ancestors of
    for i in range(1, ancestors.shape[0]):
        joint = joints[:, i] - joints[:, ancestors[i]]
        joint_glob_rot = create_global_rot_for(rot_mat[:, i], joint)
        res_here = tf.matmul(results[ancestors[i]], joint_glob_rot)
        results.append(res_here)

    results = tf.stack(results, 1)
    new_joints = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    zeros = tf.zeros([batch_size, num_joints, 1, 1])
    rest_pose = tf.concat([joints, zeros], 2)
    init_bone = tf.matmul(results, rest_pose)
    init_bone = tf.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
    rel_joints = results - init_bone

    return new_joints, rel_joints


def batch_orthographic_projection(kp3d, camera):
    """computes reprojected 3d to 2d keypoints
    Args:
        kp3d:   [batch x K x 3]
        camera: [batch x 3]
    Returns:
        kp2d: [batch x K x 2]
    """
    camera = tf.reshape(camera, (-1, 1, 3))
    kp_trans = kp3d[:, :, :2] + camera[:, :, 1:]
    shape = kp_trans.shape

    kp_trans = tf.reshape(kp_trans, (shape[0], -1))
    kp2d = camera[:, :, 0] * kp_trans

    return tf.reshape(kp2d, shape)

def batch_align_by_pelvis(kp3d):
    """Assumes kp3d is [batch x 14 x 3] in LSP order. Then hips are id [2, 3].
       Takes mid point of these points, then subtracts it.
    Args:
        kp3d: [batch x K x 3]
    Returns:
        aligned_kp3d: [batch x K x 3]
    """
    left_id, right_id = 3, 2
    pelvis = (kp3d[:, left_id, :] + kp3d[:, right_id, :]) / 2.
    return kp3d - tf.expand_dims(pelvis, axis=1)


def batch_compute_similarity_transform(real_kp3d, pred_kp3d):
    """Computes a similarity transform (sR, trans) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, trans 3x1 translation, u scale.
        i.e. solves the orthogonal Procrustes problem.
    Args:
        real_kp3d: [batch x K x 3]
        pred_kp3d: [batch x K x 3]
    Returns:
        aligned_kp3d: [batch x K x 3]
    """
    # transpose to [batch x 3 x K]
    real_kp3d = tf.transpose(real_kp3d, perm=[0, 2, 1])
    pred_kp3d = tf.transpose(pred_kp3d, perm=[0, 2, 1])

    # 1. Remove mean.
    mean_real = tf.reduce_mean(real_kp3d, axis=2, keepdims=True)
    mean_pred = tf.reduce_mean(pred_kp3d, axis=2, keepdims=True)

    centered_real = real_kp3d - mean_real
    centered_pred = pred_kp3d - mean_pred

    # 2. Compute variance of centered_real used for scale.
    variance = tf.reduce_sum(centered_pred ** 2, axis=[-2, -1], keepdims=True)

    # 3. The outer product of centered_real and centered_pred.
    K = tf.matmul(centered_pred, centered_real, transpose_b=True)

    # 4. Solution that Maximizes trace(R'K) is R=s*V', where s, V are
    # singular vectors of K.
    with tf.device('/CPU:0'):
        # SVD is terrifyingly slow on GPUs, use cpus for this. Makes it a lot faster.
        s, u, v = tf.linalg.svd(K, full_matrices=True)

        # Construct identity that fixes the orientation of R to get det(R)=1.
        det = tf.sign(tf.linalg.det(tf.matmul(u, v, transpose_b=True)))

    det = tf.expand_dims(tf.expand_dims(det, -1), -1)
    shape = tf.shape(u)
    identity = tf.eye(shape[1], batch_shape=[shape[0]])
    identity = identity * det

    # Construct R.
    R = tf.matmul(v, tf.matmul(identity, u, transpose_b=True))

    # 5. Recover scale.
    trace = tf.linalg.trace(tf.matmul(R, K))
    trace = tf.expand_dims(tf.expand_dims(trace, -1), -1)
    scale = trace / variance

    # 6. Recover translation.
    trans = mean_real - scale * tf.matmul(R, mean_pred)

    # 7. Align
    aligned_kp3d = scale * tf.matmul(R, pred_kp3d) + trans

    return tf.transpose(aligned_kp3d, perm=[0, 2, 1])

def pckh(kp2d, kp2d_pred, threshold = 0.5):
    batch_size = kp2d.shape[0] 
    head_pos = kp2d[:, 13, :2]
    neck_pos = kp2d[:, 12, :2]
    vis = kp2d[:, 12, 2] * kp2d[:, 13, 2]
    head_bone_len = np.sum((head_pos - neck_pos) ** 2, axis=-1) * vis
    head_bone_len = np.tile(np.reshape(head_bone_len, (batch_size, 1)), kp2d_pred.shape[1])

    # Calculate distance to true joint position
    joint_dists = np.sum((kp2d[:, :, :2] - kp2d_pred) ** 2, axis=-1)
    
    # Calculaute the pckh
    
    pckh = np.sum(joint_dists <  head_bone_len * threshold, axis=0)
    num_vis = np.sum(head_bone_len > 0, axis=0)
    return pckh, num_vis

def pcp(kp2d, kp2d_pred, threshold=0.5):
    """Percentage of Correct Parts (PCP)"""
    batch_size = kp2d.shape[0] 
    # ankle->knee (left and right), knee->hip (left and right)
    leg_limbs = [(0, 1), (4,3), (1,2), (5,4)]
    # wrist->elbow (left and right), elbow->shoulder (left and right)
    arm_limbs = [(6, 7), (10, 9), (7, 8), (11, 10)]
    
    pcp_dict = {
        "num_upper_arm_correct" : 0,
        "num_upper_arm_visible" : 0,
        "num_lower_arm_correct" : 0,
        "num_lower_arm_visible" : 0,
        "num_upper_leg_correct" : 0,
        "num_upper_leg_visible" : 0,
        "num_lower_leg_correct" : 0,
        "num_lower_leg_visible" : 0
    }
    
    for i, limbs in enumerate([leg_limbs, arm_limbs]):
        for j, limb in enumerate(limbs):
            limb_len = np.sum((kp2d[:, limb[0]] - kp2d[:, limb[1]]) ** 2, axis=-1)
            # Get pairwise distance between the joints in each limb
            joint_dist_1 = np.sum((kp2d[:, limb[0], :2] - kp2d_pred[:, limb[0]]) ** 2, axis=-1)
            joint_dist_2 = np.sum((kp2d[:, limb[1], :2] - kp2d_pred[:, limb[1]]) ** 2, axis=-1)
            # Only compute on limbs that are fully visible
            vis = kp2d[:, limb[0], 2] * kp2d[:, limb[1], 2]
            num_correct = np.sum((joint_dist_1 < threshold * limb_len * vis) * (joint_dist_2 < threshold * limb_len * vis))
            if i == 0:
                if j < 2:
                    pcp_dict["num_lower_leg_correct"] += num_correct
                    pcp_dict["num_lower_leg_visible"] += np.sum(vis)
                else:
                    pcp_dict["num_upper_leg_correct"] += num_correct
                    pcp_dict["num_upper_leg_visible"] += np.sum(vis)
            else:
                if j < 2:
                    pcp_dict["num_lower_arm_correct"] += num_correct
                    pcp_dict["num_lower_arm_visible"] += np.sum(vis)
                else:
                    pcp_dict["num_upper_arm_correct"] += num_correct
                    pcp_dict["num_upper_arm_visible"] += np.sum(vis)
                    

    return pcp_dict

        

def compute_pcp_avgs(pcp_dict):
    result_dict = {}
    result_dict["upper_arm_avg"] = pcp_dict["num_upper_arm_correct"] / pcp_dict["num_upper_arm_visible"]
    result_dict["lower_arm_avg"] = pcp_dict["num_lower_arm_correct"] / pcp_dict["num_lower_arm_visible"]
    result_dict["upper_leg_avg"] = pcp_dict["num_upper_leg_correct"] / pcp_dict["num_upper_leg_visible"]
    result_dict["lower_leg_avg"] = pcp_dict["num_lower_leg_correct"] / pcp_dict["num_lower_leg_visible"]
    result_dict["total_avg"] = sum([value for key, value in pcp_dict.items() if "correct" in key]) / sum([value for key, value in pcp_dict.items() if "visible" in key])
    return result_dict