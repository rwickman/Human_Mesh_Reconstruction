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



# def accumulate_fake_disc_input(generator_outputs):
#     fake_poses, fake_shapes = [], []
#     for output in generator_outputs:
#         fake_poses.append(output[3])
#         fake_shapes.append(output[4])
#     # ignore global rotation
#     fake_poses = tf.reshape(tf.convert_to_tensor(fake_poses), [-1, self.config.NUM_JOINTS_GLOBAL, 9])[:, 1:, :]
#     fake_poses = tf.reshape(fake_poses, [-1, self.config.NUM_JOINTS * 9])
#     fake_shapes = tf.reshape(tf.convert_to_tensor(fake_shapes), [-1, self.config.NUM_SHAPE_PARAMS])

#     fake_disc_input = tf.concat([fake_poses, fake_shapes], 1)
# return fake_disc_input