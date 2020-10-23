import tensorflow as tf
import numpy as np
import pickle

from model_util import batch_rodrigues, batch_global_rigid_transformation

def convert_to_variable(param, name):
    return tf.Variable(
        tf.convert_to_tensor(
            param,
            dtype=tf.float32),
        name=name,
        trainable=False)


class SMPL(tf.keras.layers.Layer):
    def __init__(self, args):
        self._args = args
        
        # Load the SMPL model parameters
        with open(args.smpl_model, "rb") as f:
            smpl_model = pickle.load(f)
        
        # (6890, 3)_
        self._template_mesh = convert_to_variable(smpl_model["v_template"], "template_mesh")

        # (6980, 3, 10)
        self._shape_disp = convert_to_variable(smpl_model["shapedirs"], "shape_displacements")
        self._num_betas = self._shape_disp.shape[-1]
        self._shape_disp = tf.transpose(tf.reshape(self._shape_disp, [-1, self._num_betas]))
        
        
        self._smpl_joint_regressor = convert_to_variable(smpl_model["J_regressor"].T, name="smpl_joint_regressor")

        # (207, 6890, 3)
        self._pose = convert_to_variable(smpl_model['posedirs'], name='pose')
        self._pose = tf.transpose(tf.reshape(self._pose, [-1, self._pose.shape[-1]]))
        
        
        # Load the blend skinning weights (6890, 24)
        self._lbs_weights = convert_to_variable(smpl_model["weights"], name="LBS_weights")

        self._face_mesh = tf.convert_to_tensor(smpl_model["f"], dtype=tf.float32)

        # Used to get the joints
        self._joint_regressor = convert_to_variable(smpl_model["cocoplus_regressor"].T, "joint_regressor")
        
        if self._args.joint_type == "lsp":
            self._joint_regressor = self._joint_regressor[:, :14]

        self._ancestors = smpl_model["kintree_table"][0].astype(np.int32)
        self._identity = tf.eye(3)
        self._joint_transformed = None

    def __call__(self, betas, thetas, num_joints=24):
        """
        Args:
            beta: the betas that define the shape
            thetas: the thetas that define the pose 
        """
        batch_size = betas.shape[0]
        _reshape = (batch_size, self._template_mesh.shape[0], self._template_mesh.shape[1])
        
        # Linearly combine blend shapes and add to neutral pose template
        shape_comb = tf.reshape(tf.matmul(betas, self._shape_disp), _reshape)
        
        shaped_mesh = self._template_mesh + shape_comb

        # Predict the joint locations based on shaped mesh
        joints_x = tf.matmul(shaped_mesh[:, :, 0], self._smpl_joint_regressor)
        joints_y = tf.matmul(shaped_mesh[:, :, 1], self._smpl_joint_regressor)
        joints_z = tf.matmul(shaped_mesh[:, :, 2], self._smpl_joint_regressor)
        joints = tf.stack([joints_x, joints_y, joints_z], axis=2)

        # Add pose blend shapes
        rotations = tf.reshape(
            batch_rodrigues(thetas),
            [batch_size, num_joints, 3, 3])
        pose_features = tf.reshape(rotations[:, 1:, :, :] - self._identity, [batch_size, -1])
        # (N, 6890, 3)
        posed_mesh =  tf.reshape(tf.matmul(pose_features, self._pose), _reshape) + shaped_mesh

        # Get the global joint location
        self.joint_transformed, rel_joints = batch_global_rigid_transformation(rotations, joints, self._ancestors)
        
        # Skin the mesh
        weights = tf.reshape(tf.tile(self._lbs_weights, [batch_size, 1]), [batch_size, -1, 24])
        
        rel_joints = tf.reshape(rel_joints, [batch_size, num_joints, 16])
        weighted_joints = tf.reshape(tf.matmul(weights, rel_joints), [batch_size, -1, 4, 4])
        
        ones = tf.ones([batch_size, posed_mesh.shape[1], 1])
        v_posed_homo = tf.expand_dims(tf.concat([posed_mesh, ones], 2), -1)
        v_posed_homo = tf.matmul(weighted_joints, v_posed_homo)

        vertices = v_posed_homo[:, :, :3, 0]

        # Compute the joints
        joints_x = tf.matmul(vertices[:, :, 0], self._joint_regressor)
        joints_y = tf.matmul(vertices[:, :, 1], self._joint_regressor)
        joints_z = tf.matmul(vertices[:, :, 2], self._joint_regressor)
        joints = tf.stack([joints_x, joints_y, joints_z], axis=2)

        return vertices, joints, rotations