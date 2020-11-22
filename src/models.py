import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  Model
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.regularizers import l2

from model_util import load_mean_theta


class Generator(Model):
    def __init__(self, args, encoder_shape=(224, 224, 3)):
        super().__init__(name="Generator")
        self._args =  args
        
        # Create the ResNet50V2 Encoder
        self._encoder = ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=encoder_shape,
            pooling="avg")

        # Build the Regressor that will perform IEF
        self._dense_1 = layers.Dense(1024, activation="relu")
        self._dense_2 = layers.Dense(1024, activation="relu")
        self._dropout_1 = layers.Dropout(0.5)
        self._dropout_2 = layers.Dropout(0.5)
        small_xavier = tf.initializers.VarianceScaling(
            .01,
            mode='fan_avg',
            distribution='uniform')
        self._reg_out = layers.Dense(85, kernel_initializer=small_xavier)

        # Load the initial mean theta params and create
        # a trainable variable for it
        self._mean_theta = tf.Variable(
            load_mean_theta(self._args),
            name='mean_theta',
            trainable=True)

    
    def call(self, x, training):
        # Get the batch size
        batch_size = x.shape[0]

        # Encode the image
        img_features = self._encoder(x, training=training)

        # Create mean_theta for every batchmodels 
        cur_theta = tf.tile(self._mean_theta, [batch_size, 1])
        
        # Use the image encoding and currect prediction
        # to iteratively improve theta.
        # This is the IEF loop as described in the paper
        # Store intermediate thetas to support gradient back-propagation
        thetas = tf.TensorArray(tf.float32, self._args.ief_iter)
        for i in range(self._args.ief_iter):
            state = tf.concat([img_features, cur_theta], axis=1)
            
            # Run through regressor to get theta update
            state = self._dense_1(state)
            state = self._dropout_1(state, training=training)
            state = self._dense_2(state)
            state = self._dropout_2(state, training=training)
            theta_residuals = self._reg_out(state)
            
            # Update the current theta estimates by the predicted residual estimate
            # (i.e., translate in the direction of estimated true theta)
            cur_theta = cur_theta + theta_residuals
            thetas.write(i, cur_theta)
        
        return self._split_thetas(thetas.stack())

    def _split_thetas(self, thetas):
        """Split every theta into its individual components."""
        outputs = []
        for i in range(self._args.ief_iter):
            cams = thetas[i, :, :self._args.num_cam_param]
            poses = thetas[i, :, self._args.num_cam_param : self._args.num_cam_param+self._args.num_pose_param]
            shapes = thetas[i, :, self._args.num_cam_param+self._args.num_pose_param : ]
            outputs.append(tf.tuple([cams, poses, shapes]))
        #thetas.mark_used()
        return outputs

class Discriminator(Model):
    def __init__(self, args, encoder_shape=(224, 224, 3)):
        super().__init__(name="Discriminator")
        self._args = args
        
        # Create the common embedding layers for the rotations
        self._conv1d_emb_1 = layers.Conv1D(filters=32, kernel_size=1, activation="relu", name="common_emb_1")
        self._conv1d_emb_2 = layers.Conv1D(filters=32, kernel_size=1, activation="relu", name="common_emb_2")

        # Create the joint discriminators
        disc_l2_reg = l2(self._args.disc_weight_decay)
        self._joint_discs = []
        for i in range(self._args.num_joints):
            self._joint_discs.append(
                layers.Dense(
                    1,
                    kernel_regularizer=disc_l2_reg,
                    name="joint_{}_disc".format(i)))

        # Create full pose discriminator
        self._flatten = layers.Flatten()
        self._pose_dense_1 = layers.Dense(1024, kernel_regularizer=disc_l2_reg, activation="relu", name="pose_dense_1")
        self._pose_dense_2 = layers.Dense(1024, kernel_regularizer=disc_l2_reg, activation="relu", name="pose_dense_2")
        self._pose_disc_out = layers.Dense(1, kernel_regularizer=disc_l2_reg, name="pose_disc_out")

        # Create shape discriminator
        self._shape_dense_1 = layers.Dense(10, kernel_regularizer=disc_l2_reg, activation="relu", name="shape_dense_1") 
        self._shape_dense_2 = layers.Dense(5, kernel_regularizer=disc_l2_reg, activation="relu", name="shape_dense_2")
        self._shape_disc_out = layers.Dense(1, kernel_regularizer=disc_l2_reg, name="shape_disc_out")

    def __call__(self, x):
        print("x.shape:", x.shape)
        poses = x[:, :self._args.num_joints * 9]
        shapes = x[:, -self._args.num_shape_param:]
        poses = tf.reshape(poses, [poses.shape[0], self._args.num_joints, 9])

        # Embed rotation matrices
        pose_embs = self._conv1d_emb_1(poses)
        pose_embs = self._conv1d_emb_2(pose_embs)
        #print("pose_embs.shape", pose_embs.shape)
        
        
        # Call each joint embedding on individual distriminator
        joint_disc_outputs = []
        for i in range(self._args.num_joints):
            joint_disc_outputs.append(self._joint_discs[i](pose_embs[:, i, :]))
        joint_disc_outputs = tf.squeeze(tf.stack(joint_disc_outputs, 1))

        # Run full pose discriminator
        pose_embs = self._flatten(pose_embs)
        pose_embs = self._pose_dense_1(pose_embs)
        pose_embs = self._pose_dense_2(pose_embs)
        pose_disc_output = self._pose_disc_out(pose_embs)

        # Run shape discriminator
        shapes = self._shape_dense_1(shapes)
        shapes = self._shape_dense_2(shapes)
        shape_disc_output = self._shape_disc_out(shapes)
        # print("joint_disc_outputs.shape", joint_disc_outputs.shape)
        # print("pose_disc_output.shape", pose_disc_output.shape)
        # print("shape_disc_output.shape", shape_disc_output.shape)

        return tf.concat((joint_disc_outputs, pose_disc_output, shape_disc_output), 1)
        
        
            

# num_cam_param
# smpl_mean_theta_filemodels 
# encoder_shape
# num_pose_param