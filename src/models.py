import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

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
        self._dense_1 = Dense(1024, activation="relu")
        self._dense_2 = Dense(1024, activation="relu")
        self._dropout_1 = Dropout(0.5)
        self._dropout_2 = Dropout(0.5)
        small_xavier = tf.initializers.VarianceScaling(
            .01,
            mode='fan_avg',
            distribution='uniform')
        self._reg_out = Dense(85, kernel_initializer=small_xavier)

        # Load the initial mean theta params and create
        # a trainable variable for it
        self._mean_theta = tf.Variable(
            load_mean_theta(self._args),
            name='mean_theta',
            trainable=True)

    
    def call(self, x):
        # Get the batch size
        batch_size = x.shape[0]

        # Encode the image
        img_features = self._encoder(x)

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
            state = self._dropout_1(state)
            state = self._dense_2(state)
            state = self._dropout_2(state)
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
        return outputs

            
            

            
            

# num_cam_param
# smpl_mean_theta_filemodels 
# encoder_shape
# num_pose_param