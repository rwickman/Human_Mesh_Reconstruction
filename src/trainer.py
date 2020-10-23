import tensorflow as tf
import numpy as np
from smpl import SMPL

from models import Generator
from model_util import batch_orthographic_projection
from data_loader import DataLoader


tf.config.run_functions_eagerly(True)

class Trainer:
    def __init__(self, args):
        self._args = args
        # Create the generator model that will predict the SMPL paramaters
        self._generator = Generator(self._args)
        self._gen_optim = tf.optimizers.Adam(learning_rate=self._args.gen_lr)
        self._data_loader = DataLoader(args)
        self._lsp_train_ds, self._lsp_test_ds = self._data_loader.load_lsp_dataset()
        self._smpl_model = SMPL(self._args)

    def train(self):
        for train_batch in self._lsp_train_ds:
            self._train_step(train_batch)
        #foo_input = np.ones((1, 224, 224, 3))
        # global_max = 0.0
        # global_max_img = 0.0
        # global_min = 0.0
        # global_min_img = 0.0

        # for f in self._lsp_train_ds.take(150):

        #     global_min = np.min([np.min(f[1].numpy()), global_min])
            
        #     global_max = np.max([np.max(f[1].numpy()), global_max])

        #     global_max_img = np.max([np.max(f[0].numpy()), global_max_img])
        #     global_min_img = np.min([np.min(f[0].numpy()), global_min_img])
        # print("GLOBAL MIN: ", global_min)
        # print("GLOBAL MAX: ", global_max)
        # print("global_max_img: ", global_max_img)
        # print("global_min_img: ", global_min_img)

        # for f in self._lsp_train_ds.take(1):
        #     img = f[0]
        #     kp2d_true = f[1]

        # gen_out = self._generator(img)
        
        # for output in gen_out:
        #     thetas = output[1]
        #     betas = output[2]
        #     _, joints, rotations = self._smpl_model(betas, thetas)
        #     kp2d_pred = batch_orthographic_projection(joints, output[0])
        #     #print(kp2d_pred)
        #     print(kp2d_pred[0])
        #     print(kp2d_true[0])
     
    @tf.function
    def _train_step(self, train_batch):
        imgs, kp2d = train_batch
        with tf.GradientTape() as gen_tape:
            gen_out = self._generator(imgs, training=True)[-1]
            # pose
            thetas = gen_out[1]
            # shape
            betas = gen_out[2]
            # Get the SMPL parameters
            _, joints, rotations = self._smpl_model(betas, thetas)

            # Project the 3D joints to 2D joints
            kp2d_pred = batch_orthographic_projection(joints, gen_out[0])

            # Get the MAE loss of the projected 2D joints
            vis = tf.expand_dims(kp2d[:, :, 2], -1)
            kp2d_loss = tf.compat.v1.losses.absolute_difference(kp2d[:, :, :2], kp2d_pred, weights=vis)
            print("kp2d_loss: ", kp2d_loss)
            tf.print(kp2d_loss)
            gen_loss = tf.reduce_sum(kp2d_loss)
        
        gen_grads = gen_tape.gradient(gen_loss, self._generator.trainable_variables)
        print(gen_grads)
        self._gen_optim.apply_gradients(zip(gen_grads, self._generator.trainable_variables))