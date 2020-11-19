import tensorflow as tf
import numpy as np
from smpl import SMPL
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from models import Generator, Discriminator
from model_util import batch_orthographic_projection
from data_loader import DataLoader


tf.config.run_functions_eagerly(True)

class Trainer:
    def __init__(self, args):
        self._args = args
        # Create the generator model that will predict the SMPL paramaters
        self._generator = Generator(self._args)
        self._discriminator = Discriminator(self._args)
        self._gen_optim = tf.optimizers.Adam(learning_rate=self._args.gen_lr)
        self._disc_optim = tf.optimizers.Adam(learning_rate=self._args.disc_lr)
        self._data_loader = DataLoader(args)
        self._lsp_train_ds, self._lsp_test_ds = self._data_loader.load_lsp_dataset()
        self._smpl_train_ds, self._smpl_test_ds = self._data_loader.load_smpl_dataset()
        self._smpl_model = SMPL(self._args)
        self._create_summary_writer()
        self._train_data = {"steps" : 0}

    def train(self):
        for train_batch in self._lsp_train_ds:
            imgs, kp2d = train_batch
            print("imgs.shape", imgs.shape)
            print("kp2d.shape", kp2d.shape)
            gen_loss = self._train_step(train_batch)
            self._train_data["steps"] += 1
            self._write_loss(gen_loss)
     
    @tf.function
    def _train_step(self, train_batch):
        imgs, kp2d = train_batch
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_out = self._generator(imgs, training=True)
            final_gen_out = gen_out[-1]

            # Get the SMPL parameters
            rotations = []
            shapes = []
            for i in range(self._args.ief_iter):
                _, joints, cur_rotations = self._smpl_model(gen_out[i][2], gen_out[i][1])
                rotations.append(cur_rotations)
                print(tf.shape(cur_rotations))
                print(tf.shape(gen_out[i][2]))
                shapes.append(gen_out[i][2])
            
            # Project the 3D joints to 2D joints
            kp2d_pred = batch_orthographic_projection(joints, final_gen_out[0])

            # Get the MAE loss of the projected 2D joints
            vis = tf.expand_dims(kp2d[:, :, 2], -1)
            kp2d_loss = tf.compat.v1.losses.absolute_difference(kp2d[:, :, :2], kp2d_pred, weights=vis)
            kp2d_loss = kp2d_loss *  self._args.gen_2d_loss_weight
            #print("kp2d_loss: ", kp2d_loss)
            tf.print(kp2d_loss)
            gen_loss = tf.reduce_sum(kp2d_loss)

            fake_disc_input = self.accumulate_fake_disc_input(rotations, shapes)
            fake_disc_output = self._discriminator(fake_disc_input)

            gen_disc_loss = tf.reduce_mean(tf.reduce_sum((fake_disc_output - 1) ** 2, axis=1))
            gen_disc_loss = gen_disc_loss * self._args.disc_loss_weight

            disc_fake_loss = tf.reduce_mean(tf.reduce_sum(fake_disc_output ** 2, axis=1))
            disc_loss = disc_fake_loss * self._args.disc_loss_weight

            # TODO: Apply real input into discriminator
        
        gen_grads = gen_tape.gradient(gen_loss, self._generator.trainable_variables)
        self._gen_optim.apply_gradients(zip(gen_grads, self._generator.trainable_variables))

        disc_grads = disc_tape.gradient(disc_loss, self._discriminator.trainable_variables)
        self._disc_optim.apply_gradients(zip(disc_grads, self._discriminator.trainable_variables))
        
        return kp2d_loss
    
    def accumulate_fake_disc_input(self, rotations, shapes):
        fake_poses, fake_shapes = [], []
        for i in range(len(rotations)):
            fake_poses.append(rotations[i])
            fake_shapes.append(shapes[i])

        # ignore global rotation
        print(len(fake_poses))
        print(len(fake_shapes))
        fake_poses = tf.reshape(tf.convert_to_tensor(fake_poses), [-1, self._args.num_joints+1, 9])[:, 1:, :]
        fake_poses = tf.reshape(fake_poses, [-1, self._args.num_joints * 9])
        fake_shapes = tf.reshape(tf.convert_to_tensor(fake_shapes), [-1, self._args.num_shape_param])
        print(fake_poses.shape)
        print(fake_shapes.shape)
        fake_disc_input = tf.concat([fake_poses, fake_shapes], 1)
        return fake_disc_input
    
    def _create_summary_writer(self):
        gen_log_dir = "logs/fit/gen"
        self._gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
    
    def _write_loss(self, gen_loss):
        with self._gen_summary_writer.as_default():
            tf.summary.scalar("Generator Loss", gen_loss, step=self._train_data["steps"])