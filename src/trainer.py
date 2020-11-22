import tensorflow as tf
import numpy as np
from smpl import SMPL
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import gridspec
import cv2, json, os, pickle, trimesh
#from smpl_renderer import SMPLRenderer


physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from models import Generator, Discriminator
from model_util import batch_orthographic_projection, batch_rodrigues, pckh, pcp, compute_pcp_avgs
from data_loader import DataLoader
from render import Render


tf.config.run_functions_eagerly(True)

colors = {
    'pink': [197, 27, 125],
    'light_pink': [233, 163, 201],
    'green': [77, 146, 33],
    'light_green': [161, 215, 106],
    'orange': [200, 90, 39],
    'light_orange': [252, 141, 89],
    'blue': [69, 117, 180],
    'light_blue': [145, 191, 219],
    'red': [215, 48, 39],
    'purple': [118, 42, 131],
    'white': [255, 255, 255],
}

joint_colors = [
    'light_pink', 'light_pink', 'light_pink', 'pink',
    'green', 'light_green', 'light_green', 'light_green',
    'light_blue', 'light_blue', 'light_blue',
    'light_orange', 'light_orange', 'light_orange',
    'purple', 'purple',
    'red',
    'blue', 'blue',
    'orange', 'orange',
]

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
       

        # Setup saving and restoring model
        self._ckpt = tf.train.Checkpoint(
            generator=self._generator,
            discriminator=self._discriminator,
            gen_optim=self._gen_optim,
            disc_optim=self._disc_optim)
        self._ckpt_manager = tf.train.CheckpointManager(self._ckpt , directory=self._args.model_dir, max_to_keep=3)
        if self._args.load_model:
            self._ckpt.restore(self._ckpt_manager.latest_checkpoint)
        self._load_train_data()

    def train(self):
        train_ds = tf.data.Dataset.zip((self._smpl_train_ds, self._lsp_train_ds))
        b_i = 0
        #self._ckpt_manager.save()
        for i in range(self._args.epochs):
            for train_batch in train_ds:
                # imgs, kp2d = train_batch
                # print("imgs.shape", imgs.shape)
                # print("kp2d.shape", kp2d.shape)
                gen_loss, disc_loss = self._train_step(train_batch)
                self._train_data["steps"] += 1
                self._write_loss(gen_loss, disc_loss)
                b_i += 1
                #if b_i % 1000 == 0:
                #    print(b_i)
            self._ckpt_manager.save()
            self._save_train_data()
        
        if self._args.test:
            num_correct = np.zeros(14)
            num_vis = np.zeros(14)
            pcp_dict = {}
            test_len = 0
            for test_batch in self._lsp_test_ds:
                test_len += test_batch[0].shape[0]
                pckh_tuple, pcp_dict_batch = self._test_step(test_batch)
                for key, value in pcp_dict_batch.items():
                    if key not in pcp_dict:
                         pcp_dict[key] = value
                    else:
                         pcp_dict[key] += value
                

                cur_num_correct, cur_num_vis = pckh_tuple
                num_vis += cur_num_vis
                num_correct += cur_num_correct
            print("PCP DICT:", pcp_dict)
            print("PCP AVG:", compute_pcp_avgs(pcp_dict))

            print("PCK: ", num_correct / num_vis)
            print("PCK AVG:", np.sum(num_correct) / np.sum(num_vis))
        self.draw()



    @tf.function
    def _train_step(self, train_batch):
        smpl_params, lsp_data = train_batch 
        imgs, kp2d = lsp_data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_out = self._generator(imgs, training=True)
            final_gen_out = gen_out[-1]

            # Get the SMPL parameters
            rotations = []
            shapes = []
            for i in range(self._args.ief_iter):
                _, joints, cur_rotations = self._smpl_model(gen_out[i][2], gen_out[i][1])
                rotations.append(cur_rotations)
                # print(tf.shape(cur_rotations))
                # print(tf.shape(gen_out[i][2]))
                shapes.append(gen_out[i][2])
            
            # Project the 3D joints to 2D joints
            kp2d_pred = batch_orthographic_projection(joints, final_gen_out[0])

            # Get the MAE loss of the projected 2D joints
            vis = tf.expand_dims(kp2d[:, :, 2], -1)
            kp2d_loss = tf.compat.v1.losses.absolute_difference(kp2d[:, :, :2], kp2d_pred, weights=vis)
            kp2d_loss = kp2d_loss *  self._args.gen_2d_loss_weight
            print("kp2d_loss: ", kp2d_loss)
            tf.print(kp2d_loss)
            

            # Run fake input into discriminator
            fake_disc_input = self.accumulate_fake_disc_input(rotations, shapes)
            fake_disc_output = self._discriminator(fake_disc_input)

            # Apply real input into discriminator
            real_disc_input = self.accumulate_real_disc_input(smpl_params)
            real_disc_output = self._discriminator(real_disc_input)

            # Compute discriminator loss
            disc_fake_loss = tf.reduce_mean(tf.reduce_sum(fake_disc_output ** 2, axis=1))
            disc_real_loss = tf.reduce_mean(tf.reduce_sum((real_disc_output - 1) ** 2, axis=1))
            disc_loss = tf.reduce_sum([disc_real_loss, disc_fake_loss]) * self._args.disc_loss_weight
            print("disc_fake_loss", disc_fake_loss)
            print("disc_real_loss", disc_real_loss)

            # Compute generator loss
            gen_disc_loss = tf.reduce_mean(tf.reduce_sum((fake_disc_output - 1) ** 2, axis=1))
            gen_disc_loss = gen_disc_loss * self._args.disc_loss_weight
            print("gen_disc_loss", gen_disc_loss)
            gen_loss = tf.reduce_sum([kp2d_loss, gen_disc_loss])
        
        gen_grads = gen_tape.gradient(gen_loss, self._generator.trainable_variables)
        self._gen_optim.apply_gradients(zip(gen_grads, self._generator.trainable_variables))

        disc_grads = disc_tape.gradient(disc_loss, self._discriminator.trainable_variables)
        self._disc_optim.apply_gradients(zip(disc_grads, self._discriminator.trainable_variables))
        
        return gen_loss, disc_loss
    
    def accumulate_fake_disc_input(self, rotations, shapes):
        fake_poses, fake_shapes = [], []
        for i in range(len(rotations)):
            fake_poses.append(rotations[i])
            fake_shapes.append(shapes[i])

        # ignore global rotation
        # print(len(fake_poses))
        # print(len(fake_shapes))
        fake_poses = tf.reshape(tf.convert_to_tensor(fake_poses), [-1, self._args.num_joints+1, 9])[:, 1:, :]
        fake_poses = tf.reshape(fake_poses, [-1, self._args.num_joints * 9])
        fake_shapes = tf.reshape(tf.convert_to_tensor(fake_shapes), [-1, self._args.num_shape_param])
        # print(fake_poses.shape)
        # print(fake_shapes.shape)
        fake_disc_input = tf.concat([fake_poses, fake_shapes], 1)
        return fake_disc_input

    
    def accumulate_real_disc_input(self, smpl_params):
        smpl_params = tf.cast(smpl_params, tf.float32)
        real_poses = smpl_params[:, :self._args.num_pose_param]
        real_poses = batch_rodrigues(real_poses)
        real_poses = real_poses[:, 1:, :]
        real_poses = tf.reshape(real_poses, [-1, self._args.num_joints * 9])
        real_shapes = smpl_params[:, -self._args.num_shape_param:]

        real_disc_input = tf.concat([real_poses, real_shapes], 1)
        return real_disc_input
    
    def _create_summary_writer(self):
        gen_log_dir = "logs/fit/gen"
        disc_log_dir = "logs/fit/disc"
        self._gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
        self._disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)
    
    def _write_loss(self, gen_loss, disc_loss):
        with self._gen_summary_writer.as_default():
            tf.summary.scalar("Generator Loss", gen_loss, step=self._train_data["steps"])
        with self._disc_summary_writer.as_default():
            tf.summary.scalar("Discriminator Loss", disc_loss, step=self._train_data["steps"])
                
    def _load_train_data(self):
        if self._args.load_model and os.path.exists(self._args.train_json):
            with open(self._args.train_json) as f:
                 self._train_data = json.load(f)
        else:
            self._train_data = {"steps" : 0}
    
    def _save_train_data(self):
        with open(self._args.train_json, "w") as f:
            json.dump(self._train_data, f)
    

    def draw(self):
        fig, ax = plt.subplots(4, 4)
        for test_batch in self._lsp_test_ds.take(1):
            imgs, kp2d = test_batch
            gen_out = self._generator(imgs, training=False)
            final_gen_out = gen_out[-1]

            # Get the SMPL parameters
            rotations = []
            shapes = []
            for i in range(self._args.ief_iter):
                vertices, joints, cur_rotations = self._smpl_model(gen_out[i][2], gen_out[i][1])
                rotations.append(cur_rotations)
                shapes.append(gen_out[i][2])
            
            # Project the 3D joints to 2D joints
            kp2d_pred = batch_orthographic_projection(joints, final_gen_out[0])
            kp2d_pred = kp2d_pred * self._args.img_size + self._args.img_size
            parents = self._get_parent_ids()
            radius = 2 #np.meam
            kp2d_pred = np.array(kp2d_pred).astype(np.int32)
            imgs = np.array(imgs)
            imgs = ((imgs + 1) / 2 * 255).astype(np.uint8)
            vertices = np.array(vertices)
            for i in range(imgs.shape[0]):
                kp2d_img = imgs[i].copy()
                for j, pos in enumerate(kp2d_pred[i]):
                    parent_pos = kp2d_pred[i][parents[j]]
                    #print("joint_colors[j]", joint_colors[j])
                    #print("colors[joint_colors[j]]", colors[joint_colors[j]])
                    cv2.circle(kp2d_img, tuple(pos), radius, colors[joint_colors[j]], -1)
                    cv2.line(kp2d_img, tuple(pos), tuple(parent_pos), colors[joint_colors[j]])

                renderer = Render(self._args)
                #smpl_renderer = SMPLRenderer(self._args)
                img_overlay = renderer(vertices[i], img=imgs[i])
                #img_overlay = smpl_renderer(vertices[i], img=imgs[i])
                img_mesh = renderer(vertices[i], img_size=imgs[i].shape[:2])
                img_mesh_rot1 = renderer.rotated(vertices[i], 60, img_size=imgs[i].shape[:2])
                img_mesh_rot2 = renderer.rotated(vertices[i], -60, img_size=imgs[i].shape[:2])
                gs = gridspec.GridSpec(2, 3)
                gs.update(wspace=0.25, hspace=0.25)
                plt.axis('off')
                plt.clf()

                def put_image_on_axis(_img, i, title):
                    ax = plt.subplot(gs[i])
                    ax.imshow(_img)
                    ax.set_title(title)
                    ax.axis('off')

                put_image_on_axis(imgs[i], 0, 'Input image')
                put_image_on_axis(kp2d_img, 1, '2D Joint locations')
                put_image_on_axis(img_overlay, 2, '3D Mesh Overlay')
                put_image_on_axis(img_mesh, 3, '3D Mesh')
                put_image_on_axis(img_mesh_rot1, 4, 'rotated 60 degree')
                put_image_on_axis(img_mesh_rot2, 5, 'rotated -60 degree')
                plt.show()
                # ax[i//4][i%4].imshow(imgs[i])
                # ax[i//4][i%4].axis("off")

    def _get_parent_ids(self):
        parents = np.array([1, 2, 8, 9, 3, 4, 7, 8, -1, -1, 9, 10, 13, -1])
        return parents


    @tf.function
    def _test_step(self, test_batch):
        imgs, kp2d = test_batch
        gen_out = self._generator(imgs, training=True)
        final_gen_out = gen_out[-1]
        
        # Get the SMPL parameters
        rotations = []
        shapes = []
        for i in range(self._args.ief_iter):
            vertices, joints, cur_rotations = self._smpl_model(gen_out[i][2], gen_out[i][1])
            rotations.append(cur_rotations)
            shapes.append(gen_out[i][2])
        
        # Project the 3D joints to 2D joints
        kp2d_pred = batch_orthographic_projection(joints, final_gen_out[0])
        kp2d, kp2d_pred = np.array(kp2d), np.array(kp2d_pred)
    
        return pckh(kp2d, kp2d_pred), pcp(kp2d, kp2d_pred)
