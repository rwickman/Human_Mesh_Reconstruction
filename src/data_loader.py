import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import os, pickle

LSP_DATASET_SIZE = 10000

# TODO: Add data augmentations (e.g., random jitter, crop, flip, ect.)
SHUFFLE_BUFFER_SIZE = 60000
class DataLoader:
    def __init__(self, args):
        self._args = args
        #self._smpl_datafiles = self._get_smpl_datafiles()

    def load_lsp_dataset(self):
        
        # Load the original LSP dataset that contains 2000 examples
        org_lsp_joints = loadmat(os.path.join(self._args.org_lsp_dir, "joints.mat"))["joints"]
        org_lsp_joints = tf.transpose(org_lsp_joints, perm=[2, 1,0])
        org_lsp_train_joints_ds = tf.data.Dataset.from_tensor_slices(org_lsp_joints[:1600])
        org_lsp_test_joints_ds = tf.data.Dataset.from_tensor_slices(org_lsp_joints[1600:])
        org_lsp_train_imgs_ds = tf.data.Dataset.list_files(os.path.join(self._args.org_lsp_dir, "images/train/*"), shuffle=False)
        org_lsp_test_imgs_ds = tf.data.Dataset.list_files(os.path.join(self._args.org_lsp_dir, "images/test/*"), shuffle=False)

        org_lsp_train_ds = tf.data.Dataset.zip((org_lsp_train_imgs_ds, org_lsp_train_joints_ds))
        #org_lsp_train_ds = org_lsp_train_ds.map(self._preprocess)
        #org_lsp_train_ds = org_lsp_train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(self._args.batch_size)

        org_lsp_test_ds = tf.data.Dataset.zip((org_lsp_test_imgs_ds, org_lsp_test_joints_ds))
        #org_lsp_test_ds = org_lsp_test_ds.map(self._preprocess)
        #org_lsp_test_ds = org_lsp_test_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(self._args.batch_size)    

         # Load the joints (14, 3, 10000)
        lsp_joints = loadmat(os.path.join(self._args.lsp_dir, "joints.mat"))["joints"]
        
        # (10000, 14, 3)
        lsp_joints = tf.transpose(lsp_joints, perm=[2, 0, 1])
        lsp_train_joints_ds = tf.data.Dataset.from_tensor_slices(lsp_joints[:8000])
        lsp_test_joints_ds = tf.data.Dataset.from_tensor_slices(lsp_joints[8000:])
        
        # Load the training images
        lsp_train_imgs_ds = tf.data.Dataset.list_files(os.path.join(self._args.lsp_dir, "images/train/*"), shuffle=False)
        
        #lsp_train_imgs_ds = lsp_train_imgs_ds.map(self._preprocess_images)
        
        # Load the testing images
        lsp_test_imgs_ds = tf.data.Dataset.list_files(os.path.join(self._args.lsp_dir, "images/test/*"), shuffle=False)
        #lsp_test_imgs_ds = tf.data.Dataset.list_files(os.path.join("../datasets/lspet_dataset/images_real/test/*"), shuffle=False)
        lsp_train_ds = tf.data.Dataset.zip((lsp_train_imgs_ds, lsp_train_joints_ds))
        
        lsp_train_ds = lsp_train_ds.concatenate(org_lsp_train_ds)
        
        lsp_train_ds = lsp_train_ds.map(self._preprocess)
        lsp_train_ds = lsp_train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(self._args.batch_size)
        
        lsp_test_ds = tf.data.Dataset.zip((lsp_test_imgs_ds, lsp_test_joints_ds))
        lsp_test_ds = lsp_test_ds.concatenate(org_lsp_test_ds)
        lsp_test_ds = lsp_test_ds.map(self._preprocess)
        lsp_test_ds = lsp_test_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(self._args.batch_size)        
        return lsp_train_ds, lsp_test_ds

    def load_smpl_dataset(self):
        poses, shapes = self._get_smpl_data()
        smpl_data = np.concatenate((poses,shapes), axis=-1)
        train_size = int(smpl_data.shape[0] * self._args.train_split) 
        smpl_train_ds = tf.data.Dataset.from_tensor_slices(smpl_data[:train_size])
        smpl_test_ds = tf.data.Dataset.from_tensor_slices(smpl_data[train_size:])

        smpl_train_ds = smpl_train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(self._args.batch_size)
        smpl_test_ds = smpl_test_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(self._args.batch_size)
        return smpl_train_ds, smpl_test_ds
    
    def _get_smpl_data(self):
        poses = []
        shapes = []
        
        cur_loaded = 0
        # Recursively go through all the files
        for root, dirs, files in os.walk(self._args.smpl_params):
            cur_loaded += 1
            if cur_loaded > self._args.max_smpl_load:
                break
            # For each directory
            for datafile in files:
                if ".pkl" == datafile[-4:]:
                    data_path = os.path.join(root, datafile)
                    with open(data_path, "rb") as f:
                        cur_data = pickle.load(f, encoding="latin-1")
                        cur_poses = cur_data["poses"]
                        cur_shapes = np.tile(cur_data["betas"], (cur_poses.shape[0], 1))
                        #print(cur_poses.shape)
                        for i in range(cur_poses.shape[0]):
                            poses.append(cur_poses[i])
                            shapes.append(cur_shapes[i])        
        
        return np.array(poses), np.array(shapes)

    def _preprocess(self, img_path, kp2d):
        kp2d = tf.cast(kp2d, tf.float32)
        img_tensor = tf.image.decode_image(tf.io.read_file(img_path), expand_animations=False)
        img_tensor = tf.image.convert_image_dtype(img_tensor, dtype=tf.float32)
        orig_image_size = tf.cast(tf.shape(img_tensor)[:2], tf.float32)

        # #encoder_img_size = np.array()
        img_tensor = tf.image.resize(
            img_tensor,
            [self._args.img_size, self._args.img_size],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        img_tensor = tf.cast(img_tensor, tf.float32)
        
        # # normalize image from [ to [-1, 1]
        #img_tensor = (img_tensor - 127.5) / 127.5

        image_size = tf.cast(tf.shape(img_tensor)[:2], tf.float32)
        
        #kp2d_joints = (kp2d[:, :2] / orig_image_size) * image_size 
        kp2d_joints = (kp2d[:, :2] / orig_image_size[::-1]) * image_size[::-1] 
        kp2d_joints = tf.clip_by_value(kp2d_joints, 0, self._args.img_size)
        kp2d_joints =  (kp2d_joints - self._args.img_size) / self._args.img_size
        
        vis = tf.expand_dims(kp2d[:, 2], axis=-1)
        kp2d_joints = kp2d_joints * vis
        #tf.print(kp2d_joints)
        #tf.print(vis)

        img_tensor = tf.subtract(img_tensor, 0.5)
        img_tensor = tf.multiply(img_tensor, 2.0)
        
        kp2d = tf.concat([kp2d_joints, vis], axis=-1)
        return img_tensor, kp2d


    #def load_mpi_3d(self):
