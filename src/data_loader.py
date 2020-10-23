import tensorflow as tf
import scipy.io as sio
import numpy as np
import os

# TODO: Add data augmentations (e.g., random jitter, crop, flip, ect.)
SHUFFLE_BUFFER_SIZE = 60000
class DataLoader:
    def __init__(self, args):
        self._args = args

    def load_lsp_dataset(self):
        # Load the joints (14, 3, 10000)
        lsp_joints = sio.loadmat(os.path.join(self._args.lsp_dir, "joints.mat"))["joints"]
        # (10000, 14, 3)
        lsp_joints = tf.transpose(lsp_joints, perm=[2, 0, 1])
        lsp_train_joints_ds = tf.data.Dataset.from_tensor_slices(lsp_joints[:8000])
        lsp_test_joints_ds = tf.data.Dataset.from_tensor_slices(lsp_joints[8000:])
        
        # Load the training images
        lsp_train_imgs_ds = tf.data.Dataset.list_files(os.path.join(self._args.lsp_dir, "images/train/*"), shuffle=False)
        #lsp_train_imgs_ds = lsp_train_imgs_ds.map(self._preprocess_images)
        
        # Load the testing images
        lsp_test_imgs_ds = tf.data.Dataset.list_files(os.path.join(self._args.lsp_dir, "images/test/*"), shuffle=False)
        #lsp_test_imgs_ds = lsp_test_imgs_ds.map(self._preprocess_images)
        
        lsp_train_ds = tf.data.Dataset.zip((lsp_train_imgs_ds, lsp_train_joints_ds))
        lsp_train_ds = lsp_train_ds.map(self._preprocess)
        lsp_train_ds = lsp_train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(self._args.batch_size)
        

        lsp_test_ds = tf.data.Dataset.zip((lsp_test_imgs_ds, lsp_test_joints_ds))
        #lsp_test_ds = lsp_test_ds.map(self._preprocess)
        lsp_test_ds = lsp_test_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(self._args.batch_size)        
        return lsp_train_ds, lsp_test_ds


    def _preprocess(self, img_path, kp2d):
        kp2d = tf.cast(kp2d, tf.float32)
        img_tensor = tf.image.decode_image(tf.io.read_file(img_path), expand_animations=False)
        orig_image_size = tf.cast(tf.shape(img_tensor)[:2], tf.float32)

        # #encoder_img_size = np.array()
        img_tensor = tf.image.resize(
            img_tensor,
            [self._args.img_size, self._args.img_size],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        img_tensor = tf.cast(img_tensor, tf.float32)
        
        # # normalize image from [ to [-1, 1]
        img_tensor = (img_tensor - 127.5) / 127.5

        image_size = tf.cast(tf.shape(img_tensor)[:2], tf.float32)
        
        #kp2d_joints = (kp2d[:, :2] / orig_image_size) * image_size 
        kp2d_joints = (kp2d[:, :2] / orig_image_size[::-1]) * image_size[::-1] 
        kp2d_joints = tf.clip_by_value(kp2d_joints, 0, self._args.img_size)
        kp2d_joints =  (kp2d_joints - self._args.img_size) / self._args.img_size
        
        vis = tf.expand_dims(kp2d[:, 2], axis=-1)
        kp2d_joints = kp2d_joints * vis
        #tf.print(kp2d_joints)
        #tf.print(vis)
        kp2d = tf.concat([kp2d_joints, vis], axis=-1)
        return img_tensor, kp2d