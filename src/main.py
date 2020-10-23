import argparse
import numpy as np
from trainer import Trainer

def main(args):
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cam_param", type=int, default=3)
    parser.add_argument("--num_pose_param", type=int, default=72) # 24 * 3
    parser.add_argument("--num_shape_param", type=int, default=10)
    parser.add_argument("--ief_iter", type=int, default=3)
    parser.add_argument("--smpl_mean_theta_file", default="../models/neutral_smpl_mean_params.h5")
    parser.add_argument("--smpl_model", default="../models/neutral_smpl_coco_regressor.pkl")
    parser.add_argument("--lsp_dir", default="../datasets/lspet_dataset")
    parser.add_argument("--img_height", type=int, default=224)
    parser.add_argument("--img_width", type=int, default=224)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gen_lr", type=float, default=1e-5)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    parser.add_argument("--joint_type", default="lsp")
    args = parser.parse_args()
    main(args)