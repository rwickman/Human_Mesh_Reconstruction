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
    parser.add_argument("--org_lsp_dir", default="../datasets/lsp_dataset")
    parser.add_argument("--lsp_dir", default="../datasets/lspet_dataset")
    parser.add_argument("--img_height", type=int, default=224)
    parser.add_argument("--img_width", type=int, default=224)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gen_lr", type=float, default=1e-5)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    parser.add_argument("--joint_type", default="lsp")
    parser.add_argument("--gen_2d_loss_weight", type=float, default=60)
    parser.add_argument("--disc_loss_weight", type=float, default=1)
    parser.add_argument("--disc_weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_joints", type=int, default=23)
    parser.add_argument("--smpl_params", default="../datasets/neutrMosh/neutrSMPL_CMU/",
            help="Path to SMPL params.")
    parser.add_argument("--max_smpl_load", type=int, default=10,
            help="Max number of SMPL data directories to read.")
    parser.add_argument("--train_split", type=float, default=0.8,
            help="Percentage of data to use in the training set.")
    parser.add_argument("--model_dir", default="model/",
            help="Percentage of data to use in the training set.")
    parser.add_argument("--train_json", default="model/train.json",
            help="Path/filename to training json")
    parser.add_argument("--load_model", action="store_true",
        help="Load the saved models.")
    parser.add_argument("--epochs", type=int, default=5,
            help="Number of epochs to train.")
    parser.add_argument("--test", action="store_true",
        help="Test model on testing dataset")
    args = parser.parse_args()
    main(args)
