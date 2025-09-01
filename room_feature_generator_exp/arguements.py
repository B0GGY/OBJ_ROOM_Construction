import argparse

def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)

    parser.add_argument("--device", default="cuda:0", help="device to use for testing")

    parser.add_argument('--pretrain_model', default='../models_save/model_90.pt', type=str)

    return parser


args = get_args_parser()
args = args.parse_args()