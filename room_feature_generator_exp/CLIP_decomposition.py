import numpy as np
import torch
import argparse
from utils.factory import create_model_and_transforms, get_tokenizer
from prs_hook import hook_prs_logger

def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-B-16",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k", type=str)
    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/shared/group/ilsvrc", type=str, help="dataset path"
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet", help="imagenet, cub or waterbirds"
    )
    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    return parser

args = get_args_parser()
args = args.parse_args()
model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
model.to(args.device)
model.eval()
# context_length = model.context_length
# vocab_size = model.vocab_size
prs = hook_prs_logger(model, args.device)

def CLIP_encode(args, image):
    """Calculates the projected residual stream for a dataset."""
    # 将以下代码放入到新的dataloader中去
    # image = preprocess(Image.open(os.path.join(data_path))).unsqueeze(0).to(args.device)
    # image shape torch.Size([1, 3, 224, 224])
    attention_results = []
    mlp_results = []
    cls_to_cls_results = []
    # print(type(image))
    with torch.no_grad():
        prs.reinit()
        representation = model.encode_image(
            image.to(args.device), attn_method="head", normalize=False
        )
        attentions, mlps = prs.finalize(representation)
        attentions = attentions.detach().cpu().numpy()  # [b, l, n, h, d]
        mlps = mlps.detach().cpu().numpy()  # [b, l+1, d]
        attention_results.append(
            np.sum(attentions, axis=2)
        )  # Reduce the spatial dimension
        mlp_results.append(mlps)
        cls_to_cls_results.append(
            np.sum(attentions[:, :, 0], axis=2)
        )  # Store the cls->cls attention, reduce the heads

    return torch.from_numpy(np.concatenate(attention_results, axis=0)).to(args.device),torch.from_numpy(np.concatenate(mlp_results, axis=0)).to(args.device)
