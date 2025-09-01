import os.path
import torch
from hamming_test import comb
import argparse
from dataloader import ItemDataset
from torch.utils.data import DataLoader
import torch.optim.optimizer
from model import Adaptive_selector
from loss import TripletLoss
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")

    parser.add_argument(
        "--data_path", default="./datasets", type=str, help="dataset path"
    )
    parser.add_argument('-LR',
                        type=float,
                        default=0.001,
                        help="initial learning rate")
    parser.add_argument('-momentum',
                        type=float,
                        default=0.9,
                        help="momentum")
    parser.add_argument('-weightDecay',
                        type=float,
                        default=1e-4,
                        help="weightDecay")
    parser.add_argument("--num_rooms", default=7, type=int)
    parser.add_argument("--room_list", default=['Bathroom', 'Bedroom', 'Dining room', 'Kitchen', 'Laundry', 'Living room', 'Study']
                        ,type=list)
    parser.add_argument("--epoches", default=100, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save"
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")

    return parser

args = get_args_parser()
args = args.parse_args()

if __name__ == '__main__':
    train_dataset = ItemDataset(args, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    val_dataset = ItemDataset(args, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1)
    model = Adaptive_selector().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.LR, momentum=args.momentum, weight_decay=args.weightDecay)
    triplet_loss = TripletLoss(margin=0.3)

    val_path = 'datasets/My_internetdata_val.npy'
    val_target_path = 'datasets/My_internetdata_val_target.npy'
    item_list_path = 'datasets/item_list.npy'
    encoded_img_features = np.load(val_path)
    encoded_img_features_target_room = np.load(val_target_path)

    try:
        os.makedirs(os.path.join(args.output_dir))
    except:
        pass
    with open(os.path.join(args.output_dir, 'epoch_logger.txt'), 'w') as f:
        for e in range(args.epoches):

            epoch_loss = 0
            model.train()
            for a, p, n in train_loader:
                # image, target = data
                a = model(a.to(args.device)).squeeze()
                p = model(p.to(args.device)).squeeze()
                n = model(n.to(args.device)).squeeze()
                a = a/torch.norm(a, dim=-1, keepdim=True)
                p = p/torch.norm(p, dim=-1, keepdim=True)
                n = n/torch.norm(n, dim=-1, keepdim=True)
                t_loss = triplet_loss(a, p, n)

                optimizer.zero_grad()
                t_loss.backward()

                optimizer.step()
                epoch_loss += t_loss.detach().item()

            print('epoch {} loss: {}'.format(e, epoch_loss/len(train_loader)))
            if e % 10 == 0:
                model.eval()

                with torch.no_grad():
                    try:
                        os.mkdir('models_save')
                    except:
                        pass
                    torch.save(model.state_dict(), 'models_save/model_{}.pt'.format(e))
                    emb_val_list = []
                    emb_val_target_list = []
                    for emb_val, emb_val_target in val_loader:
                        emb_val = model(emb_val.to(args.device)).squeeze(-1)
                        emb_val_target = model(emb_val_target.to(args.device)).squeeze(-1)
                        emb_val = emb_val / torch.norm(emb_val, dim=-1, keepdim=True)
                        emb_val_target = emb_val_target / torch.norm(emb_val_target, dim=-1, keepdim=True)
                        emb_val_list.append(emb_val)
                        emb_val_target_list.append(emb_val_target)
                    emb_val_list = torch.cat(emb_val_list)
                    emb_val_target_list = torch.cat(emb_val_target_list)

                    for r_idx in tqdm(range(args.num_rooms)):
                        try:
                            os.makedirs(os.path.join(args.output_dir, args.room_list[r_idx]))
                        except:
                            pass
                        n, h, max_sim, comb_list = comb(emb_val_list.cpu().detach().numpy(), emb_val_target_list.cpu().detach().numpy(), item_list_path, room_idx=r_idx)
                        plt.cla()
                        plt.scatter(np.array(h)/2, n)
                        plt.savefig(os.path.join(args.output_dir, args.room_list[r_idx],'epoch_{}.png'.format(e)))
                        f.write('Epoch: {}, {}, largest sim {}, comb {}\n'.format(e, args.room_list[r_idx],max_sim, comb_list))
                    print('Validation result saved.')
        f.close()
