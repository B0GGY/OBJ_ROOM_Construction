import os
import torch
from CLIP_decomposition import CLIP_encode, preprocess
import torch.nn as nn
from arguements import args

class Adaptive_selector(nn.Module):
    def __init__(self, args, inchannel=144, outchannel=1):
        super(Adaptive_selector, self).__init__()
        self.adaptive_layer = nn.Linear(inchannel, outchannel, bias=False)

    def forward(self, x):
        out = self.adaptive_layer(x)
        return out

class FeatureExtractor:
    def __init__(self, args):
        self.head_selector = Adaptive_selector(args)
        self.args = args
        # params =

        self.head_selector.load_state_dict(torch.load(args.pretrain_model))
        self.head_selector.to(self.args.device)
        self.head_selector.eval()
        # self.tmp = []

    def get_features(self, image):
        image = preprocess(image).unsqueeze(0).to(self.args.device)
        with torch.no_grad():

            # print(image.shape)
            x, mlp = CLIP_encode(self.args, image)

            # self.tmp.append(x.squeeze().detach().cpu().numpy())
            # print(x.shape)

            # print(mlp.shape)

            # print(mlp.shape)
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            # print(x.shape)
            x = x.transpose(2, 1)


            # print(x.shape)
            out = self.head_selector(x)

            return out


if __name__ == '__main__':
    from PIL import Image
    from tqdm import tqdm
    from hamming_test import comb
    from matplotlib import pyplot as plt
    import numpy as np

    test_feature_extractor = FeatureExtractor(args)

    save_dir = 'test_output_generalizability'
    try:
        os.mkdir(save_dir)
    except:
        pass

    val_path = '../datasets/Internet_data_hamming_generalizability'
    target_path = '../datasets/Internet_data_hamming_test_generalizability'
    emb_val_list = []
    emb_target_list = []

    room_list = os.listdir(val_path)
    room_list.sort()
    # print(room_list)
    for room in tqdm(room_list):
        val_room_path = os.path.join(val_path, room, 'wobkg')
        target_room_path = os.path.join(target_path, room, 'wobkg')
        val_item_list = os.listdir(val_room_path)
        target_item_list = os.listdir(target_room_path)
        val_item_list.sort()
        target_item_list.sort()
        # print(val_item_list)
        for item in val_item_list:
            image_path = os.path.join(val_room_path, item)

            img = Image.open(image_path)

            features = test_feature_extractor.get_features(img).squeeze(-1)
            # print(features.shape)
            features = features / torch.norm(features, dim=-1, keepdim=True)
            emb_val_list.append(features)
        # print(target_item_list)
        for item in target_item_list:
            image_path = os.path.join(target_room_path, item)
            img = Image.open(image_path)
            features = test_feature_extractor.get_features(img).squeeze(-1)
            # print(features.shape)
            features = features / torch.norm(features, dim=-1, keepdim=True)
            emb_target_list.append(features)
    emb_val_list = torch.cat(emb_val_list)
    emb_target_list = torch.cat(emb_target_list)
    # np.save(os.path.join(save_dir, 'val.npy'), emb_val_list.cpu().detach().numpy())
    # np.save(os.path.join(save_dir, 'target.npy'), emb_target_list.cpu().detach().numpy())
    # item_list_path = '/home/boggy/PycharmProjects/Ada_attn_selection/datasets/item_list.npy'
    item_list_path = 'item_list_generalizability.npy'
    # print(np.load(item_list_path))
    # np.save('img_result.npy', test_feature_extractor.tmp)

    with open(os.path.join(save_dir, 'epoch_logger.txt'), 'w') as f:
        for r_idx in tqdm(range(len(room_list))):
            try:
                os.makedirs(os.path.join(save_dir, room_list[r_idx]))
            except:
                pass

            n, h, max_sim, comb_list = comb(emb_val_list.cpu().detach().numpy(),
                                            emb_target_list.cpu().detach().numpy(), item_list_path,
                                            room_idx=r_idx, room_number = len(room_list))
            plt.cla()
            plt.scatter(np.array(h)/2, n)
            np.save(os.path.join(save_dir, room_list[r_idx], 'n.npy'), n)
            np.save(os.path.join(save_dir, room_list[r_idx], 'h.npy'), h)
            plt.savefig(os.path.join(save_dir, room_list[r_idx], 'epoch_{}.png'.format('test')))
            f.write('Combo result: {}, largest sim {}, comb {}\n'.format( room_list[r_idx], max_sim,
                                                                      comb_list))
        f.close()
