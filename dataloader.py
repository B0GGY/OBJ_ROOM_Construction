from torch.utils.data import Dataset
import os
import random
import numpy as np


class ItemDataset(Dataset):
    def __init__(self, args, split='train'):
        self.split = split
        self.arg = args
        if self.split == 'train':
            self.dataset_file = np.load(os.path.join(args.data_path, 'My_internetdata_train.npy'))  # [b,l,h,d]
            self.dataset_file = self.dataset_file.reshape(self.dataset_file.shape[0], -1, self.dataset_file.shape[-1])
        if self.split =='val':
            self.dataset_file = np.load(os.path.join(args.data_path, 'My_internetdata_val.npy'))  # [b,l,h,d]
            self.dataset_file = self.dataset_file.reshape(self.dataset_file.shape[0], -1, self.dataset_file.shape[-1])
            self.dataset_file_target = np.load(os.path.join(args.data_path, 'My_internetdata_val_target.npy'))  # [b,l,h,d]
            self.dataset_file_target = self.dataset_file_target.reshape(self.dataset_file_target.shape[0], -1, self.dataset_file_target.shape[-1])
        self.items_per_room_n = int(self.dataset_file.shape[0]/args.num_rooms)

    def __len__(self):
        return self.dataset_file.shape[0]

    def anchor_positive_mask_generation(self, r_idx, data, rand_item_num = True, selected_item_n = 5, selected_list = None):
        potential_idx_set = set([i for i in range(r_idx*self.items_per_room_n,(r_idx+1)*self.items_per_room_n)])
        if rand_item_num:
            selected_item_n_a = random.randint(3, 15)
            selected_item_n_p = random.randint(3, 15)
        else:
            selected_item_n_a = selected_item_n
            selected_item_n_p = selected_item_n
        selected_idx_list_a = random.sample(sorted(potential_idx_set), selected_item_n_a)

        selected_idx_list_p = random.sample(sorted(potential_idx_set-set(selected_idx_list_a)), selected_item_n_p)
        anchor = np.mean(data[selected_idx_list_a], axis=0)
        positive = np.mean(data[selected_idx_list_p], axis=0)

        return anchor, positive

    def negative_mask_generation(self, r_idx, data, difficulty_p = 0.1, rand_item_num = True, selected_item_n = 5, selected_list = None):
        # difficulty means the portion of items from the positive room
        if rand_item_num:
            selected_item_n_n = random.randint(3, 15)
        else:
            selected_item_n_n = selected_item_n
        difficulty = int(selected_item_n_n*difficulty_p)
        selected_idx_list = []
        positive_idx_set = set([i for i in range(r_idx * self.items_per_room_n, (r_idx + 1) * self.items_per_room_n)])
        negative_idx_set = set([i for i in range(self.__len__())]) - positive_idx_set
        selected_idx_list.extend(random.sample(sorted(positive_idx_set), difficulty))
        selected_idx_list.extend(random.sample(sorted(negative_idx_set), selected_item_n_n-difficulty))
        negative = np.mean(data[selected_idx_list], axis=0)
        return negative

    def __getitem__(self, idx):
        if self.split =='train':
            ap_room_idx = random.randint(0, self.arg.num_rooms-1)
            anchor, positive = self.anchor_positive_mask_generation(ap_room_idx, self.dataset_file)
            negative = self.negative_mask_generation(ap_room_idx, self.dataset_file, difficulty_p=0.8)

            return anchor.transpose((1,0)), positive.transpose((1,0)), negative.transpose((1,0))
        elif self.split == 'val':
            val_feature_embs = self.dataset_file[idx]
            val_target_feature_embs = self.dataset_file_target[idx]
            # print(val_feature_embs.shape)
            return val_feature_embs.transpose((1,0)), val_target_feature_embs.transpose((1,0))
        else:
            raise TypeError



