from itertools import combinations
import numpy as np


def cal_hamming(a,b):
    # a = np.array(list(map(int, a)))
    # b = np.array(list(map(int, b)))
    a = np.array(a)
    b = np.array(b)
    smstr = np.nonzero(a - b)
    hamming_dist = np.shape(smstr[0])[0]
    return hamming_dist


def comb(encoded_img_features, encoded_img_features_target_room, item_list_path, room_idx, obj_num = 5, room_number = 7):
    # obj_num = 5
    # room_idx = 2  # 0-6 for ['Living Room', 'Bedroom', 'Kitchen', 'Bathroom', 'Study', 'Laundry', 'Dining Room']
    gt_code = ''
    for i in range(room_number):
        if i == room_idx:
            gt_code += '1' * obj_num
        else:
            gt_code += '0' * obj_num
    gt_code = list(map(int, gt_code))
    # print(gt_code)
    # print(cal_hamming(gt_code, '0'*35))
    max_sim = 0
    max_sim_comb = None
    num_list = [j for j in range(obj_num*room_number)]

    item_name_list = np.load(item_list_path)
    e = np.array(gt_code).reshape(-1, 1)
    e = np.repeat(e, 512, axis=1)
    t_ = np.sum(encoded_img_features_target_room * e, axis=0) / obj_num
    ham_list = []
    norm_list = []
    for i in combinations(num_list, obj_num):
        initial_code = [0]*(obj_num*room_number)
        for j in i:
            initial_code[j] = 1
        # print(initial_code)
        e = np.array(initial_code).reshape(-1, 1)
        e = np.repeat(e, 512, axis=1)
        t = np.sum(encoded_img_features*e, axis=0)/obj_num
        similarity_value = np.sqrt(t@t_)
        norm_list.append(similarity_value)#np.linalg.norm(t)
        if similarity_value > max_sim:
            max_sim = similarity_value
            max_sim_comb = initial_code
        ham_list.append(cal_hamming(gt_code, initial_code))
    idx_list = [index for (index,value) in enumerate(max_sim_comb) if value == 1]
    # print('largest sim {}, comb {}'.format(max_sim, [item_name_list[item_name] for item_name in idx_list]))
    return norm_list, ham_list, max_sim, [item_name_list[item_name] for item_name in idx_list]
