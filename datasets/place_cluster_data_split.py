
category_list = {'None item':[], 'Table':[[8,22],[262,276]], 'Refrigerator':[[39,68]], 'Kitchen table':[[72,84]], 'Sink':[[90, 104]],
                 'Other item1':[[105, 117]], 'Other item2':[[133,145]], 'Stove':[[118,132]], 'TV':[[167,178]], 'Coffee table':[[184,194],[239,251]],
                 'Airconditioner':[[201,209]], 'Sofa':[[214,235]], 'Nightstand':[[298,307]], 'Bed':[[309,329]], 'Shelf1':[[335,347]], 'Study desk':[[357,368]]}
data_split = {}

none_item_range = [j for j in range(369)]

for item in category_list.keys():
    data_split[item] = []
    img_ranges = category_list[item]
    for img_range in img_ranges:
        start_idx = img_range[0]
        end_idx = img_range[1]
        data_split[item].extend([i for i in range(start_idx, end_idx+1)])
        none_item_range = list(set(none_item_range)-set([i for i in range(start_idx, end_idx+1)]))
data_split['None item'].extend(none_item_range)

