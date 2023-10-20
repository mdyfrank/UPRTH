import collections
import json
import sys


def dict_to_txt(dic, filename):
    with open(filename, 'w') as f:
        for k, v in dic.items():
            line = str(k) + ' ' + ' '.join(v) + '\n'
            f.write(line)
    return


def quantize(val, k=0.5):
    a = val // k * k
    b = val % k
    output = a if b < k / 2 else a + k
    return output

def load_json(file='./metadata/all.json'):
    bT_dict = collections.defaultdict(list)
    cpd_dict= collections.defaultdict(list)
    with open(file) as json_file:
        line = json_file.readline()
        while line != '':
            data = json.loads(line)
            # print(data)
            iid = data['asin'] #str
            boughtTogether = data['related']['boughtTogether']
            compared = data['related']['compared']
            if boughtTogether!=[]:
                bT_dict[iid] = boughtTogether
            if compared!=[]:
                cpd_dict[iid] = compared
            line = json_file.readline()
    return bT_dict, cpd_dict

def load_dict(file='all.txt'):
    ui_dict = collections.defaultdict(list)
    ic_dict = collections.defaultdict(list)
    ir_dict = collections.defaultdict(list)
    u_map = {}
    i_map = {}
    u_idx, i_idx = 0, 0
    with open(file, 'r') as f:
        line = f.readline().strip()
        while line != '':
            line = line.split()
            user, item, ctgry, rating = line[0], line[1], line[4], line[2]
            if user not in u_map:
                u_map[user] = u_idx
                u_idx += 1
            if item not in i_map:
                i_map[item] = i_idx #imap[str] = int
                i_idx += 1
            mapped_user = u_map[user] #int
            mapped_item = i_map[item] #int
            ui_dict[mapped_user].append(str(mapped_item))
            ic_dict[mapped_item] = [ctgry]
            ir_dict[mapped_item].append(eval(rating))
            line = f.readline().strip()
    for i, ratings in ir_dict.items():
        ir_dict[i] = [str(quantize(sum(ratings) / len(ratings)))]
    dict_to_txt(ui_dict, 'user_item.txt')
    dict_to_txt(ic_dict, 'item_category.txt')
    dict_to_txt(ir_dict, 'item_rating.txt')
    return ui_dict, i_map

def save_bT(bT_dict,imap):
    new_dict = collections.defaultdict(list)
    for v, items in bT_dict.items():
        if v in imap:
            for i in items:
                if i in i_map and i!=v:
                    new_dict[imap[v]].append(str(imap[i]))
    dict_to_txt(new_dict, 'boughtTogether.txt')
    return new_dict



def split(dic):
    train_dic = {}
    test_dic = {}
    train_count = 0
    test_count = 0
    for u, items in dic.items():
        # print(u,items,items[0])
        train_count += 1
        train_dic[str(u)] = [items[0]]
        if len(items) > 1:
            test_dic[str(u)] = items[1:]
            test_count += len(items) - 1
    # print(train_dic)
    dict_to_txt(train_dic, 'train.txt')
    dict_to_txt(test_dic, 'test.txt')
    return train_count, test_count


bT_dict, cpd_dict = load_json()
ui_dict, i_map = load_dict()
# print(bT_dict)
# print(i_map)
print(save_bT(bT_dict,i_map))
print(cpd_dict)
print(split(ui_dict))
