import collections
import math
import sys
import json

def dict_to_txt(dic, filename):
    with open(filename, 'w') as f:
        for k, v in dic.items():
            line = str(k) + ' ' + ' '.join(v) + '\n'
            f.write(line)
    return


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

def quantize(val, k=0.5):
    a = val // k * k
    b = val % k
    output = a if b < k / 2 else a + k
    return output


def load_dense_dict_2(file='all.txt'):
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
                i_map[item] = i_idx
                i_idx += 1
            mapped_user = u_map[user]
            mapped_item = i_map[item]
            ui_dict[mapped_user].append(str(mapped_item))
            ic_dict[mapped_item] = [ctgry]
            ir_dict[mapped_item].append(eval(rating))
            line = f.readline().strip()
    for i, ratings in ir_dict.items():
        ir_dict[i] = [str(quantize(sum(ratings) / len(ratings)))]
    return ui_dict,ic_dict,ir_dict

def load_dense_dict(file='all.txt'):
    ui_dict = collections.defaultdict(list)
    ic_dict = collections.defaultdict(list)
    ir_dict = collections.defaultdict(list)
    with open(file, 'r') as f:
        line = f.readline().strip()
        while line != '':
            line = line.split()
            user, item, ctgry, rating = line[0], line[1], line[4], line[2]
            ui_dict[user].append(item)
            ic_dict[item] = [ctgry]
            ir_dict[item].append(eval(rating))
            line = f.readline().strip()
    for i, ratings in ir_dict.items():
        ir_dict[i] = [str(quantize(sum(ratings) / len(ratings)))]
    return ui_dict,ic_dict,ir_dict

def load_dict(file='all.txt', k=1 , r=3):
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
            mapped_user = u_map[user]
            if len(ui_dict[mapped_user]) < k + u_idx%r:
                if item not in i_map:
                    i_map[item] = i_idx
                    i_idx += 1
                mapped_item = i_map[item]
                ui_dict[mapped_user].append(str(mapped_item))
                ic_dict[mapped_item] = [ctgry]
                ir_dict[mapped_item].append(eval(rating))
            line = f.readline().strip()
    for i, ratings in ir_dict.items():
        ir_dict[i] = [str(quantize(sum(ratings) / len(ratings)))]
    dict_to_txt(ui_dict, 'user_item.txt')
    dict_to_txt(ic_dict, 'item_category.txt')
    dict_to_txt(ir_dict, 'item_rating.txt')
    print(u_idx,i_idx)
    return ui_dict


def split_1(dic,ratio = 0.7):
    train_dic = {}
    test_dic = {}
    train_count = 0
    test_count = 0
    for u, items in dic.items():
        if len(items) == 1:
            train_count += 1
            train_dic[str(u)] = [items[0]]
        else:
            pos = int(len(items)*ratio//1)
            test_dic[str(u)] = items[pos:]
            test_count += len(items) - pos
            train_dic[str(u)] = items[:pos]
            train_count += pos
    dict_to_txt(train_dic, 'train.txt')
    dict_to_txt(test_dic, 'test.txt')
    return train_count, test_count

def split_2(dic,ratio = 0.1,k = 3):
    train_dic = {}
    test_dic = {}
    train_count = 0
    test_count = 0
    ctr = 0
    for u, items in dic.items():
        if len(items) > k:
            # train_count += len(items)
            # train_dic[str(u)] = items
            continue
        else:
            train_count += 1
            train_dic[str(u)] = [items[0]]
            if len(items) > 1:
                test_dic[str(u)] = items[1:]
                test_count += len(items) - 1
    dict_to_txt(train_dic, 'train.txt')
    dict_to_txt(test_dic, 'test.txt')
    return train_count, test_count

def split(dic):
    train_dic = {}
    test_dic = {}
    train_count = 0
    test_count = 0
    for u, items in dic.items():
        train_count += 1
        train_dic[str(u)] = [items[0]]
        if len(items) > 1:
            test_dic[str(u)] = items[1:]
            test_count += len(items) - 1
    dict_to_txt(train_dic, 'train.txt')
    dict_to_txt(test_dic, 'test.txt')
    return train_count, test_count

def save_ii_relation(bT_dict,imap, filename):
    new_dict = collections.defaultdict(list)
    for v, items in bT_dict.items():
        if v in imap:
            for i in items:
                if i in i_map and i!=v:
                    new_dict[imap[v]].append(str(imap[i]))
    dict_to_txt(new_dict, filename)
    return new_dict

def sparse_dict(ui_dict,ic_dict,ir_dict,k = 3):
    new_ui_dict = collections.defaultdict(list)
    new_ic_dict = {}
    new_ir_dict = {}
    u_map = {}
    i_map = {}
    u_idx, i_idx = 0, 0
    for u,items in ui_dict.items():
        if len(items) <= k:
            u_map[u] = u_idx
            u_idx += 1
            for i in items: # str
                if i not in i_map:
                    i_map[i] = i_idx
                    i_idx += 1
                new_ic_dict[i_map[i]] = ic_dict[i]
                new_ir_dict[i_map[i]] = ir_dict[i]
                new_ui_dict[u_map[u]].append(str(i_map[i]))
    dict_to_txt(new_ui_dict, 'user_item.txt')
    dict_to_txt(new_ic_dict, 'item_category.txt')
    dict_to_txt(new_ir_dict, 'item_rating.txt')
    print(u_idx, i_idx)
    return new_ui_dict, i_map

# sparse_dict = load_dict()

bT_dict, cpd_dict = load_json()

ui_dict,ic_dict,ir_dict = load_dense_dict()
sparse_dict , i_map = sparse_dict(ui_dict,ic_dict,ir_dict)

save_ii_relation(bT_dict,i_map,'boughtTogether.txt')
save_ii_relation(cpd_dict,i_map,'comparedTogether.txt')
print(split(sparse_dict))
