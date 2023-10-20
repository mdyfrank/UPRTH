import sys

import scipy.sparse as sp
import numpy as np
from load_data import Data
from utility.parser import parse_args
from sklearn.metrics.pairwise import pairwise_distances

args = parse_args()
data = Data(path='.'+args.data_path + args.dataset, batch_size=args.batch_size)
create_mat = 1
print(args.dataset)
if create_mat:
    print('create adjacency matrix')
    ug_mat = sp.dok_matrix((data.n_users,data.n_groups), dtype=np.int8)
    ui_mat = sp.dok_matrix((data.n_users,data.n_items), dtype=np.int8)

    for idx in range(len(data.user_group_src)):
        ug_mat[data.user_group_src[idx], data.user_group_dst[idx]] = 1
    ug_mat.tocsr()
    for idx in range(len(data.user_item_src)):
        ui_mat[data.user_item_src[idx], data.user_item_dst[idx]] = 1
    ui_mat.tocsr()

    # print(np.dot(ug_mat , ug_mat.T)[2])
    # print(ug_mat.sum(axis = 1)[2])
    user_similarity_group_side = np.dot(ug_mat , ug_mat.T) / (ug_mat.sum(axis = 1))
    # print(user_similarity_group_side[2].tolist())
    # sp.save_npz(args.dataset+'_user_similarity_group_side', user_similarity_group_side)
    user_similarity_item_side = np.dot(ui_mat , ui_mat.T) / (ui_mat.sum(axis = 1))
    # sp.save_npz(args.dataset+'_user_similarity_item_side', user_similarity_item_side)
else:
    print('load saved adjacency matrix')
    ug_mat=sp.load_npz(args.dataset+'_ug_mat.npz')
    ui_mat=sp.load_npz(args.dataset+'_ui_mat.npz')
    user_similarity_group_side = sp.load_npz(args.dataset+'_user_similarity_group_side.npz')
    user_similarity_item_side = sp.load_npz(args.dataset+'_user_similarity_item_side.npz')
    print(user_similarity_group_side)
    print(user_similarity_item_side)

# print(ug_mat)
# print(ui_mat)
# print(ug_mat.shape,ui_mat.shape)
print(user_similarity_group_side)
print(user_similarity_item_side)
# print(user_similarity_group_side)
# print(user_similarity_item_side)

def predict(matrix, similarity, type='group'):
    if type == 'group':
        mean_user_rating = matrix.mean(axis=1)
        ratings_diff = (matrix - mean_user_rating)
        print(mean_user_rating.shape)
        print(ratings_diff.shape)
        print(np.array(np.abs(similarity).sum(axis=1)).shape)
        pred = mean_user_rating + similarity.dot(ratings_diff)/ np.array(np.abs(similarity).sum(axis=1))
    return pred

user_prediction = predict(ug_mat, user_similarity_group_side)
user_prediction_item = predict(ug_mat, user_similarity_item_side)
np.save(args.dataset+'_user_prediction_group_similarity', user_prediction)
np.save(args.dataset+'_user_prediction_item_similarity', user_prediction_item)