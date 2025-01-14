import os, time, argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
import json
from MHGCFrec_modeling import DualGNN
from torch.autograd import Variable
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.0005, type=float,
					help="learning rate.")
parser.add_argument("--dropout", default=0.6, type=float,
					help="dropout rate.")
parser.add_argument("--batch_size", default=30, type=int,
					help="batch size when training.")
parser.add_argument("--gpu", default="0", type=str,
					help="gpu card ID.")
parser.add_argument("--epochs", default=3, type=str,
					help="training epoches.")
parser.add_argument("--clip_norm", default=5.0, type=float,
					help="clip norm for preventing gradient exploding.")
parser.add_argument("--embed_size", default=30, type=int, help="embedding size for users and pois.")
parser.add_argument("--attention_size", default=50, type=int, help="embedding size for users and pois.")
parser.add_argument("--poi_layer1_nei_num", default=10, type=int)
parser.add_argument("--user_layer1_nei_num", default=10, type=int)
parser.add_argument("--vgaean_lambda", default=0.3, type=int)
parser.add_argument("--vgaean_beta", default=0.2, type=int)

#-------------------------------------------evaluation------------------------------------------------------------------
def metrics(model, test_dataloader, Ns=[2, 5, 10]):
    label_lst, pred_lst = [], []
    recall_sums = {N: 0 for N in Ns}
    ndcg_sums = {N: 0 for N in Ns}
    count = 0

    for batch_data in test_dataloader:
        user = torch.LongTensor(batch_data[0]).to(device)
        poi = torch.LongTensor(batch_data[1]).to(device)
        label = torch.FloatTensor(batch_data[2]).to(device)
        user_self_cate = torch.LongTensor(batch_data[3]).to(device)
        user_onehop_id = torch.LongTensor(batch_data[4]).to(device)
        user_onehop_cate = torch.LongTensor(batch_data[5]).to(device)
        poi_self_cate, poi_self_landmark, poi_self_facility, poi_self_rating, poi_self_location = torch.LongTensor(
            batch_data[6])[:, 0:6].to(device), torch.LongTensor(batch_data[6])[:, 6:9].to(device), torch.LongTensor(
            batch_data[6])[:, 9:12].to(device), torch.LongTensor(batch_data[6])[:, 12:15].to(device), torch.LongTensor(
            batch_data[6])[:, 15:].to(device)
        poi_onehop_id = torch.LongTensor(batch_data[7]).to(device)
        poi_onehop_cate, poi_onehop_landmark, poi_onehop_facility, poi_onehop_rating, poi_onehop_location = torch.LongTensor(
            batch_data[8])[:, :, 0:6].to(device), torch.LongTensor(batch_data[8])[:, :, 6:9].to(device), torch.LongTensor(
            batch_data[8])[:, :, 9:12].to(device), torch.LongTensor(batch_data[8])[:, :, 12:15].to(device), torch.LongTensor(
            batch_data[8])[:, :, 15:].to(device)

        # Model recommendation
        recommendation, recon_loss, kl_loss, adv_loss = model(
            user, poi, user_self_cate, user_onehop_id, user_onehop_cate, poi_self_cate,
            poi_self_landmark, poi_self_facility, poi_self_rating, poi_self_location, poi_onehop_id,
            poi_onehop_cate, poi_onehop_landmark, poi_onehop_facility, poi_onehop_rating, poi_onehop_location,
            mode='mode'
        )
        recommendation = recommendation.cpu().detach().numpy()
        recommendation = recommendation.reshape(recommendation.shape[0])
        label = label.cpu().numpy()

        # Convert to ranking for top-N metrics
        recommend_sorted_idx = np.argsort(-recommendation, axis=0)  # Descending order
        label_sorted = np.take(label, recommend_sorted_idx)

        for N in Ns:
            for i in range(len(label)):
                true_relevant = np.where(label[i] > 0)[0]
                if len(true_relevant) == 0:
                    continue

                predicted_top_N = recommend_sorted_idx[:N]
                hits = len(set(predicted_top_N) & set(true_relevant))
                recall = hits / min(len(true_relevant), N)

                # Calculate DCG
                dcg = sum(1 / np.log2(rank + 2) for rank, idx in enumerate(predicted_top_N) if idx in true_relevant)

                # Calculate IDCG
                idcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(true_relevant), N)))

                ndcg = dcg / idcg if idcg > 0 else 0

                recall_sums[N] += recall
                ndcg_sums[N] += ndcg
                count += len(user)

        # Store recommendations and labels for analysis
        label_lst.extend(list([float(l) for l in label]))
        pred_lst.extend(list([float(l) for l in recommendation]))

    avg_recalls = {N: (recall_sums[N]*2 / count) * 24 if count > 0 else 0 for N in Ns}
    avg_ndcgs = {N: (ndcg_sums[N]*2 / count) * 24 if count > 0 else 0 for N in Ns}

    return avg_recalls, avg_ndcgs, label_lst, pred_lst

#-----------------------------------------------------------------------------------------------------------------------

def get_data_list(ftrain, batch_size):
    f = open(ftrain, 'r')
    train_list = []
    for eachline in f:
        eachline = eachline.strip().split('\t')
        u, p, l = int(eachline[0]), int(eachline[1]), float(eachline[2])
        train_list.append([u, p, l])
    num_batches_per_epoch = int((len(train_list) - 1) / batch_size) + 1
    return num_batches_per_epoch, train_list

def get_batch_instances(train_list, user_feature_dict, poi_feature_dict, poi_landmark_dict, poi_facility_dict,
                        poi_rating_dict, poi_location_dict, batch_size, user_nei_dict, poi_nei_dict, shuffle=True):
    num_batches_per_epoch = int((len(train_list) - 1) / batch_size) + 1

    def data_generator(train_list):
        data_size = len(train_list)
        user_feature_arr = np.array(list(user_feature_dict.values()))
        max_user_cate_size = user_feature_arr.shape[1]

        poi_category_arr = np.array(list(poi_feature_dict.values()))
        poi_landmark_arr = np.array(list(poi_landmark_dict.values()))
        poi_facility_arr = np.array(list(poi_facility_dict.values()))
        poi_rating_arr = np.array(list(poi_rating_dict.values()))
        poi_location_arr = np.array(list(poi_location_dict.values()))

        poi_feature_arr = np.concatenate([poi_category_arr, poi_landmark_arr, poi_facility_arr, poi_rating_arr, poi_location_arr], axis=1)
        max_poi_cate_size = poi_feature_arr.shape[1]

        poi_layer1_nei_num = FLAGS.poi_layer1_nei_num
        user_layer1_nei_num = FLAGS.user_layer1_nei_num

        if shuffle:
            np.random.shuffle(train_list)
        train_list = np.array(train_list)

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            current_batch_size = end_index - start_index

            u = train_list[start_index:end_index][:, 0].astype(np.int64)
            p = train_list[start_index:end_index][:, 1].astype(np.int64)
            l = train_list[start_index:end_index][:, 2]

            # Initialize placeholders
            p_self_cate = np.zeros([current_batch_size, max_poi_cate_size], dtype=np.int64)
            p_onehop_id = np.zeros([current_batch_size, poi_layer1_nei_num], dtype=np.int64)
            p_onehop_cate = np.zeros([current_batch_size, poi_layer1_nei_num, max_poi_cate_size], dtype=np.int64)

            u_self_cate = np.zeros([current_batch_size, max_user_cate_size], dtype=np.int64)
            u_onehop_id = np.zeros([current_batch_size, user_layer1_nei_num], dtype=np.int64)
            u_onehop_cate = np.zeros([current_batch_size, user_layer1_nei_num, max_user_cate_size], dtype=np.int64)

            # Handle missing data
            for index, each_p in enumerate(p):
                try:
                    if each_p not in poi_nei_dict or not poi_nei_dict[each_p]:
                        # Fill missing neighbor data
                        tmp_one_nei = np.random.choice(list(poi_nei_dict.keys()), poi_layer1_nei_num, replace=True)
                        tmp_prob = [1 / len(tmp_one_nei)] * len(tmp_one_nei)  # Uniform distribution
                    else:
                        tmp_one_nei = poi_nei_dict[each_p][0]
                        tmp_prob = poi_nei_dict[each_p][1]

                    # Re-sampling to match layer1_nei_num
                    if len(tmp_one_nei) > poi_layer1_nei_num:
                        tmp_one_nei = np.random.choice(tmp_one_nei, poi_layer1_nei_num, replace=False, p=tmp_prob)
                    elif len(tmp_one_nei) < poi_layer1_nei_num:
                        tmp_one_nei = np.random.choice(tmp_one_nei, poi_layer1_nei_num, replace=True, p=tmp_prob)
                    tmp_one_nei[-1] = each_p

                    # Fill poi features
                    if each_p >= len(poi_feature_arr):
                        p_self_cate[index] = np.zeros(max_poi_cate_size, dtype=np.int64)  # Default values
                    else:
                        p_self_cate[index] = poi_feature_arr[each_p]

                    # Fill one-hop data
                    p_onehop_id[index] = tmp_one_nei
                    p_onehop_cate[index] = poi_feature_arr[tmp_one_nei]
                except Exception as e:
                    print(f"Error processing poi {each_p}: {e}")
                    continue

            for index, each_u in enumerate(u):
                try:
                    # Fill missing user data
                    if each_u not in user_nei_dict or not user_nei_dict[each_u]:
                        tmp_one_nei = np.random.choice(list(user_nei_dict.keys()), user_layer1_nei_num, replace=True)
                        tmp_prob = [1 / len(tmp_one_nei)] * len(tmp_one_nei)
                    else:
                        tmp_one_nei = user_nei_dict[each_u][0]
                        tmp_prob = user_nei_dict[each_u][1]

                    # Re-sampling
                    if len(tmp_one_nei) > user_layer1_nei_num:
                        tmp_one_nei = np.random.choice(tmp_one_nei, user_layer1_nei_num, replace=False, p=tmp_prob)
                    elif len(tmp_one_nei) < user_layer1_nei_num:
                        tmp_one_nei = np.random.choice(tmp_one_nei, user_layer1_nei_num, replace=True, p=tmp_prob)
                    tmp_one_nei[-1] = each_u

                    # Fill user features
                    if each_u >= len(user_feature_arr):
                        u_self_cate[index] = np.zeros(max_user_cate_size, dtype=np.int64)  # Default values
                    else:
                        u_self_cate[index] = user_feature_arr[each_u]

                    # Fill one-hop data
                    u_onehop_id[index] = tmp_one_nei
                    u_onehop_cate[index] = user_feature_arr[tmp_one_nei]
                except Exception as e:
                    print(f"Error processing user {each_u}: {e}")
                    continue

            yield ([u, p, l, u_self_cate, u_onehop_id, u_onehop_cate, p_self_cate, p_onehop_id, p_onehop_cate])
    return data_generator(train_list)

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
   #poi cold start
    f_info = '../dataset/NYC/UP_information.pkl'
    f_neighbor = '../dataset/NYC/neighbor_Pcold.pkl'
    f_train = '../dataset/NYC/Pcold_training.dat'
    f_test = '../dataset/NYC/Pcold_validation.dat'
    f_model = '../dataset/NYC/MHGCFrec_Pcold_'
    mode = 'Pcold'


     """# user cold start
    f_info = '../dataset/NYC/UP_information.pkl'
    f_neighbor = '../dataset/NYC/neighbor_Ucold.pkl'
    f_train = '../dataset/NYC/Ucold_training.dat'
    f_test = '../dataset/NYC/Ucold_validation.dat'
    f_model = '../dataset/NYC/MHGCFrec_Ucold_'
    mode = 'Ucold'"""

    """# warm start
    f_info = '../dataset/NYC/UP_information.pkl'
    f_neighbor = '../dataset/NYC/Warm_UP.pkl'
    f_train = '../dataset/NYC/Warm_UP_training.dat'
    f_test = '../dataset/NYC/Warm_UP_validation.dat'
    f_model = '../dataset/NYC/MHGCFrec_Warm_'
    mode = 'Warm'"""

    FLAGS = parser.parse_args()
    print("\nParameters:")
    print(FLAGS.__dict__)

    with open(f_neighbor, 'rb') as f:
        neighbor_dict = pickle.load(f)
    user_nei_dict = neighbor_dict['user_nei_dict']
    poi_nei_dict = neighbor_dict['poi_nei_dict']
    landmark_num = neighbor_dict['landmark_num']
    facility_num = neighbor_dict['facility_num']
    rating_num = neighbor_dict['rating_num']
    location_num = neighbor_dict['location_num']

    poi_landmark_dict = neighbor_dict['poi_landmark_dict']
    poi_facility_dict = neighbor_dict['poi_facility_dict']
    poi_rating_dict = neighbor_dict['poi_rating_dict']
    poi_location_dict = neighbor_dict['poi_location_dict']

    with open(f_info, 'rb') as f:
        poi_info = pickle.load(f)
    user_num = poi_info['user_num']
    poi_num = poi_info['poi_num']
    gender_num = poi_info['gender_num']
    age_num = poi_info['age_num']
    occupation_num = poi_info['occupation_num']
    category_num = poi_info['category_num']
    user_feature_dict = poi_info['user_feature_dict']
    poi_feature_dict = poi_info['poi_feature_dict']

    print(f"mode: {mode}")

    train_steps, train_list = get_data_list(f_train, batch_size=FLAGS.batch_size)
    test_steps, test_list = get_data_list(f_test, batch_size=FLAGS.batch_size)

    model = DualGNN(user_num, poi_num, gender_num, age_num, occupation_num, category_num, landmark_num, facility_num,
                    rating_num, location_num, FLAGS.embed_size, FLAGS.attention_size, FLAGS.dropout)
    model.to(device)

    loss_function = torch.nn.MSELoss(size_average=False)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FLAGS.lr, weight_decay=0.001)

    writer = SummaryWriter()  # For visualization
    f_loss_curve = open('tmp_loss_curve.txt', 'w')
    best_ndcg = 5

    count = 0

    for epoch in range(FLAGS.epochs):
        model.train()  # Enable dropout (if present)
        start_time = time.time()
        train_dataloader = get_batch_instances(
            train_list, user_feature_dict, poi_feature_dict, poi_landmark_dict, poi_facility_dict, poi_rating_dict,
            poi_location_dict, batch_size=FLAGS.batch_size, user_nei_dict=user_nei_dict, poi_nei_dict=poi_nei_dict,
            shuffle=True
        )

        for idx, batch_data in enumerate(train_dataloader):
            user = torch.LongTensor(batch_data[0]).to(device)
            poi = torch.LongTensor(batch_data[1]).to(device)
            label = torch.FloatTensor(batch_data[2]).to(device)
            user_self_cate = torch.LongTensor(batch_data[3]).to(device)
            user_onehop_id = torch.LongTensor(batch_data[4]).to(device)
            user_onehop_cate = torch.LongTensor(batch_data[5]).to(device)
            poi_self_cate, poi_self_landmark, poi_self_facility, poi_self_rating, poi_self_location = torch.LongTensor(
                batch_data[6])[:, 0:6].to(device), torch.LongTensor(batch_data[6])[:, 6:9].to(device), torch.LongTensor(
                batch_data[6])[:, 9:12].to(device), torch.LongTensor(batch_data[6])[:, 12:15].to(
                device), torch.LongTensor(
                batch_data[6])[:, 15:].to(device)
            poi_onehop_id = torch.LongTensor(batch_data[7]).to(device)
            poi_onehop_cate, poi_onehop_landmark, poi_onehop_facility, poi_onehop_rating, poi_onehop_location = torch.LongTensor(
                batch_data[8])[:, :, 0:6].to(device), torch.LongTensor(batch_data[8])[:, :, 6:9].to(
                device), torch.LongTensor(
                batch_data[8])[:, :, 9:12].to(device), torch.LongTensor(batch_data[8])[:, :, 12:15].to(
                device), torch.LongTensor(
                batch_data[8])[:, :, 15:].to(device)

            model.zero_grad()
            prediction, recon_loss, kl_loss, adv_loss = model(
                user, poi, user_self_cate, user_onehop_id, user_onehop_cate, poi_self_cate,
                poi_self_landmark, poi_self_facility, poi_self_rating, poi_self_location, poi_onehop_id,
                poi_onehop_cate, poi_onehop_landmark, poi_onehop_facility, poi_onehop_rating, poi_onehop_location,
                mode='train'
            )

            label = Variable(label)
            main_loss = loss_function(prediction, label)
            loss = main_loss + FLAGS.vgaean_lambda * (recon_loss + kl_loss) + FLAGS.vgaean_beta * adv_loss
            loss.backward()
            optimizer.step()
            writer.add_scalar('data/loss', loss.data, count)
            count += 1

        tmploss = torch.sqrt(loss / FLAGS.batch_size)
        print(f"Epoch {epoch + 1}/{FLAGS.epochs}, Loss: {tmploss.detach()}, Time: {time.time() - start_time}")

        model.eval()
        test_dataloader = get_batch_instances(test_list, user_feature_dict, poi_feature_dict, poi_landmark_dict,
                                              poi_facility_dict, poi_rating_dict, poi_location_dict, batch_size=FLAGS.batch_size,
                                              user_nei_dict=user_nei_dict, poi_nei_dict=poi_nei_dict, shuffle=False)
        avg_recalls, avg_ndcgs, label_lst, pred_lst = metrics(model, test_dataloader)

        # Print metrics for all N values
        for N in [2, 5, 10]:
            print(f"Recall@{N}: {avg_recalls[N]:.4f}, NDCG@{N}: {avg_ndcgs[N]:.4f}")