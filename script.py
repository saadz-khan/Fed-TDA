import json,os

import pandas as pd
import torch
import copy
import numpy as np

from fedavg.server import Server
from fedavg.client import Client
from fedavg.models import CNN_Model,weights_init_normal, ReTrainModel,MLP
from utils import get_data

from collections import Counter
from sklearn.preprocessing import MinMaxScaler

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Check the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of available GPUs:", num_gpus)

    # Set the device to the first GPU if available, else CPU
    if num_gpus > 0:
        device = torch.device("cuda:0")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU.")

def min_max_norm(train_datasets, test_dataset, cat_columns, label):

    train_data = None
    for key in train_datasets.keys():
        train_datasets[key]['tag'] = key
        train_data = pd.concat([train_data, train_datasets[key]])
    test_dataset['tag'] = key+1
    data = pd.concat([train_data, test_dataset])

    min_max = MinMaxScaler()
    con = []

    # select continue columns
    for c in data.columns:
        if c not in cat_columns and c not in [label, 'tag']:
            con.append(c)

    data[con] = min_max.fit_transform(data[con])

    # one-hot encode discrete columns
    data = pd.get_dummies(data, columns=cat_columns)

    for key in train_datasets.keys():
        c_data = data[data['tag'] == key]
        c_data = c_data.drop(columns=['tag'])
        train_datasets[key] = c_data

    test_dataset = data[data['tag'] == key+1]
    test_dataset = test_dataset.drop(columns=['tag'])

    return train_datasets, test_dataset

def model_init(conf, train_datasets, test_dataset, device):

    ### init weight of every client node
    client_weight = {}
    if conf["is_init_avg"]:
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)
    print("The aggregate weight of each node is", client_weight)

    clients = {}

    ## init train model
    if conf['model_name'] == "mlp":
        n_input = test_dataset.shape[1] - 1
        model = MLP(n_input, 512, conf["num_classes"][conf['which_dataset']])
    elif conf['model_name'] == 'cnn':
        model = CNN_Model()
    model.apply(weights_init_normal)

    if torch.cuda.is_available():
        model.cuda(device=device)

    server = Server(conf, model, test_dataset, device)
    print("Server init finish!")

    for key in train_datasets.keys():
        clients[key] = Client(conf, copy.deepcopy(server.global_model), train_datasets[key], device)
    print("Clients init finish！")

    # save model
    if not os.path.isdir(conf["model_dir"]):
        os.mkdir(conf["model_dir"])

    return clients, server, client_weight

def train_and_eval(clients, server, client_weight):
    # fed_avg train
    max_score = 0
    score_list = []
    loss_list = []
    for e in range(conf["global_epochs"]):

        clients_models = {}
        for key in clients.keys():
#             print('training client {}...'.format(key))
            model_k = clients[key].local_train(server.global_model)
            clients_models[key] = copy.deepcopy(model_k)

    #         acc, loss = test_model(clients_models[key], test_dataset)
    #         print("client %d,Epoch %d, global_acc: %f, global_loss: %f\n" % (key, e, acc, loss))


        # fed_avg agra
        server.model_aggregate(clients_models, client_weight)

        # evaluate global model
    #     acc, loss, auc_roc, f1 = server.model_eval()
        acc, loss, auc_roc = server.model_eval()
        loss_list.append(loss)

        if conf['num_classes'][conf['which_dataset']] == 2:
            score_list.append(auc_roc)
            print("Epoch %d, global_loss: %f, auc_roc: %f" % (e, loss, auc_roc))
            if auc_roc > max_score:
                torch.save(server.global_model.state_dict(),
                           os.path.join(conf["model_dir"], "global-model.pth"))
                for key in clients.keys():
                    torch.save(clients[key].local_model.state_dict(),
                               os.path.join(conf["model_dir"], "local-model{}.pth".format(key)))
#                 torch.save(server.global_model.state_dict(),
#                            os.path.join(conf["model_dir"], "model-epoch{}.pth".format(e)))
    #             print("model save done !")
                max_score = auc_roc
                maxe = e
        else:
            score_list.append(acc)
            print("Epoch %d, global_loss: %f, acc: %f" % (e, loss, acc))

            # save best model
            if acc > max_score:
                torch.save(server.global_model.state_dict(),
                           os.path.join(conf["model_dir"], "globalmodel-epoch{}.pth".format(e)))
                for key in clients.keys():
                    torch.save(clients[key].local_model.state_dict(),
                               os.path.join(conf["model_dir"], "local-model{}-epoch{}.pth".format(key, e)))
#                 torch.save(server.global_model.state_dict(),
#                            os.path.join(conf["model_dir"], "model-epoch{}.pth".format(e)))
    #             print("model save done !")
                max_score = acc
                maxe = e

    print('max score = {0}, epoch = {1}'.format(max_score, maxe))

    return max_score, loss_list, score_list

def base_train(conf, dataset_name, b, clients_num, path, label_name, device):

    train_files_path_list = [path + "b={}/".format(b) + label_name + "_{}.csv".format(i) for i in range(clients_num)]
    print("path of partition data:\n" + str(train_files_path_list))

    # read file
    train_datasets = {}
    for i in range(len(train_files_path_list)):
        train_datasets[i] = pd.read_csv(train_files_path_list[i])
        print(train_datasets[i][label_name].value_counts())
    test_dataset = pd.read_csv(path + '{}_test.csv'.format(dataset_name))
    print("shape of test dataset: " + str(test_dataset.shape))

    train_datasets, test_dataset = min_max_norm(
        train_datasets, test_dataset,
        conf['discrete_columns'][dataset_name],
        conf['label_column'])

    clients, server, client_weight = model_init(conf, train_datasets, test_dataset, device)

    max_score, loss_list, score_list = train_and_eval(clients, server, client_weight)

    return max_score, loss_list, score_list

def get_random_data(syn_data, aug_numbers,label, ratio):
    """
    select augment data from synthesis data
    """

    aug_data = None
    for i in range(len(aug_numbers)):

        aug_i = syn_data[syn_data[label] == i]
        if aug_i.shape[0] >= aug_numbers[i]:
#             aug_data = pd.concat([aug_data, aug_i.sample(aug_numbers[i])])
            aug_data = pd.concat([aug_data, aug_i.sample(int(ratio*len(aug_i)))])
        else:
            print('label {} has no enough synthetic data'.format(i))

    return aug_data

def random_aug(train_datasets, path, dataset_name, label, label_num, ratio, aug_type='same_number'):
    """
    random select
    """
    labels_dis = []

    for key in train_datasets.keys():
        label_dis = []
        for i in range(label_num):
            label_i = len(train_datasets[key][train_datasets[key][label] == i])
            label_dis.append(label_i)
        labels_dis.append(label_dis)
    labels_dis = np.array(labels_dis)
    print(labels_dis)
    total_dis = np.sum(labels_dis, axis=0)
    print(total_dis)
    aug_numbers = total_dis - labels_dis

    if aug_type == 'same_number':

        for key in train_datasets.keys():
#             syn_data = pd.read_csv('./data/clinical/syn_data/clinical_syn_{}.csv'.format(key))
            syn_data = pd.read_csv('{0}/{1}_syn_{2}.csv'.format(path, dataset_name, key))
            aug_data = get_random_data(syn_data, aug_numbers[key],label, ratio)
            train_datasets[key] = pd.concat([train_datasets[key], aug_data])
            print(train_datasets[key].shape)

    return train_datasets

def augment_train(conf, dataset_name, b, clients_num, path, label_name, label_num, augment_path, ratio, device):

    train_files_path_list = [path + "b={}/".format(b) + label_name + "_{}.csv".format(i) for i in range(clients_num)]
    print("path of partition data:\n" + str(train_files_path_list))

    # read file
    train_datasets = {}
    for i in range(len(train_files_path_list)):
        train_datasets[i] = pd.read_csv(train_files_path_list[i])
        print(train_datasets[i][label_name].value_counts())
    test_dataset = pd.read_csv(path + '{}_test.csv'.format(dataset_name))
    print("shape of test dataset: " + str(test_dataset.shape))

    train_datasets = random_aug(train_datasets, augment_path, dataset_name, label_name, label_num, ratio)

    train_datasets, test_dataset = min_max_norm(
        train_datasets, test_dataset,
        conf['discrete_columns'][dataset_name],
        conf['label_column'])

    clients, server, client_weight = model_init(conf, train_datasets, test_dataset, device)

    max_score, loss_list, score_list = train_and_eval(clients, server, client_weight)

    return max_score, loss_list, score_list

conf = {

	#type of data: tabular, image
	"data_type" : "tabular",

	#select model from mlp,simple-cnn,vgg
	"model_name" : "mlp",

	#fed_ccvr
	"no-iid": "",

	"global_epochs" : 100,

	"local_epochs" : 1,

	# Dirichlet param
	"beta" : 0.05,

	"batch_size" : 65,

	"weight_decay":1e-5,

    #learning rate
	"lr" : 0.001,

	"momentum" : 0.9,

	"num_parties":5,

    # if set weight of different clients even
	"is_init_avg": True,

    # percentage of eval dataset in total train dataset
	"split_ratio": 0.2,

    # name of the column using as label in ml mission
	"label_column": "label",

    # path to save model
	"model_dir":"./save_model/clinical",

    # name of saved model
	"model_file":"model.pth",

    # which dataset is using in current mission
    "which_dataset": "clinical",

    "num_classes": {
        "clinical": 2,
        "credit": 2,
        "tb": 2,
        "covtype": 7,
        "intrusion": 10,
    },
    "discrete_columns": {
        "adult":[
            'workclass',
            'education',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'gender',
            'native_country'
        ],
        "intrusion":['protocol_type', 'service', 'flag'],
        "credit":[],
        "covtype":
            ['Wilderness_Area4', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Soil_Type40', 'Soil_Type1',
             'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
             'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
             'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
             'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
             'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',
             'Soil_Type38', 'Soil_Type39'],
        "clinical":["anaemia","diabetes","high_blood_pressure","sex","smoking"]
    }
}

augment_train(conf,
              "clinical",
              0.05,
              5,
              "./data/clinical/",
              "label",
              2,
              "./data/clinical/syn",
              ratio=1,
              device=device)