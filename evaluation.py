import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from utils import accuracy_0, accuracy_1, accuracy_2, accuracy_20p, accuracy_25p, accuracy_33p
from utils import split_dataset2, CFG, double_mean_squared_error, split_dataset2_mauricio, save_to_disk_mauricio_kfold, assert_dataset, split_dataset4_mauricio
import random
import os
from enum import Enum
import argparse

class Network(Enum):
    FCRN = 0
    CNN = 1
    TRANSFORMER = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Network[s]
        except KeyError:
            raise ValueError()

parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument('-n','--network', type=Network.from_string, choices=list(Network))
args = parser.parse_args()

network = args.network

template=""
if network is Network.FCRN:
    template="models/model_mlp_best_10fold{fold}.keras"
    db_folder = "./DBM16_15_k10/"
    CFG["PAD_HEIGHT"] = 16
elif network is Network.CNN:
    template="models/model_cnn_best_10fold{fold}.keras"
    db_folder = "./DBM24_15_k10/"
    CFG["PAD_HEIGHT"] = 24
elif network is Network.TRANSFORMER:
    template="models/model_trans_best_10fold{fold}.keras"
    db_folder = "./DBM48_15_k10/"
    CFG["PAD_HEIGHT"] = 48



filenameDB="./Dataset/conta_linhas4.db"
imagesDirectory="./Dataset"

CFG["PAD_WIDTH"] = 256
CFG["ALTURA"] = CFG["PAD_HEIGHT"]//2-1
CFG["BATCH_SIZE"] = 512
CFG["DB_EXTRA_PAD"] = 10
FOLDS = 10

keras.utils.set_random_seed(42)
random.seed(42)
tf.random.set_seed(42)

if not os.path.isdir(db_folder):
    save_to_disk_mauricio_kfold(filenameDB, imagesDirectory, db_folder, ori_db="FVC_SMALL", max_labels=15, folds=FOLDS)
assert_dataset(db_folder,FOLDS,50/FOLDS,50/FOLDS,50*(FOLDS-2)/FOLDS)

def augmentation(ds):
    
    return ds

def load_data(fold_folder):
    #ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    #train_ds = split_dataset2(ds_train,'full',False)
    val_ds = split_dataset2(ds_val,'full',False)
    test_ds = split_dataset2(ds_test,'full',False)

    x_train = []
    y_train = []

    x_val = []
    y_val = []

    x_test = []
    y_test = []

    # for x, y in train_ds.unbatch():
    #     x_train.append(x.numpy())
    #     y_train.append(y.numpy())
    # x_train = np.array(x_train)
    # y_train = np.array(y_train)

    for x, y in val_ds.unbatch():
        x_val.append(x.numpy())
        y_val.append(y.numpy())
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    for x, y in test_ds.unbatch():
        x_test.append(x.numpy())
        y_test.append(y.numpy())
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test

def load_data_cat(fold_folder):
    #ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    #train_ds = split_dataset2_mauricio(ds_train,'full',False)
    val_ds = split_dataset2_mauricio(ds_val,'full',False)
    test_ds = split_dataset2_mauricio(ds_test,'full',False)

    x1_train = []
    x2_train = []
    y_train = []

    x1_val = []
    x2_val = []
    y_val = []

    x1_test = []
    x2_test = []
    y_test = []

    # for x1, x2, y in train_ds.unbatch():
    #     x1_train.append(x1.numpy())
    #     x2_train.append(x2.numpy())
    #     y_train.append(y.numpy())
    x1_train = np.array(x1_train)
    x2_train = np.array(x2_train)/255
    y_train = np.array(y_train)

    for x1,x2, y in val_ds.unbatch():
        x1_val.append(x1.numpy())
        x2_val.append(x2.numpy())
        y_val.append(y.numpy())
    x1_val = np.array(x1_val)
    x2_val = np.array(x2_val)/255
    y_val = np.array(y_val)

    for x1,x2, y in test_ds.unbatch():
        x1_test.append(x1.numpy())
        x2_test.append(x2.numpy())
        y_test.append(y.numpy())
    x1_test = np.array(x1_test)
    x2_test = np.array(x2_test)/255
    y_test = np.array(y_test)

    #y_train = np.eye(16)[y_train]
    y_val = np.eye(16)[y_val]
    y_test = np.eye(16)[y_test]

    return x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test

def load_data2(fold_folder):
    #ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    #train_ds = split_dataset2_mauricio(ds_train,'full',False)
    val_ds = split_dataset2_mauricio(ds_val,'full',False)
    test_ds = split_dataset2_mauricio(ds_test,'full',False)

    x1_train = []
    x2_train = []
    y_train = []

    x1_val = []
    x2_val = []
    y_val = []

    x1_test = []
    x2_test = []
    y_test = []

    #for x1, x2, y in train_ds.unbatch():
    #    x1_train.append(x1.numpy())
    #    x2_train.append(x2.numpy())
    #    y_train.append(y.numpy())
    x1_train = np.array(x1_train)
    x2_train = np.array(x2_train)/255
    y_train = np.array(y_train)

    for x1,x2, y in val_ds.unbatch():
        x1_val.append(x1.numpy())
        x2_val.append(x2.numpy())
        y_val.append(y.numpy())
    x1_val = np.array(x1_val)
    x2_val = np.array(x2_val)/255
    y_val = np.array(y_val)

    for x1,x2, y in test_ds.unbatch():
        x1_test.append(x1.numpy())
        x2_test.append(x2.numpy())
        y_test.append(y.numpy())
    x1_test = np.array(x1_test)
    x2_test = np.array(x2_test)/255
    y_test = np.array(y_test)

    return x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test

def load_data3(fold_folder):
    #ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    #train_ds = split_dataset3_mauricio(ds_train,'full',False)
    val_ds = split_dataset3_mauricio(ds_val,'full',False)
    test_ds = split_dataset3_mauricio(ds_test,'full',False)

    x1_train = []
    x2_train = []
    x3_train = []
    y_train = []

    x1_val = []
    x2_val = []
    x3_val = []
    y_val = []

    x1_test = []
    x2_test = []
    x3_test = []
    y_test = []

    # for x1, x2, x3, y in train_ds.unbatch():
    #     x1_train.append(x1.numpy())
    #     x2_train.append(x2.numpy())
    #     x3_train.append(x3.numpy())
    #     y_train.append(y.numpy())
    # x1_train = np.array(x1_train)
    # x2_train = np.array(x2_train)/255
    # x3_train = np.array(x3_train)
    # y_train = np.array(y_train)

    for x1,x2,x3, y in val_ds.unbatch():
        x1_val.append(x1.numpy())
        x2_val.append(x2.numpy())
        x3_val.append(x3.numpy())
        y_val.append(y.numpy())
    x1_val = np.array(x1_val)
    x2_val = np.array(x2_val)/255
    x3_val = np.array(x3_val)
    y_val = np.array(y_val)

    for x1,x2,x3, y in test_ds.unbatch():
        x1_test.append(x1.numpy())
        x2_test.append(x2.numpy())
        x3_test.append(x3.numpy())
        y_test.append(y.numpy())
    x1_test = np.array(x1_test)
    x2_test = np.array(x2_test)/255
    x3_test = np.array(x3_test)
    y_test = np.array(y_test)

    return x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val, x1_test, x2_test, x3_test, y_test

def load_data4(fold_folder):
    #ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    #train_ds = split_dataset4_mauricio(ds_train,'full',False)
    val_ds = split_dataset4_mauricio(ds_val,'full',False)
    test_ds = split_dataset4_mauricio(ds_test,'full',False)

    x1_train = []
    x2_train = []
    x3_train = []
    y_train = []

    x1_val = []
    x2_val = []
    x3_val = []
    y_val = []

    x1_test = []
    x2_test = []
    x3_test = []
    y_test = []

    # for x1, x2, x3, y in train_ds.unbatch():
    #     x1_train.append(x1.numpy())
    #     x2_train.append(x2.numpy())
    #     x3_train.append(x3.numpy())
    #     y_train.append(y.numpy())
    # x1_train = np.array(x1_train)
    # x2_train = np.array(x2_train)/255
    # x3_train = np.array(x3_train)
    # y_train = np.array(y_train)

    for x1,x2,x3, y in val_ds.unbatch():
        x1_val.append(x1.numpy())
        x2_val.append(x2.numpy())
        x3_val.append(x3.numpy())
        y_val.append(y.numpy())
    x1_val = np.array(x1_val)
    x2_val = np.array(x2_val)/255
    x3_val = np.array(x3_val)
    y_val = np.array(y_val)

    for x1,x2,x3, y in test_ds.unbatch():
        x1_test.append(x1.numpy())
        x2_test.append(x2.numpy())
        x3_test.append(x3.numpy())
        y_test.append(y.numpy())
    x1_test = np.array(x1_test)
    x2_test = np.array(x2_test)/255
    x3_test = np.array(x3_test)
    y_test = np.array(y_test)

    return x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val, x1_test, x2_test, x3_test, y_test

def load_data4_mini(fold_folder):
    #ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    #train_ds = split_dataset4_mauricio(ds_train,'full',False)
    val_ds = split_dataset4_mauricio(ds_val,'full',False)
    test_ds = split_dataset4_mauricio(ds_test,'full',False)

    x1_train = []
    x2_train = []
    x3_train = []
    y_train = []

    x1_val = []
    x2_val = []
    x3_val = []
    y_val = []

    x1_test = []
    x2_test = []
    x3_test = []
    y_test = []

    # for x1, x2, x3, y in train_ds.unbatch():
    #     x1_train.append(x1.numpy())
    #     x2_train.append(x2.numpy())
    #     x3_train.append(x3.numpy())
    #     y_train.append(y.numpy())
    # x1_train = np.array(x1_train)
    # x2_train = np.array(x2_train)/255
    # x3_train = np.array(x3_train)
    # y_train = np.array(y_train)

    for x1,x2,x3, y in val_ds.unbatch():
        #x1_val.append(x1.numpy())
        #x2_val.append(x2.numpy())
        x3_val.append(x3.numpy())
        y_val.append(y.numpy())
    #x1_val = np.array(x1_val)
    #x2_val = np.array(x2_val)/255
    x3_val = np.array(x3_val)
    y_val = np.array(y_val)

    for x1,x2,x3, y in test_ds.unbatch():
        #x1_test.append(x1.numpy())
        #x2_test.append(x2.numpy())
        x3_test.append(x3.numpy())
        y_test.append(y.numpy())
    #x1_test = np.array(x1_test)
    #x2_test = np.array(x2_test)/255
    x3_test = np.array(x3_test)
    y_test = np.array(y_test)

    return x1_train, x2_train, x3_train, y_train, x1_val, x2_val, x3_val, y_val, x1_test, x2_test, x3_test, y_test

class MLP(tf.keras.Model):
    def __init__(self, d_in, d_hidden1, d_hidden2, d_hidden3, d_hidden4, d_hidden5, d_hidden6):
        super(MLP, self).__init__()
        self.d_in = d_in

        self.linear1 = layers.Dense(d_hidden1, activation='relu')
        self.linear2 = layers.Dense(d_hidden2, activation=None)  # No activation for residuals
        self.linear3 = layers.Dense(d_hidden3, activation=None)  # No activation for residuals
        self.linear4 = layers.Dense(d_hidden4, activation=None)  # No activation for residuals
        self.linear5 = layers.Dense(d_hidden5, activation=None)  # No activation for residuals
        self.linear6 = layers.Dense(d_hidden6, activation=None)  # No activation for residuals

        #self.projection1 = layers.Dense(d_hidden1, activation=None)
        self.projection2 = layers.Dense(d_hidden2, activation=None)
        self.projection3 = layers.Dense(d_hidden3, activation=None)
        self.projection4 = layers.Dense(d_hidden4, activation=None)
        self.projection5 = layers.Dense(d_hidden5, activation=None)
        self.projection6 = layers.Dense(d_hidden6, activation=None)
        #self.linear6 = layers.Dense(d_out, activation=None)
        self.dropout = layers.Dropout(0.2)

    def call(self, inputs, training=False):
        x = self.linear1(inputs)
        x = self.dropout(x, training=training)
        
        x_res = x
        x = tf.nn.relu(self.linear2(x))
        x = self.dropout(x, training=training)
        if x_res.shape[-1] != x.shape[-1]:
            x_res = self.projection2(x_res)
        x += x_res

        x_res = x
        x = tf.nn.relu(self.linear3(x))
        x = self.dropout(x, training=training)
        if x_res.shape[-1] != x.shape[-1]:
            x_res = self.projection3(x_res)
        x += x_res

        x_res = x
        x = tf.nn.relu(self.linear4(x))
        x = self.dropout(x, training=training)
        if x_res.shape[-1] != x.shape[-1]:
            x_res = self.projection4(x_res)
        x += x_res

        x_res = x
        x = tf.nn.relu(self.linear5(x))
        x = self.dropout(x, training=training)
        if x_res.shape[-1] != x.shape[-1]:
            x_res = self.projection5(x_res)
        x += x_res

        x_res = x
        x = tf.nn.relu(self.linear6(x))
        x = self.dropout(x, training=training)
        # if x_res.shape[-1] != x.shape[-1]:
        #     x_res = self.projection6(x_res)
        x += x_res

        return x
            
    # Define the config so the model can be serialized
    def get_config(self):
        return {
            'd_in': self.d_in,
            'd_hidden1': self.linear1.units,
            'd_hidden2': self.linear2.units,
            'd_hidden3': self.linear3.units,
            'd_hidden4': self.linear4.units,
            'd_hidden5': self.linear5.units,
            'd_hidden6': self.linear6.units
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

custom_objects = {'double_mean_squared_error':double_mean_squared_error,'accuracy_0': accuracy_0, 'accuracy_1': accuracy_1, 'accuracy_2': accuracy_2, 'accuracy_20p': accuracy_20p, 'accuracy_25p': accuracy_25p, 'accuracy_33p':accuracy_33p, 'max_labels':15,'MLP': MLP}

test_eval = []
for fold in range(FOLDS):
    print("fold " + str(fold))

    
    
    filename=template.format(fold=fold)
    if not os.path.exists(filename):
        print("model does not exists")
        continue

    fold_folder = db_folder+str(fold)
    model = keras.models.load_model(filename,custom_objects=custom_objects)

    if network is Network.FCRN:
        X1_train, X2_train, Y_train, X1_val, X2_val, Y_val, X1_test,  X2_test, Y_test = load_data2(fold_folder)
        evaluate = model.evaluate(X2_test,Y_test)
    elif network is Network.CNN:
        X1_train, X2_train, X3_train, Y_train, X1_val, X2_val, X3_val, Y_val, X1_test,  X2_test, X3_test, Y_test = load_data4_mini(fold_folder)
        evaluate = model.evaluate(X3_test,Y_test)
    elif network is Network.TRANSFORMER:
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(fold_folder)
        evaluate = model.evaluate(X_test,Y_test)            

    test_eval.append(evaluate)

print(np.array(test_eval)[:,1])
print("---------------------------------")
print(np.mean(np.array(test_eval)[:,1]))
print(np.mean(np.array(test_eval)[:,2]))
print(np.mean(np.array(test_eval)[:,3]))
print("---------------------------------")
print(np.std(np.array(test_eval)[:,1]))
print(np.std(np.array(test_eval)[:,2]))
print(np.std(np.array(test_eval)[:,3]))
