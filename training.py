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

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument('-n','--network', type=Network.from_string, choices=list(Network))
parser.add_argument('-m','--filename', default="models/model_10fold{fold}",type=str, help="Template for model location")
parser.add_argument('-i','--db_folder', default='./DBM24_15_k10/',type=str, help="Folder containing model inputs")
parser.add_argument('-a','--pad_height', default=24,type=int, help="Model inputs height")
parser.add_argument('-f','--folds', default=10,type=int, help="Input dataset folds")
args = parser.parse_args()

network = args.network

template=args.filename
db_folder = args.db_folder
CFG["PAD_HEIGHT"] = args.pad_height

filenameDB="./Dataset/conta_linhas4.db"
imagesDirectory="./Dataset"

CFG["PAD_WIDTH"] = 256
CFG["ALTURA"] = CFG["PAD_HEIGHT"]//2-1
CFG["BATCH_SIZE"] = 512
CFG["DB_EXTRA_PAD"] = 10
FOLDS = args.folds

keras.utils.set_random_seed(42)
random.seed(42)
tf.random.set_seed(42)

NUM_EPOCHS = 1000

image_shape = (CFG["PAD_HEIGHT"],CFG["PAD_WIDTH"],1)

# iniciando com a mesma semente 
keras.utils.set_random_seed(123)
random.seed(123)
tf.random.set_seed(123)

#early stop e checkpoint
patience = 100
checkpoint_filepath="weights.hdf5"
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                mode='min')

checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, 
                monitor="val_loss", mode="min", 
                save_best_only=True, save_weights_only=True, verbose=0)

if not os.path.isdir(db_folder):
    save_to_disk_mauricio_kfold(filenameDB, imagesDirectory, db_folder, ori_db="FVC_SMALL", max_labels=15, folds=FOLDS)
assert_dataset(db_folder,FOLDS,50/FOLDS,50/FOLDS,50*(FOLDS-2)/FOLDS)

def augmentation(ds):
    
    return ds

def load_data(fold_folder):
    ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    train_ds = split_dataset2(ds_train,'full',False)
    val_ds = split_dataset2(ds_val,'full',False)
    test_ds = split_dataset2(ds_test,'full',False)

    x_train = []
    y_train = []

    x_val = []
    y_val = []

    x_test = []
    y_test = []

    for x, y in train_ds.unbatch():
        x_train.append(x.numpy())
        y_train.append(y.numpy())
    x_train = np.array(x_train)
    y_train = np.array(y_train)

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
    ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    train_ds = split_dataset2_mauricio(ds_train,'full',False)
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

    for x1, x2, y in train_ds.unbatch():
        x1_train.append(x1.numpy())
        x2_train.append(x2.numpy())
        y_train.append(y.numpy())
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

    y_train = np.eye(16)[y_train]
    y_val = np.eye(16)[y_val]
    y_test = np.eye(16)[y_test]

    return x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test

def load_data2(fold_folder):
    ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    train_ds = split_dataset2_mauricio(ds_train,'full',False)
    val_ds = split_dataset2_mauricio(ds_val,'full',False)
    test_ds = split_dataset2_mauricio(ds_test,'full',False)

    #x1_train = []
    x2_train = []
    y_train = []

    #x1_val = []
    x2_val = []
    y_val = []

    #x1_test = []
    x2_test = []
    y_test = []

    for x1, x2, y in train_ds.unbatch():
       #x1_train.append(x1.numpy())
       x2_train.append(x2.numpy())
       y_train.append(y.numpy())
    #x1_train = np.array(x1_train)
    x2_train = np.array(x2_train)/255
    y_train = np.array(y_train)

    for x1,x2, y in val_ds.unbatch():
        #x1_val.append(x1.numpy())
        x2_val.append(x2.numpy())
        y_val.append(y.numpy())
    #x1_val = np.array(x1_val)
    x2_val = np.array(x2_val)/255
    y_val = np.array(y_val)

    for x1,x2, y in test_ds.unbatch():
        #x1_test.append(x1.numpy())
        x2_test.append(x2.numpy())
        y_test.append(y.numpy())
    #x1_test = np.array(x1_test)
    x2_test = np.array(x2_test)/255
    y_test = np.array(y_test)

    return x2_train, y_train, x2_val, y_val, x2_test, y_test

def load_data3(fold_folder):
    ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    train_ds = split_dataset3_mauricio(ds_train,'full',False)
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

    for x1, x2, x3, y in train_ds.unbatch():
        x1_train.append(x1.numpy())
        x2_train.append(x2.numpy())
        x3_train.append(x3.numpy())
        y_train.append(y.numpy())
    x1_train = np.array(x1_train)
    x2_train = np.array(x2_train)/255
    x3_train = np.array(x3_train)
    y_train = np.array(y_train)

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
    ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    train_ds = split_dataset4_mauricio(ds_train,'full',False)
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

    for x1, x2, x3, y in train_ds.unbatch():
        x1_train.append(x1.numpy())
        x2_train.append(x2.numpy())
        x3_train.append(x3.numpy())
        y_train.append(y.numpy())
    x1_train = np.array(x1_train)
    x2_train = np.array(x2_train)/255
    x3_train = np.array(x3_train)
    y_train = np.array(y_train)

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
    ds_train = tf.data.Dataset.list_files(fold_folder+"/train/*/*.bmp", shuffle=True,seed=42)
    ds_val = tf.data.Dataset.list_files(fold_folder+"/val/*/*.bmp", shuffle=True,seed=42)
    ds_test = tf.data.Dataset.list_files(fold_folder+"/test/*/*.bmp", shuffle=True,seed=42)

    train_ds = split_dataset4_mauricio(ds_train,'full',False)
    val_ds = split_dataset4_mauricio(ds_val,'full',False)
    test_ds = split_dataset4_mauricio(ds_test,'full',False)

    #x1_train = []
    #x2_train = []
    x3_train = []
    y_train = []

    #x1_val = []
    #x2_val = []
    x3_val = []
    y_val = []

    #x1_test = []
    #x2_test = []
    x3_test = []
    y_test = []

    for x1, x2, x3, y in train_ds.unbatch():
        # x1_train.append(x1.numpy())
        # x2_train.append(x2.numpy())
        x3_train.append(x3.numpy())
        y_train.append(y.numpy())
    # x1_train = np.array(x1_train)
    # x2_train = np.array(x2_train)/255
    x3_train = np.array(x3_train)
    y_train = np.array(y_train)

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

    return  x3_train, y_train, x3_val, y_val, x3_test, y_test

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
        #self.projection6 = layers.Dense(d_hidden6, activation=None)
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

def build_FCRN(nu=256):
    data_input = tf.keras.Input(shape=(1,CFG["PAD_WIDTH"]), name="data_input")
    fcrn = MLP(d_in=CFG["PAD_WIDTH"], d_hidden1=nu, d_hidden2=nu, d_hidden3=nu, d_hidden4=nu/2, d_hidden5=nu/2, d_hidden6=nu/2)
    data_output = fcrn(data_input)
    data_output = tf.reshape(data_output,(-1,128))

    #data_output = layers.Dense(16,activation='relu')(data_output)

    output = layers.Dense(1)(data_output)
    output = layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-0.1, clip_value_max=15.0))(output)
    model = keras.models.Model(inputs=data_input,outputs=output)
    return model

def build_CNN(nf=64):
    crop_input = tf.keras.Input(shape=image_shape, name="crop_input")
    cnn = keras.Sequential([
        layers.Input(shape=image_shape),
        layers.Conv2D(nf, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(nf, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(nf, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(nf, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(nf*2, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(nf*2, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(nf*2, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(nf*2, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(nf*4, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(nf*4, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(nf*4, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(nf*4, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu')
    ])
    crop_output = cnn(crop_input)

    output = layers.Dense(1)(crop_output)
    output = layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-0.1, clip_value_max=15.0))(output)
    model = keras.models.Model(inputs=crop_input,outputs=output)
    return model

def transformer_model(image_shape, patch_size, num_patches , projection_dim = 512, transformer_units = 512, transformer_layers = 4, num_classes = 16, num_attention_heads=2, reg=True):    
    inputs = layers.Input(shape=image_shape)
    
    # Spatial Attention Mechanism
    attention_map = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    attention_map = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(attention_map)
    attended_inputs = layers.Multiply()([inputs, attention_map])

    # Patch extraction and embedding
    patches = layers.Conv2D(
        filters=projection_dim, 
        kernel_size=patch_size, 
        strides=(patch_size[0] // 2, patch_size[1] // 2), 
        padding='valid'
    )(attended_inputs)  # Creates non-overlapping patches
    patches = layers.Reshape((num_patches, projection_dim))(patches)  # Reshape into sequence

    # Add positional encoding
    positional_encoding = tf.range(start=0, limit=num_patches, delta=1)
    positional_encoding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positional_encoding)
    patches_with_pos = patches + positional_encoding

    # Class Token as a Learnable Embedding (Replacing tf.Variable)
    class_token = layers.Embedding(
        input_dim=1,  # Only one class token
        output_dim=projection_dim
    )(tf.zeros((1,)))  # Provide a dummy input

    # Class Token (optional)
    class_token = tf.expand_dims(class_token, axis=0)
    batch_size = tf.shape(patches_with_pos)[0]
    class_tokens = tf.tile(class_token, [batch_size, 1, 1])
    patches_with_pos = tf.concat([class_tokens, patches_with_pos], axis=1)

    # Transformer layers
    for _ in range(transformer_layers):
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(num_heads=num_attention_heads, key_dim=projection_dim)(patches_with_pos, patches_with_pos)
        attention_output = layers.Dropout(0.1)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(patches_with_pos + attention_output)

        # Feed-forward network
        ffn_output = layers.Dense(transformer_units, activation='relu')(attention_output)
        ffn_output = layers.Dense(projection_dim)(ffn_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        patches_with_pos = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)

    # Token Reduction
    reduced_patches = layers.Dense(projection_dim // 2, activation="relu")(patches_with_pos)
    reduced_patches = layers.LayerNormalization(epsilon=1e-6)(reduced_patches)

    # Global Context Aggregation - Output tensor of size 128
    class_representation = layers.GlobalAveragePooling1D()(reduced_patches)
    #class_representation = layers.Dense(128, activation='relu')(class_representation)  # Final 128-dimension tensor

    return tf.keras.Model(inputs=inputs, outputs=class_representation)

def build_Transformer(p=16, H=48, projection_dim = 512, transformer_units = 512, N = 4, num_classes = 16, nh=2, reg=True):
    crop_input = tf.keras.Input(shape=image_shape, name="crop_input")
    patch_size = (H,p)
    num_patches = ((image_shape[0] - patch_size[0]) // (patch_size[0] // 2) + 1) * ((image_shape[1] - patch_size[1]) // (patch_size[1] // 2) + 1)
    transformer = transformer_model(image_shape,patch_size,num_patches,projection_dim, transformer_units, N, num_classes, nh, reg)
    crop_output = transformer(crop_input)

    output = layers.Dense(128, activation='relu')(crop_output)
    output = layers.LayerNormalization(epsilon=1e-6)(output)
    output = layers.Dense(64, activation='relu')(output)

    output = layers.Dense(1)(output)
    output = layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-0.1, clip_value_max=15.0))(output)
    model = keras.models.Model(inputs=crop_input, outputs=output)
    return model



custom_objects = {'double_mean_squared_error':double_mean_squared_error,'accuracy_0': accuracy_0, 'accuracy_1': accuracy_1, 'accuracy_2': accuracy_2, 'accuracy_20p': accuracy_20p, 'accuracy_25p': accuracy_25p, 'accuracy_33p':accuracy_33p, 'max_labels':15,'MLP': MLP}

test_eval = []
seeds = [42, 123, 9000]
for fold in range(FOLDS):
    print("fold " + str(fold))
    
    filename=template.format(fold=fold)
    if (os.path.exists(filename+".keras")):
        print("model exists")
        continue

    fold_folder = db_folder+str(fold)

    if network is Network.FCRN:
        model = build_FCRN()
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data2(fold_folder)
    elif network is Network.CNN:
        model = build_CNN()
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data4_mini(fold_folder)
    elif network is Network.TRANSFORMER:
        model = build_Transformer()
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(fold_folder)

    val_evals = []
    for seed in seeds:
        if (os.path.exists(filename+f"_s{seed}.keras")):
            print("model s exists")
            continue
        keras.utils.set_random_seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

        if network is Network.FCRN:
            model = build_FCRN()            
        elif network is Network.CNN:
            model = build_CNN()
        elif network is Network.TRANSFORMER:
            model = build_Transformer()

        model.compile(optimizer=keras.optimizers.Adamax(beta_2=0.99), loss=double_mean_squared_error, metrics=[accuracy_0, accuracy_1, accuracy_2, accuracy_20p, accuracy_25p, accuracy_33p])

        model.fit(X_train, Y_train, batch_size=CFG["BATCH_SIZE"], epochs=NUM_EPOCHS, validation_data=(X_val,Y_val), callbacks=[early_stopping, checkpoint], verbose=0)           

        val_eval = model.evaluate(X_val,Y_val)
        val_evals.append(val_eval[1])
        model.save(filename+f"_s{seed}.keras")

    if len(val_evals) != 0:
        best_val = val_evals.index(max(val_evals))
        seed = seeds[best_val]
        model = keras.models.load_model(filename+f"_s{seed}.keras",custom_objects=custom_objects)
        evaluate = model.evaluate(X_test,Y_test)        
        test_eval.append(evaluate)

        model.save(filename+".keras")

print(np.array(test_eval)[:,1])
print("---------------------------------")
print(np.mean(np.array(test_eval)[:,1]))
print(np.mean(np.array(test_eval)[:,2]))
print(np.mean(np.array(test_eval)[:,3]))
print("---------------------------------")
print(np.std(np.array(test_eval)[:,1]))
print(np.std(np.array(test_eval)[:,2]))
print(np.std(np.array(test_eval)[:,3]))
