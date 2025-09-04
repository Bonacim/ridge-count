
import numpy as np
import cv2
import scipy
import math
import psutil
import locale
import os
import sqlite3
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.ticker import MaxNLocator, MultipleLocator
import pandas as pd

try:
    from keras.src.utils import metrics_utils
except ModuleNotFoundError:
    from keras.utils import metrics_utils
from keras import backend

def rotate(pt, radians, origin):
    x, y = pt
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy

def accuracy_0(y_true, y_pred):
    DELTA = 0
    [
        y_pred,
        y_true,
    ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
        [y_pred, y_true]
    )
    is_cat = len(y_true.shape) != 1 and y_true.shape[1] != 1
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    if not is_cat:
        y_pred = tf.round(y_pred)
        y_pred = tf.maximum(y_pred,0)
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)
        return tf.cast(tf.equal(y_true, y_pred), backend.floatx())
    else:
        true_indices = tf.argmax(y_true, axis=1)
        pred_indices = tf.argmax(y_pred, axis=1)
        correct_predictions = tf.less_equal(tf.abs(true_indices - pred_indices), DELTA)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def accuracy_1(y_true, y_pred):
    DELTA = 1
    [
        y_pred,
        y_true,
    ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
        [y_pred, y_true]
    )
    is_cat = len(y_true.shape) != 1 and y_true.shape[1] != 1
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    if not is_cat:
        y_pred = tf.round(y_pred)
        y_pred = tf.maximum(y_pred,0)
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)
        diff = tf.subtract(y_true, y_pred)
        diff_abs = tf.math.abs(diff)
        return tf.cast(tf.math.less_equal(diff_abs,DELTA), backend.floatx())
    else:
        true_indices = tf.argmax(y_true, axis=1)
        pred_indices = tf.argmax(y_pred, axis=1)
        correct_predictions = tf.less_equal(tf.abs(true_indices - pred_indices), DELTA)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def accuracy_2(y_true, y_pred):
    DELTA = 2
    [
        y_pred,
        y_true,
    ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
        [y_pred, y_true]
    )
    is_cat = len(y_true.shape) != 1 and y_true.shape[1] != 1
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    if not is_cat:
        y_pred = tf.round(y_pred)
        y_pred = tf.maximum(y_pred,0)
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)
        diff = tf.subtract(y_true, y_pred)
        diff_abs = tf.math.abs(diff)
        return tf.cast(tf.math.less_equal(diff_abs,DELTA), backend.floatx())
    else:
        true_indices = tf.argmax(y_true, axis=1)
        pred_indices = tf.argmax(y_pred, axis=1)
        correct_predictions = tf.less_equal(tf.abs(true_indices - pred_indices), DELTA)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def accuracy_20p(y_true, y_pred):
    DELTA = 0.2
    [
        y_pred,
        y_true,
    ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
        [y_pred, y_true]
    )
    is_cat = len(y_true.shape) != 1 and y_true.shape[1] != 1
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    if not is_cat:
        y_pred = tf.round(y_pred)
        y_pred = tf.maximum(y_pred,0)
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)
        diff = tf.subtract(y_true, y_pred)
        deltas = tf.multiply(y_true,DELTA)
        deltas_floor = tf.floor(deltas)
        diff_abs = tf.math.abs(diff)
        return tf.cast(tf.math.less_equal(diff_abs,deltas_floor), backend.floatx())
    else:
        true_indices = tf.argmax(y_true, axis=1)
        pred_indices = tf.argmax(y_pred, axis=1)
        margin = tf.cast(tf.floor(DELTA * tf.cast(true_indices, tf.float32)), tf.int64)
        correct_predictions = tf.less_equal(tf.abs(true_indices - pred_indices), margin)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def accuracy_25p(y_true, y_pred):
    DELTA = 0.25
    [
        y_pred,
        y_true,
    ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
        [y_pred, y_true]
    )
    is_cat = len(y_true.shape) != 1 and y_true.shape[1] != 1
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    if not is_cat:
        y_pred = tf.round(y_pred)
        y_pred = tf.maximum(y_pred,0)
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)
        diff = tf.subtract(y_true, y_pred)
        deltas = tf.multiply(y_true,DELTA)
        deltas_floor = tf.floor(deltas)
        diff_abs = tf.math.abs(diff)
        return tf.cast(tf.math.less_equal(diff_abs,deltas_floor), backend.floatx())
    else:
        true_indices = tf.argmax(y_true, axis=1)
        pred_indices = tf.argmax(y_pred, axis=1)
        margin = tf.cast(tf.floor(DELTA * tf.cast(true_indices, tf.float32)), tf.int64)
        correct_predictions = tf.less_equal(tf.abs(true_indices - pred_indices), margin)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def accuracy_33p(y_true, y_pred):
    DELTA = 1/3
    [
        y_pred,
        y_true,
    ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
        [y_pred, y_true]
    )
    is_cat = len(y_true.shape) != 1 and y_true.shape[1] != 1
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    if not is_cat:
        y_pred = tf.round(y_pred)
        y_pred = tf.maximum(y_pred,0)
        if y_true.dtype != y_pred.dtype:
            y_pred = tf.cast(y_pred, y_true.dtype)
        diff = tf.subtract(y_true, y_pred)
        deltas = tf.multiply(y_true,DELTA)
        deltas_floor = tf.floor(deltas)
        diff_abs = tf.math.abs(diff)
        return tf.cast(tf.math.less_equal(diff_abs,deltas_floor), backend.floatx())
    else:
        true_indices = tf.argmax(y_true, axis=1)
        pred_indices = tf.argmax(y_pred, axis=1)
        margin = tf.cast(tf.floor(DELTA * tf.cast(true_indices, tf.float32)), tf.int64)
        correct_predictions = tf.less_equal(tf.abs(true_indices - pred_indices), margin)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def load_database(db="./conta_linhas_multi.db", imgDir="./Dataset", labelLimit=None, ori="MANUAL", filterDB=False, sanitized=True):
    def sqlite_power(x,n):
        return int(x)**n
    def sqlite_sqrt(x):
        return np.sqrt(x)

    #loadImages
    connection = sqlite3.connect(db)
    connection.create_function("POWER", 2, sqlite_power)
    connection.create_function("SQRT", 1, sqlite_sqrt)
    cursor = connection.cursor()
    params = []

    query = "SELECT id, filename, coordenada_1_x, coordenada_1_y, coordenada_2_x, coordenada_2_y, label FROM output WHERE SQRT(POWER(coordenada_1_x-coordenada_2_x,2) + POWER(coordenada_1_y-coordenada_2_y,2)) <= 256"
    if sanitized:
        query = query + " AND label != -1"
    if labelLimit != None:
        query = query + " AND label <= ?"
        params.append(labelLimit)
    if filterDB and (ori == "MIXED" or ori == 'FVC_SMALL'):
        query = query + " AND db == ?"
        params.append("MANUAL" if ori == 'MIXED' else ori)
    query = query + " ORDER BY id ASC"
    cursor.execute(query,params)
    rows = cursor.fetchall()
    
    ext = ".bmp"
    if ori == "FVC" or ori == "FVC_SMALL":
        ext = ".tif"

    path = [os.path.join(imgDir, "MANUAL" if ori == 'MIXED' else ori, row[1] + ext) for row in rows]
    coords = [row[2:6] for row in rows]
    label = [row[6] for row in rows]
    ids = [row[0] for row in rows]
    df = pd.DataFrame(data = {'path': path,'coords': coords,'label':label, "id": ids})
    return df

def process_path(file_path):
  label = tf.strings.split(file_path, os.sep)[-2]
  raw = tf.io.read_file(file_path)
  crop = tf.image.decode_bmp(raw, channels = CFG["CHANNELS"])
  crop = tf.image.convert_image_dtype(crop,tf.float32)
  return crop, int(label)

def load_np_array(path):
    # Load the NumPy array from the file path
    return np.load(path).astype(np.float32)

def np_load_wrapper(path_tensor):
    # Convert tensor to string
    path = path_tensor.numpy().decode('utf-8')
    # Load NumPy array
    return load_np_array(path)

def tf_load_np_array(file_path):
    # Use tf.py_function to call the wrapper function
    return tf.py_function(func=np_load_wrapper, inp=[file_path], Tout=tf.float32)

def process_path_mauricio(file_path):
  split_path = tf.strings.split(file_path, os.sep)
  label = split_path[-2]

  file_name_no_ext = tf.strings.regex_replace(split_path[-1], r'\.bmp$', '')
  new_file_name = tf.strings.join([file_name_no_ext, '.npy'])
  new_file_path = tf.strings.reduce_join([tf.strings.reduce_join(split_path[:-1], separator=os.sep), new_file_name], separator=os.sep)

  img_path = file_path
  data_path = new_file_path
  data = tf_load_np_array(data_path)
  #data = tf.convert_to_tensor(np.load(data_path.numpy().decode('utf-8')), dtype=tf.float32)
  raw = tf.io.read_file(img_path)
  crop = tf.image.decode_bmp(raw, channels = CFG["CHANNELS"])
  crop = tf.image.convert_image_dtype(crop,tf.float32)
  return crop, data, int(label)

def crop_image(image, label):
  x = CFG["DB_EXTRA_PAD"]
  y = CFG["DB_EXTRA_PAD"]
  w = tf.shape(image)[1]-2*CFG["DB_EXTRA_PAD"]
  h = tf.shape(image)[0]-2*CFG["DB_EXTRA_PAD"]
  if len(tf.shape(image)) < 3:
    image = tf.expand_dims(image, axis=-1)
  crop = tf.image.crop_to_bounding_box(image, x, y, h, w)
  crop = tf.image.resize_with_crop_or_pad(crop, CFG["PAD_HEIGHT"], CFG["PAD_WIDTH"])
  if CFG["CHANNELS"] == 3:
    first_channel = crop[:, :, 0:1]
    image_shape = tf.shape(first_channel)
    zero_channels = tf.zeros([image_shape[0], image_shape[1], 2], dtype=crop.dtype)
    crop = tf.concat([first_channel, zero_channels], axis=-1)
  return crop, label

def crop_image_mauricio(image, data, label):
  x = CFG["DB_EXTRA_PAD"]
  y = CFG["DB_EXTRA_PAD"]
  w = tf.shape(image)[1]-2*CFG["DB_EXTRA_PAD"]
  h = tf.shape(image)[0]-2*CFG["DB_EXTRA_PAD"]
  if len(tf.shape(image)) < 3:
    image = tf.expand_dims(image, axis=-1)
  crop = tf.image.crop_to_bounding_box(image, x, y, h, w)
  crop = tf.image.resize_with_crop_or_pad(crop, CFG["PAD_HEIGHT"], CFG["PAD_WIDTH"])
  if CFG["CHANNELS"] == 3:
    first_channel = crop[:, :, 0:1]
    image_shape = tf.shape(first_channel)
    zero_channels = tf.zeros([image_shape[0], image_shape[1], 2], dtype=crop.dtype)
    crop = tf.concat([first_channel, zero_channels], axis=-1)
  return crop, data, label

def crop_image_mauricio3(image, data, label):
  x = CFG["DB_EXTRA_PAD"]
  y = CFG["DB_EXTRA_PAD"]
  w = tf.shape(image)[1]-2*CFG["DB_EXTRA_PAD"]
  h = tf.shape(image)[0]-2*CFG["DB_EXTRA_PAD"]
  if len(tf.shape(image)) < 3:
    image = tf.expand_dims(image, axis=-1)
  crop = tf.image.crop_to_bounding_box(image, x, y, h, w)
  crop = tf.image.resize_with_crop_or_pad(crop, CFG["PAD_HEIGHT"], CFG["PAD_WIDTH"])
  if CFG["CHANNELS"] == 3:
    first_channel = crop[:, :, 0:1]
    image_shape = tf.shape(first_channel)
    zero_channels = tf.zeros([image_shape[0], image_shape[1], 2], dtype=crop.dtype)
    crop = tf.concat([first_channel, zero_channels], axis=-1)
 
  crop2 = tf.image.crop_to_bounding_box(image, x, y, h, w)
  crop2 = tf.image.resize_with_crop_or_pad(crop2, 2*CFG["PAD_HEIGHT"], CFG["PAD_WIDTH"])
  if CFG["CHANNELS"] == 3:
    first_channel = crop2[:, :, 0:1]
    image_shape = tf.shape(first_channel)
    zero_channels = tf.zeros([image_shape[0], image_shape[1], 2], dtype=crop.dtype)
    crop2 = tf.concat([first_channel, zero_channels], axis=-1)
  return crop, data, crop2, label


def crop_image_mauricio4(image, data, label):
  x = CFG["DB_EXTRA_PAD"]+CFG["PAD_HEIGHT"]//4
  y = 0#CFG["DB_EXTRA_PAD"]
  w = tf.shape(image)[1]-0*CFG["DB_EXTRA_PAD"]
  h = tf.shape(image)[0]-2*CFG["DB_EXTRA_PAD"]-CFG["PAD_HEIGHT"]//2
  if len(tf.shape(image)) < 3:
    image = tf.expand_dims(image, axis=-1)
  crop = tf.image.crop_to_bounding_box(image, x, y, h, w)
  crop = tf.image.resize_with_crop_or_pad(crop, CFG["PAD_HEIGHT"]//2, CFG["PAD_WIDTH"])
  if CFG["CHANNELS"] == 3:
    first_channel = crop[:, :, 0:1]
    image_shape = tf.shape(first_channel)
    zero_channels = tf.zeros([image_shape[0], image_shape[1], 2], dtype=crop.dtype)
    crop = tf.concat([first_channel, zero_channels], axis=-1)
 
  x = CFG["DB_EXTRA_PAD"]
  y = 0#CFG["DB_EXTRA_PAD"]
  w = tf.shape(image)[1]-0*CFG["DB_EXTRA_PAD"]
  h = tf.shape(image)[0]-2*CFG["DB_EXTRA_PAD"]
  crop2 = tf.image.crop_to_bounding_box(image, x, y, h, w)
  crop2 = tf.image.resize_with_crop_or_pad(crop2, CFG["PAD_HEIGHT"], CFG["PAD_WIDTH"])
  if CFG["CHANNELS"] == 3:
    first_channel = crop2[:, :, 0:1]
    image_shape = tf.shape(first_channel)
    zero_channels = tf.zeros([image_shape[0], image_shape[1], 2], dtype=crop.dtype)
    crop2 = tf.concat([first_channel, zero_channels], axis=-1)
  return crop, data, crop2, label


def split_dataset2(ds, split='train', augment=False, augmentation=None, channels=1):
    AUTOTUNE = tf.data.AUTOTUNE
    DATASET_LEN = ds.cardinality().numpy()
    if channels == 1:
        channels = 0
    CFG["CHANNELS"] = channels
    if split == 'train':
        ds_split = ds.take(int(0.8*DATASET_LEN))
    elif split == 'test':
        ds_split = ds.skip(int(0.8*DATASET_LEN))
    elif split == 'full':
        ds_split = ds
    else:
        raise Exception('Split deve ser: train, test ou full. Obteve: ' + str(split))

    ds_split = ds_split.map(process_path)
    
    if augment:
        ds_split = augmentation(ds_split)

    ds_split = ds_split.map(crop_image)

    ds_split = ds_split.cache()
    ds_split = ds_split.shuffle(buffer_size=4096, seed=42, reshuffle_each_iteration = True)
    ds_split = ds_split.batch(CFG["BATCH_SIZE"])
    ds_split = ds_split.prefetch(buffer_size=AUTOTUNE)

    return ds_split

def split_dataset2_mauricio(ds, split='train', augment=False, augmentation=None, channels=1):
    AUTOTUNE = tf.data.AUTOTUNE
    DATASET_LEN = ds.cardinality().numpy()
    if channels == 1:
        channels = 0
    CFG["CHANNELS"] = channels
    if split == 'train':
        ds_split = ds.take(int(0.8*DATASET_LEN))
    elif split == 'test':
        ds_split = ds.skip(int(0.8*DATASET_LEN))
    elif split == 'full':
        ds_split = ds
    else:
        raise Exception('Split deve ser: train, test ou full. Obteve: ' + str(split))

    ds_split = ds_split.map(process_path_mauricio)
    
    if augment:
        ds_split = augmentation(ds_split)

    ds_split = ds_split.map(crop_image_mauricio)

    ds_split = ds_split.cache()
    ds_split = ds_split.shuffle(buffer_size=4096, seed=42, reshuffle_each_iteration = True)
    ds_split = ds_split.batch(CFG["BATCH_SIZE"])
    ds_split = ds_split.prefetch(buffer_size=AUTOTUNE)

    return ds_split

def split_dataset3_mauricio(ds, split='train', augment=False, augmentation=None, channels=1):
    AUTOTUNE = tf.data.AUTOTUNE
    DATASET_LEN = ds.cardinality().numpy()
    if channels == 1:
        channels = 0
    CFG["CHANNELS"] = channels
    if split == 'train':
        ds_split = ds.take(int(0.8*DATASET_LEN))
    elif split == 'test':
        ds_split = ds.skip(int(0.8*DATASET_LEN))
    elif split == 'full':
        ds_split = ds
    else:
        raise Exception('Split deve ser: train, test ou full. Obteve: ' + str(split))

    ds_split = ds_split.map(process_path_mauricio)
    
    if augment:
        ds_split = augmentation(ds_split)

    ds_split = ds_split.map(crop_image_mauricio3)

    ds_split = ds_split.cache()
    ds_split = ds_split.shuffle(buffer_size=4096, seed=42, reshuffle_each_iteration = True)
    ds_split = ds_split.batch(CFG["BATCH_SIZE"])
    ds_split = ds_split.prefetch(buffer_size=AUTOTUNE)

    return ds_split

def split_dataset4_mauricio(ds, split='train', augment=False, augmentation=None, channels=1):
    AUTOTUNE = tf.data.AUTOTUNE
    DATASET_LEN = ds.cardinality().numpy()
    if channels == 1:
        channels = 0
    CFG["CHANNELS"] = channels
    if split == 'train':
        ds_split = ds.take(int(0.8*DATASET_LEN))
    elif split == 'test':
        ds_split = ds.skip(int(0.8*DATASET_LEN))
    elif split == 'full':
        ds_split = ds
    else:
        raise Exception('Split deve ser: train, test ou full. Obteve: ' + str(split))

    ds_split = ds_split.map(process_path_mauricio)
    
    if augment:
        ds_split = augmentation(ds_split)

    ds_split = ds_split.map(crop_image_mauricio4)

    ds_split = ds_split.cache()
    ds_split = ds_split.shuffle(buffer_size=4096, seed=42, reshuffle_each_iteration = True)
    ds_split = ds_split.batch(CFG["BATCH_SIZE"])
    ds_split = ds_split.prefetch(buffer_size=AUTOTUNE)

    return ds_split


def save_to_disk_mauricio_kfold(filenameDB, imagesDirectory, db_folder, ori_db="MANUAL", channels=1, max_labels=255, folds=10, filterDB=False,sanitized = True,FVCmatchFVC_SMALL = True):
    df = load_database(db=filenameDB, imgDir = imagesDirectory, ori=ori_db, sanitized = sanitized,filterDB=filterDB)    
    print(df.head())
    files = df["path"].unique()
    if ori_db == "FVC" or ori_db == "FVC_SMALL":
        files = list({f.split('/')[-1].split('_')[0] for f in files})
        files.sort()
    random.seed(42)
    if ori_db == "FVC" and FVCmatchFVC_SMALL:
        files_1half = list({f for f in files if int(f) <= 50})
        files_2half = list({f for f in files if int(f) > 50})
        files_1half.sort()
        files_2half.sort()
        random.shuffle(files_1half)
        random.shuffle(files_2half)
        kf = KFold(n_splits=folds)
        new_files = []
        files_1half=np.array(files_1half)
        files_2half=np.array(files_2half)
        for train_index, test_index in kf.split(files_1half):
            test_1half = files_1half[test_index]
            test_2half = files_2half[test_index]
            new_files.extend(test_1half)
            new_files.extend(test_2half)
        files = new_files
    else:
        random.shuffle(files)
    files=np.array(files)
    kf = KFold(n_splits=folds)
    train_folds = []
    test_folds = []
    val_folds = []
    if not os.path.isdir(db_folder):
        os.mkdir(db_folder)
    val_split = int(1.0/folds*len(files))
    for i, (train_index, test_index) in enumerate(kf.split(files)):
        train_files, test_files = files[train_index], files[test_index]
        val_files = []
        if i != (folds - 1):        
            val_files = train_files[i*val_split:(i+1)*val_split]
            train_files = np.concatenate((train_files[:i*val_split],train_files[(i+1)*val_split:]))
        else:
            val_files = train_files[:val_split]
            train_files = train_files[val_split:]
        fold_folder = db_folder+str(i)
        if not os.path.isdir(fold_folder):
            os.mkdir(fold_folder)
        with open(fold_folder+'/train.txt','w') as file:
            file.writelines(f"{e}\n" for e in train_files)
        with open(fold_folder+'/val.txt','w') as file:
            file.writelines(f"{e}\n" for e in val_files)
        with open(fold_folder+'/test.txt','w') as file:
            file.writelines(f"{e}\n" for e in test_files)
        train_folds.append(train_files)
        val_folds.append(val_files)
        test_folds.append(test_files)
    for index, row in df.iterrows():        
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        p1 = row["coords"][:+2]
        p2 = row["coords"][+2:+4]
        largura = CFG["ALTURA"]
        pad = int(np.fix(largura+1))+int(2*CFG["DB_EXTRA_PAD"])#margem de recorte
        minX = min(p1[0], p2[0]) - pad
        minY = min(p1[1], p2[1]) - pad
        maxX = max(p1[0], p2[0]) + pad
        maxY = max(p1[1], p2[1]) + pad

        ang = math.atan2(p2[1]-p1[1],p2[0]-p1[0]) #angulo rad

        padTop = 0 if minY >= 0 else np.abs(minY)
        padBot = 0 if maxY < img.shape[0] else maxY-img.shape[0]
        padLeft = 0 if minX >= 0 else np.abs(minX)
        padRight = 0 if maxX < img.shape[1] else maxX-img.shape[1]


        img = cv2.copyMakeBorder(img, padTop, padBot, padLeft, padRight, cv2.BORDER_REFLECT)

        blk_crop = img[padTop+minY:padTop+maxY+1][:,padLeft+minX:padLeft+maxX+1] #crop

        rotim = scipy.ndimage.rotate(blk_crop, ang / np.pi * 180, reshape=True, order=5, mode='constant')#rotaciona

        #calcula novas coordenadas
        h, w = blk_crop.shape[:2]
        ori = [w/2,h/2]
        q1 = [pad if p1[0] <= p2[0] else w-pad, pad if p1[1] <= p2[1] else h-pad]
        q2 = [w-pad if p1[0] <= p2[0] else pad, h-pad if p1[1] <= p2[1] else pad]
        q1 = rotate(q1,ang,ori)
        q2 = rotate(q2,ang,ori)

        h_new, w_new = rotim.shape[:2]
        xoffset, yoffset = (w_new - w)/2, (h_new - h)/2
        q1 = [q1[0]+xoffset,q1[1]+yoffset]
        q2 = [q2[0]+xoffset,q2[1]+yoffset]

        #crop x
        x1=int(np.fix(q1[0]))
        x2=int(np.fix(q2[0]))

        #crop y
        offset = largura

        extra_pad = CFG["DB_EXTRA_PAD"]
        
        #crop final
        crop = rotim[h_new//2-offset-extra_pad:h_new//2+offset+1+extra_pad][:,x1-extra_pad:x2+extra_pad]

        #mauricio
        p1 = np.array(p1)[..., np.newaxis]
        p2 = np.array(p2)[..., np.newaxis]
        step = 256
        steps = np.linspace(1, 0, step).reshape((1, 1, step))
        points = p1 * steps + p2 * (1 - steps)
        x = points[:, 0]
        y = points[:, 1]
        # get 2x2 square around points
        x1 = np.clip(np.floor(x), 0, img.shape[1] - 1).astype(int)
        y1 = np.clip(np.floor(y), 0, img.shape[0] - 1).astype(int)
        x2 = np.clip(x1 + 1, 0, img.shape[1] - 1)
        y2 = np.clip(y1 + 1, 0, img.shape[0] - 1)
        # interpolate pixel values
        r1 = (x2 - x) * img[y1, x1] + (x - x1) * img[y1, x2]
        r2 = (x2 - x) * img[y2, x1] + (x - x1) * img[y2, x2]
        fxy = (y2 - y) * r1 + (y - y1) * r2
        res = fxy.astype(np.float32)


        file = row["path"]
        if ori_db == "FVC" or ori_db == "FVC_SMALL":
            file = file.split('/')[-1].split('_')[0]
        if channels == 3:
            crop = cv2.cvtColor(crop,cv2.COLOR_GRAY2RGB)
        for fold in range(folds):
            fold_folder = db_folder+str(fold)

            if file in test_folds[fold]:
                label_folder = fold_folder + "/test/"
            if file in val_folds[fold]:
                label_folder = fold_folder + "/val/"
            elif file in train_folds[fold]:
                label_folder = fold_folder + "/train/"
            if not os.path.isdir(label_folder):
                os.mkdir(label_folder)
            label = row["label"]
            if label > max_labels:
                label = max_labels
            label_folder = label_folder+str(label)+"/"
            savepath_crop = label_folder+str(row["id"])+".bmp"
            savepath_data = label_folder+str(row["id"])+".npy"
            if not os.path.isdir(label_folder):
                os.mkdir(label_folder)
            cv2.imwrite(savepath_crop,crop)
            np.save(savepath_data, res)

def double_mean_squared_error(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    diff = tf.math.subtract(y_pred, y_true)
    double_diff = tf.math.scalar_mul(2,diff)
    squared_double_diff = tf.math.multiply(double_diff,double_diff)
    return backend.mean(squared_double_diff, axis=-1)

CFG = {
    "PAD_HEIGHT": 16,
    "PAD_WIDTH": 256,
    "ALTURA": 7,
    "BATCH_SIZE": 512,
    "DB_EXTRA_PAD": 10
}

def assert_dataset(db_folder, folds = 10, test_len = 5, val_len = 5, train_len = 40):
    tests = []
    vals = []
    trains = []
    for fold in range(folds):
        with open(f"{db_folder}{fold}/test.txt", 'r') as file:
            test = [int(line.strip().replace("-","").replace(".tif","")) for line in file]
        with open(f"{db_folder}{fold}/train.txt", 'r') as file:
            train = [int(line.strip().replace("-","").replace(".tif","")) for line in file]
        with open(f"{db_folder}{fold}/val.txt", 'r') as file:
            val = [int(line.strip().replace("-","").replace(".tif","")) for line in file]
        tests.append(test)
        trains.append(train)
        vals.append(val)

        
        assert len(test) == test_len
        assert len(train) == train_len
        assert len(val) == val_len

        assert not set(test).intersection(train)
        assert not set(test).intersection(val)
        assert not set(train).intersection(val)

    for fold in range(folds):
        test = tests[fold]
        prev_val = vals[fold-1]
        assert test == prev_val

        combined_val = [item for sublist in vals for item in sublist]
        assert len(combined_val) == len(set(combined_val))

        combined_test = [item for sublist in tests for item in sublist]
        assert len(combined_test) == len(set(combined_test))
