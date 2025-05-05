import sys
import os
import numpy as np
from ..pipeline.process_data import ProcessData
from ..utils.create_list_files import create_list_files
import ast  # Abstract Syntax Trees
from sklearn.model_selection import train_test_split
from mne.decoding import CSP
from sklearn.metrics import classification_report, accuracy_score
from .cov import CalculateCovariance as cov, Normalize as norm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization,
    Activation, AveragePooling2D, SeparableConv2D,
    Flatten, Dense, Dropout
)

if len(sys.argv) < 2:
    print("plese enter list to be analysed.")
    sys.exit(1)

# Get the argument (excluding the script name itself)
arg = sys.argv[1]
# Convert string to list
subjects = ast.literal_eval(arg)  # Safer than eval()
# Get the argument (excluding the script name itself)
arg = sys.argv[2]
# Convert string to list
runs = ast.literal_eval(arg)  # Safer than eval()

root = os.getenv('MNE_DATA')
print (f"searching data in folder {root}")

list_files = create_list_files(subjects=subjects, runs=runs, root=root)

if list_files is None or len(list_files) == 0:
    print("No files opened")
    sys.exit(1)
my_process_data = ProcessData()
excluded_channels = ['AF9', 'AF10','AF5', 'AF1','AF2', 'AF6','F9', 'F10','FT9', 'FT10','A1', 'A2','M1', 'M2','TP9', 'TP10','P9', 'P10','PO5', 'PO1','PO2', 'PO6','PO9', 'PO10','O9', 'O10']
my_process_data.config_montage(n_components = 5, excluded_channels = excluded_channels)

for item in list_files:
    if os.path.isdir(item):
        my_process_data.add_files_from_folder(folder = item)
    else:
        my_process_data.add_file(filename = item)

train_model, test_model  = my_process_data.define_test_train(percentage=0.80)
X_train, y_train = my_process_data.generate_data(train_model)
X_test, y_test = my_process_data.generate_data(test_model)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
n_classes = len(le.classes_)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

n_channels = X_test.shape[1]
n_times = X_test.shape[2]
n_classes = len (np.unique(y_train))

y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, n_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, n_classes)

input_shape = (n_channels, n_times, 1)  # 'channels_last' format
inputs = Input(shape=input_shape)
x = Conv2D(8, (1, 64), padding='same')(inputs)
x = BatchNormalization()(x)
x = DepthwiseConv2D((n_channels, 1), depth_multiplier=2, use_bias=False, padding='valid')(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = AveragePooling2D((1, 4))(x)
x = Dropout(0.25)(x)
x = SeparableConv2D(16, (1, 16), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = AveragePooling2D((1, 8))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
outputs = Dense(n_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

model.fit(
    X_train, y_train_cat,
    epochs=30,
    batch_size=16,
    validation_data=(X_val, y_val_cat)
)

val_loss, val_acc = model.evaluate(X_val, y_val_cat)
print(f"Validation accuracy: {val_acc:.2%}")

preds = model.predict(X_val)
pred_classes = np.argmax(preds, axis=1)
val_loss, val_acc = model.evaluate(X_test, y_test_cat)
print(f"Test accuracy: {val_acc:.2%}")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test_cat, y_pred_classes))