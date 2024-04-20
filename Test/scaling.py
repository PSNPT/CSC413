#############################################################################################
#############################################################################################
#############################################################################################
# IMPORTS
#############################################################################################
#############################################################################################
#############################################################################################
import os


# LINKING 
from dotenv import load_dotenv
load_dotenv()
ROOT = os.getenv("ROOT")
import sys
sys.path.append(ROOT)
# LINKING

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(os.environ['LD_LIBRARY_PATH'])
#print(os.environ['CUDNN_PATH'])
#print(os.environ['LD_LIBRARY_PATH'])

from tqdm import tqdm
import h5py
import pickle
import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ProgbarLogger, ReduceLROnPlateau
from keras.layers import LayerNormalization, BatchNormalization, MultiHeadAttention
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, Add, Flatten, Activation, Conv1D, Layer, LSTM
from keras.optimizers import Adam
from keras.metrics import AUC, Precision, Recall, F1Score


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
# from tensorflow.keras.layers import LayerNormalization, BatchNormalization, MultiHeadAttention
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Dropout, Dense, Add, Flatten, Activation, Conv1D, Layer
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import layers, models

FEATURES = ['Close','Open', 'High', 'Low', 'Volume',
            'SMA_30', 'EMA_14', 'BBANDS_H', 'BBANDS_L',
            'BBANDS_MAVG', 'VWAP', 'AROON_25', 'AROONU_25', 'AROOND_25', 'RSI_14', 'StochRSI', 'MFI', 'Williams_R',
            '1-Year Real Interest Rate', '1-Year Expected Inflation',
            'Personal Saving Rate', 'Sticky Price Consumer Price Index',
            'Unemployment Rate', '5-Year Forward Inflation Expectation Rate',
            'sentiment',
            "resid_63","seasonal_63","trend_63","resid_252","seasonal_252","trend_252"]

#############################################################################################
#############################################################################################
#############################################################################################
# DATASET HELPER
#############################################################################################
#############################################################################################
#############################################################################################
def construct_dataset(FEATURES,  seq_length, TICKERS):

    X, Y = [], []

    for ticker_symbol in tqdm(TICKERS):
        data = pd.read_csv(ROOT + f"Data/TEMP/{ticker_symbol}.csv", index_col=0)
        data = data[FEATURES]
        xs = []
        ys = []

        # Check for infinite values
        if data.isin([np.inf, -np.inf]).any().any():
            print(f"\nSkipping {ticker_symbol}, inf detected\n")
            continue

        for i in range(len(data)-seq_length-1):
            x = data.iloc[i:(i+seq_length)]
            y = (data.iloc[i+seq_length, data.columns.get_loc('Close')] > 0).astype(int)

            xs.append(x)
            ys.append(y)

        X.extend(np.array(xs))
        Y.extend(np.array(ys))


    return np.array(X), np.array(Y)

#############################################################################################
#############################################################################################
#############################################################################################
# CAUSAL CNN
#############################################################################################
#############################################################################################
#############################################################################################
def build_causal_cnn_model(input_shape, dropout=0.2):
    # Input layer
    input_layer = Input(shape=input_shape)
    LS = 256
    # Causal Convolutional Layers with increasing dilation rate
    x = Conv1D(filters=LS//8, kernel_size=3, dilation_rate=1, padding='causal', activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Conv1D(filters=LS//4, kernel_size=3, dilation_rate=2, padding='causal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Conv1D(filters=LS//2, kernel_size=3, dilation_rate=4, padding='causal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Conv1D(filters=LS//1, kernel_size=3, dilation_rate=8, padding='causal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    # Flatten or pooling layer
    x = Flatten()(x)

    # Fully Connected Layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Building the model
    model = Model(inputs=input_layer, outputs=output)
    return model

#############################################################################################
#############################################################################################
#############################################################################################
# MAIN
#############################################################################################
#############################################################################################
#############################################################################################

import gc  # Import the Python garbage collector

class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()  # Perform garbage collection
        keras.backend.clear_session()
        print(f"\nGarbage collection done after epoch {epoch}\n")

if __name__ == "__main__":
    
    print(f"Listing GPU Information")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs available:", gpus)
    else:
        print("No GPU found!")

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    print("Done\n")

    print("Initializing Profiler")
    tensorboard_callback = TensorBoard(log_dir=ROOT + 'logs', histogram_freq=1)
    print("Done\n")

    # Forcing GPU
    with tf.device('/GPU:0'):
        # FLAGS
        PLOT = False

        RESULTS = []
        TIME_STEP = 120
        LEN_FEAT = len(FEATURES)
        BATCH_SIZE = 128
        EPOCH = 10
        LR = 0.001
        DROPOUT = 0.5
        CACHED = False
        RAW = False

        TRAIN_X = []
        TRAIN_Y = []
        TEST_X = []
        TEST_Y = []
        USED = []

        print(f"Loading raw data")
        TICKERS = []
        for file in os.listdir(ROOT + f"Data/Combined"):
            TICKERS.append(file[:-4])
        print(len(TICKERS))

        import random
        for ticker_symbol in tqdm(random.sample(TICKERS, 20)):

            data_path = f"{ROOT}/Data/TEMP/{ticker_symbol}.csv"
            data = pd.read_csv(data_path, index_col=0)
            if data.isin([np.inf, -np.inf]).any().any():
                print(f"\nSkipping {ticker_symbol}, inf detected\n")
                continue
            data = data[FEATURES]
            
            USED.append(ticker_symbol)
            # Keep the first 90% of the rows for train
            num_rows = int(len(data) * 0.9)
            
            train = data.iloc[:num_rows]
            TRAIN_X.append(train.to_numpy())
            TRAIN_Y.append((train.iloc[TIME_STEP:, data.columns.get_loc('Close')] > 0).astype(int).to_numpy())

            test = data.iloc[num_rows:]
            TEST_X.append(test.to_numpy())
            TEST_Y.append((test.iloc[TIME_STEP:, data.columns.get_loc('Close')] > 0).astype(int).to_numpy())

        print("Done\n")
        TRAIN_X = np.array(TRAIN_X)
        TEST_X = np.array(TEST_X)
        TRAIN_Y = np.array(TRAIN_Y)
        TEST_Y = np.array(TEST_Y)

        for culling in ['BASE', 'Z', 'IQR']:
            for mode in ['MINMAX', 'STANDARD', 'ROBUST']:
                print(f"Tranforming and scaling using: Scalers/scaler_{len(FEATURES)}_IQR_ROBUST.pkl")
                X_train = np.vstack(TRAIN_X)
                X_test = np.vstack(TEST_X)

                print(f"X_train shape: {X_train.shape}")
                print(f"X_test shape: {X_test.shape}")

                with open(ROOT + f'Scalers/scaler_{len(FEATURES)}_{culling}_{mode}.pkl', 'rb') as file:
                    scaler = pickle.load(file)

                TRAIN_X_TRANSFORMED = []
                TEST_X_TRANSFORMED = []

                for data in TRAIN_X:
                    TRAIN_X_TRANSFORMED.append(scaler.transform(data))
                for data in TEST_X:
                    TEST_X_TRANSFORMED.append(scaler.transform(data))

                print("Done\n")

                print(f"Defining Generators")
                def train_data_window_generator(seq_length):
                    for idx, data in enumerate(TRAIN_X_TRANSFORMED):

                        num_windows = len(data) - seq_length

                        for i in range(num_windows):

                            window = data[i:i + seq_length]
                            label = TRAIN_Y[idx][i]
                            yield (window, label)

                def test_data_window_generator(seq_length):
                    for idx, data in enumerate(TEST_X_TRANSFORMED):

                        num_windows = len(data) - seq_length

                        for i in range(num_windows):

                            window = data[i:i + seq_length]
                            label = TEST_Y[idx][i]
                            yield (window, label)

                print("Done\n")

                print(f"Defining Models")
                # Assemble models
                MODELS = {
                    'CNN' : build_causal_cnn_model((TIME_STEP, LEN_FEAT), DROPOUT),
                }
                print("Done\n")

                print(f"Refreshing Datasets...")
                dataset_train = tf.data.Dataset.from_generator(
                    lambda: train_data_window_generator(TIME_STEP),
                    output_types=(tf.float32, tf.int32),  # Tuple of (features, label)
                    output_shapes=((TIME_STEP, len(FEATURES)), ())  # Shape of features and label
                )

                dataset_test = tf.data.Dataset.from_generator(
                    lambda: test_data_window_generator(TIME_STEP),
                    output_types=(tf.float32, tf.int32),  # Tuple of (features, label)
                    output_shapes=((TIME_STEP, len(FEATURES)), ())  # Shape of features and label
                )

                windows = len(TRAIN_X_TRANSFORMED) * (len(TRAIN_X_TRANSFORMED[0]) - TIME_STEP)
                datapoints = len(TRAIN_X_TRANSFORMED)
                total_windows = windows * datapoints
                print(f"Windows: {windows}")
                print(f"Tickers: {datapoints}")
                print(f"Steps per epoch: {windows//BATCH_SIZE}")
                
                dataset_train = dataset_train.cache().shuffle(windows, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)
                dataset_test = dataset_test.cache().batch(BATCH_SIZE, drop_remainder=True)

                print("Done\n")
                
                print(f"Beginning Training")
                for key in MODELS:

                    print(f"Training {key}\n")
                    
                    model = MODELS[key]

                    # Compile the model
                    optimizer = Adam(learning_rate=LR)
        
                    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall()], jit_compile=False) # <- Memory leak!
                    
                    csv_logger = CSVLogger(ROOT + f"/Test/{culling}_{mode}.csv")
                    gc_callback = GarbageCollectorCallback()

                    print("Parameters for training:")
                    print(f"Batch Size: {BATCH_SIZE}")
                    print(f"Epochs: {EPOCH}")
                    print(f"Caching and shuffling (~1-2 minutes, 16GB)")
                    history = model.fit(dataset_train, validation_data=dataset_test, verbose=1, epochs=EPOCH,
                    callbacks=[csv_logger, gc_callback])
                    print("Done\n")

                    print(f"Printing and Saving Performance Statistics")
                    mes = model.evaluate(dataset_test)
                    print(mes)
                    print("Done\n")