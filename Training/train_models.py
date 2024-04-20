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

from sklearn.preprocessing import RobustScaler


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
# BASELINE LSTM
#############################################################################################
#############################################################################################
#############################################################################################
def build_LSTM_model(input_shape, dropout=0.2):

    # Input layers
    LS = 256
    # Input layer
    input_layer = Input(shape=input_shape)

    # First LSTM layer
    x = LSTM(LS//8, return_sequences=True)(input_layer)
    x = LayerNormalization()(x) 
    x = Dropout(dropout)(x)
    x = LSTM(LS//4, return_sequences=True)(x)
    x = LayerNormalization()(x) 
    x = Dropout(dropout)(x)
    x = LSTM(LS//2, return_sequences=True)(x)
    x = LayerNormalization()(x) 
    x = Dropout(dropout)(x)
    x = LSTM(LS//1, return_sequences=False)(x)
    x = LayerNormalization()(x) 
    x = Dropout(dropout)(x)
    # Flatten or pooling layer
    x = Flatten()(x)

    # Fully Connected Layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Building the model
    model = Model(inputs=input_layer, outputs=output)

    # Building the model
    return model

#############################################################################################
#############################################################################################
#############################################################################################
# POSITIONAL ENCODER
#############################################################################################
#############################################################################################
#############################################################################################
class PositionalEncoding(Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # Apply sine to even indices in the array
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cosine to odd indices in the array
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model
        })
        return config
    

#############################################################################################
#############################################################################################
#############################################################################################
# TRANSFORMER TEMPORAL
#############################################################################################
#############################################################################################
#############################################################################################
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)

    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    x = Flatten()(x)

    # Change to sigmoid activation for binary classification
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

#############################################################################################
#############################################################################################
#############################################################################################
# TRANSFORMER TEMPORAL SPATIAL
#############################################################################################
#############################################################################################
#############################################################################################
class KerasTranspose(Layer):

    def call(self, x):
        return tf.transpose(x, perm=[0, 2, 1])
    

def create_spacetimeformer(input_shape, num_heads=4, d_model=64, ff_dim=256, num_transformer_blocks=3, dropout=0.2):
    inputs = Input(shape=input_shape)

    # Applying spatial and temporal encodings
    time_steps, features = input_shape
    x = PositionalEncoding(time_steps, features)(inputs)
    x = KerasTranspose()(x)
    x = PositionalEncoding(features, time_steps)(x)
    x = KerasTranspose()(x)

    # Transformer blocks with separate spatial and temporal attention
    for _ in range(num_transformer_blocks):
        # Temporal Attention (along time axis)
        x = LayerNormalization(epsilon=1e-6)(x)
        temporal_attn_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout)(x, x)
        x = Add()([x, temporal_attn_output])
        

        # Spatial Attention (along feature axis)
        # Reshape or permute dimensions as needed to treat features as sequence
        x = LayerNormalization(epsilon=1e-6)(x)
        x_perm = KerasTranspose()(x)
        spatial_attn_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout)(x_perm, x_perm)
        x_perm = Add()([x_perm, spatial_attn_output])
        x = KerasTranspose()(x_perm)  # Restore original dimensions


        # Feed-forward network
        x = LayerNormalization(epsilon=1e-6)(x)
        ffn = Dense(ff_dim, activation="relu")(x)
        ffn = Dropout(dropout)(x)
        ffn = Dense(inputs.shape[-1])(x)
        x = Add()([x, ffn])

    # Aggregating information
    # Flatten or pooling layer
    x = Flatten()(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model




#############################################################################################
#############################################################################################
#############################################################################################
# MAIN
#############################################################################################
#############################################################################################
#############################################################################################
import time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_train_batch_begin(self, batch, logs=None):
        self.start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.times.append(time.time() - self.start_time)
        print(f" Batch {batch} took {self.times[-1]:.3f} seconds")

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
        BATCH_SIZE = 1024
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

        for ticker_symbol in tqdm(TICKERS):

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

        #print(f"Tranforming and scaling using: Scalers/scaler_{len(FEATURES)}_IQR_ROBUST.pkl")
        X_train = np.vstack(TRAIN_X)
        X_test = np.vstack(TEST_X)

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")

        #with open(ROOT + f'Scalers/scaler_{len(FEATURES)}_IQR_ROBUST.pkl', 'rb') as file:
        #    scaler = pickle.load(file)

        scaler = RobustScaler()
        scaler.fit(X_train)

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
            'LSTM' : build_LSTM_model((TIME_STEP, LEN_FEAT), DROPOUT),
            'CNN' : build_causal_cnn_model((TIME_STEP, LEN_FEAT), DROPOUT),
            "TRANSFORMER" : build_transformer_model(input_shape=(TIME_STEP, LEN_FEAT),
                                                    head_size=256,
                                                    num_heads=4,
                                                    ff_dim=512,
                                                    num_layers=4,
                                                    dropout=DROPOUT),
            "SPACE" : create_spacetimeformer(input_shape=(TIME_STEP, LEN_FEAT),
            num_heads=4,
            d_model=256,
            ff_dim=512,
            num_transformer_blocks=4,
            dropout=DROPOUT
            )
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
        
        # Absolute hack of a loader. File loads into memory even with specifying a cache location, yet is not accessed as memory.
        # Fix by double calling. First loads the stored cache, second sets it to memory. 
        # Delete both for no RAM storage, the first for explicitly generating the cache every time this program is used.
        dataset_train = dataset_train.cache('train.cache').cache().shuffle(windows, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)
        dataset_test = dataset_test.cache('test.cache').cache().batch(BATCH_SIZE, drop_remainder=True)

        print("Done\n")
        
        print(f"Beginning Training")
        for key in MODELS:

            print(f"Training {key}\n")
            
            model = MODELS[key]

            # Compile the model
            optimizer = Adam(learning_rate=LR)
 
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall()], jit_compile=False) # <- Memory leak!

            # Define the checkpoint directory and file format
            checkpoint_filepath = ROOT + f'Models/{key}.keras'

            # Create a ModelCheckpoint callback that saves the model's weights only when there is an improvement in 'val_accuracy' (or any other metric of your choice)
            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )
            
            csv_logger = CSVLogger(ROOT + f"/performance/{key}.csv")
            gc_callback = GarbageCollectorCallback()

            print("Parameters for training:")
            print(f"Batch Size: {BATCH_SIZE}")
            print(f"Epochs: {EPOCH}")
            print(f"Caching and shuffling (~1-2 minutes, 16GB)")
            history = model.fit(dataset_train, validation_data=dataset_test, verbose=1, epochs=EPOCH,
            callbacks=[model_checkpoint_callback, csv_logger, gc_callback])
            print("Done\n")

            print(f"Printing and Saving Performance Statistics")
            mes = model.evaluate(dataset_test)
            print(mes)
            print("Done\n")