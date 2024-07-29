import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

# Define global variables
BATCH_SIZE = 8
IMG_SIZE = (256, 256)

# Load the CSV file containing ship segmentations
train_csv = pd.read_csv('./data/train_ship_segmentations_v2.csv')

# Filter out images with no ships
train_csv = train_csv[train_csv['EncodedPixels'].notnull()]

# Function to decode RLE (Run Length Encoding)
def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# Data augmentation using ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen_args = dict(horizontal_flip=True,
                     vertical_flip=True,
                     rotation_range=90,
                     zoom_range=0.2,
                     shear_range=0.2,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Function to augment both image and mask
def augment_data(image, mask):
    seed = np.random.randint(1e6)
    aug_image = image_datagen.random_transform(image, seed=seed)
    mask = np.expand_dims(mask, axis=-1)  # Add channel dimension to the mask
    aug_mask = mask_datagen.random_transform(mask, seed=seed)
    aug_mask = np.squeeze(aug_mask, axis=-1)  # Remove channel dimension after transformation
    return aug_image, aug_mask

# Custom data generator with augmentation
class AugmentedShipDataGenerator(Sequence):
    def __init__(self, df, img_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[indexes]
        images, masks = self.__data_generation(batch_df)
        return images, masks

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_df):
        images = []
        masks = []
        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.img_dir, row['ImageId'])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            mask = rle_decode(row['EncodedPixels'])
            mask = cv2.resize(mask, self.img_size)
            image, mask = augment_data(image, mask)  # Apply augmentation
            images.append(image)
            masks.append(mask)
        images = np.array(images) / 255.0
        masks = np.array(masks).reshape(-1, self.img_size[0], self.img_size[1], 1)
        return images, masks

# Split data into training and validation sets
train_df, val_df = train_test_split(train_csv, test_size=0.2, random_state=42)

# Create augmented data generators
train_gen = AugmentedShipDataGenerator(train_df, './data/train_v2', batch_size=BATCH_SIZE, img_size=IMG_SIZE)
val_gen = AugmentedShipDataGenerator(val_df, './data/train_v2', batch_size=BATCH_SIZE, img_size=IMG_SIZE)

# Define a small U-Net model for semantic segmentation
def small_unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Instantiate the model
model = small_unet_model()

# Define the dice coefficient metric
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Define the dice loss function
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Compile the model with Adam optimizer and dice loss
model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient])

# Train the model
history = model.fit(train_gen, validation_data=val_gen, epochs=3)

# Save the trained model
model.save('./model/unet_model.keras')
