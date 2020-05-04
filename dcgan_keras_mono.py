# -- coding: utf-8 --
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, ZeroPadding2D
# import os
# from keras.datasets import fashion_mnist
from pathlib import Path
from keras.optimizers import Adam
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout
import matplotlib.pyplot as plt

from mainprocess import load_data
import tensorflow as tf
from keras.backend import tensorflow_backend as KTF

BATCH_SIZE = 32
NUM_EPOCH = 2000
GENERATED_IMAGE_PATH = Path('generated_images/dcgan/') # 生成画像の保存先
latent_dim = 100
size = 100
mode = "face"

seed = 0
np.random.seed(seed)
tf.random.set_random_seed(seed)

# refernce from: https://qiita.com/God_KonaBanana/items/293d49e3c34601a1810b
def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128*25*25))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((25, 25, 128), input_shape=(128*25*25,)))
    model.add(UpSampling2D((2, 2)))
    model.add(Convolution2D(64, (5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Convolution2D(1, (5,5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(64, (5, 5),
                            strides=2,
                            padding='same',
                            input_shape=(size, size, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(128, (5, 5), strides=2))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(256, (5, 5), strides=2))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def train():
    """
    (X_train, y_train), (_, _) = fashion_mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1)
    print(X_train.shape)
    """
    X_train, datagen = load_data(mode=mode)
    print(X_train.shape)

    discriminator = discriminator_model()
    d_opt = Adam(lr=1e-5, beta_1=0.1)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    # generator+discriminator （discriminator部分の重みは固定）
    discriminator.trainable = False
    generator = generator_model()
    dcgan = Sequential([generator, discriminator])
    g_opt = Adam(lr=2e-4, beta_1=0.5)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)

    for epoch in range(NUM_EPOCH):
        d_gen = datagen.flow(X_train, batch_size=BATCH_SIZE, seed=seed)
        for index in range(num_batches):
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            # image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            image_batch = d_gen.next()
            generated_images = generator.predict(noise, verbose=0)

            # 生成画像を出力
            if index % 500 == 0:

                # generate images and shape
                generated_images_plot = generated_images.astype('float32') * 127.5 + 127.5
                generated_images_plot = generated_images_plot.reshape((BATCH_SIZE, size, size))

                plt.figure(figsize=(8, 4))
                plt.suptitle('epoch=%04d,index=%04d' % (epoch, index), fontsize=20)
                for i in range(BATCH_SIZE):
                    plt.subplot(4, 8, i + 1)
                    plt.imshow(generated_images_plot[i])
                    plt.gray()
                    # eliminate ticks
                    plt.xticks([]), plt.yticks([])

                # save images
                GENERATED_IMAGE_PATH.mkdir(parents=True, exist_ok=True)
                filename = GENERATED_IMAGE_PATH / Path(f"{mode}_%04d_%04d.png" % (epoch,index))
                plt.savefig(filename)
                plt.close()

            # discriminatorを更新
            X = np.concatenate((image_batch, generated_images))
            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            # generatorを更新
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, [1]*BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

        generator.save_weights(f'generator_{mode}_dcgan.h5')
        discriminator.save_weights(f'discriminator_{mode}_dcgan.h5')

if __name__ == '__main__':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    KTF.set_session(session)
    train()