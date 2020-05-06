# -- coding: utf-8 --
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Reshape, Input, Lambda, Add, Subtract, Multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, ZeroPadding2D, AveragePooling2D, Convolution2DTranspose
from keras import regularizers
from keras.initializers import RandomNormal
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
GENERATED_IMAGE_PATH = Path('generated_images/pggan/') # 生成画像の保存先
weight_path = Path("weight")
latent_dim = 100
size = 256
FINAL_STAGE = int(np.log2(256))
mode = "extracted"
weight_path.mkdir(parents=True, exist_ok=True)
seed = 0
np.random.seed(seed)
tf.random.set_random_seed(seed)

l2_penalty = 0.00001
rand_stddev = 0.02

num_feat = lambda res, fmap_decay=1.0, def_ch=8192, max_ch=128: min(int(def_ch / (2.0 ** ((res-1) * fmap_decay))), max_ch) # channel(素子)数の計算

# refernce from: https://qiita.com/God_KonaBanana/items/293d49e3c34601a1810b
def build_block(stage=1, final_stage=1):
    res = 2 ** (stage + 1)
    input = Input(shape=(res, res, num_feat(stage)))
    __x = Lambda(lambda x: x)(input)
    reg = regularizers.l2(l2_penalty)
    rand_init = RandomNormal(stddev=rand_stddev)
    if stage != 1:
        __x = Convolution2DTranspose(num_feat(stage), (3, 3), strides=(2,2), padding='same', kernel_initializer=rand_init,
                            kernel_regularizer=reg, bias_regularizer=reg)(__x)
    else:
        __x = Convolution2D(num_feat(stage), (3, 3), padding='same', kernel_initializer=rand_init,
                            kernel_regularizer=reg, bias_regularizer=reg)(__x)
    out_x = LeakyReLU(0.2)(BatchNormalization()(__x))

    if stage == final_stage:
        __rgb = Lambda(lambda x: x)(out_x)
    else:
        __rgb = UpSampling2D()(out_x)
    __rgb = Convolution2D(3, (1, 1), padding="same", kernel_initializer=rand_init,
                         kernel_regularizer=reg, bias_regularizer=reg)(__rgb)
    return Model([input], [out_x, __rgb], name=f"gen_stage_{stage}")

def merge_img(stage=1, alpha=0.0):
    if stage == 1:
        raise ValueError
    else:
        res = 2 ** (stage + 1)
        input_h = Input(shape=(res, res, 3), name=f"torgbH_{stage}")
        input_l = Input(shape=(res, res, 3), name=f"torgbL_{stage}")

        __sub = Subtract()([input_l, input_h])
        t = min(max(alpha,0.0),1.0)
        _tsub = Lambda(lambda x: t * x, output_shape=(res,res,3))(__sub)
        outputs = Add()([input_l, _tsub])
        return Model([input_l, input_h], [outputs], name="toRGB")

def generator_split(stage=1):
    reg = regularizers.l2(l2_penalty)
    rand_init = RandomNormal(stddev=rand_stddev)
    # grand_input
    inputs = Input(shape=(latent_dim,), name="grand_input")

    # local input model
    __input = Input(shape=(latent_dim,))
    __x = Dense(units=num_feat(stage) * 16, kernel_initializer=rand_init, kernel_regularizer=reg, bias_regularizer=reg)(__input)
    __x = LeakyReLU(0.2)(BatchNormalization()(__x))
    __output = Reshape((4, 4, num_feat(stage)), input_shape=(num_feat(stage),))(__x)
    input_model = Model([__input], [__output], name="input_of_generator")

    __output = input_model(inputs)
    # block builder
    models = []
    output_list = []
    for i in range(1,stage+1):
        models.append(build_block(stage=i, final_stage=stage))

    if stage == 1:
        outputs = models[0](__output)[1]
    else:
        output_list.append(models[0](__output))

        for i in range(1, stage):
            output_list.append(models[i](output_list[i-1][0]))

        outputs = [output_list[-2][1], output_list[-1][1]]

    model = Model([inputs], outputs, name="generator_block")
    return model

def build_generator(stage=1, alpha=0.5):
    __z = Input(shape=(latent_dim,), name="generator_input")
    gen_split = generator_split(stage)
    if stage == 1:
        img = gen_split(__z)
    else:
        gen_merge = merge_img(stage=stage, alpha=alpha)
        __lx, __hx = gen_split(__z)
        img = gen_merge([__lx, __hx])
    gen_model = Model([__z], [img])
    return load_gen_weight(gen_model)

def dis_stage_n(stage=1):
    reg = regularizers.l2(l2_penalty)
    rand_init = RandomNormal(stddev=rand_stddev)
    res = 2**(stage+1)

    input = Input((res, res, num_feat(stage)))
    __x = Convolution2D(num_feat(stage), (2,2), strides=(2,2), padding="same", kernel_initializer=rand_init,
                         kernel_regularizer=reg, bias_regularizer=reg)(input)
    output = LeakyReLU(0.2)(BatchNormalization()(__x))
    return Model([input], [output], name=f"dis_stage_{stage}_n")

def dis_stage_a(stage=1):
    if stage == 1:
        raise ValueError
    res = 2 ** (stage + 1)
    input = Input(shape=(res, res, 3))
    output = AveragePooling2D(padding="same")(input)
    return Model([input], [output], name=f"dis_stage_{stage}_a")

def dis_stage_r(stage=1):
    reg = regularizers.l2(l2_penalty)
    rand_init = RandomNormal(stddev=rand_stddev)
    res = 2 ** (stage + 1)
    input = Input(shape=(res, res, 3))
    __x = Convolution2D(num_feat(stage), (1, 1), padding="same", kernel_initializer=rand_init,
                        kernel_regularizer=reg, bias_regularizer=reg)(input)
    output = LeakyReLU(0.2)(BatchNormalization()(__x))
    return Model([input], [output], name=f"dis_stage_{stage}_r")


def dis_load(stage=1):
    res = 2 ** (stage + 1)
    input = Input(shape=(res, res, 3))
    if stage == 1:
        dis_r = dis_stage_r(stage)
        output = [dis_r(input)]
    else:
        dis_r = dis_stage_r(stage)
        dis_n = dis_stage_n(stage)
        dis_a = dis_stage_a(stage)
        dis_r1 = dis_stage_r(stage - 1)
        __hx = dis_r(input)
        __hx = dis_n(__hx)
        __lx = dis_a(input)
        __lx = dis_r1(__lx)
        output = [__lx, __hx]

    return Model([input], output, name=f"dis_load_{stage}")

def dis_merge(stage=1, alpha=0.5):
    if stage == 1:
        raise ValueError
    else:
        res_1 = 2 ** (stage)  # 2 ** (stage + 1)
        nfs = num_feat(stage)
        input_h = Input(shape=(res_1, res_1, nfs), name=f"DmergeH_{stage}")
        input_l = Input(shape=(res_1, res_1, nfs), name=f"DmergeL_{stage}")

        __sub = Subtract()([input_l, input_h])
        t = min(max(alpha,0.0),1.0)
        _tsub = Lambda(lambda x: t * x, output_shape=(res_1,res_1,nfs))(__sub)
        outputs = Add()([input_l, _tsub])
        return Model([input_l, input_h], [outputs], name="Dmerge")

def dis_final(stage=1):
    reg = regularizers.l2(l2_penalty)
    rand_init = RandomNormal(stddev=rand_stddev)
    res_r1 = 2 ** stage
    input = Input((res_r1, res_r1, num_feat(stage)))
    __x = Flatten()(input)
    __x = Dense(units=num_feat(stage), kernel_initializer=rand_init,
                kernel_regularizer=reg, bias_regularizer=reg, name="dis_final_dense_1")(__x)
    __x = LeakyReLU(0.2)(BatchNormalization()(__x))
    __x = Dense(units=1, kernel_initializer=rand_init,
                kernel_regularizer=reg, bias_regularizer=reg, name="dis_final_dense_2")(__x)
    output = Activation('sigmoid')(__x)
    return Model(input, output, name=f"dis_final_{stage}")

def build_discriminator(stage=1, alpha=0.5):
    res = 2 ** (stage + 1)
    input = Input((res, res, 3), name="dis_grand_input")

    loader = dis_load(stage)
    models = dis_stage_n(1)
    if stage == 1:
        __x = loader(input)
        __x = models(__x)
    else:
        __lx, __hx = loader(input)
        merge = dis_merge(stage, alpha=alpha)
        __x = merge([__lx, __hx])
        for i in range(stage-1,0,-1):
            model = dis_stage_n(i)
            __x = model(__x)

    finalize = dis_final(1)
    output = finalize(__x)
    dis_model = Model([input], [output], name="discriminator")
    return load_dis_weight(dis_model)

def build_pggan(generator, discriminator):
    input = Input(shape=(latent_dim,), name="generator_input")
    __x = generator(input)
    output = discriminator(__x)
    return Model([input], [output], name="pggan")

def __check_and_load_weight(model):
    fname = weight_path / Path(model.name + ".hdf5")
    if fname.exists():
        model.load_weights(fname.name)
    return model

def load_gen_weight(generator):
    for i, l in enumerate(generator.layers):
        if "block" in l.name:
            for j, m in enumerate(l.layers):
                if "stage" in m.name:
                    generator.layers[i].layers[j] = __check_and_load_weight(m)
    return generator

def load_dis_weight(discriminator):
    for i, l in enumerate(discriminator.layers):
        if "load" in l.name:
            for j, m in enumerate(l.layers):
                if not "input" in m.name:
                    discriminator.layers[i].layers[j] = __check_and_load_weight(m)

        if "stage" in l.name or "final" in l.name:
            discriminator.layers[i] = __check_and_load_weight(l)
    return discriminator

def __save_weight(model):
    fname = weight_path / Path(model.name + ".hdf5")
    model.save_weights(fname.name)

def save_gen_weight(generator):
    for i, l in enumerate(generator.layers):
        if "block" in l.name:
            for j, m in enumerate(l.layers):
                if "stage" in m.name:
                    __save_weight(m)

def save_dis_weight(discriminator):
    for i, l in enumerate(discriminator.layers):
        if "load" in l.name:
            for j, m in enumerate(l.layers):
                if not "input" in m.name:
                    __save_weight(m)

        if "stage" in l.name or "final" in l.name:
            __save_weight(l)

def train(stage=1, num_epoch=100):
    res = 2**(stage+1)
    X_train, datagen = load_data(mode=mode, size=res)
    print(X_train.shape)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    d_opt = Adam(lr=1e-5, beta_1=0.1)
    g_opt = Adam(lr=2e-4, beta_1=0.5)

    print(f"Stage:{stage}, NUM_EPOCH:{num_epoch}")
    print('Number of batches:', num_batches)
    for epoch in range(num_epoch):
        alpha = epoch / float(num_epoch)

        discriminator = build_discriminator(stage, alpha=alpha)
        discriminator.trainable = True
        discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

        # generator+discriminator （discriminator部分の重みは固定）
        discriminator.trainable = False
        generator = build_generator(stage=stage, alpha=alpha)
        pggan = build_pggan(generator, discriminator)
        pggan.compile(loss='binary_crossentropy', optimizer=g_opt)

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
                generated_images_plot = generated_images_plot.reshape((BATCH_SIZE, res, res, 3)).astype("uint8")

                plt.figure(figsize=(8, 4))
                plt.suptitle('stage=%02d,epoch=%04d,index=%04d' % (stage, epoch, index), fontsize=20)
                for i in range(BATCH_SIZE):
                    plt.subplot(4, 8, i + 1)
                    plt.imshow(generated_images_plot[i])
                    # plt.gray()
                    # eliminate ticks
                    plt.xticks([]), plt.yticks([])

                # save images
                GENERATED_IMAGE_PATH.mkdir(parents=True, exist_ok=True)
                filename = GENERATED_IMAGE_PATH / Path(f"{mode}_%02d_%04d_%04d.png" % (stage,epoch,index))
                plt.savefig(filename)
                plt.close()

            # discriminatorを更新
            X = np.concatenate((image_batch, generated_images))
            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            # generatorを更新
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = pggan.train_on_batch(noise, [1]*BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))

            save_gen_weight(generator)
            save_dis_weight(discriminator)

if __name__ == '__main__':
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    KTF.set_session(session)

    for stage in range(1,FINAL_STAGE):
        num_epoch = min(100, 2**(stage + 1))
        train(stage=stage, num_epoch=num_epoch)