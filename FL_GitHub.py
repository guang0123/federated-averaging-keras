from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.datasets import fashion_mnist
from keras.optimizers import SGD
import keras.backend as KB

import numpy as np
import matplotlib.pyplot as plt

import os
import time


def LeNet(input_shape=(28, 28, 1), class_num=10):
    model = Sequential()
    model.add(Conv2D(6, (5, 5),
                     input_shape=input_shape,
                     strides=(1, 1),
                     padding='valid',
                     data_format='channels_last',
                     activation='relu',))  # [None,24,24,6]
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2)))  # [None,12,12,6]

    model.add(Conv2D(16, (5, 5),
                     strides=(1, 1),
                     padding='valid',
                     data_format='channels_last',
                     activation='relu',))  # [None,8,8,16]
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(2, 2))  # [None,4,4,16]

    model.add(Flatten(data_format='channels_last'))  # [None,256]
    model.add(Dense(168, activation='relu'))  # [None,168]
    model.add(Dense(84, activation='relu'))  # [None,84]
    model.add(Dense(class_num, activation='softmax',))  # [None,10]
    # model.summary()
    sgd = SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def generate_data(K, frac_norm):
    dataset = fashion_mnist.load_data()
    (X_train, Y_train), (X_test, Y_test) = dataset
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test = to_categorical(Y_test, num_classes=10)

    X_train_client = {}
    Y_train_client = {}
    start = 0
    for i in range(K):
        X_train_client[i] = X_train[start:start+frac[i]]
        Y_train_client[i] = Y_train[start:start+frac[i]]
        start = start+frac[i]
    return X_train_client, Y_train_client, X_test, Y_test


def draw(save_path):
    plt.figure()
    acc = np.loadtxt(save_path+'acc.csv', delimiter=',')
    plt.plot(acc)
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.savefig(save_path+'acc.png')


if __name__ == "__main__":
    K = 20  # total number of clients
    N = 4  # the nuumber of clients participated in each update
    frac = [1000]*10 + [2000]*10
    frac_norm = [i/sum(frac) for i in frac]  # weighted coefficient
    X_train_client, Y_train_client, X_test, Y_test = generate_data(
        K, frac_norm)
    out_epoch = 1000  # the total epoch for global model average
    in_epoch = 5  # the epoch for local model update

    save_path = 'result/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    all_model = []
    for i in range(K):
        all_model.append(LeNet())

    all_model[-1].save(save_path+'init_model.h5')

    for i in range(K):
        all_model[i] = load_model(save_path+'init_model.h5')

    print('----BEGIN')
    all_weights = [model.get_weights() for model in all_model]
    acc_list = []
    loss_list = []
    norm_list = []

    for t in range(out_epoch):
        #if (t+1) % 5 == 0:
        #    lr = KB.get_value(all_model[0].optimizer.lr)
        #    for i in range(K):
        #        KB.set_value(all_model[i].optimizer.lr, lr * 0.5)

        arm = np.random.choice(K, size=N, replace=False, p=frac_norm)
        start_time = time.clock()
        for a in arm:
            all_model[a].fit(x=X_train_client[a], y=Y_train_client[a],
                             batch_size=32, epochs=in_epoch, verbose=1)
        print('**TRAIN, ', time.clock() - start_time)

        start_time = time.clock()
        for a in arm:
            all_weights[a] = all_model[a].get_weights()
        print('**GET, ', time.clock() - start_time)

        start_time = time.clock()
        layer_num = len(all_weights[0])
        ave_weights = []
        for layer in range(layer_num):
            init = np.zeros_like(all_weights[0][layer])
            for index, model in enumerate(all_weights):
                init += frac_norm[index]*model[layer]
            ave_weights.append(init)
        print('**AVE, ', time.clock() - start_time)

        start_time = time.clock()
        for i in range(K):
            all_model[i].set_weights(ave_weights)
        print('**SET, ', time.clock() - start_time)

        start_time = time.clock()
        loss, acc = all_model[0].evaluate(x=X_test, y=Y_test)
        print('**TEST, ', time.clock() - start_time)

        loss_list.append(loss)
        acc_list.append(acc)
        norm_list.append([np.linalg.norm(i) for i in ave_weights])
        print('----', t, arm, loss, acc)
        print(norm_list[-1])

        if t % 10 == 0:
            np.savetxt(save_path+'loss.csv', loss_list, delimiter=',')
            np.savetxt(save_path+'acc.csv', acc_list, delimiter=',')
            np.savetxt(save_path+'norm.csv', norm_list, delimiter=',')
            all_model[0].save(save_path+'model_'+str(t)+'.h5')

    np.savetxt(save_path+'loss.csv', loss_list, delimiter=',')
    np.savetxt(save_path+'acc.csv', acc_list, delimiter=',')
    all_model[0].save(save_path+'model.h5')
    np.savetxt(save_path+'norm.csv', norm_list, delimiter=',')

    draw(save_path)
