import numpy as np
import collections
from tqdm import tqdm
import math
import torch
def dataGeneratorWithProcessor(ex, slice_size, processor=None, test_stride=1, file_type='mat', normalize=False, mean_val=None, std_val=None, add_padding=False, padding_type='zero', crop=0, use_preamble=False, conv2d=False):
    stride = test_stride
    if file_type.lower() == 'mat':
        if use_preamble:
            from file_reader import read_file_mat_preamble as read_file
        elif conv2d:
            from file_reader import read_file_mat_2d as read_file
        else:
            from file_reader import read_file_mat as read_file
    elif file_type.lower() == 'pickle':
        from file_reader import read_file
    else:
        raise Exception('File not support!')
    if use_preamble:
        ex_data, preamble, samples_in_example = read_file(ex)
    else:
        ex_data, samples_in_example = read_file(ex)
    
    if ex_data is not None and samples_in_example > crop and crop > 0:
        samples_in_example = crop
        ex_data = ex_data[0:crop, :]

    if samples_in_example <= slice_size + 1:
        if add_padding and ex_data is not None and samples_in_example > 0:
            if padding_type == 'zero':
                # method 1: zero padding
                pad_len = slice_size + 2 - samples_in_example
                ex_data = np.pad(ex_data , ((int(math.floor(pad_len/2)), int(math.ceil(pad_len/2))), (0,0)), 'constant', constant_values=0)
            elif padding_type == 'stride':
                # method 2: stride concatenation
                l=int(np.sqrt(slice_size))
                slice_size_new = slice_size+2
                stride = l*(samples_in_example-l)/(slice_size_new-l)
                num_stride = int(np.ceil((samples_in_example-l)/stride) + 1)
                stride = np.round(stride)
                idx = np.array([])
                for i in range(num_stride):
                    idx = np.hstack((idx,np.arange(l)+stride*i)) if idx.size else np.arange(l)+stride*i
                offset = slice_size_new-idx.shape[0]
                idx = np.hstack((idx,np.arange(slice_size_new-offset,slice_size_new))) if offset > 0 else idx[:slice_size_new]
                idx[idx >= samples_in_example] = samples_in_example - 1
                ex_data = ex_data[idx]
                
            samples_in_example = ex_data.shape[0]
        else:
            return None
    
    if normalize is True:
        if mean_val is None:
            mean_val = 0.0
        if std_val is None:
            std_val = 1.0
        
        if not conv2d:
            ex_data = (ex_data - mean_val) / std_val
        else:
            ex_data = (ex_data - np.mean(mean_val)) / np.mean(std_val)
        if use_preamble:
            preamble = (preamble - mean_val) / std_val

    num_slices = int(math.floor((samples_in_example - slice_size)/stride + 1))
    if not conv2d:
        X = np.zeros((num_slices, slice_size, 2), dtype='float32')
    else:
        X = np.zeros((num_slices, slice_size, 2, 1), dtype='float32')
    if use_preamble:
        P = np.zeros((num_slices, preamble.shape[0], 2), dtype='float32')
    for s in range(num_slices):
        # Here we should use the normalization method in NewDataGenerator
        IQ_slice = ex_data[s*stride: s*stride + slice_size, :]
        if not conv2d:
            X[s,:,:] = IQ_slice
        else:
            X[s,:,:,:] = IQ_slice
        if use_preamble:
            P[s,:,:] = preamble
    
    X_processed = X
    if use_preamble:
        preamble_processed = P
   
    if use_preamble:
        return [X_processed, preamble_processed]
    else:
        return X_processed
    
def compute_accuracy(ex_list, labels, device_ids, slice_size, model, batch_size=16, vote_type='majority', processor=None, test_stride=1, file_type='mat', normalize=False, mean_val=None, std_val=None, add_padding=False, padding_type='zero',crop=0, use_preamble=False, conv2d=False):
    total_examples = 0
    correct_examples = 0
    total_slices = 0
    correct_slices = 0
    preds_slice = {}
    preds_exp = {}

    model.eval()
    for ex in tqdm(ex_list):
        real_label = device_ids[labels[ex]]
        if use_preamble:
            [X, preamble] = dataGeneratorWithProcessor(ex, slice_size, processor, test_stride, file_type, normalize=normalize, mean_val=mean_val, std_val=std_val, add_padding=add_padding, padding_type=padding_type, crop=crop, use_preamble=use_preamble)
        else:
            X = dataGeneratorWithProcessor(ex, slice_size, processor, test_stride, file_type, normalize=normalize, mean_val=mean_val, std_val=std_val, add_padding=add_padding, padding_type=padding_type, crop=crop, conv2d=conv2d)

        if X is None:
            continue
            
        # TODO: HERE we could make it fit to the batch size
        num_slices = X.shape[0]
        #total_slices = total_slices + num_slices
        
        num_batches = int(np.ceil(float(num_slices)/float(batch_size)))
        preds = np.zeros((num_slices,len(device_ids)))
        for i in range(num_batches):
            X_batch = X[i*batch_size:min((i+1)*batch_size,num_slices),:,:]
            data = torch.from_numpy(np.array(X_batch.transpose(0,2,1))).cuda()
            
            preds[i*batch_size:min((i+1)*batch_size,num_slices),:] = model(data).cpu().data.numpy()
            
        #print 'preds: ', preds
        # just for test
        Y = np.argmax(preds, axis=1)

        slices_in_example = X.shape[0]
        counter_dict = collections.Counter(Y)
        correct_in_example = counter_dict[real_label]
        
        total_slices = total_slices + slices_in_example
        total_examples = total_examples + 1

        correct_slices += correct_in_example
        if vote_type == 'majority':
            if counter_dict.most_common(1)[0][0] == real_label:
                correct_examples = correct_examples + 1
                preds_exp[ex] = 1
            else:
                preds_exp[ex] = 0
        if vote_type == 'prob_sum':
            tot_prob = preds.sum(axis=0)
            #print 'tot_prob: ', tot_prob
            predicted_class = np.argmax(tot_prob)
            if predicted_class == real_label:
                correct_examples = correct_examples + 1
            preds_exp[ex] = [real_label,predicted_class,tot_prob] 
        if vote_type == 'log_prob_sum': # it is probably different, but I don0t remember the formula...
            log_preds = np.log(preds)
            tot_prob = log_preds.sum(axis=0)
            predicted_class = np.argmax(tot_prob)
            if predicted_class == real_label:
                correct_examples = correct_examples + 1
                preds_exp[ex] = 1
            else:
                preds_exp[ex] = 0
        preds_slice[ex] = (slices_in_example,correct_in_example)
    
    Preds = {}
    Preds['preds_slice'] = preds_slice
    Preds['preds_exp'] = preds_exp
    acc_slice = float(correct_slices)/float(total_slices)
    acc_ex = float(correct_examples)/float(total_examples)
    
    # print(total_slices)
    # print ("Per-slice accuracy: %.3f ; Per-example accuracy: %.3f" %(acc_slice, acc_ex))
    return acc_slice, acc_ex, Preds


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="test script for models evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s', '--slice_size', type=int, default=1024)
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('-nf', '--num_filters', type=int, default=128)
    parser.add_argument('-wp', '--weights_path', default='/scratch/luca.angioloni/conv1d/')
    parser.add_argument('-vt', '--vote_type', default='majority')

    args = parser.parse_args()

    import keras.models as models
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.layers.convolutional import Conv1D, MaxPooling1D
    from keras.optimizers import Adam

    from keras.utils import multi_gpu_model

    import tensorflow as tf

    import pickle

    def getModel(slice_size, classes):
        """A dummy model to test the functionalities of the Data Generator"""
        model = models.Sequential()
        model.add(Conv1D(args.num_filters,7,activation='relu', padding='same', input_shape=(slice_size, 2)))
        model.add(Conv1D(args.num_filters,5,activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(args.num_filters,7,activation='relu', padding='same'))
        model.add(Conv1D(args.num_filters,5,activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(args.num_filters,7,activation='relu', padding='same'))
        model.add(Conv1D(args.num_filters,5,activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()

        return model

    # Take one random 1k dataset just to test the Data Generator
    base_path = "/scratch/RFMLS/dataset10K/random_1K/0/"
    stats_path = '/scratch/RFMLS/dataset10KStats/random_1K/0/'

    weights_path = args.weights_path

    file = open(base_path + "label.pkl",'r')
    labels = pickle.load(file)

    file = open(stats_path + "/device_ids.pkl", 'r')
    device_ids = pickle.load(file)

    file = open(stats_path + "/stats.pkl", 'r')
    stats = pickle.load(file)

    file = open(base_path + "partition.pkl",'r')
    partition = pickle.load(file)

    ex_list = partition['train']
    val_list = partition['val']
    test_list = partition['test']

    slice_size = args.slice_size
    batch_size = 256  # this is a reasonable batch size for our hardware, singe this batch will be slit in 8 GPUs, each one with batch size 128
    K = args.k
    files_per_IO = 500000  # this is more or less the number of files that the discovery cluster can fit into RAM (we sould check better)

    # if you want to quick test that it all works with a small portion of the dataset
    #
    # ex_list = ex_list[0:int(len(ex_list)*0.01)]
    # val_list = val_list[0:int(len(val_list)*0.01)]

    # Transform the model so that it can run on multiple GPUs
    cpu_net = None
    with tf.device("/cpu:0"):
        cpu_net = getModel(slice_size, len(device_ids))  # the model should live into the CPU and is then updated with the results of the GPUs

    cpu_net.load_weights(weights_path)

    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    net = multi_gpu_model(cpu_net, gpus=8)
    net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # The model needs to be compiled again for the multi gpu process

    acc_slice, acc_ex, preds = compute_accuracy(test_list, labels, device_ids, slice_size, K, net, vote_type=args.vote_type)
    print ("Per-slice accuracy: %.3f ; Per-example accuracy: %.3f" %(acc_slice, acc_ex))

