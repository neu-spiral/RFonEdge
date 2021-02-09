from scipy.fftpack import fft
import scipy.io as spio
import pickle
import numpy as np


def read_file_mat_fft(file):
        # Real hard work here
    mat_data = spio.loadmat(file)
    if mat_data.has_key('complexSignal'):
        complex_data = mat_data['complexSignal']  # try to use views here also
    elif mat_data.has_key('f_sig'):
        complex_data = mat_data['f_sig']
    complex_data = fft(complex_data)
    real_data = np.reshape(complex_data.real, (complex_data.shape[1], 1))
    imag_data = np.reshape(complex_data.imag, (complex_data.shape[1], 1))
    samples_in_example =  real_data.shape[0]
    ex_data = np.concatenate((real_data,imag_data), axis=1)
    return ex_data, samples_in_example


def read_file_mat(file, fft=False):
    # Real hard work here
    mat_data = spio.loadmat(file)
    if 'complexSignal' in mat_data:
        complex_data = mat_data['complexSignal']  # try to use views here also
    elif 'f_sig' in mat_data:
        complex_data = mat_data['f_sig']
    
    real_data = np.reshape(complex_data.real, (complex_data.shape[1], 1))
    imag_data = np.reshape(complex_data.imag, (complex_data.shape[1], 1))
    samples_in_example =  real_data.shape[0]
    ex_data = np.concatenate((real_data,imag_data), axis=1)
    return ex_data, samples_in_example


def read_file_mat_2d(file):
    # Real hard work here
    mat_data = spio.loadmat(file)
    if mat_data.has_key('complexSignal'):
        complex_data = mat_data['complexSignal']  # try to use views here also
    elif mat_data.has_key('f_sig'):
        complex_data = mat_data['f_sig']
    
    real_data = np.reshape(complex_data.real, (complex_data.shape[1], 1, 1))
    imag_data = np.reshape(complex_data.imag, (complex_data.shape[1], 1, 1))
    samples_in_example =  real_data.shape[0]
    ex_data = np.concatenate((real_data,imag_data), axis=1)
    return ex_data, samples_in_example


def read_file_mat_preamble(file, fft=False):
    # Real hard work here
    mat_data = spio.loadmat(file)
    if mat_data.has_key('complexSignal'):
        complex_data = mat_data['complexSignal']  # try to use views here also
    elif mat_data.has_key('f_sig'):
        complex_data = mat_data['f_sig']
    
    real_data = np.reshape(complex_data.real, (complex_data.shape[1], 1))
    imag_data = np.reshape(complex_data.imag, (complex_data.shape[1], 1))
    samples_in_example =  real_data.shape[0]
    ex_data = np.concatenate((real_data,imag_data), axis=1)
    
    sample_rate = mat_data['fs']
#    if sample_rate == 200000000:
    # preamble is located between 2us and 18 us
    # preamble length is 3200 at 200MS/s sample rate, and 320 at 20MS/s
    preamble_start = int(sample_rate/1E6 * 2) # 400 # 
    preamble_end = preamble_start + 320 # int(sample_rate/1E6 * 18)
    
    # first half of preamble is short training sequence and second half is long training sequence
    preamble = ex_data[preamble_start:preamble_end,:]
    ex_data = ex_data[preamble_end:,:]
    samples_in_example =  ex_data.shape[0]
    
    return ex_data, preamble, samples_in_example
#    else:
#        print("Sample rate is %d" % sample_rate)
#        return None, None, 0


def read_file(file):
    # Real hard work here
    with open(file,'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        pickle_data = u.load()
    #pickle_data = pickle.load(open(file, 'rb'))
    key_len = len(pickle_data.keys())
    if key_len == 1:
        #complex_data = pickle_data[pickle_data.keys()[0]]
        complex_data = pickle_data[next(iter(pickle_data))]
        
    elif key_len == 0:
        return None, 0
    else:
        # TODO: add support to 'result' folder
        raise Exception("{} {} Key length not equal to 1!".format(file, str(pickle_data.keys())))
        pass

    if complex_data.shape[0] == 0:
        # print complex_data.shape
        return None, 0

    real_data = np.expand_dims(complex_data.real, axis=1)
    imag_data = np.expand_dims(complex_data.imag, axis=1)
    samples_in_example =  real_data.shape[0]
    ex_data = np.concatenate((real_data,imag_data), axis=1)
    return ex_data, samples_in_example
