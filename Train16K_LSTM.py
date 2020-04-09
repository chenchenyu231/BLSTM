#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #GPU 3
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot
import librosa
import Preprocess as pre
import hdf5storage
import numpy as np
import time
import sys
import tensorflow as tf
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential,Model, model_from_json
from keras.layers import Activation,Dense,Dropout,Flatten,ConvLSTM2D,LSTM,TimeDistributed,Input,Bidirectional,Conv1D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam, Nadam, Adamax
import glob
from tqdm import tqdm
from natsort import natsorted
# from keras.utils import print_summary
# G_RAM Control
# Gpu_Control = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5) # Assign 50% G_RAM in this program
Gpu_Control = tf.GPUOptions(allow_growth=True) # G_RAM will auto control it's range
sess = tf.Session(config = tf.ConfigProto(gpu_options=Gpu_Control))

tf.keras.backend.set_session(sess)

def before_train(train_A_dir, train_B_dir, model_dir, output_dir, tensorboard_log_dir):
    
    sampling_rate = 16000
    num_mcep = 80 #24
    frame_period = 5.0

    print('Preprocessing Data...')

    start_time = time.time()
    
    # list_A= pre.load_data_list(train_A_dir)
    # list_B= pre.load_data_list(train_B_dir)
    # print(list_A[1])
    train_A_dir=str(train_A_dir)+'\*\*.wav'    #當層 \*.wav 內有一層\*\*.wav
    train_B_dir=str(train_B_dir)+'\*\*.wav'
    list_A=glob.glob(train_A_dir)
    list_B=glob.glob(train_B_dir)
    tta=natsorted(list_A)
    ttb=natsorted(list_B)
    
    wavs_A = pre.load_wavs(wav_list = tta, sr = sampling_rate)   
    wavs_B = pre.load_wavs(wav_list = ttb, sr = sampling_rate)

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = pre.world_encode_data(wavs = wavs_A, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
    log_f0s_mean_A, log_f0s_std_A = pre.logf0_statistics(f0s_A)
    print('Log Pitch A')
    print('Mean: %f, Std: %f' %(log_f0s_mean_A, log_f0s_std_A))
    
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = pre.world_encode_data(wavs = wavs_B, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
    log_f0s_mean_B, log_f0s_std_B = pre.logf0_statistics(f0s_B)
    print('Log Pitch B')
    print('Mean: %f, Std: %f' %(log_f0s_mean_B, log_f0s_std_B))


    coded_sps_A_transposed = pre.transpose_in_list(lst = coded_sps_A)
    coded_sps_B_transposed = pre.transpose_in_list(lst = coded_sps_B)
    
    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = pre.coded_sps_normalization_fit_transoform(coded_sps = coded_sps_A_transposed)
    print("Input data fixed.")
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = pre.coded_sps_normalization_fit_transoform(coded_sps = coded_sps_B_transposed)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A = log_f0s_mean_A, std_A = log_f0s_std_A, mean_B = log_f0s_mean_B, std_B = log_f0s_std_B)
    np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A = coded_sps_A_mean, std_A = coded_sps_A_std, mean_B = coded_sps_B_mean, std_B = coded_sps_B_std)

    end_time = time.time()
    time_elapsed = end_time - start_time
    
    
    Para_name=['sampling_rate', 'num_mcep', 'frame_period',
               'coded_sps_A_norm', 'coded_sps_B_norm', 'coded_sps_A', 'coded_sps_B',
               'coded_sps_A_mean', 'coded_sps_A_std', 'coded_sps_B_mean', 'coded_sps_B_std',
               'log_f0s_mean_A', 'log_f0s_std_A', 'log_f0s_mean_B', 'log_f0s_std_B']
    
#    Para_num=len(Para_name) 
    Local_Var=locals()
    Para_data=[Local_Var[para_index] for para_index in Para_name]
    
    Para=dict(zip(Para_name, Para_data))
        
    print('Preprocessing Done.')
    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    return Para

def mini_train(Para):

    np.random.seed(random_seed)
    globals().update(Para)
    n_frames = 128 #origin 128

    dataset_A, dataset_B = pre.sample_data(dataset_A = coded_sps_A_norm, 
                                               dataset_B = coded_sps_B_norm,
                                               frames_per_sample = n_frames,
                                               s_type='parallel',
                                               perm=[1,0])
    return dataset_A,dataset_B


def train(Para,num_epochs,path,validation_A_dir,output_dir,model_name):

   
    n_frames = 128
    dataset_A, dataset_B = pre.sample_data(dataset_A = Para["coded_sps_A_norm"], 
                                               dataset_B = Para["coded_sps_B_norm"],
                                               frames_per_sample = n_frames,
                                               s_type='parallel',
                                               perm=[1,0])
##validation 
    n_samples = dataset_A.shape[0]
    



##build model
    # model = Sequential()
    #Inputs = Input(shape=(80,None))
    Inputs = Input(shape=(None,80))
    
    a= Conv1D(128,kernel_size=5, strides=2,activation='relu')(Inputs)
    b= Conv1D(256,kernel_size=5, strides=2,activation='relu')(a)
    c= Conv1D(512,kernel_size=5, strides=2,activation='relu')(b)
    x = Bidirectional(LSTM(512, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True, return_state=False, stateful=False))(c)
    y = Bidirectional(LSTM(512, activation='relu', recurrent_activation='hard_sigmoid', return_sequences=True, return_state=False, stateful=False))(x)
    
    d= Conv1D(1024,kernel_size=5, strides=2,activation='relu')(y)
    e= Conv1D(512,kernel_size=5, strides=2,activation='relu')(d)
    f= Conv1D(80,kernel_size=5, strides=2,activation='linear')(e)
 #   Outputs = Dense(80, activation = 'linear')(d)

 #   model = Model(inputs=Inputs, outputs=Outputs)
    model = Model(inputs=Inputs, outputs=f)
    print(model.summary())
    
    
    # model.add(Bidirectional(LSTM(128,input_dim=(None, return_sequences=True)))
    # model.add(Bidirectional(LSTM(64, return_sequences=True)))
    
    
    sgd=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False);
    #sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0);
    adagrad=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0);
    adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0);
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0);
    adamax=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0);
    nadam=Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004);

    OptimIz=nadam;

    model.compile(
	loss='mean_squared_error',   #mean_squared_error
	optimizer=OptimIz,
	metrics=['accuracy'])
						
    checkpointer = ModelCheckpoint(
        filepath=path+model_name+".hdf5",
        monitor="loss",
        mode="min",
        verbose=1,
        save_best_only=True)
    

    StartTime= time.time()

    print('indata.shape:',dataset_A.shape)
    print('outdata.shape:',dataset_B.shape)
    data_A=dataset_A.reshape(dataset_A.shape[0],dataset_A.shape[2],dataset_A.shape[1])
    data_B=dataset_B.reshape(dataset_B.shape[0],dataset_B.shape[2],dataset_B.shape[1])
    Hist=model.fit(data_A, data_B, batch_size=1, epochs=num_epochs,verbose=1,callbacks=[checkpointer],validation_split=0.1,shuffle=False)
    
    #do waveform reconstruction
    sampling_rate = 16000
    num_mcep = 80 #24
    frame_period = 5.0
    
    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A')
        if not os.path.exists(validation_A_output_dir):
            os.makedirs(validation_A_output_dir)
    
    test_A_dir=str(validation_A_dir)+'\*\*.wav'
    Eva_list_A=glob.glob(test_A_dir) 
    for filepath in tqdm(Eva_list_A,desc='Generating'):
        filedir=os.path.basename(os.path.dirname(filepath))
        outpath=os.path.join(validation_A_output_dir,filedir)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
                        
        wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
        wav = pre.wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
        f0, timeaxis, sp, ap = pre.world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
        f0_converted = pre.pitch_conversion(f0 = f0, mean_log_src = Para["log_f0s_mean_A"], std_log_src = Para["log_f0s_std_A"], mean_log_target = Para["log_f0s_mean_B"], std_log_target = Para["log_f0s_std_B"])
        coded_sp = pre.world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
        coded_sp_transposed = coded_sp.T
        coded_sp_norm = (coded_sp_transposed - Para["coded_sps_A_mean"]) / Para["coded_sps_A_std"]
        data_Tes=np.array([coded_sp_norm])
        
        data_Tes_new=data_Tes.reshape(data_Tes.shape[0],data_Tes.shape[2],data_Tes.shape[1])
        data_Ans= model.predict(data_Tes_new, batch_size=1, verbose=1, steps=None)
        #data_Ans = model.test(inputs = data_Tes, direction = 'A2B')[0]
        
        coded_sp_converted_norm = data_Ans
        
        coded_sp_converted_norm_new=coded_sp_converted_norm.reshape(coded_sp_converted_norm.shape[2],coded_sp_converted_norm.shape[1])
                        
        coded_sp_converted = coded_sp_converted_norm_new * Para["coded_sps_B_std"] + Para["coded_sps_B_mean"]
        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = pre.world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
        wav_transformed = pre.world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                    
        librosa.output.write_wav(os.path.join(outpath,os.path.basename(filepath)),wav_transformed, sampling_rate)
        
        
   
    
    
    
    
    print(model.summary())
    with open(path+model_name+".json", "w") as f:
        f.write(model.to_json())

    EndTime = time.time()
    print('time is {} sec'.format(EndTime-StartTime))


if __name__ == '__main__':


    train_A_dir = r'C:\Users\chenyu\Desktop\Voice_Converter_CycleGAN-stable - 複製\data\Source'
    train_B_dir = r'C:\Users\chenyu\Desktop\Voice_Converter_CycleGAN-stable - 複製\data\Target'
    model_dir = './model/debug_lstm/'
    model_name = 'BlSTM_Wang_0408.ckpt'
    initial_model = None
    num_epochs = 100
    random_seed = 0
    validation_A_dir = r'C:\Users\chenyu\Desktop\Voice_Converter_CycleGAN-stable - 複製\data\Test'#None if argv.validation_A_dir == 'None' or argv.validation_A_dir == 'none' else argv.validation_A_dir
    #validation_B_dir = argv.validation_B_dir#None if argv.validation_B_dir == 'None' or argv.validation_B_dir == 'none' else argv.validation_B_dir
    output_dir = '.\outputBLSTM_Wang0408'
    tensorboard_log_dir = './log'
    
   
    Pre_Data=before_train(train_A_dir = train_A_dir,
                          train_B_dir = train_B_dir,
                          model_dir = model_dir,
                          output_dir = output_dir,
                          tensorboard_log_dir = tensorboard_log_dir)
    
    #dataset_A, dataset_B = mini_train(Para=Pre_Data)
    
    train(Para=Pre_Data,num_epochs = num_epochs,path=model_dir,validation_A_dir=validation_A_dir,output_dir = output_dir,model_name=model_name)