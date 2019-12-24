"""
Associative Compression Network based on https://arxiv.org/pdf/1804.02476v2.pdf

Strongly referenced ACN implementation and blog post from:
http://jalexvig.github.io/blog/associative-compression-networks/

Base VAE referenced from pytorch examples:
https://github.com/pytorch/examples/blob/master/vae/main.py
"""

# TODO conv
# TODO load function
# daydream function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
from glob import glob
import numpy as np
from copy import deepcopy, copy

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import transforms
from torch.nn.utils.clip_grad import clip_grad_value_

from utils import create_new_info_dict, save_checkpoint, create_mnist_datasets, seed_everything
from utils import plot_example, plot_losses, count_parameters
from utils import set_model_mode, kl_loss_function, write_log_files
from utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic

from pixel_cnn import GatedPixelCNN
from acn_models import VQVAE
from IPython import embed


def create_conv_acn_pcnn_models(info, model_loadpath='', dataset_name='FashionMNIST'):
    '''
    load details of previous model if applicable, otherwise create new models
    '''
    train_cnt = 0
    epoch_cnt = 0

    # use argparse device no matter what info dict is loaded
    preserve_args = ['device', 'batch_size', 'save_every_epochs',
                     'base_filepath', 'model_loadpath', 'perplexity',
                     'use_pred']
    largs = info['args']
    # load model if given a path
    if model_loadpath !='':
        _dict = torch.load(model_loadpath, map_location=lambda storage, loc:storage)
        dinfo = _dict['info']
        pkeys = info.keys()
        for key in dinfo.keys():
            if key not in preserve_args or key not in pkeys:
                info[key] = dinfo[key]
        train_cnt = info['train_cnts'][-1]
        epoch_cnt = info['epoch_cnt']
        info['args'].append(largs)

    # setup loss-specific parameters for data
    # data going into dml should be bt -1 and 1
    rescale = lambda x: (x - 0.5) * 2.
    rescale_inv = lambda x: (0.5 * x) + 0.5


    # transform is dependent on loss type
    dataset_transforms = transforms.Compose([transforms.ToTensor(), rescale])
    data_output = create_mnist_datasets(dataset_name=info['dataset_name'],
                                                     base_datadir=info['base_datadir'],
                                                     batch_size=info['batch_size'],
                                                     dataset_transforms=dataset_transforms)
    data_dict, size_training_set, num_input_chans, num_output_chans, hsize, wsize = data_output
    info['size_training_set'] = size_training_set
    info['num_input_chans'] = num_input_chans
    info['num_output_chans'] = num_input_chans
    info['hsize'] = hsize
    info['wsize'] = wsize

    # pixel cnn architecture is dependent on loss
    # for dml prediction, need to output mixture of size nmix
    info['nmix'] =  (2*info['nr_logistic_mix']+info['nr_logistic_mix'])*info['num_output_chans']
    info['output_dim']  = info['nmix']
    # last layer for pcnn
    info['last_layer_bias'] = 0.0

    # setup models
    # acn prior with vqvae embedding
    vq_conv_model = VQVAE(
                          input_size=info['input_channels'],
                          output_size=info['output_dim'],
                          encoder_output_size=info['encoder_output_size'],
                          last_layer_bias=info['last_layer_bias'],
                          num_clusters=info['num_vqk'],
                          num_z=info['num_z'],
                               ).to(info['device'])

    model_dict = {'vq_conv_model':vq_conv_model}
    parameters = []
    for name,model in model_dict.items():
        parameters+=list(model.parameters())
        print('created %s model with %s parameters' %(name,count_parameters(model)))

    model_dict['opt'] = optim.Adam(parameters, lr=info['learning_rate'])

    if args.model_loadpath !='':
       for name,model in model_dict.items():
            model_dict[name].load_state_dict(_dict[name+'_state_dict'])
    return model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv

def clip_parameters(model_dict, clip_val=10):
    for name, model in model_dict.items():
        if 'model' in name:
            clip_grad_value_(model.parameters(), clip_val)
    return model_dict

def run(train_cnt, model_dict, data_dict, phase, device, rec_loss_type, dropout_rate):
    st = time.time()
    run = rec_running = loss_running = vq_running = commit_running =  0.0
    data_loader = data_dict[phase]
    model_dict = set_model_mode(model_dict, phase)
    vq_model = model_dict['vq_conv_model']
    num_batches = len(data_loader)
    for idx, (data, label, batch_index) in enumerate(data_loader):
        target = data = data.to(device)
        bs,c,h,w = target.shape
        model_dict['opt'].zero_grad()
        # dropout on input is different than what I've done in the pcnn_acn
        # model - there I only did dropout on the pcnn_decoder tf
        data = F.dropout(data, p=dropout_rate, training=True, inplace=False)
        x_d, z_e_x, z_q_x, latents = vq_model(data)
        # latents are 84x4x4
        # z is 84x64
        # input into dml should be bt -1 and 1
        # in sum-based dml FashionMNIST that works at 240k examples the
        # kl is at ~8 and dml_loss is around 2100
        rec_loss = discretized_mix_logistic_loss(x_d, target, nr_mix=info['nr_logistic_mix'], reduction=info['reduction'])

        rec_loss.backward(retain_graph=True)
        # encourage vq embedding space to be good
        vq_loss = F.mse_loss(z_q_x, z_e_x.detach(), reduction=info['reduction'])
        commit_loss = F.mse_loss(z_e_x, z_q_x.detach(), reduction=info['reduction'])
        commit_loss *= info['vq_commitment_beta']

        loss = rec_loss+vq_loss+commit_loss
        if phase == 'train':
            clip_grad_value_(vq_model.parameters(), 5)
            loss.backward(retain_graph=True)
            model_dict['opt'].step()
        run+=bs
        rec_running+=rec_loss.item()
        loss_running+=loss.item()
        vq_running+=vq_loss.item()
        commit_running+=commit_loss.item()
        # add batch size because it hasn't been added to train cnt yet
        if phase == 'train':
            train_cnt+=bs
        if idx == num_batches-2:
            # store example near end for plotting
            yhat_batch = sample_from_discretized_mix_logistic(x_d, info['nr_logistic_mix'])
            example = {'data':data.detach().cpu(), 'target':target.detach().cpu(), 'yhat':yhat_batch.detach().cpu()}
        if not idx % 10:
            loss_avg = {rec_loss_type:rec_running/run,
                        'loss':loss_running/run,
                        'vq':vq_running/run, 'commit':commit_running/run}
            print(train_cnt, idx, loss_avg)

    model_dict['vq_conv_model'] = vq_model
    # store average loss for return
    loss_avg = {rec_loss_type:rec_running/run,
                        'loss':loss_running/run,
                        'vq':vq_running/run, 'commit':commit_running/run}
    print("finished %s after %s secs at cnt %s"%(phase,
                                                time.time()-st,
                                                train_cnt,
                                                ))
    print(loss_avg)
    return model_dict, data_dict, loss_avg, example

def train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info, rescale_inv):
    print('starting training routine')
    base_filepath = info['base_filepath']
    base_filename = os.path.split(info['base_filepath'])[1]
    while train_cnt < info['num_examples_to_train']:
        print('starting epoch %s on %s'%(epoch_cnt, info['device']))
        model_dict, data_dict, train_loss_avg, train_example = run(train_cnt,
                                                                       model_dict,
                                                                       data_dict,
                                                                       phase='train',
                                                                       device=info['device'],
                                                                       rec_loss_type=info['rec_loss_type'],
                                                                       dropout_rate=info['dropout_rate'])
        epoch_cnt +=1
        train_cnt +=info['size_training_set']
        if not epoch_cnt % info['save_every_epochs'] or epoch_cnt == 1:
            # make a checkpoint
            print('starting valid phase')
            model_dict, data_dict, valid_loss_avg, valid_example = run(train_cnt,
                                                                           model_dict,
                                                                           data_dict,
                                                                           phase='valid',
                                                                           device=info['device'],
                                                                           rec_loss_type=info['rec_loss_type'],
                                                                           dropout_rate=info['dropout_rate'])
            for loss_key in valid_loss_avg.keys():
                for lphase in ['train_losses', 'valid_losses']:
                    if loss_key not in info[lphase].keys():
                        info[lphase][loss_key] = []
                info['valid_losses'][loss_key].append(valid_loss_avg[loss_key])
                info['train_losses'][loss_key].append(train_loss_avg[loss_key])

            # store model
            state_dict = {}
            for key, model in model_dict.items():
                state_dict[key+'_state_dict'] = model.state_dict()

            info['train_cnts'].append(train_cnt)
            info['epoch_cnt'] = epoch_cnt
            state_dict['info'] = info

            ckpt_filepath = os.path.join(base_filepath, "%s_%010dex.pt"%(base_filename, train_cnt))
            train_img_filepath = os.path.join(base_filepath,"%s_%010d_train_rec.png"%(base_filename, train_cnt))
            valid_img_filepath = os.path.join(base_filepath, "%s_%010d_valid_rec.png"%(base_filename, train_cnt))
            plot_filepath = os.path.join(base_filepath, "%s_%010dloss.png"%(base_filename, train_cnt))
            train_example['target'] = rescale_inv(train_example['target'])
            train_example['yhat'] = rescale_inv(train_example['yhat'])
            plot_example(train_img_filepath, train_example, num_plot=10)
            plot_example(valid_img_filepath, valid_example, num_plot=10)
            save_checkpoint(state_dict, filename=ckpt_filepath)

            plot_losses(info['train_cnts'],
                        info['train_losses'],
                        info['valid_losses'], name=plot_filepath, rolling_length=1)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='train acn')
    # operatation options
    parser.add_argument('-l', '--model_loadpath', default='', help='load model to resume training or sample')
    parser.add_argument('-ll', '--load_last_model', default='',  help='load last model from directory from directory')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=394)
    parser.add_argument('--num_threads', default=2)
    parser.add_argument('-se', '--save_every_epochs', default=10, type=int)
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--input_channels', default=1, type=int, help='num of channels of input')
    parser.add_argument('--target_channels', default=1, type=int, help='num of channels of target')
    parser.add_argument('--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-e', '--exp_name', default='vqvae', help='name of experiment')
    parser.add_argument('-dr', '--dropout_rate', default=0.0, type=float)
    # sum obviously trains on fashion mnist after < 1e6 examples, but it isn't
    # obvious to me at this point that mean will train (though it does on normal
    # mnist)
    parser.add_argument('-r', '--reduction', default='sum', type=str, choices=['sum', 'mean'])
    parser.add_argument('--output_projection_size', default=32, type=int)
    parser.add_argument('--rec_loss_type', default='dml', type=str, help='name of loss. options are dml', choices=['dml'])
    parser.add_argument('--nr_logistic_mix', default=10, type=int)
    # acn model setup
    parser.add_argument('-cl', '--code_length', default=64, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)

    #parser.add_argument('-kl', '--kl_beta', default=.5, type=float, help='scale kl loss')
    parser.add_argument('--last_layer_bias', default=0.0, help='bias for output decoder - should be 0 for dml')
    parser.add_argument('--encoder_output_size', default=2048, help='output as a result of the flatten of the encoder - found experimentally')
    #parser.add_argument('--encoder_output_size', default=6272, help='output as a result of the flatten of the encoder - found experimentally')
    parser.add_argument('-sm', '--sample_mean', action='store_true', default=False)
    # vq model setup
    parser.add_argument('--vq_commitment_beta', default=0.25, help='scale for loss 3 in vqvae - how hard to enforce commitment to cluster')
    parser.add_argument('--num_vqk', default=512, type=int)
    parser.add_argument('--num_z', default=64, type=int)
    # dataset setup
    parser.add_argument('-d',  '--dataset_name', default='FashionMNIST', help='which mnist to use', choices=['MNIST', 'FashionMNIST'])
    parser.add_argument('--model_savedir', default='../model_savedir', help='save checkpoints here')
    parser.add_argument('--base_datadir', default='../dataset/', help='save datasets here')
    # tsne info
    parser.add_argument('--tsne', action='store_true', default=False)
    parser.add_argument('-p', '--perplexity', default=10, type=int, help='perplexity used in scikit-learn tsne call')
    parser.add_argument('-ut', '--use_pred', default=False, action='store_true',  help='plot tsne with pred image instead of target')
    # walk-thru
    parser.add_argument('-w', '--walk', action='store_true', default=False, help='walk between two images in latent space')
    parser.add_argument('-st', '--start_label', default=0, type=int, help='start latent walk image from label')
    parser.add_argument('-ed', '--end_label', default=5, type=int, help='end latent walk image from label')
    parser.add_argument('-nw', '--num_walk_steps', default=40, type=int, help='number of steps in latent space between start and end image')
    args = parser.parse_args()
    # note - when reloading model, this will use the seed given in args - not
    # the original random seed
    seed_everything(args.seed, args.num_threads)
    # get naming scheme
    if args.load_last_model != '':
        # load last model from this dir
        base_filepath = args.load_last_model
        args.model_loadpath = sorted(glob(os.path.join(base_filepath, '*.pt')))[-1]
    elif args.model_loadpath != '':
        # use full path to model
        base_filepath = os.path.split(args.model_loadpath)[0]
    else:
        # create new base_filepath
        #if args.use_batch_norm:
        #    bn = '_bn'
        #else:
        #    bn = ''
        if args.dropout_rate > 0:
            do='_do%s'%args.dropout_rate
        else:
            do=''
        args.exp_name += '_'+args.dataset_name + '_'+args.rec_loss_type+do
        base_filepath = os.path.join(args.model_savedir, args.exp_name)
    print('base filepath is %s'%base_filepath)

    info = create_new_info_dict(vars(args), base_filepath)
    model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv = create_conv_acn_pcnn_models(info, args.model_loadpath)
    if args.tsne:
        call_tsne_plot(model_dict, data_dict, info)
    if args.walk:
        latent_walk(model_dict, data_dict, info)
    # only train if we weren't asked to do anything else
    if not max([args.tsne, args.walk]):
        write_log_files(info)
        train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info, rescale_inv)

