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
from torch import nn, optim
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import transforms

from utils import create_new_info_dict, save_checkpoint, create_mnist_datasets, seed_everything
from utils import plot_example, plot_losses, count_parameters
from utils import set_model_mode, kl_loss_function, write_log_files
from utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic

from pixel_cnn import GatedPixelCNN
from acn_models import PriorNetwork, ACNVQVAEres
from IPython import embed


def create_models(info, model_loadpath='', dataset_name='FashionMNIST'):
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
    if info['rec_loss_type'] == 'dml':
        # data going into dml should be bt -1 and 1
        rescale = lambda x: (x - 0.5) * 2.
        rescale_inv = lambda x: (0.5 * x) + 0.5
    if info['rec_loss_type'] == 'bce':
        rescale = lambda x: x
        rescale_inv = lambda x: x


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
    if info['rec_loss_type'] == 'dml':
        info['nmix'] =  (2*info['nr_logistic_mix']+info['nr_logistic_mix'])*info['num_output_chans']
        info['output_dim']  = info['nmix']
        # last layer for pcnn
        info['last_layer_bias'] = 0.0
    if info['rec_loss_type'] == 'bce':
        # last layer for pcnn
        info['last_layer_bias'] = 0.5
        info['output_dim']  = info['num_output_chans']
    # setup models
    vq_acn_model = ConvEncodeDecodeLargeVQVAE(code_len=info['code_length'],
                               input_size=info['input_channels'],
                               output_size=info['output_dim'],
                               encoder_output_size=info['encoder_output_size'],
                               last_layer_bias=info['last_layer_bias'],
                               num_clusters=info['num_vqk'],
                               num_z=info['num_z'],
                               ).to(info['device'])

    prior_model = PriorNetwork(size_training_set=info['size_training_set'],
                               code_length=info['code_length'], k=info['num_k']).to(info['device'])

    pcnn_decoder = GatedPixelCNN(
                                 input_dim=info['input_channels'],
                                 output_dim=info['output_dim'],
                                 dim=info['pixel_cnn_dim'],
                                 n_layers=info['num_pcnn_layers'],
                                 float_condition_size=info['code_length'],
                                 spatial_condition_size=info['input_channels'],
                                 # output dim is same as deconv output in this
                                 # case
                                 last_layer_bias=info['last_layer_bias'],
                                 use_batch_norm=info['use_batch_norm'],
                                 output_projection_size=info['output_projection_size']).to(info['device'])

    model_dict = {'vq_acn_model':vq_acn_model, 'prior_model':prior_model, 'pcnn_decoder':pcnn_decoder}
    parameters = []
    for name,model in model_dict.items():
        parameters+=list(model.parameters())
        print('created %s model with %s parameters' %(name,count_parameters(model)))
    model_dict['opt'] = optim.Adam(parameters, lr=info['learning_rate'])

    if args.model_loadpath !='':
       for name,model in model_dict.items():
            model_dict[name].load_state_dict(_dict[name+'_state_dict'])
    return model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv

def forward_pass(model_dict, data, label, batch_index, phase, info):
    target = data = data.to(info['device'])
    bs,c,h,w = target.shape
    model_dict['opt'].zero_grad()
    data = F.dropout(data, p=info['dropout_rate'], training=True, inplace=False)
    vq_dml, z, u_q, s_q, z_e_x, z_q_x, latents = model_dict['vq_acn_model'](data)
    if phase == 'train':
        # fit knn during training
        model_dict['prior_model'].codes[batch_index] = u_q.detach().cpu().numpy()
        model_dict['prior_model'].fit_knn(model_dict['prior_model'].codes)
    u_p, s_p = model_dict['prior_model'](u_q)
    vq_yhat_batch = sample_from_discretized_mix_logistic(vq_dml.detach(), info['nr_logistic_mix'], only_mean=info['sample_mean'])
    pcnn_dml = model_dict['pcnn_decoder'](x=target, float_condition=z, spatial_condition=vq_yhat_batch.detach())
    # dont sample pcnn dml unless necessary
    return model_dict, data, target, z, u_q, s_q, u_p, s_p, vq_dml, z_e_x, z_q_x, latents, vq_yhat_batch, pcnn_dml

def account_losses(loss_dict):
    loss_avg = {}
    for key in loss_dict.keys():
        if key != 'running':
            loss_avg[key] = loss_dict[key]/loss_dict['running']
    return loss_avg

def run(train_cnt, model_dict, data_dict, phase, info):
    st = time.time()
    # track losses for this epoch
    loss_dict = {'running': 0,
                 'kl':0,
                 'vq_%s'%info['rec_loss_type']:0,
                 'pcnn_%s'%info['rec_loss_type']:0,
                 'vq':0,
                 'commit':0,
                  }
    data_loader = data_dict[phase]
    model_dict = set_model_mode(model_dict, phase)
    num_batches = len(data_loader)
    for idx, (data, label, batch_index) in enumerate(data_loader):
        bs = data.shape[0]
        fp_out = forward_pass(model_dict, data, label, batch_index, phase, info)
        model_dict, data, target, z, u_q, s_q, u_p, s_p, vq_dml, z_e_x, z_q_x, latents, vq_yhat_batch, pcnn_dml = fp_out
        kl = kl_loss_function(u_q, s_q, u_p, s_p, reduction=info['reduction'])
        # pixel cnn reconstruction with spatial reconditioning
        # now passing spatial condition as one channel - pcnn can only control z
        vq_rec_loss = discretized_mix_logistic_loss(vq_dml, target, nr_mix=info['nr_logistic_mix'], reduction=info['reduction'])/2.0
        pcnn_rec_loss = discretized_mix_logistic_loss(pcnn_dml, target, nr_mix=info['nr_logistic_mix'], reduction=info['reduction'])/2.0
        # VQ Losses
        vq_loss = F.mse_loss(z_q_x, z_e_x.detach(), reduction=info['reduction'])
        commit_loss = F.mse_loss(z_e_x, z_q_x.detach(), reduction=info['reduction'])
        commit_loss *= info['vq_commitment_beta']
        loss = kl+vq_rec_loss+pcnn_rec_loss+vq_loss+commit_loss
        if phase == 'train':
            loss.backward()
            model_dict['opt'].step()

        loss_dict['running']+=bs
        loss_dict['loss']+=loss.item()
        loss_dict['kl']+= kl.item()
        loss_dict['vq_%s'%info['rec_loss_type']]+=vq_rec_loss.item()
        loss_dict['pcnn_%s'%info['rec_loss_type']]+=pcnn_rec_loss.item()
        loss_dict['vq']+= vq_loss.item()
        loss_dict['commit']+= commit_loss.item()
        # add batch size because it hasn't been added to train cnt yet
        if phase == 'train':
            train_cnt+=bs
        if idx == num_batches-2:
            pcnn_yhat_batch = sample_from_discretized_mix_logistic(pcnn_dml.detach(), info['nr_logistic_mix'], only_mean=info['sample_mean'])
            # store example near end for plotting
            example = {'data':data.detach().cpu(),
                       'target':target.detach().cpu(),
                       'vq_yhat':vq_yhat_batch.detach().cpu(),
                       'pcnn_yhat':pcnn_yhat_batch.cpu(),
                       }
        if not idx % 10:
            print(train_cnt, idx, account_losses(loss_dict))

    # store average loss for return
    loss_avg = account_losses(loss_dict)
    print(loss_avg)
    print("finished %s after %s secs at cnt %s"%(phase,
                                                time.time()-st,
                                                train_cnt,
                                                ))
    return model_dict, data_dict, loss_avg, example

def train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info, rescale_inv):
    print('starting training routine')
    base_filepath = info['base_filepath']
    base_filename = os.path.split(info['base_filepath'])[1]
    while train_cnt < info['num_examples_to_train']:
        print('starting epoch %s on %s'%(epoch_cnt, info['device']))
        model_dict, data_dict, train_loss_avg, train_example = run_acn(train_cnt, model_dict, data_dict, phase='train', info=info)
        epoch_cnt +=1
        train_cnt +=info['size_training_set']
        if not epoch_cnt % info['save_every_epochs'] or epoch_cnt ==1:
            # make a checkpoint
            print('starting valid phase')
            model_dict, data_dict, valid_loss_avg, valid_example = run_acn(train_cnt, model_dict, data_dict, phase='valid', info=info)
            # track losses
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
            # plot example images
            for ex in [train_example, valid_example]:
                for key, data in ex.items():
                    train_example[key] = rescale_inv(data)
            plot_example(train_img_filepath, train_example, num_plot=10)
            plot_example(valid_img_filepath, valid_example, num_plot=10)
            # save model
            save_checkpoint(state_dict, filename=ckpt_filepath)
            # plot losses
            plot_losses(info['train_cnts'],
                        info['train_losses'],
                        info['valid_losses'], name=plot_filepath, rolling_length=1)

def call_tsne_plot(model_dict, data_dict, info):
    from utils import tsne_plot
    # always be in eval mode
    model_dict = set_model_mode(model_dict, 'valid')
    with torch.no_grad():
        for phase in ['valid', 'train']:
            data_loader = data_dict[phase]
            for idx, (data, label, batch_index) in enumerate(data_loader):
                fp_out = forward_pass(model_dict, data, label, batch_index, phase, info)
                model_dict, data, target, z, u_q, s_q, u_p, s_p, vq_dml, z_e_x, z_q_x, latents, vq_yhat_batch, pcnn_dml = fp_out
                X = u_q.cpu().numpy()
                images = target[:,0].cpu().numpy()
                T = 'target'
                color = label
                param_name = '_%s_P%s_%s.html'%(phase, info['perplexity'], T)
                html_path = info['model_loadpath'].replace('.pt', param_name)
                tsne_plot(X=X, images=images, color=color,
                          perplexity=info['perplexity'],
                          html_out_path=html_path, serve=False)
                break

def sample(model_dict, data_dict, info):
    from copy import deepcopy
    model_dict = set_model_mode(model_dict, 'valid')
    output_savepath = args.model_loadpath.replace('.pt', '')
    with torch.no_grad():
        for phase in ['train', 'valid']:
            data_loader = data_dict[phase]
            with torch.no_grad():
                for idx, (data, label, batch_index) in enumerate(data_loader):
                    bs = min([data.shape[0], 10])
                    fp_out = forward_pass(model_dict, data, label, batch_index, phase, info)
                    model_dict, data, target, z, u_q, s_q, u_p, s_p, vq_dml, z_e_x, z_q_x, latents, vq_yhat_batch, pcnn_dml = fp_out
                    pcnn_yhat_batch = sample_from_discretized_mix_logistic(pcnn_dml.detach(), info['nr_logistic_mix'], only_mean=info['sample_mean'])
                   # teacher forced version
                    print('data', data.min(), data.max())
                    print('target', target.min(), target.max())
                    # create blank canvas for autoregressive sampling
                    np_target = data.detach().cpu().numpy()
                    np_vq_yhat = deepcopy(vq_yhat_batch.detach().cpu().numpy())
                    np_pcnn_yhat = pcnn_yhat_batch.detach().cpu().numpy()
                    if args.use_zero_as_canvas:
                        print('using zero output as sample canvas')
                        canvas = torch.zeros_like(vq_yhat_batch)
                        st_can = '_zc'
                    else:
                        print('using deconv output as sample canvas')
                        canvas = vq_yhat_batch
                        st_can = '_dc'
                    for i in range(canvas.shape[1]):
                        for j in range(canvas.shape[2]):
                            print('sampling row: %s'%j)
                            for k in range(canvas.shape[3]):
                                output = model_dict['pcnn_decoder'](x=canvas, float_condition=z, spatial_condition=vq_yhat_batch.detach())
                                # output should be bt -1 and 1 for canvas
                                asam = sample_from_discretized_mix_logistic(output.detach(), info['nr_logistic_mix'], only_mean=info['sample_mean'])
                                canvas[:,i,j,k] = asam[:,i,j,k]
                    output = sample_from_discretized_mix_logistic(output.detach(), info['nr_logistic_mix'], only_mean=info['sample_mean'])
                    np_output = output.detach().cpu().numpy()
                    print('vq_yhat_batch', vq_yhat_batch.min(), vq_yhat_batch.max())
                    print('pcnn_yhat_batch', pcnn_yhat_batch.min(), pcnn_yhat_batch.max())
                    print('canvas', canvas.min(), canvas.max())
                    f,ax = plt.subplots(bs, 4, sharex=True, sharey=True, figsize=(3,bs))
                    for idx in range(bs):
                        ax[idx,0].matshow(np_target[idx,0], cmap=plt.cm.gray)
                        ax[idx,1].matshow(np_vq_yhat[idx,0], cmap=plt.cm.gray)
                        ax[idx,2].matshow(np_pcnn_yhat[idx,0], cmap=plt.cm.gray)
                        ax[idx,3].matshow(np_output[idx,0], cmap=plt.cm.gray)
                        ax[idx,0].set_title('true')
                        ax[idx,1].set_title('vq')
                        ax[idx,2].set_title('tf')
                        ax[idx,3].set_title('sam')
                        ax[idx,0].axis('off')
                        ax[idx,1].axis('off')
                        ax[idx,2].axis('off')
                        ax[idx,3].axis('off')
                    iname = output_savepath + st_can +'_sample_%s.png'%phase
                    print('plotting %s'%iname)
                    plt.savefig(iname)
                    plt.close()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='train acn')
    # operatation options
    parser.add_argument('-l', '--model_loadpath', default='', help='load model to resume training or sample')
    parser.add_argument('-ll', '--load_last_model', default='',  help='load last model from directory from directory')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=394)
    parser.add_argument('--num_threads', default=2)
    parser.add_argument('-se', '--save_every_epochs', default=5, type=int)
    parser.add_argument('-bs', '--batch_size', default=84, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--input_channels', default=1, type=int, help='num of channels of input')
    parser.add_argument('--target_channels', default=1, type=int, help='num of channels of target')
    parser.add_argument('--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-e', '--exp_name', default='pcnn_acn_vq', help='name of experiment')
    parser.add_argument('-dr', '--dropout_rate', default=0.0, type=float)
    parser.add_argument('-r', '--reduction', default='sum', type=str, choices=['sum', 'mean'])
    # batch norm resulted in worse outcome in pixel-cnn-only model
    parser.add_argument('--use_batch_norm', default=False, action='store_true')
    parser.add_argument('--output_projection_size', default=32, type=int)
    # right now, still using float input for bce (which seemes to work) --
    # should actually convert data to binary...
    # if discretized mixture of logistics, we can predict pixel values. shape
    # changes are required for output sampling
    parser.add_argument('--rec_loss_type', default='dml', type=str, help='name of loss. options are bce or dml', choices=['bce', 'dml'])
    parser.add_argument('--nr_logistic_mix', default=10, type=int)
    # acn model setup
    parser.add_argument('-cl', '--code_length', default=64, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('--pixel_cnn_dim', default=64, type=int, help='pixel cnn dimension')
    parser.add_argument('--last_layer_bias', default=0.0, help='bias for output decoder - should be 0 for dml')
    parser.add_argument('--num_classes', default=10, help='num classes for class condition in pixel cnn')
    parser.add_argument('--encoder_output_size', default=1024, help='output as a result of the flatten of the encoder - found experimentally')
    parser.add_argument('--num_pcnn_layers', default=8, help='num layers for pixel cnn')
    # vq model setup
    parser.add_argument('--vq_commitment_beta', default=0.25, help='scale for loss 3 in vqvae - how hard to enforce commitment to cluster')
    parser.add_argument('--num_vqk', default=512, type=int)
    parser.add_argument('--num_z', default=64, type=int)
    # dataset setup
    parser.add_argument('-d',  '--dataset_name', default='FashionMNIST', help='which mnist to use', choices=['MNIST', 'FashionMNIST'])
    parser.add_argument('--model_savedir', default='../model_savedir', help='save checkpoints here')
    parser.add_argument('--base_datadir', default='../dataset/', help='save datasets here')
    # sampling info
    parser.add_argument('-s', '--sample', action='store_true', default=False)
    parser.add_argument('-sm', '--sample_mean', action='store_true', default=False)
    parser.add_argument('-z', '--use_zero_as_canvas', action='store_true', default=False)
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
        if args.use_batch_norm:
            bn = '_bn'
        else:
            bn = ''
        if args.dropout_rate > 0:
            do='_do%s'%args.dropout_rate
        else:
            do=''
        lo = '_r%s'%args.reduction
        pl = '_%s'%args.pixel_cnn_dim
        args.exp_name += '_'+args.dataset_name + '_'+args.rec_loss_type+bn+do+pl+lo
        base_filepath = os.path.join(args.model_savedir, args.exp_name)
    print('base filepath is %s'%base_filepath)

    info = create_new_info_dict(vars(args), base_filepath)
    model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv = create_models(info, args.model_loadpath)
    if args.tsne:
        call_tsne_plot(model_dict, data_dict, info)
    if args.sample:
        # limit batch size
        sample(model_dict, data_dict, info)
    if args.walk:
        latent_walk(model_dict, data_dict, info)
    # only train if we weren't asked to do anything else
    if not max([args.sample, args.tsne, args.walk]):
        write_log_files(info)
        train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info, rescale_inv)

