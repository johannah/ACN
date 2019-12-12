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
import numpy as np
from copy import deepcopy, copy
from torch import nn, optim
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from IPython import embed

from utils import create_new_info_dict, save_checkpoint, create_mnist_datasets, seed_everything
from utils import plot_example, plot_losses
from utils import set_model_mode, kl_loss_function

from pixel_cnn import GatedPixelCNN
from acn_models import ConvEncoder, PriorNetwork

def create_conv_acn_pcnn_models(info, model_loadpath=''):
    '''
    load details of previous model if applicable, otherwise create new models
    '''
    train_cnt = 0
    epoch_cnt = 0

    # use argparse device no matter what info dict is loaded
    preserve_args = ['device', 'batch_size', 'save_every_epochs',
                     'base_filepath', 'model_loadpath', 'perplexity',
                     'use_targets']
    largs = info['args']
    #preserve_dict = {}
    #for key in preserve_args:
    #    preserve_dict[key] = info[key]

    if model_loadpath !='':
        #tmlp =  model_loadpath+'.tmp'
        #os.system('cp %s %s'%(model_loadpath, tmlp))
        _dict = torch.load(model_loadpath, map_location=lambda storage, loc:storage)
        dinfo = _dict['info']
        pkeys = info.keys()
        for key in dinfo.keys():
            if key not in preserve_args or key not in pkeys:
                info[key] = dinfo[key]
        train_cnt = info['train_cnts'][-1]
        epoch_cnt = info['epoch_cnt']
        info['args'].append(largs)

    ## use argparse device no matter what device is loaded
    #for key in preserve_args:
    #    info[key] = preserve_dict[key]

    encoder_model = ConvEncoder(info['code_length'], input_size=info['input_channels'],
                            encoder_output_size=info['encoder_output_size']).to(info['device'])
    prior_model = PriorNetwork(size_training_set=info['size_training_set'],
                               code_length=info['code_length'], k=info['num_k']).to(info['device'])
    pcnn_decoder = GatedPixelCNN(input_dim=info['target_channels'],
                                  dim=info['possible_values'],
                                  n_layers=info['num_pcnn_layers'],
                                  n_classes=info['num_classes'],
                                  float_condition_size=info['code_length'],
                                  last_layer_bias=info['last_layer_bias']).to(info['device'])

    model_dict = {'encoder_model':encoder_model, 'prior_model':prior_model, 'pcnn_decoder':pcnn_decoder}
    parameters = []
    for name,model in model_dict.items():
        parameters+=list(model.parameters())
    model_dict['opt'] = optim.Adam(parameters, lr=info['learning_rate'])

    if args.model_loadpath !='':
       for name,model in model_dict.items():
            model_dict[name].load_state_dict(_dict[name+'_state_dict'])
    return model_dict, info, train_cnt, epoch_cnt


def run_acn(train_cnt, model_dict, data_dict, phase, device):
    st = time.time()
    run = rec_running = kl_running = sum_running = 0.0
    data_loader = data_dict[phase]
    model_dict = set_model_mode(model_dict, phase)
    for idx, (data, label, batch_index) in enumerate(data_loader):
        target = data = data.to(device)
        bs,c,h,w = target.shape
        model_dict['opt'].zero_grad()
        z, u_q, s_q = model_dict['encoder_model'](data)
        # add the predicted codes to the input
        yhat_batch = torch.sigmoid(model_dict['pcnn_decoder'](x=data, float_condition=z))
        model_dict['prior_model'].codes[batch_index] = u_q.detach().cpu().numpy()
        model_dict['prior_model'].fit_knn(model_dict['prior_model'].codes)
        u_p, s_p = model_dict['prior_model'](u_q)
        kl = kl_loss_function(u_q, s_q, u_p, s_p)
        kl = info['kl_beta']*kl.view(bs*info['code_length']).sum(dim=-1).mean()
        rec_loss = F.binary_cross_entropy(yhat_batch, target, reduction='none')
        rec_loss = rec_loss.view(bs,c*h*w).sum(dim=-1).mean()
        loss = kl+rec_loss
        if phase == 'train':
            loss.backward()
            model_dict['opt'].step()
        run+=bs
        kl_running+= kl.item()
        rec_running+= rec_loss.item()
        # add batch size because it hasn't been added to train cnt yet
        if phase == 'train':
            train_cnt+=bs
    example = {'data':data, 'target':target, 'yhat':yhat_batch}
    loss_avg = {'kl':kl_running/bs, 'rec':rec_running/run, 'loss':(rec_running+kl_running)/run}
    print("finished %s after %s secs at cnt %s"%(phase,
                                                time.time()-st,
                                                train_cnt,
                                                ))
    print(loss_avg)
    return model_dict, data_dict, loss_avg, example

def train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info):
    base_filepath = info['base_filepath']
    base_filename = os.path.split(info['base_filepath'])[1]
    while train_cnt < info['num_examples_to_train']:
        print('starting epoch %s on %s'%(epoch_cnt, info['device']))
        model_dict, data_dict, train_loss_avg, train_example = run_acn(train_cnt, model_dict, data_dict, phase='train', device=info['device'])
        epoch_cnt +=1
        train_cnt +=info['size_training_set']
        if not epoch_cnt % info['save_every_epochs']:
            # make a checkpoint
            print('starting valid phase')
            model_dict, data_dict, valid_loss_avg, valid_example = run_acn(train_cnt, model_dict, data_dict, phase='valid', device=info['device'])
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

            plot_example(train_img_filepath, train_example, num_plot=5)
            plot_example(valid_img_filepath, valid_example, num_plot=5)
            save_checkpoint(state_dict, filename=ckpt_filepath)

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
            for idx, (data, label, batch_idx) in enumerate(data_loader):
                target = data = data.to(info['device'])
                # yhat_batch is bt 0-1
                z, u_q, s_q = model_dict['encoder_model'](data)
                u_p, s_p = model_dict['prior_model'](u_q)
                yhat_batch = torch.sigmoid(model_dict['pcnn_decoder'](x=target, float_condition=z))
                X = u_q.cpu().numpy()
                if info['use_targets']:
                    images = target[:,0].cpu().numpy()
                    T = 'target'
                else:
                    images = np.round(yhat_batch.cpu().numpy()[:,0], 0).astype(np.int32)
                    T = 'pred'
                color = label
                param_name = '_%s_P%s_%s.html'%(phase, info['perplexity'], T)
                html_path = info['model_loadpath'].replace('.pt', param_name)
                tsne_plot(X=X, images=images, color=color,
                          perplexity=info['perplexity'],
                          html_out_path=html_path, serve=False)
                break

def sample(model_dict, data_dict, info):
    from skvideo.io import vwrite
    model_dict = set_model_mode(model_dict, 'valid')
    output_savepath = args.model_loadpath.replace('.pt', '')
    with torch.no_grad():
        for phase in ['train', 'valid']:
            data_loader = data_dict[phase]
            with torch.no_grad():
                for idx, (data, label, batch_idx) in enumerate(data_loader):
                    target = data = data.to(info['device'])
                    bs = data.shape[0]
                    z, u_q, s_q = model_dict['encoder_model'](data)
                    # teacher forced version
                    yhat_batch = torch.sigmoid(model_dict['pcnn_decoder'](x=target, float_condition=z))
                    # create blank canvas for autoregressive sampling
                    canvas = torch.zeros_like(target)
                    building_canvas = []
                    for i in range(canvas.shape[1]):
                        for j in range(canvas.shape[2]):
                            print('sampling row: %s'%j)
                            for k in range(canvas.shape[3]):
                                output = torch.sigmoid(model_dict['pcnn_decoder'](x=canvas, float_condition=z))
                                canvas[:,i,j,k] = output[:,i,j,k].detach()
                                # add frames for video
                                if not k%5:
                                    building_canvas.append(deepcopy(canvas[0].detach().cpu().numpy()))

                    f,ax = plt.subplots(bs, 3, sharex=True, sharey=True, figsize=(3,bs))
                    nptarget = target.detach().cpu().numpy()
                    npoutput = output.detach().cpu().numpy()
                    npyhat = yhat_batch.detach().cpu().numpy()
                    for idx in range(bs):
                        ax[idx,0].imshow(nptarget[idx,0], cmap=plt.cm.viridis)
                        ax[idx,0].set_title('true')
                        ax[idx,1].imshow(npyhat[idx,0], cmap=plt.cm.viridis)
                        ax[idx,1].set_title('tf')
                        ax[idx,2].imshow(npoutput[idx,0], cmap=plt.cm.viridis)
                        ax[idx,2].set_title('sam')
                        ax[idx,0].axis('off')
                        ax[idx,1].axis('off')
                        ax[idx,2].axis('off')
                    iname = output_savepath + '_sample_%s.png'%phase
                    print('plotting %s'%iname)
                    plt.savefig(iname)
                    plt.close()

                    # make movie
                    building_canvas = (np.array(building_canvas)*255).astype(np.uint8)
                    print('writing building movie')
                    mname = output_savepath + '_build_%s.mp4'%phase
                    vwrite(mname, building_canvas)
                    print('finished %s'%mname)
                    # only do one batch
                    break

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='train acn')
    # operatation options
    parser.add_argument('-l', '--model_loadpath', default='', help='load model to resume training or sample')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=394)
    parser.add_argument('--num_threads', default=2)
    parser.add_argument('-se', '--save_every_epochs', default=5, type=int)
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--input_channels', default=1, type=int, help='num of channels of input')
    parser.add_argument('--target_channels', default=1, type=int, help='num of channels of target')
    parser.add_argument('--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-e', '--exp_name', default='pcnn_acn', help='name of experiment')
    # acn model setup
    parser.add_argument('-cl', '--code_length', default=64, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('-kl', '--kl_beta', default=.5, type=float, help='scale kl loss')
    parser.add_argument('--possible_values', default=1, help='num values that the pcnn output can take')
    parser.add_argument('--last_layer_bias', default=0.5, help='bias for output decoder')
    parser.add_argument('--num_classes', default=10, help='num classes for class condition in pixel cnn')
    parser.add_argument('--encoder_output_size', default=2048, help='output as a result of the flatten of the encoder - found experimentally')
    parser.add_argument('--num_pcnn_layers', default=12, help='num layers for pixel cnn')
    # dataset setup
    parser.add_argument('-d',  '--dataset_name', default='FashionMNIST', help='which mnist to use', choices=['MNIST', 'FashionMNIST'])
    parser.add_argument('--model_savedir', default='../model_savedir', help='save checkpoints here')
    parser.add_argument('--base_datadir', default='../dataset/', help='save datasets here')
    # sampling info
    parser.add_argument('-s', '--sample', action='store_true', default=False)
    parser.add_argument('-tf', '--teacher_force', action='store_true', default=False)
    # tsne info
    parser.add_argument('-t', '--tsne', action='store_true', default=False)
    parser.add_argument('-p', '--perplexity', default=3, type=int, help='perplexity used in scikit-learn tsne call')
    parser.add_argument('-ut', '--use_targets', default=False, action='store_true',  help='plot tsne with true target image instead of tf pred')
    args = parser.parse_args()
    if args.sample:
        # limit the batch size when sampling
        args.batch_size = min([args.batch_size, 10])

    seed_everything(args.seed, args.num_threads)
    # get naming scheme
    args.exp_name += '_'+args.dataset_name
    base_filepath = os.path.join(args.model_savedir, args.exp_name)

    data_dict, size_training_set, nchans, hsize, wsize = create_mnist_datasets(dataset_name=args.dataset_name, base_datadir=args.base_datadir, batch_size=args.batch_size)
    info = create_new_info_dict(vars(args), size_training_set, base_filepath)
    model_dict, info, train_cnt, epoch_cnt = create_conv_acn_pcnn_models(info, args.model_loadpath)

    if args.sample:
        # limit batch size
        sample(model_dict, data_dict, info)
    elif args.tsne:
        call_tsne_plot(model_dict, data_dict, info)
    else:
        train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info)

