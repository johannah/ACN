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
from acn_models import ConvEncoder, PriorNetwork
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

    # setup models
    encoder_model = ConvEncoder(info['code_length'], input_size=info['input_channels'],
                            encoder_output_size=info['encoder_output_size']).to(info['device'])
    prior_model = PriorNetwork(size_training_set=info['size_training_set'],
                               code_length=info['code_length'], k=info['num_k']).to(info['device'])
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

    pcnn_decoder = GatedPixelCNN(input_dim=info['num_input_chans'],
                                 output_dim=info['output_dim'],
                                 dim=info['pixel_cnn_dim'],
                                 n_layers=info['num_pcnn_layers'],
                                 float_condition_size=info['code_length'],
                                 last_layer_bias=info['last_layer_bias'],
                                 use_batch_norm=info['use_batch_norm'],
                                 output_projection_size=info['output_projection_size']).to(info['device'])

    model_dict = {'encoder_model':encoder_model, 'prior_model':prior_model, 'pcnn_decoder':pcnn_decoder}
    parameters = []
    for name,model in model_dict.items():
        parameters+=list(model.parameters())
        print('created %s model with %s parameters' %(name,count_parameters(model)))
    model_dict['opt'] = optim.Adam(parameters, lr=info['learning_rate'])

    if args.model_loadpath !='':
       for name,model in model_dict.items():
            model_dict[name].load_state_dict(_dict[name+'_state_dict'])
    return model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv


def run_acn(train_cnt, model_dict, data_dict, phase, device, rec_loss_type, dropout_rate):
    st = time.time()
    run = rec_running = kl_running = loss_running = 0.0
    data_loader = data_dict[phase]
    model_dict = set_model_mode(model_dict, phase)
    num_batches = len(data_loader)
    for idx, (data, label, batch_index) in enumerate(data_loader):
        target = data = data.to(device)
        bs,c,h,w = target.shape
        model_dict['opt'].zero_grad()
        z, u_q, s_q = model_dict['encoder_model'](data)
        if phase == 'train':
            # fit knn during training
            model_dict['prior_model'].codes[batch_index] = u_q.detach().cpu().numpy()
            model_dict['prior_model'].fit_knn(model_dict['prior_model'].codes)
        u_p, s_p = model_dict['prior_model'](u_q)
        # calculate loss
        kl = kl_loss_function(u_q, s_q, u_p, s_p, reduction=info['reduction'])
        data = F.dropout(data, p=dropout_rate, training=True, inplace=False)
        yhat_batch = model_dict['pcnn_decoder'](x=data, float_condition=z)
        if rec_loss_type  == 'bce':
            rec_loss = F.binary_cross_entropy(torch.sigmoid(yhat_batch), target, reduction='none')
            rec_loss = rec_loss.view(bs,c*h*w).sum(dim=-1).mean()
        if rec_loss_type == 'dml':
            # TODO - what normalization is needed here
            # input into dml should be bt -1 and 1
            # pcnn starts at kl:6 dml:3017 with sum reduction on Fashion MNIST -
            # not sure if this model will train yet
            rec_loss = discretized_mix_logistic_loss(yhat_batch, target, nr_mix=info['nr_logistic_mix'], reduction=info['reduction'])
        loss = kl+rec_loss
        if phase == 'train':
            loss.backward()
            model_dict['opt'].step()
        run+=bs
        kl_running+= kl.item()
        rec_running+= rec_loss.item()
        loss_running+= loss.item()
        # add batch size because it hasn't been added to train cnt yet
        if phase == 'train':
            train_cnt+=bs
        if idx == num_batches-2:
            # store example near end for plotting
            example = {'data':data.detach().cpu(), 'target':target.detach().cpu(), 'yhat':yhat_batch.detach().cpu()}
        if not idx % 10:
            loss_avg = {'kl':kl_running/run, rec_loss_type:rec_running/run, 'loss':loss_running/run}
            print(idx, loss_avg)

    # store average loss for return
    loss_avg = {'kl':kl_running/run, info['rec_loss_type']:rec_running/run, 'loss':loss_running/run}
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
        model_dict, data_dict, train_loss_avg, train_example = run_acn(train_cnt,
                                                                       model_dict,
                                                                       data_dict,
                                                                       phase='train',
                                                                       device=info['device'],
                                                                       rec_loss_type=info['rec_loss_type'],
                                                                       dropout_rate=info['dropout_rate'])
        epoch_cnt +=1
        train_cnt +=info['size_training_set']
        if not epoch_cnt % info['save_every_epochs']:
            # make a checkpoint
            print('starting valid phase')
            model_dict, data_dict, valid_loss_avg, valid_example = run_acn(train_cnt,
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


            if info['rec_loss_type'] == 'dml':
                train_example['yhat'] = sample_from_discretized_mix_logistic(train_example['yhat'], info['nr_logistic_mix'])
                valid_example['yhat'] = sample_from_discretized_mix_logistic(valid_example['yhat'], info['nr_logistic_mix'])
            train_example['target'] = rescale_inv(train_example['target'])
            train_example['yhat'] = rescale_inv(train_example['yhat'])
            plot_example(train_img_filepath, train_example, num_plot=10)
            plot_example(valid_img_filepath, valid_example, num_plot=10)
            save_checkpoint(state_dict, filename=ckpt_filepath)

            plot_losses(info['train_cnts'],
                        info['train_losses'],
                        info['valid_losses'], name=plot_filepath, rolling_length=1)


def latent_walk(model_dict, data_dict, info):
    from skvideo.io import vwrite
    from imageio import imwrite
    model_dict = set_model_mode(model_dict, 'valid')
    output_savepath = args.model_loadpath.replace('.pt', '')
    phase = 'train'
    data_loader = data_dict[phase]
    bs = args.num_walk_steps
    with torch.no_grad():
        for walki in range(10):
            for idx, (data, label, batch_idx) in enumerate(data_loader):
                # limit number of samples
                lim = min([data.shape[0], 10])
                target = data = data[:lim].to(info['device'])
                z, u_q, s_q = model_dict['encoder_model'](data)
                break
            _,c,h,w=target.shape
            latents = torch.zeros((bs,z.shape[1]))
            si = 0; ei = 1
            sl = label[si].item(); el = label[ei].item()
            print('walking from %s to %s'%(sl, el))
            for code_idx in range(z.shape[1]):
                s = z[si,code_idx]
                e = z[ei,code_idx]
                code_walk = torch.linspace(s,e,bs)
                latents[:,code_idx] = code_walk
            latents = latents.to(info['device'])
            ## create blank canvas for autoregressive sampling
            canvas = torch.zeros((bs,c,h,w))
            for i in range(canvas.shape[1]):
                for j in range(canvas.shape[2]):
                    print('sampling row: %s'%j)
                    for k in range(canvas.shape[3]):
                        output = model_dict['pcnn_decoder'](x=canvas, float_condition=latents)
                        if info['rec_loss_type'] == 'bce':
                            canvas[:,i,j,k] = torch.sigmoid(output[:,i,j,k].detach())
                        if info['rec_loss_type'] == 'dml':
                            sample_dml = sample_from_discretized_mix_logistic(output, info['nr_logistic_mix'], only_mean=info['sample_mean'])
                            canvas[:,i,j,k] = sample_dml[:,i,j,k].detach()
            npst = target[si:si+1].detach().cpu().numpy()
            npen = target[ei:ei+1].detach().cpu().numpy()
            npwalk = canvas.detach().cpu().numpy()
            # add multiple frames of each sample as a hacky way to make the
            # video more interpretable to humans
            walk_video = np.concatenate((npst))
            walk_strip = npst[0,0]
            for ww in range(npwalk.shape[0]):
                walk_video = np.concatenate((walk_video,
                                             npwalk[ww:ww+1], npwalk[ww:ww+1],
                                             ))
                walk_strip = np.concatenate((walk_strip, npwalk[ww:ww+1,0]))
            walk_video = np.concatenate((walk_video,
                                         npen,
                                         ))
            walk_strip = np.concatenate((walk_strip, npen[0,0]))
            walk_video = (walk_video*255).astype(np.uint8)
            mname = output_savepath + '%s_s%s_e%s_walk.mp4'%(walki,sl,el)
            iname = output_savepath + '%s_s%s_e%s_walk_strip.png'%(walki,sl,el)
            imwrite(iname, walk_strip)
            ## make movie
            print('writing walk movie')
            vwrite(mname, walk_video)
            print('finished %s'%mname)

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
                if info['rec_loss_type'] == 'bce':
                    assert target.max() <=1
                    assert target.min() >=0
                    yhat_batch = torch.sigmoid(model_dict['pcnn_decoder'](x=target, float_condition=z))
                elif info['rec_loss_type'] == 'dml':
                    assert target.max() <=1
                    assert target.min() >=-1
                    yhat_batch_dml = model_dict['pcnn_decoder'](x=target, float_condition=z)
                    yhat_batch = sample_from_discretized_mix_logistic(yhat_batch_dml, info['nr_logistic_mix'], only_mean=info['sample_mean'])
                else:
                    raise ValueError('invalid rec_loss_type')
                X = u_q.cpu().numpy()
                if info['use_pred']:
                    images = np.round(yhat_batch.cpu().numpy()[:,0], 0).astype(np.int32)
                    T = 'pred'
                else:
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
    from skvideo.io import vwrite
    model_dict = set_model_mode(model_dict, 'valid')
    output_savepath = args.model_loadpath.replace('.pt', '')
    with torch.no_grad():
        for phase in ['train', 'valid']:
            data_loader = data_dict[phase]
            with torch.no_grad():
                for idx, (data, label, batch_idx) in enumerate(data_loader):
                    bs = min([data.shape[0], 10])
                    target = data = data[:bs].to(info['device'])
                    z, u_q, s_q = model_dict['encoder_model'](data)
                    # teacher forced version
                    print('data', data.min(), data.max())
                    print('target', target.min(), target.max())
                    if info['rec_loss_type'] == 'bce':
                        assert target.max() <=1
                        assert target.min() >=0
                        yhat_batch = torch.sigmoid(model_dict['pcnn_decoder'](x=target, float_condition=z))
                    elif info['rec_loss_type'] == 'dml':
                        assert target.max() <=1
                        assert target.min() >=-1
                        yhat_batch_dml = model_dict['pcnn_decoder'](x=target, float_condition=z)
                        yhat_batch = sample_from_discretized_mix_logistic(yhat_batch_dml, info['nr_logistic_mix'], only_mean=info['sample_mean'])
                    else:
                        raise ValueError('invalid rec_loss_type')
                    # create blank canvas for autoregressive sampling
                    canvas = torch.zeros_like(target)
                    building_canvas = []
                    for i in range(canvas.shape[1]):
                        for j in range(canvas.shape[2]):
                            print('sampling row: %s'%j)
                            for k in range(canvas.shape[3]):
                                output = model_dict['pcnn_decoder'](x=canvas, float_condition=z)
                                if info['rec_loss_type'] == 'bce':
                                    # output should be bt 0 and 1 for canvas
                                    canvas[:,i,j,k] = torch.sigmoid(output[:,i,j,k].detach())
                                if info['rec_loss_type'] == 'dml':
                                    output = sample_from_discretized_mix_logistic(output.detach(), info['nr_logistic_mix'], only_mean=info['sample_mean'])
                                    # output should be bt -1 and 1 for canvas
                                    #print(output[:,i,j,k].min(), output[:,i,j,k].max())
                                    canvas[:,i,j,k] = output[:,i,j,k]
                                # add frames for video
                                if not k%5:
                                    building_canvas.append(deepcopy(canvas[0].detach().cpu().numpy()))

                    print('canvas', canvas.min(), canvas.max())
                    print('yhat_batch', yhat_batch.min(), yhat_batch.max())
                    f,ax = plt.subplots(bs, 3, sharex=True, sharey=True, figsize=(3,bs))
                    nptarget = data.detach().cpu().numpy()
                    npyhat = yhat_batch.detach().cpu().numpy()
                    npoutput = canvas.detach().cpu().numpy()
                    for idx in range(bs):
                        ax[idx,0].matshow(nptarget[idx,0], cmap=plt.cm.gray)
                        ax[idx,1].matshow(npyhat[idx,0]  , cmap=plt.cm.gray)
                        ax[idx,2].matshow(npoutput[idx,0], cmap=plt.cm.gray)
                        #ax[idx,0].imshow(nptarget[idx,0], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
                        #ax[idx,1].imshow(npyhat[idx,0], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
                        #ax[idx,2].imshow(npoutput[idx,0], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
                        ax[idx,0].set_title('true')
                        ax[idx,1].set_title('tf')
                        ax[idx,2].set_title('sam')
                        ax[idx,0].axis('off')
                        ax[idx,1].axis('off')
                        ax[idx,2].axis('off')
                    iname = output_savepath + '_sample_%s.png'%phase
                    print('plotting %s'%iname)
                    plt.savefig(iname)
                    plt.close()

                    ## make movie
                    #building_canvas = (np.array(building_canvas)*255).astype(np.uint8)
                    #print('writing building movie')
                    #mname = output_savepath + '_build_%s.mp4'%phase
                    #vwrite(mname, building_canvas)
                    #print('finished %s'%mname)
                    ## only do one batch
                    break

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
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--input_channels', default=1, type=int, help='num of channels of input')
    parser.add_argument('--target_channels', default=1, type=int, help='num of channels of target')
    parser.add_argument('--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-e', '--exp_name', default='pcnn_acn', help='name of experiment')
    parser.add_argument('-dr', '--dropout_rate', default=0.5, type=float)
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
    parser.add_argument('--encoder_output_size', default=2048, help='output as a result of the flatten of the encoder - found experimentally')
    parser.add_argument('--num_pcnn_layers', default=8, help='num layers for pixel cnn')
    # dataset setup
    parser.add_argument('-d',  '--dataset_name', default='FashionMNIST', help='which mnist to use', choices=['MNIST', 'FashionMNIST'])
    parser.add_argument('--model_savedir', default='../model_savedir', help='save checkpoints here')
    parser.add_argument('--base_datadir', default='../dataset/', help='save datasets here')
    # sampling info
    parser.add_argument('-s', '--sample', action='store_true', default=False)
    parser.add_argument('-sm', '--sample_mean', action='store_true', default=False)
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
    model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv = create_conv_acn_pcnn_models(info, args.model_loadpath)
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

