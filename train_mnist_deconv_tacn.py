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
from acn_models import tPTPriorNetwork, ACNres
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
    # acn prior with vqvae embedding
    acn_model = ACNres(code_len=info['code_length'],
                               input_size=info['input_channels'],
                               output_size=info['output_dim'],
                               encoder_output_size=info['encoder_output_size'],
                               hidden_size=info['hidden_size'],
                               ).to(info['device'])

    prior_model = tPTPriorNetwork(size_training_set=info['size_training_set'],
                               code_length=info['code_length'], k=info['num_k']).to(info['device'])
    prior_model.codes = prior_model.codes.to(info['device'])
    model_dict = {'acn_model':acn_model, 'prior_model':prior_model}
    parameters = []
    for name,model in model_dict.items():
        parameters+=list(model.parameters())
        print('created %s model with %s parameters' %(name,count_parameters(model)))

    model_dict['opt'] = optim.Adam(parameters, lr=info['learning_rate'])

    if args.model_loadpath !='':
       for name,model in model_dict.items():
            model_dict[name].load_state_dict(_dict[name+'_state_dict'])
    return model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv

def account_losses(loss_dict):
    ''' return avg losses for each loss sum in loss_dict based on total examples in running key'''
    loss_avg = {}
    for key in loss_dict.keys():
        if key != 'running':
            loss_avg[key] = loss_dict[key]/loss_dict['running']
    return loss_avg

def clip_parameters(model_dict, clip_val=10):
    for name, model in model_dict.items():
        if 'model' in name:
            clip_grad_value_(model.parameters(), clip_val)
    return model_dict

def forward_pass(model_dict, data, label, batch_index, phase, info):
    model_dict = set_model_mode(model_dict, phase)
    target = data = data.to(info['device'])
    bs,c,h,w = target.shape
    model_dict['opt'].zero_grad()
    data = F.dropout(data, p=info['dropout_rate'], training=True, inplace=False)
    z, u_q = model_dict['acn_model'](data)
    u_q_flat = u_q.view(bs, info['code_length'])
    if phase == 'train':
        # fit acn knn during training
        model_dict['prior_model'].update_codebook(batch_index, u_q_flat.detach())
    u_p, s_p = model_dict['prior_model'](u_q_flat)
    u_p = u_p.view(bs, 4, 7, 7)
    s_p = s_p.view(bs, 4, 7, 7)
    rec_dml =  model_dict['acn_model'].decode(z)
    return model_dict, data, target, u_q, u_p, s_p, rec_dml

def run(train_cnt, model_dict, data_dict, phase, info):
    st = time.time()
    loss_dict = {'running': 0,
             'kl':0,
             'rec_%s'%info['rec_loss_type']:0,
             'loss':0,
              }
    data_loader = data_dict[phase]
    num_batches = len(data_loader)
    for idx, (data, label, batch_index) in enumerate(data_loader):
        bs,c,h,w = data.shape
        fp_out = forward_pass(model_dict, data, label, batch_index, phase, info)
        model_dict, data, target, u_q, u_p, s_p, rec_dml = fp_out
        if idx == 0:
            log_ones = torch.zeros(bs, info['code_length']).to(info['device'])
        elif bs != log_ones.shape[0]:
            log_ones = torch.zeros(bs, info['code_length']).to(info['device'])
        kl = kl_loss_function(u_q.view(bs, info['code_length']), log_ones,
                              u_p.view(bs, info['code_length']), s_p.view(bs, info['code_length']),
                              reduction=info['reduction'])
        rec_loss = discretized_mix_logistic_loss(rec_dml, target, nr_mix=info['nr_logistic_mix'], reduction=info['reduction'])
        loss = kl+rec_loss
        loss_dict['running']+=bs
        loss_dict['loss']+=loss.item()
        loss_dict['kl']+= kl.item()
        loss_dict['rec_%s'%info['rec_loss_type']]+=rec_loss.item()
        if phase == 'train':
            model_dict = clip_parameters(model_dict)
            loss.backward()
            model_dict['opt'].step()
            train_cnt+=bs
        if idx == num_batches-2:
            # store example near end for plotting
            rec_yhat_batch = sample_from_discretized_mix_logistic(rec_dml, info['nr_logistic_mix'], only_mean=info['sample_mean'])
            example = {'data':rescale_inv(data.detach().cpu()),
                       'target':rescale_inv(target.detach().cpu()),
                       'vq_yhat':rescale_inv(rec_yhat_batch.detach().cpu())}
        if not idx % 10:
            print(train_cnt, idx, account_losses(loss_dict))

    loss_avg = account_losses(loss_dict)
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
                                                                       phase='train', info=info)
        epoch_cnt +=1
        train_cnt +=info['size_training_set']
        if not epoch_cnt % info['save_every_epochs'] or epoch_cnt == 1:
            # make a checkpoint
            print('starting valid phase')
            model_dict, data_dict, valid_loss_avg, valid_example = run(train_cnt,
                                                                           model_dict,
                                                                           data_dict,
                                                                           phase='valid', info=info)
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
            plot_example(train_img_filepath, train_example, num_plot=10)
            plot_example(valid_img_filepath, valid_example, num_plot=10)
            save_checkpoint(state_dict, filename=ckpt_filepath)

            plot_losses(info['train_cnts'],
                        info['train_losses'],
                        info['valid_losses'], name=plot_filepath, rolling_length=1)


def latent_walk(model_dict, data_dict, info):
    from skvideo.io import vwrite
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
            output = model_dict['conv_decoder'](latents)
            npst = target[si:si+1].detach().cpu().numpy()
            npen = target[ei:ei+1].detach().cpu().numpy()
            if info['rec_loss_type'] == 'dml':
                output = sample_from_discretized_mix_logistic(output, info['nr_logistic_mix'], only_mean=info['sample_mean'])
            npwalk = output.detach().cpu().numpy()
            # add multiple frames of each sample as a hacky way to make the
            # video more interpretable to humans
            walk_video = np.concatenate((npst, npst,
                                         npst, npst,
                                         npst, npst))
            for ww in range(npwalk.shape[0]):
                walk_video = np.concatenate((walk_video,
                                             npwalk[ww:ww+1], npwalk[ww:ww+1],
                                             npwalk[ww:ww+1], npwalk[ww:ww+1],
                                             npwalk[ww:ww+1], npwalk[ww:ww+1],
                                             ))
            walk_video = np.concatenate((walk_video,
                                         npen, npen,
                                         npen, npen,
                                         npen, npen))
            walk_video = (walk_video*255).astype(np.uint8)
            ## make movie
            print('writing walk movie')
            mname = output_savepath + '%s_s%s_e%s_walk.mp4'%(walki,sl,el)
            vwrite(mname, walk_video)
            print('finished %s'%mname)

def daydream(model_dict, data_dict, info):
    import matplotlib.transforms as mtrans
    from skvideo.io import vwrite
    # always be in eval mode
    num_examples = 10
    with torch.no_grad():
        for phase in ['train', 'valid']:
            data_loader = data_dict[phase]
            for idx, (data, label, batch_index) in enumerate(data_loader):
                break
            # only run data loader once
            plt_path = info['model_loadpath'].replace('.pt', '_%s_debug_daydream.png'%phase)
            mv_path = info['model_loadpath'].replace('.pt', '_%s_debug_daydream.mp4'%phase)
            f,ax = plt.subplots(num_examples, args.num_compare+2, sharex=True, sharey=True, figsize=(args.num_compare+2, num_examples))
            # always set phase == train because the prior model was trained
            # with noise added and will not perform with input without noise
            model_dict = set_model_mode(model_dict, phase=phase)
            target = data = data[:num_examples].to(info['device'])
            z, u_q = model_dict['acn_model'](data)
            rec_dml =  model_dict['acn_model'].decode(z)

            # get data for plotting later
            exyhat = sample_from_discretized_mix_logistic(rec_dml, info['nr_logistic_mix'], only_mean=True)
            npex = exyhat.cpu().detach().numpy()
            nptarget = target.cpu().detach().numpy()
            cnt = 0
            _,c,h,w = target.shape
            out_video = np.ones((args.num_compare*num_examples, h, w*2))
            for ii in range(num_examples):
                one_u_q = u_q[ii]*torch.ones((args.num_compare, 4, 7, 7)).to(info['device'])
                one_u_q_flat = one_u_q.view(args.num_compare, info['code_length'])
                one_u_p_flat, one_s_p_flat = model_dict['prior_model'](one_u_q_flat)
                print(one_u_p_flat.min(), one_u_p_flat.max())
                print(one_u_q_flat.min(), one_u_q_flat.max())
                # now we have several comparisons from this example
                # s_p is logsigma
                # 0.5 multiplier bc we parameterize the std dev not var -
                # see kld calculation - which is what defines std vs var
                if not ii%2:
                    print("sampling from prior does not work - using mean for example")
                    z_flat = one_u_q_flat+torch.exp(0.5*one_s_p_flat)*torch.randn(one_s_p_flat.shape).to(info['device'])
                else:
                    z_flat = one_u_p_flat+torch.exp(0.5*one_s_p_flat)*torch.randn(one_s_p_flat.shape).to(info['device'])
                z = z_flat.view(args.num_compare, 4, 7, 7)
                srec_dml =  model_dict['acn_model'].decode(z)
                syhat = sample_from_discretized_mix_logistic(srec_dml, info['nr_logistic_mix'], only_mean=True)
                npyhat = syhat.cpu().detach().numpy()
                ax[ii,0].matshow(nptarget[ii,0], cmap=plt.cm.gray)
                ax[ii,0].axis('off')
                ax[ii,1].matshow(npex[ii,0], cmap=plt.cm.gray)
                ax[ii,1].axis('off')
                for ic in range(args.num_compare):
                    ax[ii,ic+2].matshow(npyhat[ic,0], cmap=plt.cm.gray)
                    ax[ii,ic+2].axis('off')
                    out_video[cnt,:] = np.concatenate((npex[ii], npyhat[ic]), 2)
                    cnt+=1
            ax[0,0].set_title('true')
            ax[0,1].set_title('u_q')
            mid = 2+int(args.num_compare/2)
            ax[0,mid].set_title('u_p daydreams')
            print('saving daydream', plt_path)
            # rearange the axes for no overlap
            f.tight_layout()
            # Get the bounding boxes of the axes including text decorations
            r = f.canvas.get_renderer()
            get_bbox = lambda ax: ax.get_tightbbox(r).transformed(f.transFigure.inverted())
            bboxes = np.array(list(map(get_bbox, ax.flat)), mtrans.Bbox).reshape(ax.shape)

            #Get the minimum and maximum extent, get the coordinate half-way between those
            xmin = np.array(list(map(lambda b: b.x0, bboxes.flat))).reshape(ax.shape).min(axis=0)
            xmax = np.array(list(map(lambda b: b.x1, bboxes.flat))).reshape(ax.shape).min(axis=0)
            # min is left most extent of the subplot indexed
            # max is right most extent
            cmax = xmax[1]; cmin = xmin[2]
            cmid = cmax + (cmin-cmax)/2.0
            line = plt.Line2D((cmid,cmid),(0,1), color="r", linewidth=3)
            f.add_artist(line)
            plt.savefig(plt_path)
            plt.close()
            out_video = (((out_video+1)/2)*255).astype(np.uint8)
            inputdict = {"-r":"10"}
            vwrite(mv_path, out_video, inputdict=inputdict)
#

def call_plot(model_dict, data_dict, info):
    from utils import tsne_plot
    from utils import pca_plot
    # always be in eval mode
    model_dict = set_model_mode(model_dict, 'valid')
    with torch.no_grad():
        for phase in ['valid', 'train']:
            data_loader = data_dict[phase]
            for idx, (data, label, batch_index) in enumerate(data_loader):
                fp_out = forward_pass(model_dict, data, label, batch_index, phase, info)
                model_dict, data, target, u_q, u_p, s_p, rec_dml = fp_out
                bs = data.shape[0]
                u_q_flat = u_q.view(bs, info['code_length'])
                X = u_q_flat.cpu().numpy()
                color = label
                images = target[:,0].cpu().numpy()
                if args.tsne:
                    param_name = '_tsne_%s_P%s.html'%(phase, info['perplexity'])
                    html_path = info['model_loadpath'].replace('.pt', param_name)
                    tsne_plot(X=X, images=images, color=color,
                          perplexity=info['perplexity'],
                          html_out_path=html_path, serve=False)
                if args.pca:
                    param_name = '_pca_%s.html'%(phase)
                    html_path = info['model_loadpath'].replace('.pt', param_name)
                    pca_plot(X=X, images=images, color=color,
                              html_out_path=html_path, serve=False)
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
    parser.add_argument('-bs', '--batch_size', default=84, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--input_channels', default=1, type=int, help='num of channels of input')
    parser.add_argument('--target_channels', default=1, type=int, help='num of channels of target')
    parser.add_argument('--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-e', '--exp_name', default='deconv_acn_res_convthruout_repq_bigprior', help='name of experiment')
    parser.add_argument('-dr', '--dropout_rate', default=0.0, type=float)
    parser.add_argument('-r', '--reduction', default='sum', type=str, choices=['sum', 'mean'])
    parser.add_argument('--rec_loss_type', default='dml', type=str, help='name of loss. options are dml', choices=['dml'])
    parser.add_argument('--nr_logistic_mix', default=10, type=int)
    # acn model setup
    parser.add_argument('-cl', '--code_length', default=196, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)

    #parser.add_argument('-kl', '--kl_beta', default=.5, type=float, help='scale kl loss')
    parser.add_argument('--last_layer_bias', default=0.0, help='bias for output decoder - should be 0 for dml')
    parser.add_argument('--encoder_output_size', default=784, help='output as a result of the flatten of the encoder - found experimentally')
    #parser.add_argument('--encoder_output_size', default=6272, help='output as a result of the flatten of the encoder - found experimentally')
    parser.add_argument('-sm', '--sample_mean', action='store_true', default=False)
    # dataset setup
    parser.add_argument('-d',  '--dataset_name', default='FashionMNIST', help='which mnist to use', choices=['MNIST', 'FashionMNIST'])
    parser.add_argument('--model_savedir', default='../model_savedir', help='save checkpoints here')
    parser.add_argument('--base_datadir', default='../dataset/', help='save datasets here')
    # latent pca/tsne info
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--tsne', action='store_true', default=False)
    parser.add_argument('-p', '--perplexity', default=10, type=int, help='perplexity used in scikit-learn tsne call')
    # daydream
    parser.add_argument('-dd', '--daydream', action='store_true', default=False)
    parser.add_argument('-nc', '--num_compare', default=20, type=int, help='number of comparisons to daydream from prior')
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
        print('loading last model....')
        print(args.model_loadpath)
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
    model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv = create_models(info, args.model_loadpath)
    kldis = nn.KLDivLoss(reduction=info['reduction'])
    lsm = nn.LogSoftmax(dim=1)
    sm = nn.Softmax(dim=1)
    if args.tsne or args.pca:
        call_plot(model_dict, data_dict, info)
    if args.walk:
        latent_walk(model_dict, data_dict, info)
    if args.daydream:
        daydream(model_dict, data_dict, info)
    # only train if we weren't asked to do anything else
    if not max([args.daydream, args.tsne, args.pca, args.walk]):
        write_log_files(info)
        train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info, rescale_inv)

