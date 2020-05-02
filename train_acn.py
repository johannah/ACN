"""
Associative Compression Network based on https://arxiv.org/pdf/1804.02476v2.pdf
determined this architecture based on experiments in github.com/johannah/ACN

"""

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

from utils import create_new_info_dict, save_checkpoint, seed_everything
from utils import plot_example, plot_losses, count_parameters
from utils import set_model_mode, kl_loss_function, write_log_files
from utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic
from utils import create_mnist_datasets
from acn_models import tPTPriorNetwork, ACNresMNIST, ACNVQVAEresMNIST, ACNVQVAEresMNISTsmall

import torchvision
from IPython import embed

# setup loss-specific parameters for data
# data going into dml should be bt -1 and 1
# atari data coming in is uint 0 to 255
norm_by = 255.0
rescale = lambda x: ((x / norm_by) * 2.)-1
rescale_inv = lambda x: 255*((x+1)/2.)

def create_models(info, model_loadpath=''):
    '''
    load details of previous model if applicable, otherwise create new models
    '''
    train_cnt = 0
    epoch_cnt = 0

    # use argparse device no matter what info dict is loaded
    preserve_args = ['device', 'batch_size', 'save_every_epochs',
                     'base_filepath', 'model_loadpath', 'perplexity',
                     'num_examples_to_train'
                     ]
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
    if info['rec_loss_type'] == 'dml':
        # data going into dml should be bt -1 and 1
        rescale = lambda x: (x - 0.5) * 2.
        rescale_inv = lambda x: (0.5 * x) + 0.5
    if info['rec_loss_type'] == 'bce':
        rescale = lambda x: x
        rescale_inv = lambda x: x
    dataset_transforms = transforms.Compose([transforms.ToTensor(), rescale])
    data_output = create_mnist_datasets(dataset_name=info['dataset_name'],
                           base_datadir=info['base_datadir'],
                           batch_size=info['batch_size'],
                           dataset_transforms=dataset_transforms)
    data_dict, size_training_set, num_input_chans, num_output_chans, hsize, wsize = data_output
    info['num_input_chans'] = num_input_chans
    info['num_output_chans'] = num_input_chans
    info['hsize'] = hsize
    info['wsize'] = wsize

    if not loaded:
        info['size_training_set'] = size_training_set


    # pixel cnn architecture is dependent on loss
    # for dml prediction, need to output mixture of size nmix
    info['nmix'] =  (2*info['nr_logistic_mix']+info['nr_logistic_mix'])*info['target_channels']
    info['output_dim']  = info['nmix']

    # setup models
    # acn prior with vqvae embedding
    if info['vq_decoder']:
        acn_model = ACNVQVAEresMNIST(code_len=info['code_length'],
                               input_size=info['input_channels'],
                               output_size=info['output_dim'],
                               hidden_size=info['hidden_size'],
                               num_clusters=info['num_vqk'],
                               num_z=info['num_z'],
                               ).to(info['device'])
    else:
        acn_model = ACNresMNIST(code_len=info['code_length'],
                               input_size=info['input_channels'],
                               output_size=info['output_dim'],
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

    if model_loadpath !='':
        for name, model in model_dict.items():
            lname = name+'_state_dict'
            # some old models were saved with vq
            if lname == 'acn_model_state_dict' and lname not in _dict.keys():
                lname = 'vq_acn_model_state_dict'
            if '_model' in name:
                model_dict[name].load_state_dict(_dict[lname])
        if 'codes' in _dict.keys():
            model_dict['prior_model'].codes = _dict['codes']
        else:
            # find codes
            model_dict = set_codes_from_model(data_dict, model_dict, info)
            state_dict = {}
            for key, model in model_dict.items():
                state_dict[key+'_state_dict'] = model.state_dict()
            info['train_cnts'].append(train_cnt)
            info['epoch_cnt'] = epoch_cnt
            state_dict['codes'] = model_dict['prior_model'].codes
            state_dict['info'] = info
            save_checkpoint(state_dict, filename=model_loadpath+'.cd')

    return model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv

def set_codes_from_model(data_dict, model_dict, info):
    print('setting model codes')
    data_loader = data_dict['train']
    for idx, (data, label, batch_index) in enumerate(data_loader):
        for key in model_dict.keys():
            model_dict[key].zero_grad()

        forward_pass(model_dict, data, label, batch_index, 'train', info)
    return model_dict


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

def forward_pass(model_dict, data, label, batch_indexes, phase, info):
    # prepare data in appropriate way
    model_dict = set_model_mode(model_dict, phase)
    target = data = data.to(info['device'])
    # input is bt 0 and 1
    bs,c,h,w = data.shape
    z, u_q = model_dict['acn_model'](data)
    u_q_flat = u_q.view(bs, info['code_length'])
    if phase == 'train':
        # check that we are getting what we think we are getting from the replay
        # buffer
        assert batch_indexes.max() < model_dict['prior_model'].codes.shape[0]
        model_dict['prior_model'].update_codebook(batch_indexes, u_q_flat.detach())

    u_p, s_p = model_dict['prior_model'](u_q_flat)
    u_p = u_p.view(bs, model_dict['acn_model'].bottleneck_channels, model_dict['acn_model'].eo, model_dict['acn_model'].eo)
    s_p = s_p.view(bs, model_dict['acn_model'].bottleneck_channels, model_dict['acn_model'].eo, model_dict['acn_model'].eo)
    if info['vq_decoder']:
        rec_dml, z_e_x, z_q_x, latents =  model_dict['acn_model'].decode(z)
        return model_dict, data, target, rec_dml, u_q, u_p, s_p, rec_dml, z_e_x, z_q_x, latents
    else:
        rec_dml =  model_dict['acn_model'].decode(z)
        return model_dict, data, target, rec_dml, u_q, u_p, s_p, rec_dml

def run(train_cnt, model_dict, data_dict, phase, info):
    st = time.time()
    loss_dict = {'running': 0,
             'kl':0,
             'rec_%s'%info['rec_loss_type']:0,
             'loss':0,
              }
    if info['vq_decoder']:
        loss_dict['vq'] = 0
        loss_dict['commit'] = 0

    dataset = data_dict[phase]
    num_batches = len(dataset)//info['batch_size']
    print(phase, 'num batches', num_batches)
    set_model_mode(model_dict, phase)
    torch.set_grad_enabled(phase=='train')
    batch_cnt = 0

    data_loader = data_dict[phase]
    num_batches = len(data_loader)
    for idx, (data, label, batch_index) in enumerate(data_loader):
        for key in model_dict.keys():
            model_dict[key].zero_grad()
        fp_out = forward_pass(model_dict, data, label, batch_index, phase, info)
        if info['vq_decoder']:
            model_dict, data, target, rec_dml, u_q, u_p, s_p, rec_dml, z_e_x, z_q_x, latents = fp_out
        else:
            model_dict, data, target, rec_dml, u_q, u_p, s_p, rec_dml = fp_out
        bs,c,h,w = data.shape
        if batch_cnt == 0:
            log_ones = torch.zeros(bs, info['code_length']).to(info['device'])
        if bs != log_ones.shape[0]:
            log_ones = torch.zeros(bs, info['code_length']).to(info['device'])
        kl = kl_loss_function(u_q.view(bs, info['code_length']), log_ones,
                              u_p.view(bs, info['code_length']), s_p.view(bs, info['code_length']),
                              reduction=info['reduction'])
        rec_loss = discretized_mix_logistic_loss(rec_dml, target, nr_mix=info['nr_logistic_mix'], reduction=info['reduction'])

        if info['vq_decoder']:
            vq_loss = F.mse_loss(z_q_x, z_e_x.detach(), reduction=info['reduction'])
            commit_loss = F.mse_loss(z_e_x, z_q_x.detach(), reduction=info['reduction'])
            commit_loss *= info['vq_commitment_beta']
            loss_dict['vq']+= vq_loss.detach().cpu().item()
            loss_dict['commit']+= commit_loss.detach().cpu().item()
            loss = kl+rec_loss+commit_loss+vq_loss
        else:
            loss = kl+rec_loss

        loss_dict['running']+=bs
        loss_dict['rec_%s'%info['rec_loss_type']]+=rec_loss.detach().cpu().item()
        loss_dict['loss']+=loss.detach().cpu().item()
        loss_dict['kl']+= kl.detach().cpu().item()
        loss_dict['running']+=bs
        loss_dict['loss']+=loss.detach().cpu().item()
        loss_dict['kl']+= kl.detach().cpu().item()
        loss_dict['rec_%s'%info['rec_loss_type']]+=rec_loss.detach().cpu().item()
        if phase == 'train':
            model_dict = clip_parameters(model_dict)
            loss.backward()
            model_dict['opt'].step()
            train_cnt+=bs
        if batch_cnt == num_batches-1:
            # store example near end for plotting
            rec_yhat = sample_from_discretized_mix_logistic(rec_dml, info['nr_logistic_mix'], only_mean=info['sample_mean'], sampling_temperature=info['sampling_temperature'])
            example = {
                  'target':data.detach().cpu().numpy(),
                  'rec':rec_yhat.detach().cpu().numpy(),
                   }
        #print(prof)
        if not batch_cnt % 100:
            print(train_cnt, batch_cnt, account_losses(loss_dict))
            print(phase, 'cuda', torch.cuda.memory_allocated(device=None))
        batch_cnt+=1


    loss_avg = account_losses(loss_dict)
    torch.cuda.empty_cache()
    print("finished %s after %s secs at cnt %s"%(phase,
                                                time.time()-st,
                                                train_cnt,
                                                ))
    del data; del target
    return loss_avg, example


def train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info, rescale_inv):
    print('starting training routine')
    base_filepath = info['base_filepath']
    base_filename = os.path.split(info['base_filepath'])[1]
    while train_cnt < info['num_examples_to_train']:
        print('starting epoch %s on %s'%(epoch_cnt, info['device']))

        train_loss_avg, train_example = run(train_cnt,
                                                model_dict,
                                                data_dict,
                                                phase='train', info=info)

        epoch_cnt +=1
        train_cnt +=info['size_training_set']
        if not epoch_cnt % info['save_every_epochs'] or epoch_cnt == 1:
            # make a checkpoint
            print('starting valid phase')
            valid_loss_avg, valid_example = run(train_cnt,
                                                 model_dict,
                                                 data_dict,
                                                 phase='valid', info=info)
            for loss_key in train_loss_avg.keys():
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
            state_dict['codes'] = model_dict['prior_model'].codes
            state_dict['info'] = info
            ckpt_filepath = os.path.join(base_filepath, "%s_%010dex.pt"%(base_filename, train_cnt))
            train_img_filepath = os.path.join(base_filepath,"%s_%010d_train_rec.png"%(base_filename, train_cnt))
            valid_img_filepath = os.path.join(base_filepath, "%s_%010d_valid_rec.png"%(base_filename, train_cnt))
            plot_filepath = os.path.join(base_filepath, "%s_%010d_loss.png"%(base_filename, train_cnt))
            plot_example(train_img_filepath, train_example, num_plot=10)
            plot_example(valid_img_filepath, valid_example, num_plot=10)
            save_checkpoint(state_dict, filename=ckpt_filepath)

            plot_losses(info['train_cnts'],
                        info['train_losses'],
                        info['valid_losses'], name=plot_filepath, rolling_length=1)

        torch.cuda.empty_cache()

def call_plot(model_dict, data_dict, info, tsne, pca):
    from utils import tsne_plot
    from utils import pca_plot
    from sklearn.cluster import KMeans
    # always be in eval mode - so we dont swap neighbors
    model_dict = set_model_mode(model_dict, 'valid')
    with torch.no_grad():
        #for phase in ['valid', 'train']:
        for phase in ['train', 'valid']:
            data_loader = data_dict[phase]
            for idx, (data, label, batch_index) in enumerate(data_loader):

                fp_out = forward_pass(model_dict, data, label, batch_index, phase, info)
                if info['vq_decoder']:
                    model_dict, data, target, rec_dml, u_q, u_p, s_p, rec_dml, z_e_x, z_q_x, latents = fp_out
                else:
                    model_dict, data, target, rec_dml, u_q, u_p, s_p, rec_dml = fp_out

                bs,c,h,w=data.shape
                rec_yhat = sample_from_discretized_mix_logistic(rec_dml, info['nr_logistic_mix'], only_mean=info['sample_mean'], sampling_temperature=info['sampling_temperature'])
                data = data.detach().cpu().numpy()
                rec = rec_yhat.detach().cpu().numpy()
                u_q_flat = u_q.view(bs, info['code_length'])
                n = min([20,bs])

                n_neighbors = args.num_k
                all_neighbor_distances, all_neighbor_indexes = model_dict['prior_model'].kneighbors(u_q_flat, n_neighbors=n_neighbors)
                all_neighbor_indexes = all_neighbor_indexes.cpu().numpy()
                all_neighbor_distances = all_neighbor_distances.cpu().numpy()

                n_cols = 2+n_neighbors
                u_q_np = deepcopy(u_q.cpu().numpy())
                umin = u_q_np.min()
                umax = u_q_np.max()
                u_q_np = (u_q_np-umin)/(umax-umin)

                nbatch_index = batch_index.cpu().numpy()
                for i in np.arange(0,n*2,2):
                    # bi 5136
                    neighbor_indexes = all_neighbor_indexes[i]
                    code = u_q[i].view((1, model_dict['acn_model'].bottleneck_channels, model_dict['acn_model'].eo, model_dict['acn_model'].eo)).cpu().numpy()
                    f,ax = plt.subplots(4,n_cols)
                    ax[0,0].set_title('t%s'%nbatch_index[i])
                    ax[0,0].matshow(data[i,0])
                    ax[1,0].set_title('rec')
                    ax[1,0].matshow(rec[i,0])
                    ax[2,0].matshow(code[0,0])
                    ax[3,0].matshow(code[0,1])
                    neighbor_data = torch.stack([data_dict['train'].dataset.indexed_dataset[index][0] for index in neighbor_indexes])
                    neighbor_labels = torch.stack([data_dict['train'].dataset.indexed_dataset[index][1] for index in neighbor_indexes])
                    # u_q_flat
                    neighbor_codes_flat = model_dict['prior_model'].codes[neighbor_indexes]
                    neighbor_codes = neighbor_codes_flat.view(n_neighbors, model_dict['acn_model'].bottleneck_channels, model_dict['acn_model'].eo, model_dict['acn_model'].eo)
                    if info['vq_decoder']:
                        neighbor_rec_dml, _, _, _ =  model_dict['acn_model'].decode(neighbor_codes.to(info['device']))
                    else:
                        neighbor_rec_dml =  model_dict['acn_model'].decode(neighbor_codes.to(info['device']))
                    neighbor_rec_yhat = sample_from_discretized_mix_logistic(neighbor_rec_dml, info['nr_logistic_mix'], only_mean=info['sample_mean'], sampling_temperature=info['sampling_temperature']).cpu().numpy()
                    neighbor_data = ((neighbor_data*2)-1).cpu().numpy()
                    neighbor_codes = neighbor_codes.cpu().numpy()
                    for ni in range(n_neighbors):
                        index = all_neighbor_indexes[i,ni].item()
                        neighbor_label = neighbor_labels[ni]
                        ax[0,ni+2].set_title('%s %s'%(neighbor_label.cpu().numpy(), index))
                        ax[0,ni+2].matshow(neighbor_data[ni,0])
                        ax[1,ni+2].matshow(neighbor_rec_yhat[ni,0])
                        ax[2,ni+2].matshow(neighbor_codes[ni,0])
                        ax[3,ni+2].matshow(neighbor_codes[ni,1])

                    [ax[0,col].axis('off') for col in range(n_cols)]
                    [ax[1,col].axis('off') for col in range(n_cols)]
                    [ax[2,col].axis('off') for col in range(n_cols)]
                    [ax[3,col].axis('off') for col in range(n_cols)]
                    plt.subplots_adjust(wspace=0, hspace=0)
                    plt.tight_layout()
                    plt_path = info['model_loadpath'].replace('.pt', '_neighbors_%s_%06d_plt.png'%(phase,nbatch_index[i]))
                    print('plotting', plt_path)
                    plt.savefig(plt_path)
                    plt.close()
                X = u_q_flat.cpu().numpy()
                km = KMeans(n_clusters=10)
                y = km.fit_predict(X)
                # color points based on clustering, label, or index
                color = y#label.cpu().numpy() #y#batch_indexes
                if tsne:
                    param_name = '_tsne_%s_P%s.html'%(phase, info['perplexity'])
                    html_path = info['model_loadpath'].replace('.pt', param_name)
                    tsne_plot(X=X, images=data[:,0], color=color,
                          perplexity=info['perplexity'],
                          html_out_path=html_path, serve=False)
                if pca:
                    param_name = '_pca_%s.html'%(phase)
                    html_path = info['model_loadpath'].replace('.pt', param_name)
                    pca_plot(X=X, images=data[:,0], color=color,
                              html_out_path=html_path, serve=False)

                break

def save_latents(model_dict, data_dict, info):
    # always be in eval mode - so we dont swap neighbors
    model_dict = set_model_mode(model_dict, 'valid')
    results = {}
    all_vq_latents = []
    with torch.no_grad():
        for phase in ['valid', 'train']:
            data_loader = data_dict[phase]
            for idx, (data, label, batch_index) in enumerate(data_loader):
                fp_out = forward_pass(model_dict, data, label, batch_index, phase, info)
                if info['vq_decoder']:
                    model_dict, data, target, rec_dml, u_q, u_p, s_p, rec_dml, z_e_x, z_q_x, latents = fp_out
                else:
                    model_dict, data, target, rec_dml, u_q, u_p, s_p, rec_dml = fp_out
                bs,c,h,w=data.shape
                u_q_flat = u_q.view(bs, info['code_length'])
                n_neighbors = args.num_k
                neighbor_distances, neighbor_indexes = model_dict['prior_model'].kneighbors(u_q_flat, n_neighbors=n_neighbors)
                try:
                    if not idx:
                        all_indexes = batch_index.cpu().numpy()
                        all_acn_uq = u_q.cpu().numpy()
                        all_neighbors = neighbor_indexes.cpu().numpy()
                        all_neighbor_distances = neighbor_distances.cpu().numpy()
                        if info['vq_decoder']:
                            all_vq_latents = latents.cpu().numpy()
                    else:
                        all_indexes = np.append(all_indexes, batch_index.cpu().numpy())
                        all_acn_uq = np.vstack((all_acn_uq, u_q.cpu().numpy()))
                        all_neighbors = np.vstack((all_neighbors, neighbor_indexes.cpu().numpy()))
                        all_neighbor_distances = np.vstack((all_neighbor_distances,
                                                            neighbor_distances.cpu().numpy()))
                        if info['vq_decoder']:
                            all_vq_latents = np.vstack((all_vq_latents, latents.cpu().numpy()))
                except:
                    embed()

                print('finished', all_neighbors.shape[0])
            results['valid'] = {'index':all_indexes, 'acn_uq':all_acn_uq,
                                'neighbor_train_indexes':all_neighbors,
                                'neighbor_distances':all_neighbor_distances,
                                'vq_latents':all_vq_latents}

        base_filepath = info['base_filepath']
        base_filename = os.path.split(info['base_filepath'])[1]
        l_filepath = os.path.join(base_filepath, "%s_%010d_output"%(base_filename, info['train_cnts'][-1]))
        np.savez(l_filepath, results)

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






if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='train acn')
    # experiment parameters
    parser.add_argument('--base_datadir', default='../dataset/', help='fetch/save dataset from this dir')
    parser.add_argument('-sd', '--model_savedir', default='../model_savedir', help='save experiments in this dir')
    parser.add_argument('-l', '--model_loadpath', default='', help='load specific model to resume training or sample')
    parser.add_argument('-ll', '--load_last_model', default='',  help='load last saved model from this directory')
    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='if true, use cuda device. not setup for gpu-specific allocation')
    parser.add_argument('-ee', '--eval_experiment', default=False, action='store_true', help='if true, eval experiment by running every model in loaddir')
    parser.add_argument('--seed', default=394, help='experiment random seed')
    parser.add_argument('-bs', '--batch_size', default=256, type=int)
    parser.add_argument('--num_threads', default=2, help='how many threads to allow pytorch')
    parser.add_argument('-se', '--save_every_epochs', default=10, type=int, help='how often to validate and checkpoint model')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('-n', '--num_examples_to_train', default=500000000, type=int, help='total data points to show to model before stopping training')
    parser.add_argument('-e', '--exp_name', default='validation_acn', help='name of experiment')

    # data details
    parser.add_argument('-d',  '--dataset_name', default='FashionMNIST', help='which mnist to use', choices=['MNIST', 'FashionMNIST'])
    parser.add_argument('--input_channels', default=1, type=int, help='num of channels of input')
    parser.add_argument('--target_channels', default=1, type=int, help='num of channels of target')

    # loss parameters
    parser.add_argument('-r', '--reduction', default='sum', type=str, choices=['sum', 'mean'], help='how to reduce losses, tested with sum')
    parser.add_argument('--rec_loss_type', default='dml', type=str, help='name of loss. options are dml', choices=['dml'])
    parser.add_argument('--nr_logistic_mix', default=10, type=int, help='number of mixes used with dml loss - tested with 10')
    parser.add_argument('-sm', '--sample_mean', action='store_true', default=False, help='if true, sample mean from dml for yhat')
    parser.add_argument('-st', '--sampling_temperature', default=0.1, help='temperature to sample dml')

    # architecture choices (will not update when loading a previously trained model)
    # acn model setup
    parser.add_argument('-cl', '--code_length', default=98, type=int, help='this is dependent on architecture choice and will need to be modified if structure of acn model changes')
    parser.add_argument('-k', '--num_k', default=5, type=int, help='number of nearest neighbors to use in acn update')
    parser.add_argument('--hidden_size', default=256, type=int, help='hidden size of acn')
    parser.add_argument('-vq', '--vq_decoder', action='store_true', default=False, help='use vq decoder')
    # vq model setup
    parser.add_argument('--vq_commitment_beta', default=0.25, help='scale for loss 3 in vqvae - how hard to enforce commitment to cluster')
    parser.add_argument('--num_vqk', default=512, type=int, help='num k used in vqvae')
    parser.add_argument('--num_z', default=64, type=int, help='size of z in vqvae')
    # sampling latent pca/tsne info
    parser.add_argument('-sl', '--save_latents',  action='store_true', default=False, help='write latents and neighbors out to file')
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--tsne', action='store_true', default=False)
    parser.add_argument('-p', '--perplexity', default=10, type=int, help='perplexity used in scikit-learn tsne call')


    args = parser.parse_args()
    if args.vq_decoder:
        args.exp_name+='_vq'
    random_state = np.random.RandomState(args.seed)
    # note - when reloading model, this will use the seed given in args - not
    # the original random seed
    seed_everything(args.seed, args.num_threads)
    # get naming scheme
    loaded = True
    if args.eval_experiment:
        model_paths = sorted(glob(os.path.join(args.load_last_model, '*.pt')))
        base_filepath = args.load_last_model
        info = create_new_info_dict(vars(args), base_filepath, __file__)
        for model_loadpath in model_paths:
            print('running eval on experiment', model_loadpath)
            model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv = create_models(info, model_loadpath)
            call_plot(model_dict, data_dict, info, args.tsne, args.pca)
            save_latents(model_dict, data_dict, info)
    else:
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
            exp_num = 0
            base_filepath = os.path.join(args.model_savedir, args.exp_name, args.exp_name+'_%02d'%exp_num)
            while os.path.exists(base_filepath):
                exp_num+=1
                base_filepath = os.path.join(args.model_savedir, args.exp_name, args.exp_name+'_%02d'%exp_num)
            loaded = False
        print('base filepath is %s'%base_filepath)
        # setup loss-specific parameters for data
        info = create_new_info_dict(vars(args), base_filepath, __file__)

        model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv = create_models(info, args.model_loadpath)
        if max([args.sample, args.tsne, args.pca]):
            call_plot(model_dict, data_dict, info, args.tsne, args.pca)
            save_latents(model_dict, data_dict, info)
        elif args.save_latents:
            save_latents(model_dict, data_dict, info)
        else:
            # only train if we weren't asked to do anything else
            write_log_files(info)
            train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info, rescale_inv)
