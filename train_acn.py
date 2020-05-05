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
        if os.path.exists(model_loadpath+'.cd'):
            model_loadpath = model_loadpath+'.cd'
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

    model_dict = {'acn_model':acn_model, 'prior_model':prior_model}
    parameters = []
    for name,model in model_dict.items():
        parameters+=list(model.parameters())
        print('created %s model with %s parameters' %(name,count_parameters(model)))

    model_dict['opt'] = optim.Adam(parameters, lr=info['learning_rate'])

    if model_loadpath !='':
        for name, model in model_dict.items():
            if '_model' in name:
                lname = name+'_state_dict'
                model_dict[name].load_state_dict(_dict[lname])
        same = (_dict['codes'] == model_dict['prior_model'].codes.cpu()).sum().item()
        # make sure that loaded codes are the same
        #model_dict = set_codes_from_model(data_dict, model_dict, info)
        assert same == _dict['codes'].shape[0]* _dict['codes'].shape[1]
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
        return model_dict, data, target, rec_dml, u_q, u_p, s_p, z_e_x, z_q_x, latents
    else:
        rec_dml =  model_dict['acn_model'].decode(z)
        return model_dict, data, target, rec_dml, u_q, u_p, s_p

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
            model_dict, data, target, rec_dml, u_q, u_p, s_p, z_e_x, z_q_x, latents = fp_out
        else:
            model_dict, data, target, rec_dml, u_q, u_p, s_p = fp_out
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

def call_plot(model_dict, data_dict, info, sample, tsne, pca):
    from utils import tsne_plot
    from utils import pca_plot
    from sklearn.cluster import KMeans
    # always be in eval mode - so we dont swap neighbors
    model_dict = set_model_mode(model_dict, 'valid')
    srandom_state = np.random.RandomState(1234)
    with torch.no_grad():
        for phase in ['train', 'valid']:
            batch_index = srandom_state.randint(0,len(data_dict[phase].dataset),info['batch_size'])
            print(batch_index)
            data = torch.stack([data_dict[phase].dataset.indexed_dataset[index][0] for index in batch_index])
            label = torch.stack([data_dict[phase].dataset.indexed_dataset[index][1] for index in batch_index])
            batch_index = torch.LongTensor(batch_index)
            data = torch.FloatTensor(data)
            fp_out = forward_pass(model_dict, data, label, batch_index, 'valid', info)
            if info['vq_decoder']:
                model_dict, data, target, rec_dml, u_q, u_p, s_p, z_e_x, z_q_x, latents = fp_out
            else:
                model_dict, data, target, rec_dml, u_q, u_p, s_p = fp_out

            bs,c,h,w=data.shape
            rec_yhat = sample_from_discretized_mix_logistic(rec_dml, info['nr_logistic_mix'], only_mean=info['sample_mean'], sampling_temperature=info['sampling_temperature'])
            data = data.detach().cpu().numpy()
            rec = rec_yhat.detach().cpu().numpy()
            u_q_flat = u_q.view(bs, info['code_length'])
            # choose limited number to plot
            n = min([20,bs])
            n_neighbors = args.num_k
            if sample:
                all_neighbor_distances, all_neighbor_indexes = model_dict['prior_model'].kneighbors(u_q_flat, n_neighbors=n_neighbors)
                all_neighbor_indexes = all_neighbor_indexes.cpu().numpy()
                all_neighbor_distances = all_neighbor_distances.cpu().numpy()

                n_cols = 2+n_neighbors
                tbatch_index = batch_index.cpu().numpy()
                np_label = label.cpu().numpy()
                for i in np.arange(0,n):
                    # plot each base image
                    plt_path = info['model_loadpath'].replace('.pt', '_batch_rec_neighbors_%s_%06d_plt.png'%(phase,tbatch_index[i]))
                    # bi 5136
                    neighbor_indexes = all_neighbor_indexes[i]
                    code = u_q[i].view((1, model_dict['acn_model'].bottleneck_channels, model_dict['acn_model'].eo, model_dict['acn_model'].eo)).cpu().numpy()
                    f,ax = plt.subplots(4,n_cols)
                    ax[0,0].set_title('L%sI%s'%(np_label[i], tbatch_index[i]))
                    ax[0,0].set_ylabel('true')
                    ax[0,0].matshow(data[i,0])
                    ax[1,0].set_title('I%s'%tbatch_index[i])
                    ax[1,0].set_ylabel('rec')
                    ax[1,0].matshow(rec[i,0])
                    ax[2,0].matshow(code[0,0])
                    ax[3,0].matshow(code[0,1])

                    neighbor_data = torch.stack([data_dict['train'].dataset.indexed_dataset[index][0] for index in neighbor_indexes])
                    neighbor_label = torch.stack([data_dict['train'].dataset.indexed_dataset[index][1] for index in neighbor_indexes])
                    # u_q_flat
                    neighbor_codes_flat = model_dict['prior_model'].codes[neighbor_indexes]
                    neighbor_codes = neighbor_codes_flat.view(n_neighbors, model_dict['acn_model'].bottleneck_channels, model_dict['acn_model'].eo, model_dict['acn_model'].eo)
                    if info['vq_decoder']:
                        neighbor_rec_dml, _, _, _ =  model_dict['acn_model'].decode(neighbor_codes.to(info['device']))
                    else:
                        neighbor_rec_dml = model_dict['acn_model'].decode(neighbor_codes.to(info['device']))
                    neighbor_data = neighbor_data.cpu().numpy()
                    neighbor_rec_yhat = sample_from_discretized_mix_logistic(neighbor_rec_dml, info['nr_logistic_mix'], only_mean=info['sample_mean'], sampling_temperature=info['sampling_temperature']).cpu().numpy()
                    for ni in range(n_neighbors):
                        nindex = all_neighbor_indexes[i,ni].item()
                        nlabel = neighbor_label[ni].cpu().numpy()
                        ncode = neighbor_codes[ni].cpu().numpy()
                        ax[0,ni+2].set_title('L%sI%s'%(nlabel, nindex))
                        ax[0,ni+2].matshow(neighbor_data[ni,0])
                        ax[1,ni+2].matshow(neighbor_rec_yhat[ni,0])
                        ax[2,ni+2].matshow(ncode[0])
                        ax[3,ni+2].matshow(ncode[1])
                    ax[2,0].set_ylabel('lc0')
                    ax[3,0].set_ylabel('lc1')
                    [ax[xx,0].set_xticks([]) for xx in range(4)]
                    [ax[xx,0].set_yticks([]) for xx in range(4)]
                    for xx in range(4):
                        [ax[xx,col].axis('off') for col in range(1, n_cols)]
                    plt.subplots_adjust(wspace=0, hspace=0)
                    plt.tight_layout()
                    print('plotting', plt_path)
                    plt.savefig(plt_path)
                    plt.close()
            X = u_q_flat.cpu().numpy()
            #km = KMeans(n_clusters=10)
            #y = km.fit_predict(X)
            # color points based on clustering, label, or index
            color = label.cpu().numpy() #y #batch_indexes
            if tsne:
                param_name = '_tsne_%s_P%s.html'%(phase, info['perplexity'])
                html_path = info['model_loadpath'].replace('.pt', param_name)
                if not os.path.exists(html_path):
                    tsne_plot(X=X, images=data[:,0], color=color,
                          perplexity=info['perplexity'],
                          html_out_path=html_path, serve=False)
            if pca:
                param_name = '_pca_%s.html'%(phase)
                html_path = info['model_loadpath'].replace('.pt', param_name)
                if not os.path.exists(html_path):
                    pca_plot(X=X, images=data[:,0], color=color,
                              html_out_path=html_path, serve=False)


def save_latents(l_filepath, model_dict, data_dict, info):
    # always be in eval mode - so we dont swap neighbors
    all_vq_latents = []
    with torch.no_grad():
        for phase in ['train','valid']:
            if not os.path.exists(l_filepath+'_%s.npz'%phase):
                model_dict = set_model_mode(model_dict, 'valid')
                data_loader = data_dict[phase]
                for idx, (data, label, batch_index) in enumerate(data_loader):
                    fp_out = forward_pass(model_dict, data, label, batch_index, phase, info)
                    if info['vq_decoder']:
                        model_dict, data, target, rec_dml, u_q, u_p, s_p, z_e_x, z_q_x, latents = fp_out
                    else:
                        model_dict, data, target, rec_dml, u_q, u_p, s_p = fp_out
                    bs,c,h,w=data.shape
                    u_q_flat = u_q.view(bs, info['code_length'])
                    n_neighbors = info['num_k']
                    neighbor_distances, neighbor_indexes = model_dict['prior_model'].kneighbors(u_q_flat, n_neighbors=n_neighbors)
                    if not idx:
                        all_indexes = batch_index.cpu().numpy()
                        all_labels = label.cpu().numpy()
                        all_acn_uq = u_q.cpu().numpy()
                        all_neighbors = neighbor_indexes.cpu().numpy()
                        all_neighbor_distances = neighbor_distances.cpu().numpy()
                        if info['vq_decoder']:
                            all_vq_latents = latents.cpu().numpy()
                    else:
                        all_indexes = np.append(all_indexes, batch_index.cpu().numpy())
                        all_labels = np.append(all_labels, label.cpu().numpy())
                        all_acn_uq = np.vstack((all_acn_uq, u_q.cpu().numpy()))
                        all_neighbors = np.vstack((all_neighbors, neighbor_indexes.cpu().numpy()))
                        all_neighbor_distances = np.vstack((all_neighbor_distances,
                                                            neighbor_distances.cpu().numpy()))
                        if info['vq_decoder']:
                            all_vq_latents = np.vstack((all_vq_latents, latents.cpu().numpy()))
                    print('finished save latents', all_neighbors.shape[0])
                    np.savez(l_filepath+'_'+phase,
                                   index=all_indexes,
                                   labels=all_labels,
                                   acn_uq=all_acn_uq,
                                   neighbor_train_indexes=all_neighbors,
                                   neighbor_distances=all_neighbor_distances,
                                   vq_latents=all_vq_latents)


    train_results = np.load(l_filepath+'_train.npz')
    valid_results = np.load(l_filepath+'_valid.npz')
    return train_results, valid_results

def classify_latents(basepath, train_results, valid_results):
    from sklearn.neighbors import KNeighborsClassifier
    uqv_fp = basepath+'_valid_acn_cm.png'
    uqt_fp = basepath+'_train_acn_cm.png'
    print('finding knn on latents')
    tds,c,h,w = train_results['acn_uq'].shape
    vds,c,h,w = valid_results['acn_uq'].shape
    uq_train_X = train_results['acn_uq'].reshape((tds,c*h*w))
    uq_valid_X = valid_results['acn_uq'].reshape((vds,c*h*w))

    train_y = train_results['labels']
    valid_y = valid_results['labels']
    label_names = sorted(set(valid_y))

    uq_knn = KNeighborsClassifier(n_neighbors=info['num_k'])
    uq_knn.fit(uq_train_X, train_y)
    if not os.path.exists(uqv_fp):
        uq_pred_valid_y = uq_knn.predict(uq_valid_X)
        plot_confusion_matrix(valid_y, uq_pred_valid_y, label_names, normalize=False, filename=uqv_fp)

    if not os.path.exists(uqt_fp):
        uq_pred_train_y = uq_knn.predict(uq_train_X)
        plot_confusion_matrix(train_y, uq_pred_train_y, label_names, normalize=False, filename=uqt_fp)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, filename='confusion.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.utils.multiclass import unique_labels
    np.set_printoptions(precision=2)
    print("starting plotting confusion", filename)
    acc = accuracy_score(y_true, y_pred)
    if not title:
        if normalize:
            title = 'Normalized Acc %.02f'%acc
        else:
            title = 'Acc %.02f'%acc

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [classes[n] for n in list(unique_labels(y_true, y_pred))]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("finished plotting confusion", filename)
    return cm

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
        print('evaling experiment')
        if args.load_last_model != '':
            model_paths = sorted(glob(os.path.join(args.load_last_model, '*.pt')))
            if not len(model_paths):
                print('did not find any models at %s'%args.load_last_model)
            base_filepath = args.load_last_model
        elif args.model_loadpath != '':
            base_filepath = os.path.split(args.model_loadpath)[0]
            model_paths = [args.model_loadpath]
        else:
            print('provide model or directory load path')
            sys.exit()
        for model_loadpath in model_paths:
            info = create_new_info_dict(vars(args), base_filepath, __file__)
            info['model_loadpath'] = model_loadpath
            print('running eval on experiment', model_loadpath)
            model_dict, data_dict, info, train_cnt, epoch_cnt, rescale, rescale_inv = create_models(info, model_loadpath)
            if max([args.sample, args.tsne, args.pca]):
                call_plot(model_dict, data_dict, info, args.sample, args.tsne, args.pca)
            base_filepath = info['base_filepath']
            base_filename = os.path.split(info['model_loadpath'])[1].replace('.pt', '').replace('.cd', '')
            l_filepath = os.path.join(base_filepath, "%s_output"%(base_filename))
            if args.save_latents:
                train_latent_dict, valid_latent_dict = save_latents(l_filepath, model_dict, data_dict, info)
                classify_latents(l_filepath, train_latent_dict, valid_latent_dict)

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
        if max([args.sample, args.tsne, args.pca, args.save_latents]):
            call_plot(model_dict, data_dict, info, args.sample, args.tsne, args.pca)
            if args.save_latents:
                base_filepath = info['base_filepath']
                base_filename = os.path.split(info['model_loadpath'])[1]
                l_filepath = os.path.join(base_filepath, "%s_%010d_output"%(base_filename, info['train_cnts'][-1]))
                train_latents_dict, valid_latents_dict = save_latents(l_filepath, model_dict, data_dict, info)
                classify_latents(l_filepath, train_latent_dict, valid_latent_dict)
        else:
            # only train if we weren't asked to do anything else
            write_log_files(info)
            train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info, rescale_inv)
