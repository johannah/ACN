import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from IPython import embed


class IndexedDataset(Dataset):
    def __init__(self, dataset_function, path, train=True, download=True, transform=transforms.ToTensor()):
        """ class to provide indexes into the data
        """
        self.indexed_dataset = dataset_function(path,
                             download=download,
                             train=train,
                             transform=transform)

    def __getitem__(self, index):
        data, target = self.indexed_dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.indexed_dataset)

def save_checkpoint(state, filename='model.pt'):
    print("starting save of model %s" %filename)
    torch.save(state, filename)
    print("finished save of model %s" %filename)

def create_mnist_datasets(dataset_name, base_datadir, batch_size):
    dataset = eval('datasets.'+dataset_name)
    datadir = os.path.join(base_datadir, dataset_name)
    train_data = IndexedDataset(dataset, path=datadir,
                                train=True, download=True,
                                transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data = IndexedDataset(dataset, path=datadir,
                               train=False, download=True,
                               transform=transforms.ToTensor())
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    data_dict = {'train':train_loader, 'valid':valid_loader}
    nchans,hsize,wsize = data_dict['train'].dataset[0][0].shape
    size_training_set = len(train_data)
    return data_dict, size_training_set, nchans, hsize, wsize

def create_new_info_dict(arg_dict, size_training_set, base_filepath):
    if not os.path.exists(base_filepath):
        os.makedirs(base_filepath)
    info = {'train_cnts':[],
            'train_losses':{},
            'valid_losses':{},
            'save_times':[],
            'args':[arg_dict],
            'last_save':0,
            'last_plot':0,
            'epoch_cnt':0,
            'size_training_set':size_training_set,
            'base_filepath':base_filepath,
             }
    for arg,val in arg_dict.items():
        info[arg] = val
    if info['cuda']:
        info['device'] = 'cuda'
    else:
        info['device'] = 'cpu'
    return info

def seed_everything(seed=394, max_threads=2):
    torch.manual_seed(394)
    torch.set_num_threads(max_threads)

def plot_example(img_filepath, example, plot_on=['target', 'yhat'], num_plot=10):
    '''
    img_filepath: location to write .png file
    example: dict with torch images of the same shape [bs,c,h,w] to write
    plot_on: list of keys of images in example dict to write
    num_plot: limit the number of examples from bs to this int
    '''
    for cnt, pon in enumerate(plot_on):
        bs,c,h,w = example[pon].shape
        num_plot = min([bs, num_plot])
        eimgs = example[pon].view(bs,c,h,w)[:num_plot]
        if not cnt:
            comparison = eimgs
        else:
            comparison = torch.cat([comparison, eimgs])
    save_image(comparison.cpu(), img_filepath, nrow=num_plot)
    print('writing comparison image: %s img_path'%img_filepath)

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_losses(train_cnts, train_losses, test_losses, name='loss_example.png', rolling_length=4):
    f,ax=plt.subplots(1,1,figsize=(3,3))
    test_cmap = matplotlib.cm.get_cmap('Blues')
    train_cmap = matplotlib.cm.get_cmap('Greens')
    color_idxs = np.linspace(.3,.75,num=len(train_losses.keys()))
    test_colors = np.array([test_cmap(ci) for ci in color_idxs])
    train_colors = np.array([train_cmap(ci) for ci in color_idxs])
    for idx, key in enumerate(sorted(train_losses.keys())):
        ax.plot(rolling_average(train_cnts, rolling_length),
                rolling_average(train_losses[key], rolling_length),
                label='train %s'%key, lw=1,
                c=train_colors[idx])
        ax.plot(rolling_average(train_cnts, rolling_length),
                rolling_average(test_losses[key], rolling_length),
                label='test %s'%key, lw=1,
                c=test_colors[idx])
        ax.scatter(rolling_average(train_cnts, rolling_length),
               rolling_average(train_losses[key], rolling_length),
                s=4, c=tuple(train_colors[idx][None]), marker='x')
        ax.scatter(rolling_average(train_cnts, rolling_length),
               rolling_average(test_losses[key], rolling_length),
                s=4, c=tuple(test_colors[idx][None]), marker='o')
    ax.legend()
    plt.savefig(name)
    plt.close()

def tsne_plot(X, images, color, num_clusters=30, perplexity=5, serve_port=8104, html_out_path='mpld3.html', serve=False):
    from sklearn.manifold import TSNE
    import mpld3
    from skimage.transform import resize

    print('computing TSNE')
    Xtsne = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)
    x = Xtsne[:,0]
    y = Xtsne[:,1]
    # get color from kmeans cluster
    #print('computing KMeans clustering')
    #Xclust = KMeans(n_clusters=num_clusters).fit_predict(Xtsne)
    #c = Xclust
    # Create list of image URIs
    html_imgs = []
    print('adding hover images')
    for nidx in range(images.shape[0]):
        f,ax = plt.subplots()
        ax.imshow(resize(images[nidx], (180,180)))
        dd = mpld3.fig_to_dict(f)
        img = dd['axes'][0]['images'][0]['data']
        html = '<img src="data:image/png;base64,{0}">'.format(img)
        html_imgs.append(html)
        plt.close()

    # Define figure and axes, hold on to object since they're needed by mpld3
    fig, ax = plt.subplots(figsize=(8,8))
    # Make scatterplot and label axes, title
    sc = ax.scatter(x, y, s=100,alpha=0.7, c=color, edgecolors='none')
    plt.title("TSNE")
    # Create the mpld3 HTML tooltip plugin
    tooltip = mpld3.plugins.PointHTMLTooltip(sc, html_imgs)
    # Connect the plugin to the matplotlib figure
    mpld3.plugins.connect(fig, tooltip)
    #plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom())
    # Uncomment to save figure to html file
    out=mpld3.fig_to_html(fig)
    print('writing tsne image to %s'%html_out_path)
    fpath = open(html_out_path, 'w')
    fpath.write(out)
    # display is used in jupyter
    #mpld3.display()
    if serve==True:
        mpld3.show(port=serve_port, open_browser=False)

##################################################################

def set_model_mode(model_dict, phase):
    for name, model in model_dict.items():
        print('setting', name, phase)
        if name != 'opt':
            if phase == 'valid':
                model_dict[name].eval()
            else:
                model_dict[name].train()
    return model_dict

def kl_loss_function(u_q, s_q, u_p, s_p):
    ''' reconstruction loss + coding cost
     coding cost is the KL divergence bt posterior and conditional prior
     Args:
         u_q: mean of model posterior
         s_q: log std of model posterior
         u_p: mean of conditional prior
         s_p: log std of conditional prior

     reduction is sum over elements, then mean over batch
     Returns: loss
     '''
    acn_KLD = (s_p-s_q-0.5 + ((2*s_q).exp() + (u_q-u_p).pow(2)) / (2*(2*s_p).exp()))
    return acn_KLD

