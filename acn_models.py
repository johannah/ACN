import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.nn import functional as F
import torch
from IPython import embed
# fast vq from rithesh
from functions import vq, vq_st

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar

# rithesh version
# https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/vqvae.py
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class ACNVQVAEresMNISTsmall(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       hidden_size=256,
                 num_clusters=512, num_z=32):

        super(ACNVQVAEresMNISTsmall, self).__init__()
        self.code_len = code_len
        self.num_clusters = num_clusters
        self.num_z = num_z
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.eo = 7
        fsos = first_stage_output_size = 128
        encoder_match_size = ems = hidden_size
        bc = self.bottleneck_channels = 2
        self.encoder = nn.Sequential(
                nn.Conv2d(input_size, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, bc, 1, 1, 0),
                )
        self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(bc, 16, 1, 1, 0),
            nn.Conv2d(16, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            ResBlock(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            )
        self.codebook = VQEmbedding(num_clusters, hidden_size)
        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def vq_encode(self, mu):
        z_e_x = self.conv_layers(mu)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def forward(self, frames):
        # x is bs,hidden_size,h,w
        # mu is 256,1,11,11
        mu = self.encoder(frames)
        z = self.reparameterize(mu)
        return z, mu

    def decode(self, z):
        z_e_x, latents = self.vq_encode(z)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, latents



class ACNVQVAEresMNIST(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       hidden_size=256,
                 num_clusters=512, num_z=32):

        super(ACNVQVAEresMNIST, self).__init__()
        self.code_len = code_len
        self.num_clusters = num_clusters
        self.num_z = num_z
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.eo = 7
        fsos = first_stage_output_size = 128
        encoder_match_size = ems = hidden_size
        bc = self.bottleneck_channels = 2
        self.encoder = nn.Sequential(
                nn.Conv2d(input_size, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, bc, 1, 1, 0),
                )
        self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(bc, 16, 1, 1, 0),
            nn.Conv2d(16, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            ResBlock(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            )
        self.codebook = VQEmbedding(num_clusters, hidden_size)
        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def vq_encode(self, mu):
        z_e_x = self.conv_layers(mu)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def forward(self, frames):
        # x is bs,hidden_size,h,w
        # mu is 256,1,11,11
        mu = self.encoder(frames)
        z = self.reparameterize(mu)
        return z, mu

    def decode(self, z):
        z_e_x, latents = self.vq_encode(z)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, latents

class ACNresMNIST(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       hidden_size=256,
                 ):

        super(ACNresMNIST, self).__init__()
        self.code_len = code_len
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.eo = 7
        fsos = first_stage_output_size = 128
        encoder_match_size = ems = hidden_size
        self.frame_encoder = nn.Sequential(
                               nn.Conv2d(input_size, fsos, 1, 1, 0),
                               nn.BatchNorm2d(fsos),
                               nn.ReLU(True),
                               nn.Conv2d(fsos, fsos, 1, 1, 0),
                               nn.ReLU(True),
                             )
        bc = self.bottleneck_channels = 2
        self.encoder = nn.Sequential(
                nn.Conv2d(fsos, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, bc, 1, 1, 0),
                )
        self.decoder = nn.Sequential(
                  nn.ConvTranspose2d(bc, 16, 1, 1, 0),
                  nn.ConvTranspose2d(16, hidden_size, 1, 1, 0),
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )

        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def forward(self, frames):
        x = self.frame_encoder(frames)
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        return z, mu

    def decode(self, z):
        x_tilde = self.decoder(z)
        return x_tilde



class ACNVQVAEres(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       hidden_size=256,
                 num_clusters=512, num_z=32):

        super(ACNVQVAEres, self).__init__()
        self.code_len = code_len
        self.num_clusters = num_clusters
        self.num_z = num_z
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.eo = 8
        fsos = first_stage_output_size = 128
        encoder_match_size = ems = hidden_size
        self.frame_encoder = nn.Sequential(
                               nn.Conv2d(input_size, fsos, 1, 1, 0),
                               nn.BatchNorm2d(fsos),
                               nn.ReLU(True),
                               nn.Conv2d(fsos, fsos, 1, 1, 0),
                               nn.ReLU(True),
                             )
        bc = self.bottleneck_channels = 1
        self.encoder = nn.Sequential(
                nn.Conv2d(fsos, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 0),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 0),
                nn.Conv2d(hidden_size, hidden_size, 2, 1, 0),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, bc, 1, 1, 0),
                )
        self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 2, 1, 0),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 0),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 0),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(bc, 16, 1, 1, 0),
            nn.Conv2d(16, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            ResBlock(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            )
        self.codebook = VQEmbedding(num_clusters, hidden_size)
        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def vq_encode(self, mu):
        z_e_x = self.conv_layers(mu)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def forward(self, frames):
        x = self.frame_encoder(frames)
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        return z, mu

    def decode(self, z):
        z_e_x, latents = self.vq_encode(z)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, latents

class tPTPriorNetwork(nn.Module):
    def __init__(self, size_training_set, code_length, n_hidden=512, k=5, random_seed=4543):
        super(tPTPriorNetwork, self).__init__()
        self.rdn = np.random.RandomState(random_seed)
        self.k = k
        self.size_training_set = size_training_set
        self.code_length = code_length
        self.input_layer = nn.Linear(code_length, n_hidden)
        self.skipin_to_2 = nn.Linear(n_hidden, n_hidden)
        self.skipin_to_3 = nn.Linear(n_hidden, n_hidden)
        self.skip1_to_out = nn.Linear(n_hidden, n_hidden)
        self.skip2_to_out = nn.Linear(n_hidden, n_hidden)
        self.h1 = nn.Linear(n_hidden, n_hidden)
        self.h2 = nn.Linear(n_hidden, n_hidden)
        self.h3 = nn.Linear(n_hidden, n_hidden)
        self.fc_mu = nn.Linear(n_hidden, self.code_length)
        self.fc_s = nn.Linear(n_hidden, self.code_length)

        # needs to be a param so that we can load
        self.codes = nn.Parameter(torch.FloatTensor(self.rdn.standard_normal((self.size_training_set, self.code_length))), requires_grad=False)
        # start off w/ default batch size - this will change automatically if
        # different input is given
        batch_size = 64
        n_neighbors = 5
        self.neighbors = torch.LongTensor((batch_size, n_neighbors))
        self.distances = torch.FloatTensor((batch_size, n_neighbors))
        self.batch_indexer = torch.LongTensor(torch.arange(batch_size))

    def update_codebook(self, indexes, values):
        assert indexes.min() >= 0
        assert indexes.max() < self.codes.shape[0]
        self.codes[indexes] = values

    def kneighbors(self, test, n_neighbors):
        with torch.no_grad():
            device = test.device
            bs = test.shape[0]
            return_size = (bs,n_neighbors)
            # dont recreate unless necessary
            if self.neighbors.shape != return_size:
                print('updating prior sizes')
                self.neighbors = torch.LongTensor(torch.zeros(return_size, dtype=torch.int64))
                self.distances = torch.zeros(return_size)
                self.batch_indexer = torch.LongTensor(torch.arange(bs))
            if device != self.codes.device:
                print('transferring prior to %s'%device)
                self.neighbors = self.neighbors.to(device)
                self.distances = self.distances.to(device)
                self.codes = self.codes.to(device)

            for bidx in range(test.shape[0]):
                dists = torch.norm(self.codes-test[bidx], dim=1)
                self.distances[bidx], self.neighbors[bidx] = dists.topk(n_neighbors, largest=False)
                del dists
        #print('kn', bidx, torch.cuda.memory_allocated(device=None))
        return self.distances.detach(), self.neighbors.detach()

    def batch_pick_close_neighbor(self, codes):
        '''
        :code latent activation of training
        '''
        neighbor_distances, neighbor_indexes = self.kneighbors(codes, n_neighbors=self.k)
        bsize = neighbor_indexes.shape[0]
        if self.training:
            # randomly choose neighbor index from top k
            chosen_neighbor_index = torch.LongTensor(self.rdn.randint(0,neighbor_indexes.shape[1],size=bsize))
        else:
            chosen_neighbor_index = torch.LongTensor(torch.zeros(bsize, dtype=torch.int64))
        return self.codes[neighbor_indexes[self.batch_indexer, chosen_neighbor_index]]

    def forward(self, codes):
        previous_codes = self.batch_pick_close_neighbor(codes)
        return self.encode(previous_codes)

    def encode(self, prev_code):
        """
        The prior network was an
        MLP with three hidden layers each containing 512 tanh
        units
        - and skip connections from the input to all hidden
        layers and
        - all hiddens to the output layer.
        """
        i = torch.tanh(self.input_layer(prev_code))
        # input goes through first hidden layer
        _h1 = torch.tanh(self.h1(i))

        # make a skip connection for h layers 2 and 3
        _s2 = torch.tanh(self.skipin_to_2(_h1))
        _s3 = torch.tanh(self.skipin_to_3(_h1))

        # h layer 2 takes in the output from the first hidden layer and the skip
        # connection
        _h2 = torch.tanh(self.h2(_h1+_s2))

        # take skip connection from h1 and h2 for output
        _o1 = torch.tanh(self.skip1_to_out(_h1))
        _o2 = torch.tanh(self.skip2_to_out(_h2))
        # h layer 3 takes skip connection from prev layer and skip connection
        # from nput
        _o3 = torch.tanh(self.h3(_h2+_s3))

        out = _o1+_o2+_o3
        mu = self.fc_mu(out)
        logstd = self.fc_s(out)
        return mu, logstd



