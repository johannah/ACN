# ACN

Implementation of Associative Compression Networks for Representation Learning ([ACN](https://arxiv.org/abs/1804.02476)) by Graves, Menick, and van den Oord.

### Train fashion MNIST with ACN encoder and PixelCNN decoder:  
```
python train_mnist_pcnn_acn.py -c
```
 
### Sample from a trained model:   
```
python train_mnist_pcnn_acn.py -s -l path_to_model.pt
```

### Make a TSNE plot (saved as html)
```
python train_mnist_pcnn_acn.py --tsne -l path_to_model.pt
```
fashion
6176
5667
0404
9793


| ACN | ACN-VQ | 
| --- | --- |
| [](results/mnist_acn/mnist_acn_validation_01_0024000000ex_batch_rec_neighbors_valid_006176_plt.png) | [](mnist_acnvq/mnist_acn_vq_vq_00_0024600000ex_batch_rec_neighbors_valid_006176_plt.png) |   

