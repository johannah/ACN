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
| ![fashion-acn-0404](https://github.com/johannah/ACN/blob/master/results/fashion_acn/fashion_acn_validation_00_0032400000ex_batch_rec_neighbors_valid_000404_plt.png) | ![fashion-acnvq-0404](https://github.com/johannah/ACN/blob/master/results/fashion_acnvq/fashion_acnvq_validation_small_vq_01_0078000000ex_batch_rec_neighbors_valid_000404_plt.png) |    
| ![fashion-acn-9793](https://github.com/johannah/ACN/blob/master/results/fashion_acn/fashion_acn_validation_00_0032400000ex_batch_rec_neighbors_valid_009793_plt.png) | ![fashion-acnvq-9793](https://github.com/johannah/ACN/blob/master/results/fashion_acnvq/fashion_acnvq_validation_small_vq_01_0078000000ex_batch_rec_neighbors_valid_009793_plt.png) |    
| ![fashion-acn-6176](https://github.com/johannah/ACN/blob/master/results/fashion_acn/fashion_acn_validation_00_0032400000ex_batch_rec_neighbors_valid_006176_plt.png) | ![fashion-acnvq-6176](https://github.com/johannah/ACN/blob/master/results/fashion_acnvq/fashion_acnvq_validation_small_vq_01_0078000000ex_batch_rec_neighbors_valid_006176_plt.png) |    
| ![fashion-acn-5667](https://github.com/johannah/ACN/blob/master/results/fashion_acn/fashion_acn_validation_00_0032400000ex_batch_rec_neighbors_valid_005667_plt.png) | ![fashion-acnvq-5667](https://github.com/johannah/ACN/blob/master/results/fashion_acnvq/fashion_acnvq_validation_small_vq_01_0078000000ex_batch_rec_neighbors_valid_005667_plt.png) |    
| ![mnist-acn-6176](https://github.com/johannah/ACN/blob/master/results/mnist_acn/mnist_acn_validation_01_0024000000ex_batch_rec_neighbors_valid_006176_plt.png) | ![mnist-acnvq-6176](https://github.com/johannah/ACN/blob/master/results/mnist_acnvq/mnist_acn_vq_vq_00_0024600000ex_batch_rec_neighbors_valid_006176_plt.png) |   
