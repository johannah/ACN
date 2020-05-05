# ACN

Implementation of Associative Compression Networks for Representation Learning ([ACN](https://arxiv.org/abs/1804.02476)) by Graves, Menick, and van den Oord. We also tried using a VQ-VAE style decoder (named acnvq). See below for results. 

### Train fashion_mnist acn
```
python train_acn.py --dataset_name FashionMNIST
```

### Train mnist acn-vq
```
python train_acn.py --dataset_name FashionMNIST --vq_decoder
```
 
### Plot reconstructions of neighbors, pca, tsne from a trained model:   
```
python train_acn.py -l path_to_model.pt --pca --tsne
```
# Results

| Data Type | Eval | ACN | ACN-VQ |   
| --- | --- | --- | --- | 
| MNIST | KNN Accuracy | 97% | 97% |  
| MNIST | PCA | [mnist-acn-pca](results/mnist_acn/mnist_acn_validation_01_0024000000ex_pca_valid.html) | [mnist-acnvq-pca](results/mnist_acnvq/mnist_acn_vq_vq_00_0024600000ex_pca_valid.html) |   
| MNIST | TSNE | [mnist-acn-tsne](results/mnist_acn/mnist_acn_validation_01_0024000000ex_tsne_valid_P10.html) | [mnist-acnvq-pca](results/mnist_acnvq/mnist_acn_vq_vq_00_0024600000ex_tsne_valid_P10.html) |   
| Fashion | PCA | [fashion-acn-pca](results/fashion_acn/fashion_acn_validation_00_0032400000ex_pca_valid.html) | [mnist-acnvq-pca](results/fashion_acnvq/fashion_acnvq_validation_small_vq_01_0078000000ex_pca_valid.html) |   
| Fashion | TSNE | [fashion-acn-tsne](results/fashion_acn/fashion_acn_validation_00_0032400000ex_tsne_valid_P10.html) | [mnist-acnvq-pca](results/fashion_acnvq/fashion_acnvq_validation_small_vq_01_0078000000ex_tsne_valid_P10.html) |   


| ACN | ACN-VQ | 
| --- | --- |
| ![fashion-acn-0404](results/fashion_acn/fashion_acn_validation_00_0032400000ex_batch_rec_neighbors_valid_000404_plt.png) | ![fashion-acnvq-0404](results/fashion_acnvq/fashion_acnvq_validation_small_vq_01_0078000000ex_batch_rec_neighbors_valid_000404_plt.png) |    

| ![fashion-acn-9793](results/fashion_acn/fashion_acn_validation_00_0032400000ex_batch_rec_neighbors_valid_009793_plt.png) | ![fashion-acnvq-9793](results/fashion_acnvq/fashion_acnvq_validation_small_vq_01_0078000000ex_batch_rec_neighbors_valid_009793_plt.png) |    

| ![fashion-acn-6176](results/fashion_acn/fashion_acn_validation_00_0032400000ex_batch_rec_neighbors_valid_006176_plt.png) | ![fashion-acnvq-6176](results/fashion_acnvq/fashion_acnvq_validation_small_vq_01_0078000000ex_batch_rec_neighbors_valid_006176_plt.png) |    

| ![fashion-acn-5667](results/fashion_acn/fashion_acn_validation_00_0032400000ex_batch_rec_neighbors_valid_005667_plt.png) | ![fashion-acnvq-5667](results/fashion_acnvq/fashion_acnvq_validation_small_vq_01_0078000000ex_batch_rec_neighbors_valid_005667_plt.png) |    

| ![mnist-acn-6176](results/mnist_acn/mnist_acn_validation_01_0024000000ex_batch_rec_neighbors_valid_006176_plt.png) | ![mnist-acnvq-6176](results/mnist_acnvq/mnist_acn_vq_vq_00_0024600000ex_batch_rec_neighbors_valid_006176_plt.png) |   

