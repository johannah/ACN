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

