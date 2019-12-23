## train_mnist_pcnn_acn.py

This model trains a ACN with a pixel cnn decoder, using the "float condition" (acn z latent)
as conditioning into the pixel cnn. I was not able to successfully sample from this architecture, despite a fair 
amount of hyperparameter search. The learned latents are also not very well separated. 
Ultimately - adding spatial conditioning via a simple deconv allowed the pixel cnn 
to be effectively trained (see train_mnist_*deconv*.py models for more info)

[TSNE Embedding]()

![Loss curve](https://github.com/ "Loss curve")
![Sampled Reconstructions](github.com "Samples")

