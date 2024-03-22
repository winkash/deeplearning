# deeplearning
Deep Learning Models for text and image

Deep & Cross Network (DCN)
What is Deep & Cross Network (DCN)? DCN was designed to learn explicit and bounded-degree cross features more effectively. It starts with an input layer (typically an embedding layer), followed by a cross network containing multiple cross layers that models explicit feature interactions, and then combines with a deep network that models implicit feature interactions.\

Cross Network. This is the core of DCN. It explicitly applies feature crossing at each layer, and the highest polynomial degree increases with layer depth. The following figure shows the 
-th cross layer

Deep Network. It is a traditional feedforward multilayer perceptron (MLP).

![alt_text](https://camo.githubusercontent.com/f59de32a658b2f1524975da7d055e870968671e3b68c5d38179e97b0d8f523ca/687474703a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d3157744455435636622d656574556e575643416d635068386d4a46757435455564)

Neural Network with Pytorch
Neural networks comprise of layers/modules that perform operations on data. The torch.nn namespace provides all the building blocks you need to build your own neural network. Every module in PyTorch subclasses the nn.Module. A neural network is a module itself that consists of other modules (layers). This nested structure allows for building and managing complex architectures easily.



