# deeplearning
Deep Learning Models for text and image

Deep & Cross Network (DCN)
What is Deep & Cross Network (DCN)? DCN was designed to learn explicit and bounded-degree cross features more effectively. It starts with an input layer (typically an embedding layer), followed by a cross network containing multiple cross layers that models explicit feature interactions, and then combines with a deep network that models implicit feature interactions.\

Cross Network. This is the core of DCN. It explicitly applies feature crossing at each layer, and the highest polynomial degree increases with layer depth. The following figure shows the 
-th cross layer

Deep Network. It is a traditional feedforward multilayer perceptron (MLP).



