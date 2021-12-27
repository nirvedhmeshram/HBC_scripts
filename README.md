# HBC_scripts
Implementations of hbc ops
In order to work around the cumsum op, we have directly formed the offsets in the model. The interface of the model still remains the same.

Here is the pytorch implementation: hbc_pytorch.py

The new ops added are
torch.arange and tensor.unsqueeze 

Here is the tensorflow implementation hbc_tensorflow.py

The new ops added are
tf.range and tf.expand_dims 

Caution:
This implementation only supports the current use case with 
lengths = torch.tensor([[1], [1], [1], [1], [1]])

Later in order so support size of 5000 as is the case in FBGMM benchmark we changed the tensorflow op and the new implementation is hbc_tensorflow_largesize.py
