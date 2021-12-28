# HBC_scripts
Implementations of hbc ops
In order to work around the cumsum op, we have directly formed the offsets in the model. The interface of the model still remains the same.

Here is the pytorch implementation: hbc_torch.py

The new ops added are
torch.arange and tensor.unsqueeze 

Here is the tensorflow implementation hbc_tensorflow.py

The new ops added are
tf.range and tf.expand_dims 

Caution:
This implementation only supports the current use case with 
lengths = torch.tensor([[1], [1], [1], [1], [1]])

#Later in order so support size of 5000 as is the case in FBGMM benchmark we changed the tensorflow op and the new implementation is hbc_tensorflow_largesize.py

# Here are the introduction to run this script using IREE for cuda (follow the IREE documentation to do same thing for CPU using dylib)
python hbc_tensorflow_largesize.py # creates the hbc_sm_v2 folder which has the model
python make_mlir_frommodel.py # creates the MLIR in MHLO
iree-translate  --iree-input-type=mhlo -iree-mlir-to-vm-bytecode-module \
-iree-hal-target-backends=cuda  mhlo_hbc_sm_v2_large.mlir  -o iree_hbc_cuda.vmfb # translate to byte code
iree-run-module --driver=cuda --module_file=iree_hbc_cuda.vmfb \
--entry_function=__call__ --function_input="5000xi64"=21 --function_input="5000x1xi64=1" --function_input="5000x1xf32"=0.2 #run the byte code, this will create a large output
~/iree-build-release/iree/tools/iree-benchmark-module --driver=cuda --module_file=iree_hbc_cuda.vmfb \
--entry_function=__call__ --function_input="5000xi64"=21 --function_input="5000x1xi64=1" --function_input="5000x1xf32"=0.2 #benchmark

