#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import numpy as np
import torch
from torch.nn import functional as f
import distiller
import torchvision.transforms as transforms
from examples.automated_deep_compression import siamese
import torch
from examples.automated_deep_compression.cka import centering


msglogger = logging.getLogger()


def cache_featuremaps(module, input, output, intermediate_fms):
    """Create a cached dictionary of each layer's input and output feature-maps.

    We use this with distiller.FMReconstructionChannelPruner
    """

    # Make a permanent copy of the input and output feature-maps
    X = input[0]
    n_points_per_layer = 10
    X = X.detach().cpu().clone()
    #X = X
    Y = output.detach().cpu().clone()

    # Preprocess the outputs: transpose the batch and channel dimensions, create a flattened view, and transpose.
    # The outputs originally have shape: (batch size, num channels, feature-map width, feature-map height).
    # Y = Y.transpose(0,1).contiguous()
    # Y = Y.view(Y.size(0), -1)
    # Y = Y.t()
    # Y i storch.Size([128, 16, 32, 32])

    # Y = Y.view(Y.size(0), Y.size(1), -1) # torch.Size([128, 16, 1024])
    # Y = Y.transpose(2, 1)  # torch.Size([128, 1024, 16])
    # Y = Y.contiguous().view(-1, Y.size(2))  # torch.Size([131072, 16])

    intermediate_fms['output_fms'][module.distiller_name].append(Y)
    intermediate_fms['input_fms'][module.distiller_name].append(X)




def collect_intermediate_featuremap_samples(model, validate_fn, modules_names):
    """Collect pairs of input/output feature-maps.

    For reconstruction of weights, we need to collect pairs of (layer_input, layer_output)
    using a sample subset of the input dataset.  We feed this dataset-subset to the 
    """
    from functools import partial

    def install_io_collectors(m, intermediate_fms, mod_names):
        if isinstance(m, torch.nn.Conv2d) and m.distiller_name in mod_names:
            intermediate_fms['output_fms'][m.distiller_name] = []
            intermediate_fms['input_fms'][m.distiller_name] = []
            hook_handles.append(m.register_forward_hook(partial(cache_featuremaps, intermediate_fms=intermediate_fms)))


    msglogger.info("\nCollecting input/ouptput feature-map pairs for weight reconstruction")
    distiller.assign_layer_fq_names(model)
    # Register on the forward hooks, then run the forward path and collect the data
    hook_handles = []
    intermediate_fms = {"output_fms": dict(), "input_fms": dict()}
    model.apply(partial(install_io_collectors, intermediate_fms=intermediate_fms, 
                        mod_names=modules_names))
    
    validate_fn()
    
    # Unregister from the forward hooks
    for handle in hook_handles:
        handle.remove()

    def concat_tensor_list(tensor_list, dim):
        return torch.cat(tensor_list, dim=dim)

    model.intermediate_fms = {"output_fms": dict(), "input_fms": dict()}

    outputs = model.intermediate_fms['output_fms']
    inputs = model.intermediate_fms['input_fms']

    msglogger.info("Concatenating FMs...")
    for (layer_name, X), Y in zip(intermediate_fms['input_fms'].items(), intermediate_fms['output_fms'].values()):                
        inputs[layer_name] = concat_tensor_list(X, dim=0)
        outputs[layer_name] = concat_tensor_list(Y, dim=0)

    msglogger.info("Done.")
    del intermediate_fms

    print('Creating CKA dataset')
    n_samples = 128
    train_dataset = siamese.FilterDataset(outputs, n_samples=n_samples, n_examples=10000)
    siamese_args = distiller.utils.MutableNamedTuple(
        {'action': 'train',
         'epoch': 5,
         'margin': 1.0,
         'cuda': True,
         'randaug': False,
         'contra_loss': True,
         'model_file': 'siamese_cka.pkl'})

    print('Training Siamese CKA model')
    siamese_net = siamese.train(train_dataset, siamese_args)

    print('Creating Embedding dict')
    layers = outputs.keys()
    embeddings = dict()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    for layer in layers:
        layer_features = outputs[layer]
        n_channels = layer_features.shape[1]
        x_layer = []
        for c in range(n_channels):
            data_samples_idx = np.random.choice(layer_features.shape[0], n_samples, replace=False)
            data_samples = layer_features[data_samples_idx, c, :, :]
            x = data_samples.reshape(n_samples, -1).numpy()
            x = centering(np.matmul(x, x.T))
            x_layer.append(x)
        x_layer = torch.tensor(np.expand_dims(np.stack(x_layer, axis=0), axis=1), device=device)
        embeddings[layer] = siamese_net.forward_once(x_layer.float()).cpu().detach().numpy()

    import pickle
    pickle_out = open("embeddings.pickle", "wb")
    pickle.dump(embeddings, pickle_out)
    pickle_out.close()

    return embeddings