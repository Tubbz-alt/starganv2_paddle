import argparse
import copy


from munch import Munch


import numpy as np
import math

from paddorch.vision.models.wing import FAN

from paddle import fluid
import os,sys
import paddorch as porch
import joblib
input_file=sys.argv[1]

lr=1e-4
if len(sys.argv)>2:
    lr=float(sys.argv[2])
z_y_m = joblib.load(input_file)
out_dir=os.path.dirname(input_file)

y_train = np.concatenate([vv[1] for vv in z_y_m])
shuffle_indices=np.random.choice(len(y_train),size=len(y_train)*5)
y_train=y_train[shuffle_indices]
z_train = np.concatenate([vv[0] for vv in z_y_m])[shuffle_indices]
# m_out_train,m_out_train_1,m_out_train_2 = mapping_network.finetune(pyporch.FloatTensor(z_train), pyporch.LongTensor(y_train))
m_out_train = np.concatenate([vv[2] for vv in z_y_m])[shuffle_indices]




place = fluid.CUDAPlace(0)
batch_size=128
with fluid.dygraph.guard(place=place):
    import core.model

    if "afhq" in input_file:
        mapping_network_ema = core.model.MappingNetwork(16,64,
                                                        3)  # copy.deepcopy(mapping_network)
        out_model_fn = "../expr/checkpoints/afhq/100000_nets_ema.ckpt/mapping_network.pdparams"
        mapping_network_ema.load_state_dict(
            porch.load(out_model_fn))

    else:
        mapping_network_ema = core.model.MappingNetwork(16,64,
                                                        2)  # copy.deepcopy(mapping_network)
        out_model_fn = "../expr/checkpoints/celeba_hq/100000_nets_ema.ckpt/mapping_network.pdparams"
        mapping_network_ema.load_state_dict(
            porch.load(out_model_fn))

    d_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=mapping_network_ema.parameters())
    from tqdm import tqdm

    mapping_network_ema.train()
    z_train_p = porch.Tensor(z_train)
    y_train_p = porch.LongTensor(y_train)
    m_out_train_p = porch.Tensor(m_out_train)
    best_loss=100000000
    for ii in  range(100000000000) :
        st=np.random.randint(0,z_train_p.shape[0]-batch_size)
        out =mapping_network_ema(z_train_p[st:st+batch_size], y_train_p[st:st+batch_size])
        d_avg_cost=fluid.layers.mse_loss(out,m_out_train_p[st:st+batch_size]) #+fluid.layers.mse_loss(out1,m_out_train_1p)+fluid.layers.mse_loss(out2,m_out_train_2p)

        d_avg_cost.backward()
        d_optimizer.minimize(d_avg_cost)
        mapping_network_ema.clear_gradients()
        if ii%99==0:
            print("d_avg_cost", d_avg_cost.numpy())
            if best_loss>d_avg_cost.numpy():
                best_loss=d_avg_cost.numpy()
                porch.save(mapping_network_ema.state_dict(),out_model_fn)
                print("save model file:",out_model_fn)
