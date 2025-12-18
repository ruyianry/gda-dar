#!/bin/bash

python3 cifar_classify_gen.py \
    --config configs/cifar10-vgg16.yaml \
    --network_pkl /raid/crp.dssi/volume_Kubernetes/Benquan/VAE_exp/osg/pretrained/cifar10_cond_sid2a_alpha1.2-053760.pkl \
    --ratio 0.25 \
    --generator_type "edm" 2>&1 --D_pkl /raid/crp.dssi/volume_Kubernetes/Benquan/VAE_exp/osg/model/cifar10/training-state-100000.pt?download=true | tee -a "logs/log_cifarcond_${timestamp_tt}.txt"



