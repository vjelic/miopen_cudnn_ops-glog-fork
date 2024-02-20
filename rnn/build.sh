#!/usr/bin/env bash

cd cudnn_samples-linux-x86_64-8.9.7.29_cuda12-archive/src/cudnn_samples_v8/RNN_v8.0/

make clean
make all

cp RNN ../../../../
