#!/usr/bin/env bash

cd cudnn_samples_v8/RNN_v8.0/

make clean
make all

cp RNN ../..
