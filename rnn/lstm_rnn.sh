#!/usr/bin/env bash

./op_driver.py -lib cudnn -op rnnfp16 -n 1024 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 1 -t 1 -w 1 -V 0 -i 10
./op_driver.py -lib cudnn -op rnnfp16 -n 1024 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 2 -t 1 -w 1 -V 0 -i 10
./op_driver.py -lib cudnn -op rnnfp16 -n 1024 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 4 -t 1 -w 1 -V 0 -i 10
./op_driver.py -lib cudnn -op rnn     -n 1024 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 1 -t 1 -w 1 -V 0 -i 10
./op_driver.py -lib cudnn -op rnn     -n 1024 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 2 -t 1 -w 1 -V 0 -i 10
./op_driver.py -lib cudnn -op rnn     -n 1024 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 4 -t 1 -w 1 -V 0 -i 10
./op_driver.py -lib cudnn -op rnnfp16 -n  256 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 1 -t 1 -w 1 -V 0 -i 10
./op_driver.py -lib cudnn -op rnnfp16 -n  256 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 2 -t 1 -w 1 -V 0 -i 10
./op_driver.py -lib cudnn -op rnnfp16 -n  256 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 4 -t 1 -w 1 -V 0 -i 10
./op_driver.py -lib cudnn -op rnn     -n  256 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 1 -t 1 -w 1 -V 0 -i 10
./op_driver.py -lib cudnn -op rnn     -n  256 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 2 -t 1 -w 1 -V 0 -i 10
./op_driver.py -lib cudnn -op rnn     -n  256 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 4 -t 1 -w 1 -V 0 -i 10
