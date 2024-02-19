#!/usr/bin/env python3

import argparse
import subprocess


# create command line arguments
parser = argparse.ArgumentParser(description='Collect miopen and rocblas configurations.')

parser.add_argument('-lib', type=str, help='The library to use.', choices=['miopen', 'cudnn'], required=True)
parser.add_argument('-op', '--operation', type=str, help='Base argument - the operation to be performed.', choices=['rnn', 'rnnfp16'], required=True)
parser.add_argument('-F', '--forw', type=str, help='Run only Forward RNN == 1 or only Backward Data RNN == 2, Backward Weights = 4 or both == 0 (Default=0)', default=0)
parser.add_argument('-H', '--hid_h', type=str, help='Hidden State Length (Default=32)', default=32)
parser.add_argument('-L', '--seed_low', type=str, help='Least significant 32 bits of seed (Default=0)', default=0)
parser.add_argument('-M', '--seed_high', type=str, help='Most significant 32 bits of seed (Default=0)', default=0)
parser.add_argument('-P', '--dropout', type=str, help='Dropout rate (Default=0.0)', default=0.0)
parser.add_argument('-U', '--use_dropout', type=str, help='Use dropout: 1; Not use dropout: 0 (Default=0)', default=0)
parser.add_argument('-V', '--verify', type=str, help='Verify Each Layer (Default=1)', default=1)
parser.add_argument('-W', '--in_h', type=str, help='Input Length (Default=32)', default=32)
parser.add_argument('-a', '--rnnalgo', type=str, help='default, fundamental (Default=0)', default=0)
parser.add_argument('-b', '--bias', type=str, help='Use Bias (Default=0)', default=0)
parser.add_argument('-c', '--fwdtype', type=str, help='RNN forward being training or inference, Default training (Default=0)', default=0)
parser.add_argument('-f', '--datatype', type=str, help='16-bit or 32-bit fp (Default=1)', default=1)
# parser.add_argument('-h', '--help', type=str, help='Print Help Message')
parser.add_argument('-i', '--iter', type=str, help='Number of Iterations (Default=1)', default=1)
parser.add_argument('-k', '--seq_len', type=str, help='Number of iterations to unroll over (Default=10)', default=10)
parser.add_argument('-l', '--num_layer', type=str, help='Number of hidden stacks (Default=1)', default=1)
parser.add_argument('-m', '--mode', type=str, help='RNN Mode (relu, tanh, lstm, gru) (Default=tanh)', default='tanh')
parser.add_argument('-n', '--batchsize', type=str, help='Mini-batch size (Default=4)', default=4)
parser.add_argument('-o', '--dump_output', type=str, help='Dumps the output buffers (Default=0)', default=0)
parser.add_argument('-p', '--inputmode', type=str, help='linear == 0 or skip == 1, (Default=0)', default=0)
parser.add_argument('-q', '--use_padding', type=str, help='packed tensors == 0 or padded == 1, (Default=0)', default=0)
parser.add_argument('-r', '--bidirection', type=str, help='uni- or bi-direction, default uni- (Default=0)', default=0)
parser.add_argument('-t', '--time', type=str, help='Time Each Layer (Default=0)', default=0)
parser.add_argument('-w', '--wall', type=str, help='Wall-clock Time Each Layer, Requires time == 1 (Default=0)', default=0)

# evaluate arguments
args = parser.parse_args()

command = ['']
if args.lib == 'miopen':
    # ./op_driver.py -ba rnn -n 1024 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -F 1 -t 1 -w 1 -V 0
    command = ['/opt/rocm/bin/MIOpenDriver', f'{args.operation}',
                '-F', f'{args.forw}', '-H', f'{args.hid_h}',
                '-L', f'{args.seed_low}', '-M', f'{args.seed_high}',
                '-P', f'{args.dropout}', '-U', f'{args.use_dropout}',
                '-V', f'{args.verify}', '-W', f'{args.in_h}',
                '-a', f'{args.rnnalgo}', '-b', f'{args.bias}',
                '-c', f'{args.fwdtype}', '-f', f'{args.datatype}',
                '-i', f'{args.iter}', #'-h', f'{args.help}',
                '-k', f'{args.seq_len}', '-l', f'{args.num_layer}',
                '-m', f'{args.mode}', '-n', f'{args.batchsize}',
                '-o', f'{args.dump_output}', '-p', f'{args.inputmode}',
                '-q', f'{args.use_padding}', '-r', f'{args.bidirection}',
                '-t', f'{args.time}', '-w', f'{args.wall}']
    process = subprocess.run(command)

elif args.lib == 'cudnn':
    # use RNN from https://github.com/johnpzh/cudnn_samples_v8/tree/master/RNN_v8.0
    # ./RNN -dataType1 -seqLength32 -numLayers8 -inputSize224 -hiddenSize1000 -projSize1000 -miniBatch1024 -inputMode1 -dirMode0 -cellMode0 -biasMode3 -algorithm0 -mathPrecision1 -mathType0 -dropout0.0 -printWeights0
    # -algorithm          0 : (CUDNN_RNN_ALGO_STANDARD) : recurrence algorithm (0-standard, 1-persist static, 2-persist dynamic
    # -biasMode           2 : (CUDNN_RNN_DOUBLE_BIAS)   : bias type (0-no bias, 1-inp bias, 2-rec bias, 3-double bias
    # -cellMode           0 : (CUDNN_RNN_RELU)          : cell type (0-RELU, 1-TANH, 2-LSTM, 3-GRU)
    # -dataType           0 : (CUDNN_DATA_FLOAT)        : selects data format (0-FP16, 1-FP32, 2-FP64)
    # -dirMode            0 : (CUDNN_UNIDIRECTIONAL)    : recurrence pattern (0-unidirectional, 1-bidirectional)
    # -dropout            0 :                           : dropout rate
    # -hiddenSize       512 :                           : hidden size
    # -inputMode          0 : (CUDNN_LINEAR_INPUT)      : input to the RNN model (0-skip input, 1-linear input)
    # -inputSize        512 :                           : input vector size
    # -mathPrecision      0 : (CUDNN_DATA_FLOAT)        : math precision (0-FP16, 1-FP32, 2-FP64); if different than dataType => [ERROR] Inconsistent parameter: dataType does not match mathPrecision!
    # -mathType           0 : (CUDNN_DEFAULT_MATH)      : math type (0-default, 1-tensor op math, 2-tensor op math with conversion)
    # -miniBatch         64 :                           : miniBatch size
    # -numLayers          2 :                           : number of layers
    # -printWeights       0 :                           : Print weights
    # -projSize         512 : (disabled)                : LSTM cell output size
    # -seqLength         20 :                           : sequence length

    print(args.operation)
    if args.operation == 'rnn': # fp32
        dataType = 1
        mathPrecision = 1
    elif args.operation == 'rnnfp16':
        dataType = 0
        mathPrecision = 0

    if args.bias == 1:
        bias = 3
    else:
        bias = 0

    command = ['./RNN',
               '-dataType' + f'{dataType}',
               '-seqLength' + f'{args.seq_len}',
               '-numLayers' + f'{args.datatype}',
               '-inputSize' + f'{args.in_h}',
               '-hiddenSize' + f'{args.hid_h}',
               '-projSize' + f'{args.hid_h}',
               '-miniBatch' + f'{args.batchsize}',
               '-inputMode' + '1',
               '-dirMode' + f'{args.bidirection}',
               '-cellMode' + '0',
               '-biasMode' + f'{bias}',
               '-algorithm' + f'{args.rnnalgo}',
               '-mathPrecision' + f'{dataType}',
               '-mathType' + '0',
               '-dropout' + f'{args.dropout}',
               '-printWeights' + '0']
    process = subprocess.run(command)
