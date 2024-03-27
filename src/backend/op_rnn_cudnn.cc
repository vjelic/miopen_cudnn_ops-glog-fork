#include "operator.hpp"
#include <vector>
#include <iostream>
#include <iomanip>

op_rnn_cudnn::op_rnn_cudnn(void * desc) : op_rnn(desc){}
op_rnn_cudnn::~op_rnn_cudnn(){}

void print_info(const std::string& str1,
    const double& value = -1.0, const std::string& str2 = "")
{
    std::cout << std::right << std::setw(20) << str1  << ": ";
    if (value != -1)
    {
        std::cout << std::right << std::setw(8) << std::setprecision(4) << value << " ";
        std::cout << std::left  << std::setw(30) << str2  << " ";
    }
    std::cout << std::endl;
}

void op_rnn_cudnn::tune_op(){}
void op_rnn_cudnn::forward()
{
    float timeForward = 0.0;
    float timeBackwardData = 0.0;
    float timeBackwardWeights = 0.0;

    if (dataType == CUDNN_DATA_FLOAT)
    {
        rnn = new RNN_IMPL<float>();

        rnn->dataType            = dataType;
        rnn->seqLength           = seqLength;
        rnn->numLayers           = numLayers;
        rnn->inputSize           = inputSize;
        rnn->hiddenSize          = hiddenSize;
        rnn->outputSize          = outputSize;
        rnn->miniBatch           = miniBatch;
        rnn->rnnInputMode        = rnnInputMode;
        rnn->recurrencePattern   = recurrencePattern;
        rnn->rnnCellMode         = rnnCellMode;
        rnn->rnnBiasMode         = rnnBiasMode;
        rnn->rnnAlgorithm        = rnnAlgorithm;
        rnn->mathPrecision       = mathPrecision;
        rnn->mathType            = mathType;
        rnn->dropout             = dropout;
        rnn->warmupIterations    = warmupIterations;
        rnn->measureIterations   = measureIterations;

        rnn->init();
        rnn->validate();
        timeForward = rnn->forward();
        timeBackwardData = rnn->backward_data();
        timeBackwardWeights = rnn->backward_filter();

        delete rnn;
    }
    else if (dataType == CUDNN_DATA_HALF)
    {
        rnnfp16 = new RNN_IMPL<half>();

        rnnfp16->dataType            = dataType;
        rnnfp16->seqLength           = seqLength;
        rnnfp16->numLayers           = numLayers;
        rnnfp16->inputSize           = inputSize;
        rnnfp16->hiddenSize          = hiddenSize;
        rnnfp16->outputSize          = outputSize;
        rnnfp16->miniBatch           = miniBatch;
        rnnfp16->rnnInputMode        = rnnInputMode;
        rnnfp16->recurrencePattern   = recurrencePattern;
        rnnfp16->rnnCellMode         = rnnCellMode;
        rnnfp16->rnnBiasMode         = rnnBiasMode;
        rnnfp16->rnnAlgorithm        = rnnAlgorithm;
        rnnfp16->mathPrecision       = mathPrecision;
        rnnfp16->mathType            = mathType;
        rnnfp16->dropout             = dropout;
        rnnfp16->warmupIterations    = warmupIterations;
        rnnfp16->measureIterations   = measureIterations;

        rnnfp16->init();
        rnnfp16->validate();
        timeForward = rnnfp16->forward();
        timeBackwardData = rnnfp16->backward_data();
        timeBackwardWeights = rnnfp16->backward_filter();

        delete rnnfp16;
    }

    // display the measurements
    std::cout << std::endl;
    print_info("Results");
    print_info("timeForward"        , timeForward, "ms");
    print_info("timeBackwardData"   , timeBackwardData, "ms");
    print_info("timeBackwardWeights", timeBackwardWeights, "ms");
}

void op_rnn_cudnn::backward_data(){}
void op_rnn_cudnn::backward_filter(){}
void op_rnn_cudnn::backward(){}

std::string op_rnn_cudnn::get_fwd_algo_name() { return ""; }
std::string op_rnn_cudnn::get_bwd_data_name() { return ""; }
std::string op_rnn_cudnn::get_bwd_filter_name() { return ""; }

void op_rnn_cudnn::print_fwd_time(const float kernel_average_time) {}
void op_rnn_cudnn::print_bwd_time(const float kernel_average_time) {}
void op_rnn_cudnn::print_wrw_time(const float kernel_average_time) {}
