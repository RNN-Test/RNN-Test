# RNN-Test
RNN-Test: Generic Adversarial Testing Framework for Recurrent Neural Network Systems

This repository provides the source code of RNN-Test, example adversarial inputs for three models and figures not listed in the paper, mainly for the reproducibility. The code space is rough now and will be maintained continously.

## Description
RNN-Test is a generic adversarial testing framework for Recurrent Neural Network Systems(RNN), which could be adopted for variants of RNN models. Here, we evaluate on three RNN models, which are PTB language model, spell checker model and DeepSpeech ASR model. [PTB language model](https://github.com/nelken/tf) is a well known language model, basically to generate subsequent texts taking previous texts as input. [Spell checker model](https://github.com/Currie32/Spell-Checker/) is one of the widespread seq2seq models in NLP tasks, which receives a sentence with spelling mistakes as input and outputs the sentence with the mistakes corrected. [DeepSpeech ASR model](https://github.com/mozilla/DeepSpeech/) is a state-of-the-art speech-to-text RNN model, which is employed in lots of security-critical scenarios.

### Repository structure
In terms of this repository, it composes three main parts. The file structure is listed as below.
|-- RNN-Test
    |-- README.md 
    |-- src                                                       //Source code, including the model to test
    |   |-- audio                                                 //code for testing DeepSpeech ASR model
    |   |   |-- statefulRNNCell.py                                //RNN wrapper, to retrieve states
    |   |   |-- gen_adv_exp.py                                    //main procedure to generate adversarial inputs, able to output experiment data to .csv file
    |   |   |-- coverage.py                                       //boosting and computing the coverage metrics
    |   |   |-- make_checkpoint.py                                //utilities for the model
    |   |   |-- filterbanks.npy                                   //utilities for the model
    |   |   |-- tf_logits.py                                      //utilities for the model
    |   |   |-- testdata                                          //original test inputs, of .wav format
    |   |   |   |-- sample-000000.wav                             //one of the inputs
    |   |   |   |-- ...                                           //other inputs
    |   |   |-- commonvoice
    |   |       |-- labels.csv                                    //the predicted label 
    |   |       |-- ...
    |   |-- ptb                                                   //code for testing PTB language model
    |   |   |-- ptb_word_lm_wrapped.py                            //code of PTB language model, wrapped with RNN wrapper
    |   |   |-- gen_adv_exp.py                                    //same function as that of audio
    |   |   |-- coverage.py                                       //same function as that of audio
    |   |   |-- myreader.py                                       //utility to read data for the model
    |   |   |-- ckpt                                              //binary model to test
    |   |   |   |-- ... 
    |   |   |-- testdata                                          //training, valid, test data for the model
    |   |       |-- ptb.test.txt
    |   |       |-- ptb.train.txt
    |   |       |-- ptb.valid.txt
    |   |-- spell checker                                         //code for testing spell checker model
    |        |-- statefulRNNCell.py                               //RNN wrapper, to retrieve states
    |        |-- SpellChecker_wrapped.py                          //code of spell checker model, wrapped with RNN wrapper
    |        |-- gen_adv_exp.py                                   //same as above
    |        |-- coverage_sp.py                                   //same as above
    |        |-- kp=0.85,nl=2,th=0.95,11.ckpt.*                   //binary model to test
    |        |-- test_seq_short                                   //test inputs, with labels
    |        |-- books                                            //resources for training the model
    |            |-- ...
    |-- adv outputs                                               //example experimental outputs, including adversarial inputs 
    |   |-- audio
    |   |   |-- 2_state_diff_cell_state_adv_1572945539            //adversarial inputs of RNN-Test w. CS_C
    |   |       |-- adv_sample-000000.wav                         //each adversarial input
    |   |       |-- ...
    |   |-- ptb
    |   |   |-- 2_state_diff_hidden_state_adv1                    //output of its gen_adv_exp.py, including the adversarial inputs obtained
    |   |-- spell checker
    |       |-- 2_state_diff_cell_state_adv1                      //same as above
    |-- figures                                                   //figures not listed in the paper
        |-- RQ2                                                   //figures for RQ2 in the paper
        |   |-- perturbation                                      //perturbation vectors and the figures
        |       |-- audio                                         
        |       |   |-- sample-000000.eps                         //perturbation figure of sample-000000 input
        |       |   |-- sample-000000_vector.txt                  //perturbation vector of sample-000000 input before TSNE transformation
        |       |   |-- ...                                       //similar figures and vectors of other inputs
        |       |-- ptb                                           
        |       |   |-- step_0.eps                                //perturbation figure of input of step 0
        |       |   |-- step_0_vector.txt                         //perturbation vector of input of step 0 before TSNE transformation
        |       |   |-- ... 
        |       |-- sp                                            
        |           |-- step_0.eps                                //similar as above
        |           |-- step_0_vector.txt                         
        |           |-- ...
        |-- RQ3                                                   //figures for RQ3 in the paper
            |-- correlation                                       //correlation between model performance metrics and coverage value
            |   |-- audio                                         
            |   |   |-- DXSuccess rate.eps                        //success rate w.r.t value of neuron coverage
            |   |   |-- DXWER.eps                                 //WER w.r.t value of neuron coverage
            |   |   |-- hiddenSuccess rate.eps                    //success rate w.r.t value of hidden state coverage
            |   |   |-- hiddenWER.eps                             //WER w.r.t value of hidden state coverage
            |   |   |-- cell[0]Success rate.eps                   //success rate w.r.t value of cell state coverage on section 1
            |   |   |-- cell[0]WER.eps                            //WER w.r.t value of cell state coverage on section 1
            |   |   |-- ...                                       //other performance metrcs w.r.t cell state coverage on other sections
            |   |-- ptb                                           
            |   |   |-- DX_pp.eps                                 //perplexity w.r.t value of neuron coverage
            |   |   |-- hidden_pp.eps                             //similar as above
            |   |   |-- cell[0]pp.eps                             //similar as above
            |   |   |-- ...
            |   |-- sp                                            //similar as above
            |       |-- ...                                       
            |-- distribution                                      //distribution of value ranges of coverage metrics
                |-- audio                                         
                |   |-- DX_box_toge.eps                           //value range of neuron coverage
                |   |-- hidden_box_toge.eps                       //value range of hidden state coverage
                |   |-- cell[0]_box_toge.eps                      //value range of cell state coverage on section 1
                |   |-- ...                                       
                |-- ptb                                           //similar as above
                |   |-- ...
                |-- sp                                            //similar as above
                    |-- ...


## Dependencies
### Dependencies for code space of all the models:

### Other dependencies:
1. For PTB language model and spell checker model
Python 2.7

2. For DeepSpeech ASR model
Python 3.5/3.6
DeepSpeech 0.1.1

## To run the models
1. For PTB language model

2. For spell checker model

3. For DeepSpeech ASR model

