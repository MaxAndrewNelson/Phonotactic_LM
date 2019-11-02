# Phonotactic Language Model

## About
This repo contains code from Mayer and Nelson (2020) for phonotactic learning with recurrent neural language models.

## Contents
`src` contains all necessary code

`sample_data` contains files for training both feature and embedding models on an IPA transcribed version of the [CMU pronouncing dictionary](https://http://www.speech.cs.cmu.edu/cgi-bin/cmudict), and using fit models to make predictions on the nonce words used in Daland et al (2011). 
*   `corpora` contains our version of the CMU dictionary
*   `test_data` contains IPA transcribed stimuli from Daland et al (2011)
*   `features` contains a .csv file specifying an English feature set from Hayes (2009)


## Running the models
Requirements: Python 3.6+ with NumPy and Pytorch (1.0 or later)

Run `src/main.py` with the following positional arguments
*   `input_file` - path to the file containing the training data
*   `test_file` - path to the file containing held out data, after training the network will provide judgments for these words in the form of perplexity
*   `judgement_file` - path to the a file that will be created, which will contain all words in `test_file` and their corresponding perplexities
*   `feature_file` (optional) - path to a `.csv` file containing features for all phonemes in the training and test data. If not provided, an embedding model will be trained.

The following optional arguments can also be specified, but will otherwise use default values

*   `--d_emb` - dimensionality of embeddings, default 24, ignored if `feature_file` is specified.
*   `--d_hid` - dimensionality of hidden state, default 64
*   `--num_layers` - number of stacked recurrent layers, default 1
*   `--batch_size` - minibatch size, default 64
*   `--learning_rate` - initial learning rate (Adam optimizer), default 0.005
*   `--epochs` - number of epochs, default 10
*   `--tied` - boolean determing if embeddings are tied or not (Press and Wolf 2017), default TRUE, ignored if `--d_emb != --d_hid` or `feature_file` is specified.
*   `--training_split` - proportion of data to go into training, with remaining going into the dev, default 60, ignored if `dev=FALSE`
*   `--dev` - boolean determing if there will be a dev/train split on the data in `input_file`, default TRUE

Formats for training data, test data, and feature files should match those in the provided `corpora`, `test_data`, and `features` files. 

A sample version of the featural model can be run from the command line with:

`python src/main.py ./sample_data/corpora/CMU_dict_IPA ./sample_data/test_data/Daland_et_al_IPA.txt Daland_judgments.txt ./sample_data/features/english.csv`

This will fit a 23x64 RNN (there are 23 features specified in `english.csv`) on the CMU dictionary with a 60/40 dev-train split for 10 epochs and create a text file, `Daland_judgements.txt`, which contains the perplexities the fit model assigns to all words listed in `Daland_et_al_IPA.txt`.

The embedding models can be run by omitting the final argument above.



