JointRCM
=======

Tools for improving word embeddings with paraphrasing knowledge.

These tools are used in the following paper:

Mo Yu, Mark Dredze. Improving Lexical Embeddings with Semantic Knowledge. ACL2014 short (accepted).

(The evaluation data and some new code for the camera-ready paper will be updated after June.)

###################
#CODE DESCRIPTION:#
###################

We provide three tools in this package:

################
#word2vec_joint#
################
This tool is used for jointly training the embeddings. It can be also used for training word2vec or RCM only by shutting down one of the objective (setting the corresponding number of threads be 0).

Usage:
./word2vec_joint_lm -train lm_train_data -output model_name -cbow 1 -size 200 -window 5 -negative 15 -hs 0 -threads 12 -binary 1 -lambda 0.01 -ppfile pp_train_data -pretrain 0 -pretrain-file pre-trained_model -word2vec 1 -thread-pmm 2 -epochs 400

File description:
lm_train_data: training data for language model
model_name: the learned model (embedding) file
pp_train_data: training data for the RCM objective
pre-trained_model: pre-trained embeddings using word2vec

################
#tune_lm       #
################
This tool is used for training language models based on learned word embeddings, as described in Section 4.1 in [1].

Usage:
./tune_lm -train lm_train_data -output lm_name -cbow 1 -size 200 -window 5 -negative 15 -hs 0 -threads 12 -binary 1 -pretrain 1 -pretrain-file pre-trained_lm -word2vec 1 -thread-pmm 00 -alpha 1e-3

File description:
lm_name: the learned language model file
pre-trained_lm: pre-trained language model using word2vec

Language model file has the similar data structure to that of the embedding file used in word2vec. The word2vec model file stores the input embeddings of all words in vocabulary, and the language model file stores both the input and output embeddings.

################
#eval          #
################
This tool is used for evaluating the MRR scores on evaluation data of the paraphrasing tasks.

###################
#DATA DESCRIPTION:#
###################
Todo.

References
=======
[1] Mo Yu, Mark Dredze. Improving Lexical Embeddings with Semantic Knowledge. ACL2014 short (accepted).

For questions, comments or to report bugs, please contact gflfof@gmail.com
