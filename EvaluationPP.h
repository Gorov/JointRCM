//
//  EvaluationPP.h
//  word2vec
//
//  Created by gflfof gflfof on 14-1-22.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef word2vec_EvaluationPP_h
#define word2vec_EvaluationPP_h

#include "Paraphrase.h"

const long long max_size = 2000;         // max length of strings
const long long N = 200;                   // number of closest words
const long long max_w = 50;              // max length of vocabulary entries

typedef std::tr1::unordered_map<string, int > worddict;

int evaluate(string filename, int threshold, Paraphrase* pp);
int evaluateMRR(string filename, int threshold, Paraphrase* pp);
int evaluateMRR(string filename, int threshold, Paraphrase2* pp);

int getInner(string filename, int threshold, string filepair);

int GetLogLoss(string filename, int threshold, Paraphrase* pp);

#endif
