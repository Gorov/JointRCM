//
//  word2vec.cpp
//  word2vec
//
//  Created by gflfof gflfof on 14-1-22.
//  Copyright (c) 2014年 hit. All rights reserved.
//

//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "Paraphrase.h"
#include "EvaluationPP.h"

int main(int argc, char **argv) {
    string modelfile(argv[1]);
    string testfile(argv[2]);
    int thres = 10000;
    int analysis = 0;
    if(argc == 4)
    {
        thres = atoi(argv[3]);
    }
    if(argc == 5)
    {
        analysis = atoi(argv[4]);
    }
    printf("building pp dict...\n");
    Paraphrase* pp = new Paraphrase(testfile);
    //pp = new Paraphrase("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/ppdb-1.0-s-lexical.test");
    //pp = new Paraphrase("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/ppdb-1.0-s-lexical.wordlist100");
    printf("evaluating...\n");
    //evaluate(modelfile, thres, pp);
    evaluateMRR(modelfile, thres, pp);
    //GetLogLoss(modelfile, thres, pp);
    delete pp;
    pp = NULL;
    return 0;
}
