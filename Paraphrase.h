//
//  Paraphrase.h
//  word2vec
//
//  Created by gflfof gflfof on 14-1-22.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef word2vec_Paraphrase_h
#define word2vec_Paraphrase_h

#include <tr1/unordered_map>
#include <iostream>
#include <vector>
#include <fstream>
#include <string.h>

using namespace std;

typedef std::tr1::unordered_map<string, vector<string> > word2list;
typedef std::tr1::unordered_map<string, int> word2int;
typedef std::tr1::unordered_map<string, word2int> word2dict;

class Paraphrase2
{
public:
    word2dict ppdict;
    
    Paraphrase2(string filename)
    {
        InitDict(filename);
    }
    
    void InitDict(string filename);
    
    void GetList(string word, vector<string>& pplist);
    void GetDict(vector<string>& keys);
};

class Paraphrase
{
    public:
    word2list pplist;
    
    Paraphrase(string filename)
    {
        InitList(filename);
    }
    
    Paraphrase(Paraphrase2* ppdict)
    {
        
    }
    
    void InitList(string filename);
    
    //vector<string>* GetList(string word);
    void GetList(string word, vector<string>& retlist);
};

#endif
