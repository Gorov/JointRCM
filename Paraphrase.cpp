//
//  Paraphrase.cpp
//  word2vec
//
//  Created by gflfof gflfof on 14-1-22.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include "Paraphrase.h"
#include <sstream>

void Paraphrase::InitList(string filename)
{
    ifstream ifs(filename.c_str());
    char line_buf[1000];
    ifs.getline(line_buf, 1000, '\n');
    while (strcmp(line_buf, "")) {
        istringstream iss(line_buf);
        string key; iss >> key;
        
        vector<string> val;
        pplist[key] = val;
        while (! iss.eof()) {
            string word; iss >> word;
            pplist[key].push_back(word);
            //iss >> word;
        }
        ifs.getline(line_buf, 1000, '\n');
    }
    ifs.close();
}

/*
vector<string>* Paraphrase::GetList(string word)
{
    word2list::iterator iter = pplist.find(word);
    if (iter != pplist.end()) {
        return &iter->second;
    }
    else
    {
        return NULL;
    }
}
 */

void Paraphrase::GetList(string word, vector<string>& retlist)
{
    retlist.clear();
    word2list::iterator iter = pplist.find(word);
    if (iter != pplist.end()) {
        retlist = iter->second;
    }
}

void Paraphrase2::InitDict(string filename)
{
    ifstream ifs(filename.c_str());
    char line_buf[1000];
    ifs.getline(line_buf, 1000, '\n');
    while (strcmp(line_buf, "")) {
        istringstream iss(line_buf);
        string key; iss >> key;
        word2dict::iterator iter = ppdict.find(key);
        if (iter == ppdict.end()) {
            word2int d;
            ppdict[key] = d;
        }

        while (! iss.eof()) {
            string word; iss >> word;
            word2int::iterator it_wi = ppdict[key].find(word);
            if (it_wi == ppdict[key].end()) {
                ppdict[key][word] = 1;
            }
            
            word2dict::iterator it_wd = ppdict.find(word);
            if (it_wd == ppdict.end()) {
                word2int d;
                ppdict[word] = d;
                ppdict[word][key] = 1;
            }
            else
            {
                it_wi = ppdict[word].find(key);
                if (it_wi == ppdict[word].end()) {
                    ppdict[word][key] = 1;
                }
            }
        }
        ifs.getline(line_buf, 1000, '\n');
    }
    ifs.close();
}

void Paraphrase2::GetList(string word, vector<string>& pplist)
{
    word2dict::iterator iter = ppdict.find(word);
    pplist.clear();
    if (iter != ppdict.end()) {
        for (word2int::iterator it = iter->second.begin(); it != iter->second.end(); it++) {
            pplist.push_back(it->first);
        }
    }
}

void Paraphrase2::GetDict(vector<string>& keys)
{
}
