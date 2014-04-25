//
//  EvaluationPP.cpp
//  word2vec
//
//  Created by gflfof gflfof on 14-1-22.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "EvaluationPP.h"

//const long long max_size = 2000;         // max length of strings
//const long long N = 1;                   // number of closest words
//const long long max_w = 50;              // max length of vocabulary entries

int evaluate(string filename, int threshold, Paraphrase* pp)
{
    FILE *f;
    char st1[max_size], st2[max_size], bestw[N][max_size], ch;
    float dist, len, bestd[N];
    long long words, size, a, b, c, d, b1;
    float *M;
    char *vocab;
    
    f = fopen(filename.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    if (threshold) if (words > threshold) words = threshold;
    fscanf(f, "%lld", &size);
    vocab = (char *)malloc(words * max_w * sizeof(char));
    M = (float *)malloc(words * size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
        return -1;
    }
    worddict vocabdict;
    //FILE* flist = fopen("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/list", "w");
    for (b = 0; b < words; b++) {
        fscanf(f, "%s%c", &vocab[b * max_w], &ch);
        for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
        vocabdict[string(&vocab[b * max_w])] = (int)b;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
        //fprintf(flist, "%s\n", &vocab[b * max_w]);
    }
    fclose(f);
    //fclose(flist);

    int count = 0;
    int Total = 0; int At_top = 0; int Num_Ob = 0;
    int Ks[] = {10, 20, 50, 100, 200};
    int At_tops[] = {0, 0, 0, 0, 0};

    for (word2list::iterator iter = pp->pplist.begin(); iter != pp->pplist.end(); iter++) {
        string key = iter->first;
        vector<string> val = iter -> second;
        
        count++;
        if (count % 10 == 0) {
            printf("%d\r", count);
            fflush(stdout);
        }

        for (a = 0; a < N; a++) bestd[a] = 0;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        
        strcpy(st1, key.c_str());
        for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);

        //for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st1)) break;
        worddict::iterator it = vocabdict.find(string(st1));
        if (it != vocabdict.end() && it->second < words) b1 = it->second;
        else continue;
        
        vector<int> answers;
        vector<string>::iterator iter2;
        for (iter2 = val.begin(); iter2 != val.end(); iter2++) {
            strcpy(st2, iter2->c_str());
            for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
            it = vocabdict.find(string(st2));
            if (it != vocabdict.end() && it->second < words) answers.push_back(it->second);
            else break;
        }
        if (iter2 != val.end()) continue;
        
        Num_Ob++;
        
        for (c = 0; c < words; c++) {
            if (c == b1) continue;
            dist = 0;
            for (a = 0; a < size; a++) dist += M[a + b1 * size] * M[a + c * size];
            for (a = 0; a < N; a++) {
                if (dist > bestd[a]) {
                    for (d = N - 1; d > a; d--) {
                        bestd[d] = bestd[d - 1];
                        strcpy(bestw[d], bestw[d - 1]);
                    }
                    bestd[a] = dist;
                    strcpy(bestw[a], &vocab[c * max_w]);
                    break;
                }
            }
        }
        
        int PP_total = (int)val.size();
        //int PP_at_top = 0;
        for (vector<string>::iterator iter = val.begin(); iter != val.end(); iter++) {
            strcpy(st2, iter->c_str());
            for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
            
            for (int i = 0; i < N; i++) {
                if (!strcmp(st2, bestw[i])) {
                    At_top ++;
                    break;
                }
            }
            
            for (int k = 0; k < 5; k++) {
                for (int i = 0; i < Ks[k]; i++) {
                    if (!strcmp(st2, bestw[i])) {
                        At_tops[k] ++;
                        break;
                    }
                }
            }
        }

        Total += PP_total;
        //At_top += PP_at_top;
    }
    printf("\n");
    printf("Queries seen / total: %d %d   %.2f %% \n", Num_Ob, (int)pp->pplist.size(), (float)Num_Ob/pp->pplist.size()*100);
    printf("Mean Recall @ %lld: %d %d %.2f %% \n", N, At_top, Total, (float)At_top / Total * 100);
    for (int k = 0; k < 5; k++) {
        printf("Mean Recall @ %d: %d %d %.2f %% \n", Ks[k], At_tops[k], Total, (float)At_tops[k] / Total * 100);
    }
    return 0;
}

int LoadModel(string filename, string vocabfile, int threshold)
{
    FILE *f;
    char ch;
    float len;
    long long words, size, a, b;
    float *M;
    char *vocab;
    
    f = fopen(filename.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    if (threshold) if (words > threshold) words = threshold;
    fscanf(f, "%lld", &size);
    vocab = (char *)malloc(words * max_w * sizeof(char));
    M = (float *)malloc(words * size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
        return -1;
    }
    worddict vocabdict;
    FILE* flist = fopen(vocabfile.c_str(), "w");
    for (b = 0; b < words; b++) {
        fscanf(f, "%s%c", &vocab[b * max_w], &ch);
        for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
        vocabdict[string(&vocab[b * max_w])] = (int)b;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
        fprintf(flist, "%s\n", &vocab[b * max_w]);
    }
    fclose(f);
    fclose(flist);
    return 0;
}

int evaluateMRR(string filename, int threshold, Paraphrase* pp)
{
    FILE *f;
    char st1[max_size], st2[max_size], bestw[N][max_size], ch;
    float dist, len, bestd[N];
    long long words, size, a, b, c, d, b1;
    float *M;
    char *vocab;
    
    f = fopen(filename.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    if (threshold) if (words > threshold) words = threshold;
    fscanf(f, "%lld", &size);
    vocab = (char *)malloc(words * max_w * sizeof(char));
    M = (float *)malloc(words * size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
        return -1;
    }
    worddict vocabdict;
    //FILE* flist = fopen("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/list", "w");
    for (b = 0; b < words; b++) {
        fscanf(f, "%s%c", &vocab[b * max_w], &ch);
        for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
        worddict::iterator iter = vocabdict.find(string(&vocab[b* max_w]));
        if (iter == vocabdict.end()) {
            vocabdict[string(&vocab[b * max_w])] = (int)b;
        }
        //vocabdict[string(&vocab[b * max_w])] = (int)b;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
        //fprintf(flist, "%s\n", &vocab[b * max_w]);
    }
    fclose(f);
    //fclose(flist);
    
    int count = 0;
    int Total = 0; int Num_Ob = 0;
    double score_total = 0.0;
    
    for (word2list::iterator iter = pp->pplist.begin(); iter != pp->pplist.end(); iter++) {
        string key = iter->first;
        vector<string> val = iter -> second;
        
        count++;
        if (count % 10 == 0) {
            printf("%d\r", count);
            fflush(stdout);
        }
        
        for (a = 0; a < N; a++) bestd[a] = 0;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        
        strcpy(st1, key.c_str());
        for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);
        
        //for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st1)) break;
        worddict::iterator it = vocabdict.find(string(st1));
        if (it != vocabdict.end() && it->second < words) b1 = it->second;
        else continue;
        
        vector<int> answers;
        vector<string>::iterator iter2;
        for (iter2 = val.begin(); iter2 != val.end(); iter2++) {
            strcpy(st2, iter2->c_str());
            for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
            it = vocabdict.find(string(st2));
            if (it != vocabdict.end() && it->second < words) answers.push_back(it->second);
            else break;
            dist = 0;
            for (a = 0; a < size; a++) dist += M[a + b1 * size] * M[a + it->second * size];
            //printf("%lf\n", dist);
        }
        if (iter2 != val.end()) continue;
        
        Num_Ob++;
        for (c = 0; c < words; c++) {
            if (c == b1) continue;
            dist = 0;
            for (a = 0; a < size; a++) dist += M[a + b1 * size] * M[a + c * size];
            for (a = 0; a < N; a++) {
                if (dist > bestd[a]) {
                    for (d = N - 1; d > a; d--) {
                        bestd[d] = bestd[d - 1];
                        strcpy(bestw[d], bestw[d - 1]);
                    }
                    bestd[a] = dist;
                    strcpy(bestw[a], &vocab[c * max_w]);
                    break;
                }
            }
        }
        
        double score = 0.0;
        int PP_total = (int)val.size();
        
        for (vector<string>::iterator iter = val.begin(); iter != val.end(); iter++) {
            strcpy(st2, iter->c_str());
            for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
            
            for (int i = 0; i < N; i++) {
                if (!strcmp(st2, bestw[i])) {
                    //printf("%lf\n", bestd[i]);
                    score += (double)1 / (i + 1);
                    break;
                }
            }
        }
        
        Total += PP_total;
        score /= PP_total;
        score_total += score;
    }
    printf("\n");
    printf("Queries seen / total: %d %d   %.2f %% \n", Num_Ob, (int)pp->pplist.size(), (float)Num_Ob/pp->pplist.size()*100);
    printf("MRR (threshold %lld): %.2f %d %.2f %% \n", N, score_total, Num_Ob, score_total / Num_Ob * 100);
    return 0;
}

int evaluateMRR(string filename, int threshold, Paraphrase2* pp)
{
    FILE *f;
    char st1[max_size], st2[max_size], bestw[N][max_size], ch;
    float dist, len, bestd[N];
    long long words, size, a, b, c, d, b1;
    float *M;
    char *vocab;
    
    f = fopen(filename.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    if (threshold) if (words > threshold) words = threshold;
    fscanf(f, "%lld", &size);
    vocab = (char *)malloc(words * max_w * sizeof(char));
    M = (float *)malloc(words * size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
        return -1;
    }
    worddict vocabdict;
    //FILE* flist = fopen("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/list", "w");
    for (b = 0; b < words; b++) {
        fscanf(f, "%s%c", &vocab[b * max_w], &ch);
        for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
        vocabdict[string(&vocab[b * max_w])] = (int)b;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
        //fprintf(flist, "%s\n", &vocab[b * max_w]);
    }
    fclose(f);
    //fclose(flist);
    
    int count = 0;
    int Total = 0; int Num_Ob = 0;
    double score_total = 0.0;
    
    vector<string> val;
    for (word2dict::iterator iter = pp->ppdict.begin(); iter != pp->ppdict.end(); iter++) {
        string key = iter->first;
        pp->GetList(key, val);
        //vector<string> val = iter -> second;
        
        count++;
        if (count % 10 == 0) {
            printf("%d\r", count);
            fflush(stdout);
        }
        
        for (a = 0; a < N; a++) bestd[a] = 0;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        
        strcpy(st1, key.c_str());
        for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);
        
        //for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st1)) break;
        worddict::iterator it = vocabdict.find(string(st1));
        if (it != vocabdict.end() && it->second < words) b1 = it->second;
        else continue;
        
        vector<int> answers;
        vector<string>::iterator iter2;
        for (iter2 = val.begin(); iter2 != val.end(); iter2++) {
            strcpy(st2, iter2->c_str());
            for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
            it = vocabdict.find(string(st2));
            if (it != vocabdict.end() && it->second < words) answers.push_back(it->second);
            else break;
            dist = 0;
            for (a = 0; a < size; a++) dist += M[a + b1 * size] * M[a + it->second * size];
            //printf("%lf\n", dist);
        }
        if (iter2 != val.end()) continue;
        
        Num_Ob++;
        for (c = 0; c < words; c++) {
            if (c == b1) continue;
            dist = 0;
            for (a = 0; a < size; a++) dist += M[a + b1 * size] * M[a + c * size];
            for (a = 0; a < N; a++) {
                if (dist > bestd[a]) {
                    for (d = N - 1; d > a; d--) {
                        bestd[d] = bestd[d - 1];
                        strcpy(bestw[d], bestw[d - 1]);
                    }
                    bestd[a] = dist;
                    strcpy(bestw[a], &vocab[c * max_w]);
                    break;
                }
            }
        }
        
        double score = 0.0;
        int PP_total = (int)val.size();
        
        for (vector<string>::iterator iter = val.begin(); iter != val.end(); iter++) {
            strcpy(st2, iter->c_str());
            for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
            
            for (int i = 0; i < N; i++) {
                if (!strcmp(st2, bestw[i])) {
                    printf("%lf\n", bestd[i]);
                    score += (double)1 / (i + 1);
                    break;
                }
            }
        }
        
        Total += PP_total;
        score /= PP_total;
        score_total += score;
    }
    printf("\n");
    printf("Queries seen / total: %d %d   %.2f %% \n", Num_Ob, (int)pp->ppdict.size(), (float)Num_Ob/pp->ppdict.size()*100);
    printf("MRR (threshold %lld): %.2f %d %.2f %% \n", N, score_total, Num_Ob, score_total / Num_Ob * 100);
    return 0;
}

int GetLogLoss(string filename, int threshold, Paraphrase* pp)
{
    FILE *f;
    char st1[max_size], st2[max_size], ch;
    float dist, len;
    long long words, size, a, b, c, b1;
    float *M;
    char *vocab;
    
    f = fopen(filename.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    if (threshold) if (words > threshold) words = threshold;
    fscanf(f, "%lld", &size);
    vocab = (char *)malloc(words * max_w * sizeof(char));
    M = (float *)malloc(words * size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
        return -1;
    }
    worddict vocabdict;
    //FILE* flist = fopen("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/list", "w");
    for (b = 0; b < words; b++) {
        fscanf(f, "%s%c", &vocab[b * max_w], &ch);
        for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
        worddict::iterator iter = vocabdict.find(string(&vocab[b* max_w]));
        if (iter == vocabdict.end()) {
            vocabdict[string(&vocab[b * max_w])] = (int)b;
        }
        //vocabdict[string(&vocab[b * max_w])] = (int)b;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
    fclose(f);
    //fclose(flist);
    
    int count = 0; int Num_Ob = 0;
    double score_total = 0.0;
    double numerator = 0.0, denominator = 0.0;
    
    for (word2list::iterator iter = pp->pplist.begin(); iter != pp->pplist.end(); iter++) {
        string key = iter->first;
        vector<string> val = iter -> second;
        
        count++;
        if (count % 10 == 0) {
            printf("%d\r", count);
            fflush(stdout);
        }
        
        strcpy(st1, key.c_str());
        for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);
        
        worddict::iterator it = vocabdict.find(string(st1));
        if (it != vocabdict.end() && it->second < words) b1 = it->second;
        else continue;
        
        vector<int> answers;
        vector<string>::iterator iter2;
        for (iter2 = val.begin(); iter2 != val.end(); iter2++) {
            strcpy(st2, iter2->c_str());
            for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
            it = vocabdict.find(string(st2));
            if (it != vocabdict.end() && it->second < words) answers.push_back(it->second);
            else break;
            dist = 0;
            numerator = 0.0;
            for (a = 0; a < size; a++) numerator += M[a + b1 * size] * M[a + it->second * size];
            //printf("%lf\n", dist);
        }
        if (iter2 != val.end()) continue;
        
        Num_Ob++;
        denominator = 0.0;
        for (c = 0; c < words; c++) {
            if (c == b1) continue;
            dist = 0;
            for (a = 0; a < size; a++) dist += M[a + b1 * size] * M[a + c * size];
            denominator += exp(dist);
        }
        denominator = log(denominator);
        
        double logscore = numerator - denominator;
        score_total += logscore;
    }
    printf("\n");
    printf("Queries seen / total: %d %d   %.2f %% \n", Num_Ob, (int)pp->pplist.size(), (float)Num_Ob/pp->pplist.size()*100);
    printf("Mean LogLoss: %.2f %d %.2f %% \n", score_total, Num_Ob, -score_total / Num_Ob);
    printf("Mean LogLoss (global): %.2f %d %.2f %% \n", score_total, (int)pp->pplist.size(), -score_total / (int)pp->pplist.size());
    return 0;
}

int evaluateSubMRR(string filename, int threshold, Paraphrase* pp, int semdim)
{
    FILE *f;
    char st1[max_size], st2[max_size], bestw[N][max_size], ch;
    float dist, len, bestd[N];
    long long words, size;
    long long a, b, c, d, b1;
    float *M;
    char *vocab;
    
    f = fopen(filename.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    if (threshold) if (words > threshold) words = threshold;
    fscanf(f, "%lld", &size);
    vocab = (char *)malloc(words * max_w * sizeof(char));
    M = (float *)malloc(words * size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
        return -1;
    }
    worddict vocabdict;
    //FILE* flist = fopen("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/list", "w");
    for (b = 0; b < words; b++) {
        fscanf(f, "%s%c", &vocab[b * max_w], &ch);
        for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
        worddict::iterator iter = vocabdict.find(string(&vocab[b* max_w]));
        if (iter == vocabdict.end()) {
            vocabdict[string(&vocab[b * max_w])] = (int)b;
        }
        //vocabdict[string(&vocab[b * max_w])] = (int)b;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
        //fprintf(flist, "%s\n", &vocab[b * max_w]);
    }
    fclose(f);
    //fclose(flist);
    
    int count = 0;
    int Total = 0; int Num_Ob = 0;
    double score_total = 0.0;
    
    for (word2list::iterator iter = pp->pplist.begin(); iter != pp->pplist.end(); iter++) {
        string key = iter->first;
        vector<string> val = iter -> second;
        
        count++;
        if (count % 10 == 0) {
            printf("%d\r", count);
            fflush(stdout);
        }
        
        for (a = 0; a < N; a++) bestd[a] = 0;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        
        strcpy(st1, key.c_str());
        for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);
        
        //for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st1)) break;
        worddict::iterator it = vocabdict.find(string(st1));
        if (it != vocabdict.end() && it->second < words) b1 = it->second;
        else continue;
        
        vector<int> answers;
        vector<string>::iterator iter2;
        for (iter2 = val.begin(); iter2 != val.end(); iter2++) {
            strcpy(st2, iter2->c_str());
            for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
            it = vocabdict.find(string(st2));
            if (it != vocabdict.end() && it->second < words) answers.push_back(it->second);
            else break;
            dist = 0;
            for (a = 0; a < semdim; a++) dist += M[a + b1 * size] * M[a + it->second * size];
            //printf("%lf\n", dist);
        }
        if (iter2 != val.end()) continue;
        
        Num_Ob++;
        for (c = 0; c < words; c++) {
            if (c == b1) continue;
            dist = 0;
            for (a = 0; a < semdim; a++) dist += M[a + b1 * size] * M[a + c * size];
            for (a = 0; a < N; a++) {
                if (dist > bestd[a]) {
                    for (d = N - 1; d > a; d--) {
                        bestd[d] = bestd[d - 1];
                        strcpy(bestw[d], bestw[d - 1]);
                    }
                    bestd[a] = dist;
                    strcpy(bestw[a], &vocab[c * max_w]);
                    break;
                }
            }
        }
        
        double score = 0.0;
        int PP_total = (int)val.size();
        
        for (vector<string>::iterator iter = val.begin(); iter != val.end(); iter++) {
            strcpy(st2, iter->c_str());
            for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
            
            for (int i = 0; i < N; i++) {
                if (!strcmp(st2, bestw[i])) {
                    printf("%lf\n", bestd[i]);
                    score += (double)1 / (i + 1);
                    break;
                }
            }
        }
        
        Total += PP_total;
        score /= PP_total;
        score_total += score;
    }
    printf("\n");
    printf("Queries seen / total: %d %d   %.2f %% \n", Num_Ob, (int)pp->pplist.size(), (float)Num_Ob/pp->pplist.size()*100);
    printf("MRR (threshold %lld): %.2f %d %.2f %% \n", N, score_total, Num_Ob, score_total / Num_Ob * 100);
    return 0;
}

int getInner(string filename, int threshold, string filepair)
{
    FILE *f;
    char st1[max_size], st2[max_size], ch;
    float dist, len;
    long long words, size, a, b, b1;
    float *M;
    char *vocab;
    
    f = fopen(filename.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    if (threshold) if (words > threshold) words = threshold;
    fscanf(f, "%lld", &size);
    vocab = (char *)malloc(words * max_w * sizeof(char));
    M = (float *)malloc(words * size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
        return -1;
    }
    worddict vocabdict;
    //FILE* flist = fopen("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/list", "w");
    for (b = 0; b < words; b++) {
        fscanf(f, "%s%c", &vocab[b * max_w], &ch);
        for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
        worddict::iterator iter = vocabdict.find(string(&vocab[b* max_w]));
        if (iter == vocabdict.end()) {
            vocabdict[string(&vocab[b * max_w])] = (int)b;
        }
        //vocabdict[string(&vocab[b * max_w])] = (int)b;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        //for (a = 0; a < size; a++) M[a + b * size] /= len;
        //fprintf(flist, "%s\n", &vocab[b * max_w]);
    }
    fclose(f);
    //fclose(flist);
    
    ifstream ifs(filepair.c_str());
    char line_buf[1000];
    ifs.getline(line_buf, 1000, '\n');
    while (strcmp(line_buf, "")) {
        istringstream iss(line_buf);
        string key; iss >> key;
        string val; iss >> val;
        ifs.getline(line_buf, 1000, '\n');
        
        //cout << key << "\t" << val << endl;
        strcpy(st1, key.c_str());
        for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);
        
        worddict::iterator it = vocabdict.find(string(st1));
        if (it != vocabdict.end()) b1 = it->second;
        else {
            printf("unknown\n");
            continue;
        }
        
        strcpy(st2, val.c_str());
        for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
        it = vocabdict.find(string(st2));
        if (it == vocabdict.end()) {
            printf("unknown\n");
            continue;
        }
        dist = 0;
        for (a = 0; a < size; a++) dist += M[a + b1 * size] * M[a + it->second * size];
        printf("%lf\n", dist);
    }
    ifs.close();
    return 0;
}
