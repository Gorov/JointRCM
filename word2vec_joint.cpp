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
#include <sstream>
#include "Paraphrase.h"
#include "EvaluationPP.h"



#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
    long long cn;
    int *point;
    char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5;
int num_threads = 1, min_reduce = 1, num_thread_pmm = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
long long pp_count_actual = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real lambda = 0.1, starting_lambda;
real *syn0;
real *syn1, *syn1neg, *expTable;
//real *A;

word2int upperdict;
vector<string> uppervocab;

int weight_tying = 0;

clock_t start;
char train_file2[MAX_STRING];
char pp_file[MAX_STRING];
int word2vec = 1;
int reg = 0;
int reg_in = 0;
int reg_out = 0;
int epochs = 1;

int pretrain = 0;
char pretrain_file[MAX_STRING];

Paraphrase2* pp;
Paraphrase* ppeval;
//real lambda = 0.1;
real lambda_in = 0.1;
real lambda_out = 0.1;

long long sem_dim = 0;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

int LoadEmb(string modelfile);

void InitUnigramTable() {
    int a, i;
    long long train_words_pow = 0;
    real d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
    i = 0;
    d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
    for (a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (real)table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
        }
        if (i >= vocab_size) i = (int)vocab_size - 1;
    }
}

void BuildUpperDict() {
    char tmpword[max_w];
    for (int a = 0; a < vocab_size; a++) {
        for (int c = 0; c < max_w; c++) tmpword[c] = toupper(vocab[a].word[c]);
        word2int::iterator iter = upperdict.find(string(tmpword));
        if (iter == upperdict.end()) upperdict[string(tmpword)] = (int)a;
        uppervocab.push_back(tmpword);
    }
}

int evaluateMRRout(int threshold, Paraphrase* pp)
{
    char bestw[N][max_size];
    real* M;
    float dist, len, bestd[N];
    long long words, size, a, c, d, b1;
    
    size = layer1_size;
    words = vocab_size;
    if (threshold) if (words > threshold) words = threshold;
    
    M = (float *)malloc(words * layer1_size * sizeof(float));
    memcpy(M, syn1neg, words * layer1_size * sizeof(float));
    for (int i = 0; i < words; i++) {
        len = 0;
        for (a = 0; a < size; a++) len += M[a + i * size] * M[a + i * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + i * size] /= len;
    }
    size = layer1_size;
    
    int count = 0;
    int Total = 0; int Num_Ob = 0;
    double score_total = 0.0;
    char key[max_size], answer[max_size];
    for (word2list::iterator iter = pp->pplist.begin(); iter != pp->pplist.end(); iter++) {
        strcpy(key,iter->first.c_str());
        for (a = 0; a < max_w; a++) key[a] = toupper(key[a]);
        vector<string> val = iter -> second;
        
        count++;
        if (count % 10 == 0) {
            printf("%d\r", count);
            fflush(stdout);
        }
        
        for (a = 0; a < N; a++) bestd[a] = 0;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        
        //for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st1)) break;
        worddict::iterator it = upperdict.find(string(key));
        if (it != upperdict.end() && it->second < words) b1 = it->second;
        else continue;
        
        vector<string>::iterator iter2;
        for (iter2 = val.begin(); iter2 != val.end(); iter2++) {
            strcpy(answer, iter2->c_str());
            for (a = 0; a < max_w; a++) answer[a] = toupper(answer[a]);
            it = upperdict.find(string(answer));
            if (it == upperdict.end() || it->second >= words) break;
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
                    strcpy(bestw[a], uppervocab[c].c_str());
                    break;
                }
            }
        }
        
        double score = 0.0;
        int PP_total = (int)val.size();
        
        for (int i = 0; i < N; i++) {
            if (!strcmp(answer, bestw[i])) {
                //printf("%lf\n", bestd[i]);
                score += (double)1 / (i + 1);
                break;
            }
        }
        
        Total += PP_total;
        score /= PP_total;
        score_total += score;
    }
    printf("\n");
    free(M);
    printf("Queries seen / total: %d %d   %.2f %% \n", Num_Ob, (int)pp->pplist.size(), (float)Num_Ob/pp->pplist.size()*100);
    printf("MRR (threshold %lld): %.2f %d %.2f %% \n", N, score_total, Num_Ob, score_total / Num_Ob * 100);
    return 0;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    int a, size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    size = vocab_size;
    train_words = 0;
    for (a = 0; a < size; a++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if (vocab[a].cn < min_count) {
            vocab_size--;
            free(vocab[vocab_size].word);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetWordHash(vocab[a].word);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = a;
            train_words += vocab[a].cn;
        }
    }
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    // Allocate memory for the binary tree construction
    for (a = 0; a < vocab_size; a++) {
        vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
    int a, b = 0;
    
    unsigned int hash;
    for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
        vocab[b].cn = vocab[a].cn;
        vocab[b].word = vocab[a].word;
        b++;
    } else free(vocab[a].word);
    vocab_size = b;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    for (a = 0; a < vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
    for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
    pos1 = vocab_size - 1;
    pos2 = vocab_size;
    // Following algorithm constructs the Huffman tree by adding one node at a time
    for (a = 0; a < vocab_size - 1; a++) {
        // First, find two smallest nodes 'min1, min2'
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                pos1--;
            } else {
                min1i = pos2;
                pos2++;
            }
        } else {
            min1i = pos2;
            pos2++;
        }
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                pos1--;
            } else {
                min2i = pos2;
                pos2++;
            }
        } else {
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;
    }
    // Now assign binary code to each vocabulary word
    for (a = 0; a < vocab_size; a++) {
        b = a;
        i = 0;
        while (1) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == vocab_size * 2 - 2) break;
        }
        vocab[a].codelen = i;
        vocab[a].point[0] = vocab_size - 2;
        for (b = 0; b < i; b++) {
            vocab[a].code[i - b - 1] = code[b];
            vocab[a].point[i - b] = point[b] - vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

void LearnVocabFromTrainFile() {
    char word[MAX_STRING];
    FILE *fin;
    long long a, i;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;
    AddWordToVocab((char *)"</s>");
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        i = SearchVocab(word);
        if (i == -1) {
            a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else vocab[i].cn++;
        if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);
    fclose(fin);
}

void SaveVocab() {
    long long i;
    FILE *fo = fopen(save_vocab_file, "wb");
    for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void ReadVocab() {
    long long a, i = 0;
    char c;
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        a = AddWordToVocab(word);
        fscanf(fin, "%lld%c", &vocab[a].cn, &c);
        i++;
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    file_size = ftell(fin);
    fclose(fin);
}

void InitNet() {
    long long a, b;
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    if (hs) {
        a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
        if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
            syn1[a * layer1_size + b] = 0;
    }
    if (negative>0) {
        a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
        if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
            syn1neg[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
            //syn1neg[a * layer1_size + b] = 0;
    }
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
        syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
    
    if (pretrain) LoadEmb(pretrain_file);
    //else for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
    //        syn1neg[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
    
    CreateBinaryTree();
}

void *TrainModelRegThread(void *id) {
    long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(train_file, "rb");
    fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
    while (1) {
        if (word_count - last_word_count > 10000) {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                       word_count_actual / (real)(train_words + 1) * 100,
                       word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            while (1) {
                word = ReadWordIndex(fi);
                if (feof(fi)) break;
                if (word == -1) continue;
                word_count++;
                if (word == 0) break;
                // The subsampling randomly discards frequent words while keeping the ranking same
                if (sample > 0) {
                    real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                }
                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi)) break;
        if (word_count > train_words / num_threads) break;
        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
        if (cbow) {  //train the cbow architecture
            // in -> hidden
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                c = sentence_position - window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
            }
            // NEGATIVE SAMPLING
            if (negative > 0) for (d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = word;
                    label = 1;
                } else {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (vocab_size - 1) + 1;
                    if (target == word) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
            }
            // hidden -> in
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                c = sentence_position - window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                l1 = last_word * layer1_size;
                
                //todo:
                vector<string> pplist;
                if (reg_in) {
                    // Regularizations
                    //vector<string>* pplist = pp->GetList(vocab[last_word].word);
                    pp->GetList(vocab[last_word].word, pplist);
                    //if (NULL == pplist) continue;
                    if (pplist.size() != 0){
                        unsigned long length = pplist.size();
                        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
                        for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                            int id = SearchVocab((char*)iter->c_str());
                            if (id != -1) {
                                long long l = id * layer1_size;
                                for (c = 0; c < sem_dim; c++) neu1[c] += syn0[c + l];
                            }
                            else
                            {
                                length--;
                            }
                        }
                        
                        if (length != 0) {
                            //update pp words
                            for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                                int id = SearchVocab((char*)iter->c_str());
                                if (id != -1) {
                                    long long l = id * layer1_size;
                                    for (c = 0; c < sem_dim; c++) syn0[c + l] -= alpha * lambda / length * (syn0[c + l] - syn0[c + l1]);
                                }
                            }
                            
                            //update central word
                            for (c = 0; c < sem_dim; c++) syn0[c + l1] += alpha * lambda * (neu1[c] / length - syn0[c + l1]);
                        }
                    }
                }
                for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
                //currently we assume that words in the window are unlikely to have overlaped pp-lists. Thus the embedding updated here will be unlikely to change the gradients for the regularization terms of other words.
            }
            vector<string> pplist;
            if (reg_out) {
                if(negative > 0){
                    pp->GetList(vocab[word].word, pplist);
                    if(0 != pplist.size()) {
                        l2 = word * layer1_size;
                        
                        unsigned long length = pplist.size();
                        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                        for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                            int id = SearchVocab((char*)iter->c_str());
                            if (id != -1) {
                                long long l = id * layer1_size;
                                for (c = 0; c < layer1_size; c++) neu1e[c] += syn1neg[c + l];
                            }
                            else
                            {
                                length--;
                            }
                        }
                        
                        if (length != 0) {
                            //update pp words
                            for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                                int id = SearchVocab((char*)iter->c_str());
                                if (id != -1) {
                                    long long l = id * layer1_size;
                                    for (c = 0; c < layer1_size; c++) syn1neg[c + l] -= alpha * lambda / length * (syn1neg[c + l] - syn1neg[c + l2]);
                                }
                            }
                            
                            //update central word
                            for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += alpha * lambda * (neu1e[c] / length - syn1neg[c + l2]);
                        }
                    }
                }
            }
        }
        else if(!cbow) {  //train skip-gram
            for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                c = sentence_position - window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                l1 = last_word * layer1_size;
                for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                // HIERARCHICAL SOFTMAX
                if (hs) for (d = 0; d < vocab[word].codelen; d++) {
                    f = 0;
                    l2 = vocab[word].point[d] * layer1_size;
                    // Propagate hidden -> output
                    for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
                    if (f <= -MAX_EXP) continue;
                    else if (f >= MAX_EXP) continue;
                    else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                    // 'g' is the gradient multiplied by the learning rate
                    g = (1 - vocab[word].code[d] - f) * alpha;
                    // Propagate errors output -> hidden
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                    // Learn weights hidden -> output
                    for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
                }
                // NEGATIVE SAMPLING
                if (negative > 0 && !weight_tying) for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = table[(next_random >> 16) % table_size];
                        if (target == 0) target = next_random % (vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    l2 = target * layer1_size;
                    f = 0;
                    for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                    for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
                }
                // WEIGHT-TYING NEGATIVE SAMPLING
                if (weight_tying > 0) for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = table[(next_random >> 16) % table_size];
                        if (target == 0) target = next_random % (vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    l2 = target * layer1_size;
                    f = 0;
                    for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn0[c + l2];
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                    for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn0[c + l2];
                    for (c = 0; c < layer1_size; c++) syn0[c + l2] += g * syn0[c + l1];
                }
                // Learn weights input -> hidden
                for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
                vector<string> pplist;
                if (reg_in) {
                    // Regularizations
                    //vector<string>* pplist = pp->GetList(vocab[last_word].word);
                    pp->GetList(vocab[last_word].word, pplist);
                    //if (NULL == pplist) continue;
                    if (pplist.size() == 0) continue;
                    
                    unsigned long length = pplist.size();
                    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                    for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                        int id = SearchVocab((char*)iter->c_str());
                        if (id != -1) {
                            long long l = id * layer1_size;
                            for (c = 0; c < layer1_size; c++) neu1e[c] += syn0[c + l];
                        }
                        else
                        {
                            length--;
                        }
                    }
                
                    if (length == 0) {
                        continue;
                    }
                
                    //update pp words
                    for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                        int id = SearchVocab((char*)iter->c_str());
                        if (id != -1) {
                            long long l = id * layer1_size;
                            for (c = 0; c < layer1_size; c++) syn0[c + l] -= alpha * lambda / length * (syn0[c + l] - syn0[c + l1]);
                        }
                    }
                    
                    //update central word
                    for (c = 0; c < layer1_size; c++) syn0[c + l1] += alpha * lambda * (neu1e[c] / length - syn0[c + l1]);
                }
                
                if (reg_out) {
                    if(negative > 0 && !weight_tying){
                        // Regularizations
                        //vector<string>* pplist = pp->GetList(vocab[word].word);
                        pp->GetList(vocab[word].word, pplist);
                        //if (NULL == pplist) continue;
                        if(0 == pplist.size()) continue;
                        l2 = word * layer1_size;
                        
                        unsigned long length = pplist.size();
                        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                        for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                            int id = SearchVocab((char*)iter->c_str());
                            if (id != -1) {
                                long long l = id * layer1_size;
                                for (c = 0; c < layer1_size; c++) neu1e[c] += syn1neg[c + l];
                            }
                            else
                            {
                                length--;
                            }
                        }
                        
                        if (length == 0) {
                            continue;
                        }
                        
                        //update pp words
                        for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                            int id = SearchVocab((char*)iter->c_str());
                            if (id != -1) {
                                long long l = id * layer1_size;
                                for (c = 0; c < layer1_size; c++) syn1neg[c + l] -= alpha * lambda / length * (syn1neg[c + l] - syn1neg[c + l2]);
                            }
                        }
                        
                        //update central word
                        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += alpha * lambda * (neu1e[c] / length - syn1neg[c + l2]);
                    }
                    if(weight_tying > 0){
                        // Regularizations
                        //vector<string>* pplist = pp->GetList(vocab[word].word);
                        pp->GetList(vocab[word].word, pplist);
                        //if (NULL == pplist) continue;
                        if(0 == pplist.size()) continue;
                        l2 = word * layer1_size;
                        
                        unsigned long length = pplist.size();
                        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                        for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                            int id = SearchVocab((char*)iter->c_str());
                            if (id != -1) {
                                long long l = id * layer1_size;
                                for (c = 0; c < layer1_size; c++) neu1e[c] += syn0[c + l];
                            }
                            else
                            {
                                length--;
                            }
                        }
                        
                        if (length == 0) {
                            continue;
                        }
                        
                        //update pp words
                        for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                            int id = SearchVocab((char*)iter->c_str());
                            if (id != -1) {
                                long long l = id * layer1_size;
                                for (c = 0; c < layer1_size; c++) syn0[c + l] -= alpha * lambda / length * (syn0[c + l] - syn0[c + l2]);
                            }
                        }
                        
                        //update central word
                        for (c = 0; c < layer1_size; c++) syn0[c + l2] += alpha * lambda * (neu1e[c] / length - syn0[c + l2]);
                    }
                    if (hs) {
                        //vector<string>* pplist = pp->GetList(vocab[word].word);
                        //if (NULL == pplist) continue;
                        pp->GetList(vocab[word].word, pplist);
                        if (0 == pplist.size()) continue;
                        
                        unsigned long length = pplist.size();
                        l2 = vocab[word].point[vocab[word].codelen - 1] * layer1_size;
                        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                        for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                            int id = SearchVocab((char*)iter->c_str());
                            if (id != -1) {
                                long long l = vocab[id].point[vocab[id].codelen - 1] * layer1_size;
                                for (c = 0; c < layer1_size; c++) neu1e[c] += syn1[c + l];
                            }
                            else
                            {
                                length--;
                            }
                        }
                        
                        if (length == 0) {
                            continue;
                        }
                        
                        //update pp words
                        for(vector<string>::iterator iter = pplist.begin(); iter != pplist.end(); iter++){
                            int id = SearchVocab((char*)iter->c_str());
                            if (id != -1) {
                                long long l = vocab[id].point[vocab[id].codelen - 1] * layer1_size;
                                for (c = 0; c < layer1_size; c++) syn1[c + l] -= alpha * lambda / length * (syn1[c + l] - syn1[c + l2]);
                            }
                        }
                        
                        //update central word
                        for (c = 0; c < layer1_size; c++) syn1[c + l2] += alpha * lambda * (neu1e[c] / length - syn1[c + l2]);
                    }
                }
            }
        }
        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void *TrainModelRegNCEThread(void *id) {
    long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    long long word_actual_old = 0;
    //long long part = train_words / 10;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    int pp_count = 0;
    int pp_last_count = 0;
    int train_pp_total = epochs * pp->ppdict.size();
    FILE *fi = fopen(train_file, "rb");
    //FILE *fi = fopen("/Users/gflfof/Desktop/new work/phrase_embedding/trunk/trainword.txt", "rb");
    train_pp_total = 0;
//    while (1) {
//        word = ReadWordIndex(fi);
//        if (word != 0) train_pp_total++;
//        if (feof(fi)) break;
//    }
    //train_pp_total--;
    //train_pp_total *= epochs;
    //train_pp_total *= 1;
    //file_size = ftell(fi);
    
    for (int ep = 0; ep < 1; ep++) {
        fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
        
        while (1) {
            if (word_count - last_word_count > 10000) {
                word_actual_old = word_count_actual;
                word_count_actual += word_count - last_word_count; 
                //if (word_count_actual / part == word_actual_old / part + 1) {
                //    evaluateMRRout(10000, ppeval);
                //}
                last_word_count = word_count;
                if ((debug_mode > 1)) {
                    now=clock();
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                           word_count_actual / (real)(train_words + 1) * 100,
                           word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                    fflush(stdout);
                }
                alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));
                if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
            }
            if (pp_count - pp_last_count >= 5000){
                pp_count_actual += pp_count - pp_last_count;
                pp_last_count = pp_count;
                now=clock();
                printf("%cLambda: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, lambda,
                       pp_count_actual / (real)(train_pp_total + 1) * 100,
                       pp_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                lambda = starting_lambda * (1 - pp_count_actual / (real)(train_pp_total + 1));
                if (lambda < starting_lambda * 0.0001) lambda = starting_lambda * 0.0001;
            }
            if (sentence_length == 0) {
                while (1) {
                    word = ReadWordIndex(fi);
                    if (feof(fi)) break;
                    if (word == -1) continue;
                    word_count++;
                    if (word == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                    }
                    sen[sentence_length] = word;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                sentence_position = 0;
            }
            if (feof(fi)) break;
            if (sentence_length == 0) continue;
            if (word_count > train_words / num_threads) break;
            if (pp_count > train_pp_total / num_threads) break;
            word = sen[sentence_position];
            if (word == -1) continue;
            for (c = 0; c < layer1_size; c++) neu1[c] = 0;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
            //next_random = next_random * (unsigned long long)25214903917 + 11;
            b = next_random % window;
            if (cbow) {  //train the cbow architecture
                if(word2vec){
                    // in -> hidden
                    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                        c = sentence_position - window + a;
                        if (c < 0) continue;
                        if (c >= sentence_length) continue;
                        last_word = sen[c];
                        if (last_word == -1) continue;
                        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
                    }
                    // NEGATIVE SAMPLING
                    if (negative > 0) for (d = 0; d < negative + 1; d++) {
                        if (d == 0) {
                            target = word;
                            label = 1;
                        } else {
                            next_random = next_random * (unsigned long long)25214903917 + 11;
                            target = table[(next_random >> 16) % table_size];
                            if (target == 0) target = next_random % (vocab_size - 1) + 1;
                            if (target == word) continue;
                            label = 0;
                        }
                        l2 = target * layer1_size;
                        f = 0;
                        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
                        if (f > MAX_EXP) g = (label - 1) * alpha;
                        else if (f < -MAX_EXP) g = (label - 0) * alpha;
                        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
                    }
                    // hidden -> in
                    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
                        c = sentence_position - window + a;
                        if (c < 0) continue;
                        if (c >= sentence_length) continue;
                        last_word = sen[c];
                        if (last_word == -1) continue;
                        for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
                    }
                }
                if (reg_out) {
                    if(negative > 0){
                        word2dict::iterator iter_pair = pp->ppdict.find(vocab[word].word);
                        //cout << pp_count << " " << vocab[word].word << " ";
                        if (iter_pair != pp->ppdict.end()){
                            for(word2int::iterator iter = iter_pair->second.begin(); iter != iter_pair->second.end(); iter++){
                                last_word = SearchVocab((char*)iter->first.c_str());
                                if (last_word == -1) continue;
                                
                                l1 = last_word * layer1_size;
                                for (c = 0; c < layer1_size; c++) neu1[c] = 0;
                                // NEGATIVE SAMPLING // todo gradient
                                if (negative > 0 ) for (d = 0; d < negative + 1; d++) {
                                    if (d == 0) {
                                        target = word;
                                        label = 1;
                                    } else {
                                        next_random = next_random * (unsigned long long)25214903917 + 11;
                                        target = table[(next_random >> 16) % table_size];
                                        if (target == 0) target = next_random % (vocab_size - 1) + 1;
                                        if (target == word) continue;
                                        label = 0;
                                    }
                                    //cout << target << " ";
                                    //cout << vocab[target].word << " ";
                                    l2 = target * layer1_size;
                                    f = 0;
                                    for (c = 0; c < layer1_size; c++) f += syn1neg[c + l1] * syn1neg[c + l2];
                                    if (f > MAX_EXP) g = (label - 1) * lambda;
                                    else if (f < -MAX_EXP) g = (label - 0) * lambda;
                                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * lambda;
                                    for (c = 0; c < layer1_size; c++) neu1[c] += g * syn1neg[c + l2];
                                    for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn1neg[c + l1];
                                }
                                for (c = 0; c < layer1_size; c++) syn1neg[c + l1] += neu1[c];
                            }
                        }
                        //cout << endl;
                        pp_count++;
                    }
                }
            }
            sentence_position++;
            if (sentence_position >= sentence_length) {
                sentence_length = 0;
                continue;
            }
        }
        if((ep + 1) % 10 == 0 && id == 0) if (ppeval != NULL) 
        {
            //evaluateMRRout(10000, ppeval);   
        }
    }
    
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void *TrainPPDBVectorThread(void *id) {
    long long b, d, word, last_word;
    long long word_count = 0;
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    word2int::iterator it;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    long long train_words_total = epochs * pp->ppdict.size();
    pp_count_actual = epochs * pp->ppdict.size();
    word_count = 0;
    for (int ep = 0; ep < epochs; ep++) {
        //cout << ep << endl;
        for (word2dict::iterator iter_pair = pp->ppdict.begin(); iter_pair != pp->ppdict.end(); iter_pair++) {
            //fprintf(fo, "%s\n", iter_pair->first.c_str());
            word = SearchVocab((char*)iter_pair->first.c_str());
            if (word == -1) continue;
            //cout << vocab[word].word << " ";
            
            for(word2int::iterator iter = iter_pair->second.begin(); iter != iter_pair->second.end(); iter++){
                last_word = SearchVocab((char*)iter->first.c_str());
                if (last_word == -1) continue;
                
                l1 = last_word * layer1_size;
                for (c = 0; c < layer1_size; c++) neu1[c] = 0;
                // NEGATIVE SAMPLING // todo gradient
                if (negative > 0 ) for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = table[(next_random >> 16) % table_size];
                        if (target == 0) target = next_random % (vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    //cout << target << " ";
                    l2 = target * layer1_size;
                    f = 0;
                    for (c = 0; c < layer1_size; c++) f += syn1neg[c + l1] * syn1neg[c + l2];
                    if (f > MAX_EXP) g = (label - 1) * lambda;
                    else if (f < -MAX_EXP) g = (label - 0) * lambda;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * lambda;
                    for (c = 0; c < layer1_size; c++) neu1[c] += g * syn1neg[c + l2];
                    for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn1neg[c + l1];
                }
                for (c = 0; c < layer1_size; c++) syn1neg[c + l1] += neu1[c];
            }
            //cout << endl;
            word_count++;
            if (word_count % 5000 == 0) { //todo: shrink learning rate
                now=clock();
                printf("%cLambda: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, lambda,
                       word_count / (real)(train_words_total + 1) * 100,
                       word_count / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
                lambda = starting_lambda * (1 - word_count / (real)(train_words_total + 1));
                if (lambda < starting_lambda * 0.0001) lambda = starting_lambda * 0.0001;
            }
        }
        if((ep + 1) % 10 == 0) if (ppeval != NULL) 
        {
            evaluateMRRout(10000, ppeval);   
        }
        //next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
    }
    //fclose(fi);
    //fclose(fo);
    free(neu1);
    pthread_exit(NULL);
}

void *TrainPPDBVectorThreadNew(void *id) {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL); 
    long long b, d, word, last_word;
    long long word_count = 0, last_word_count = 0;
    long long l1, l2, c, target, label;
    unsigned long long next_random = (long long)id;
    word2int::iterator it;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    long long train_words_total = epochs * pp->ppdict.size();
    long long begin = pp->ppdict.size() / (long long)num_thread_pmm * ((long long)id - num_threads);
    long long end = pp->ppdict.size() / (long long)num_thread_pmm * ((long long)id + 1  - num_threads);
    if ((long long)id == num_threads + num_thread_pmm - 1) end = (long long)pp->ppdict.size();
    //pp_count_actual = epochs * pp->ppdict.size();
    
    long long* wordIds = (long long*)calloc(pp->ppdict.size(), sizeof(long long));
    string* wordList = new string[pp->ppdict.size()];
    int i = 0;
    for (word2dict::iterator iter_pair = pp->ppdict.begin(); iter_pair != pp->ppdict.end(); iter_pair++) {
        word = SearchVocab((char*)iter_pair->first.c_str());
        wordList[i] = iter_pair->first;
        wordIds[i++] = word;
    }
    for (int ep = 0; ep < epochs; ep++) {
        //cout << ep << endl;
        for (i = begin; i < end; i++) {
        //for (word2dict::iterator iter_pair = pp->ppdict.begin(); iter_pair != pp->ppdict.end(); iter_pair++) {
            word = wordIds[i];
            //word = SearchVocab((char*)wordList[i].c_str());
            if (word == -1) continue;
            //cout << vocab[word].word << " ";
            
            word2dict::iterator iter_pair = pp->ppdict.find(wordList[i]);
            
            for(word2int::iterator iter = iter_pair->second.begin(); iter != iter_pair->second.end(); iter++){
                last_word = SearchVocab((char*)iter->first.c_str());
                if (last_word == -1) continue;
                pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
                pthread_testcancel();/*the thread can be killed only here*/
                pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
                
                l1 = last_word * layer1_size;
                for (c = 0; c < layer1_size; c++) neu1[c] = 0;
                // NEGATIVE SAMPLING // todo gradient
                if (negative > 0 ) for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        target = table[(next_random >> 16) % table_size];
                        if (target == 0) target = next_random % (vocab_size - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    //cout << target << " ";
                    l2 = target * layer1_size;
                    f = 0;
                    for (c = 0; c < layer1_size; c++) f += syn1neg[c + l1] * syn1neg[c + l2];
                    if (f > MAX_EXP) g = (label - 1) * lambda;
                    else if (f < -MAX_EXP) g = (label - 0) * lambda;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * lambda;
                    for (c = 0; c < layer1_size; c++) neu1[c] += g * syn1neg[c + l2];
                    for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn1neg[c + l1];
                }
                for (c = 0; c < layer1_size; c++) syn1neg[c + l1] += neu1[c];
            }
            //cout << endl;
            word_count++;
            if ((word_count - last_word_count) % 5000 == 0) { //todo: shrink learning rate
                pp_count_actual += word_count - last_word_count;
                now=clock();
                printf("%cLambda: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, lambda,
                       pp_count_actual / (real)(train_words_total + 1) * 100,
                       pp_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
                lambda = starting_lambda * (1 - pp_count_actual / (real)(train_words_total + 1));
                if (lambda < starting_lambda * 0.0001) lambda = starting_lambda * 0.0001;
                word_count = last_word_count;
            }
        }
        if((ep + 1) % 10 == 0 && (((ep + 1) / 10)) % num_thread_pmm == ((long long)id - num_threads) ) if (ppeval != NULL) 
        {
            evaluateMRRout(10000, ppeval);   
        }
        //next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
    }
    //fclose(fi);
    //fclose(fo);
    free(wordIds);
    free(neu1);
    delete [] wordList;
    pthread_exit(NULL);
}

int LoadEmb(string modelname) {
    long long words, size, a, b;
    char ch;
    FILE *f = fopen(modelname.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    if (size != layer1_size) {
        printf("inconsistent dimensions \n");
        exit(-1);
    }
    //syn1neg = (float *)malloc(words * size * sizeof(float));
    if (syn1neg == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
        return -1;
    }
    
    char tmpword[max_w];
    int count; char ch2;
    for (b = 0; b < words; b++) {
        fscanf(f, "%s%c%d%c", tmpword, &ch, &count, &ch2);
        //fscanf(f, "%s%c", tmpword, &ch);
        int id = SearchVocab(tmpword);
        if (id == -1) continue;
        
        for (a = 0; a < size; a++) fread(&syn1neg[a + id * size], sizeof(float), 1, f);
        //for (a = 0; a < size; a++) fread(&syn1neg[a + b * size], sizeof(float), 1, f);
    }
    fclose(f);
    
    return 0;
}

void TrainModel() {
    long a, b;
    FILE *fo;
    pthread_t *pt = (pthread_t *)malloc((num_threads + num_thread_pmm) * sizeof(pthread_t));
    printf("Starting training using file %s\n", train_file);
    starting_alpha = alpha;
    starting_lambda = lambda;
    if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;
    InitNet();
    if (negative > 0) InitUnigramTable();
    start = clock();
    BuildUpperDict();
    evaluateMRRout(10000, ppeval);
    //for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelRegThread, (void *)a);
//    for (int ep = 0; ep < epochs; ep++) {
//        //TrainModelRegNCEThread(0);
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelRegNCEThread, (void *)a);
    a = num_threads;
    //pthread_create(&pt[0], NULL, TrainModelRegNCEThread, (void *)0);
    //pthread_create(&pt[num_threads], NULL, TrainPPDBVectorThread, (void *)a);
    for (a = num_threads; a < num_threads + num_thread_pmm; a++) pthread_create(&pt[a], NULL, TrainPPDBVectorThreadNew, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    for (a = num_threads; a < num_threads + num_thread_pmm; a++) pthread_cancel(pt[a]);
    
//    }
    //for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainPPDBVectorThread, (void *)a);
    //TrainPPDBVectorThread(0);
    //TrainModelRegNCEThread(0);

    fo = fopen(output_file, "wb");
    if (classes == 0) {
        // Save the word vectors
        fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
        for (a = 0; a < vocab_size; a++) {
            //fprintf(fo, "%s %d ", vocab[a].word, vocab[a].cn);
            fprintf(fo, "%s ", vocab[a].word);
            if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
            else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
            if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo);
            else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn1neg[a * layer1_size + b]);
            fprintf(fo, "\n");
        }
    }
    fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    //pp = new Paraphrase("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/ppdb-1.0-s-lexical.test");
    //pp = new Paraphrase("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/ppdb-1.0-s-lexical.wordlist100");
    //evaluate("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/vectors.wordlist100.bin", 10000, pp);
    //evaluateMRR("/Users/gflfof/Desktop/new work/phrase_embedding/trunk/vector.wordlist100.reg", 10000, pp);
    //return 0;
    int i;
    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    reg_in = 0;
    reg_out = 0;
    if (argc == 1) {
        printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
        printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
        printf("\t-hs <int>\n");
        printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\t-classes <int>\n");
        printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-cbow <int>\n");
        printf("\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
        
        //strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/trunk/text8.wordlist1.small");
        //strcpy(output_file, "/Users/gflfof/Desktop/new work/phrase_embedding/trunk/vector.wordlist100.noreg.bin");
        //strcpy(output_file, "/Users/gflfof/Desktop/new work/phrase_embedding/trunk/vector.wordlist100.noreg.bin.oldtype");
        strcpy(train_file, "/Users/gflfof/Desktop/new work/phrase_embedding/trunk/text8.wordlist100");
        strcpy(output_file, "/Users/gflfof/Desktop/new work/phrase_embedding/trunk/vector.wordlist100.regonly.bin");
        cbow = 1;
        layer1_size = 100;
        window = 5;
        negative = 15;
        hs = 0;
        sample = 1e-3;
        num_threads = 2;
        num_thread_pmm = 2;
        binary = 1;
        //lambda = 4;
        lambda = 0.2;
        
        //alpha = 0.005;
        //alpha = 0.025;
        lambda = 0.005;
        //sample = 0;
        
        weight_tying = 0;
        strcpy(pp_file, "/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/ppdb-1.0-s-lexical.wordlist100");
        reg_in = 0;
        reg_out = 0;
        word2vec = 1;
        pretrain = 0;
        strcpy(pretrain_file, "/Users/gflfof/Desktop/new work/phrase_embedding/trunk/vector.wordlist100.noreg.bin.oldtype");
        strcpy(train_file2, "/Users/gflfof/Desktop/new work/phrase_embedding/trunk/trainword.new");
        epochs = 300;
    }

    else{
        if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) epochs = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-train2", argc, argv)) > 0) strcpy(train_file2, argv[i + 1]);
        if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
        if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
        if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-weight-tying", argc, argv)) > 0) weight_tying = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-thread-pmm", argc, argv)) > 0) num_thread_pmm = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-regin", argc, argv)) > 0) reg_in = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-regout", argc, argv)) > 0) reg_out = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-ppfile", argc, argv)) > 0) strcpy(pp_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-lambda", argc, argv)) > 0) lambda = atof(argv[i + 1]);
        if ((i = ArgPos((char *)"-semdim", argc, argv)) > 0) sem_dim = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-word2vec", argc, argv)) > 0) word2vec = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-pretrain", argc, argv)) > 0) pretrain = atoi(argv[i + 1]);
        if ((i = ArgPos((char *)"-pretrain-file", argc, argv)) > 0) strcpy(pretrain_file, argv[i + 1]);
    }
    
    if (sem_dim == 0) {
        sem_dim = layer1_size;
    }
    pp_count_actual = 0;
    
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    
    //pp = new Paraphrase2("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/ppdb-1.0-s-lexical.wordlist100");
    pp = new Paraphrase2(pp_file);
    ppeval = new Paraphrase("/export/a04/moyu/gigaword_data/ppdb/new/ppdb-1.0-s-lexical.dev");
    //ppeval = new Paraphrase(pp_file);
    //pp = new Paraphrase("/Users/gflfof/Desktop/new work/phrase_embedding/trunk/ppdb-1.0-s-lexical.wordlist1");
    //pp = new Paraphrase2("/Users/gflfof/Desktop/new work/phrase_embedding/trunk/ppdb-1.0-s-lexical.wordlist1");
    //lambda = 2;
    
    TrainModel();
    cout << endl;
    cout << alpha << endl;
    cout << word_count_actual << endl;
    cout << lambda << endl;
    cout << pp_count_actual << endl;
    
    evaluateMRRout(10000, ppeval);
    //Paraphrase* ppeval = new Paraphrase("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/ppdb-1.0-s-lexical.wordlist100");
    //Paraphrase* ppeval = new Paraphrase("/Users/gflfof/Desktop/new work/phrase_embedding/PPDB/ppdb-1.0-s-lexical.wordlist100");
    //evaluate("/Users/gflfof/Desktop/new work/phrase_embedding/trunk/vector.wordlist100.noreg", 10000, pp);
    //evaluateMRR("/Users/gflfof/Desktop/new work/phrase_embedding/trunk/vector.wordlist1.noreg", 10000, ppeval);
    //evaluateMRR(output_file, 10000, ppeval);
    //evaluateMRR(output_file, 10000, pp);
    delete pp;
    pp = NULL;
    return 0;
}
