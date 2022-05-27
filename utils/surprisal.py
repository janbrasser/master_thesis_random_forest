import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import BertTokenizerFast, BertForMaskedLM
import pandas as pd
from collections import defaultdict
import glob
from wordfreq import word_frequency, zipf_frequency
import numpy as np
from nltk.tokenize import sent_tokenize

STRIDE = 200

def score_gpt(sentence):
    with torch.no_grad():
        all_log_probs = torch.tensor([], device=model.device)
        input = tokenizer(sentence, return_tensors="pt")
        labels = torch.tensor(input['input_ids'], device=model.device)
        output = model(**input, labels=labels)
        # logits for each position of the sentence for all tokens in the vocabulary
        shift_logits = output['logits'][..., :-1, :].contiguous()
        # labels from second to last word
        shift_labels = labels[..., 1:].contiguous()
        # dimensions of vocabulary
        # print(shift_labels.view(-1))
        # cross entropy:
        # arg1: Predicted unnormalized scores (often referred to as logits), shape: input length x vocab size
        # arg2:  Ground truth class indices or class probabilities
        # view: the size -1 is inferred from other dimensions
        log_probs = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                      # class indices
                                                      shift_labels.view(-1), reduction='none')
        print(log_probs)


def score_bert(sentence):
    mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
    with torch.no_grad():
        all_log_probs = []
        offset_mapping = []
        start_ind = 0
        while True:
            encodings = tokenizer(sentence[start_ind:], max_length=512, truncation=True, return_offsets_mapping=True)
            tensor_input = torch.tensor([encodings['input_ids']], device=model.device)
            mask_input = tensor_input.clone()
            offset = 1 if start_ind == 0 else STRIDE
            for i, word in enumerate(encodings['input_ids'][:-1]):
                if i < offset:
                    continue
                mask_input[:, i] = mask_id
                output = model(mask_input, labels=tensor_input)
                log_probs = torch.nn.functional.log_softmax(output['logits'][:, i], dim=-1).squeeze(0)
                all_log_probs.append(-log_probs[tensor_input[0, i]].item())
                mask_input[:, i] = word

            offset_mapping.extend([(i + start_ind, j + start_ind) for i, j in encodings['offset_mapping'][offset:-1]])
            if encodings['offset_mapping'][-2][1] + start_ind >= (len(sentence) - 1):
                break
            start_ind += encodings['offset_mapping'][-STRIDE - 1][1]

        return all_log_probs, offset_mapping


def get_word_dict():
    word_dict = dict()
    for f in sorted(glob.glob('../stimuli/daf_words/DAF*')):
        x = []
        textId = int(f[-11:-9])
        temp = pd.read_csv(f, delimiter='\t', header="infer", skiprows=1)
        temp['textId'] = textId
        temp['wordId'] = temp['Zeile'] - (temp['Zeile'][0]-1)
        assert temp['wordId'][0] == 1
        temp.reset_index(drop=True, inplace=True)
        x.append(temp)
    all_data = pd.concat(x, axis=0, join='outer', ignore_index=False, sort=False)

    return all_data


def read_word_file(file):
    temp = pd.read_csv(file, delimiter='\t', header="infer", skiprows=1)
    temp['textId'] = textId
    temp['wordId'] = temp['Zeile'] - (temp['Zeile'][0] - 1)
    assert temp['wordId'][0] == 1
    # temp.drop(['Zeile'], inplace=True)
    temp.rename(columns={"Wort": "word",
                         "Type": "type",
                         "Pos-Tag": "pos",
                         "Lemma": "lemma",
                         "Annotierte_Typefrequenz_absolut": "freq_ab"}, inplace=True)
    temp.drop('Zeile', 1, inplace=True)
    temp.reset_index(drop=True, inplace=True)
    return temp

def wf(sentence):
    w_s = sentence.split()
    wf = [word_frequency(word, "de") for word in w_s]
    zipf = [zipf_frequency(word, "de") for word in w_s]
    return wf, zipf


MODEL = "bert"

if MODEL == "bert":
    model = BertForMaskedLM.from_pretrained('bert-base-cased')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    score = score_bert

elif MODEL == 'gpt':
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    score = score_gpt

# TODO DOUBLE CHECK
STOP_CHARS = ['!', '?', '.', ';', '”', "”", ':']

word_dict = defaultdict(lambda: defaultdict(dict))
for sent_file in sorted(glob.glob('../texts_en/*')):
    print(f"working on text {sent_file}")
    all_sents = []
    textId = sent_file[-14:-4]
    surprisal = []
    with open(sent_file, encoding='utf-8') as f:
        sents = f.readlines()
        sents = sent_tokenize(sents[0], language='English')
        sents = [sent.strip() for sent in sents]
        for s in range(0, len(sents)):
            sent = sents[s]
            probs, offset = score(sent)
            words = sent.split()
            j = 0
            for i in range(0, len(words)):
                if words[i] == sent[offset[j][0]:offset[j][1]]:
                    surprisal += [probs[i]]
                    j += 1
                else:
                    concat_token = sent[offset[j][0]:offset[j][1]]
                    concat_surprisal = probs[j]
                    while concat_token != words[i]:
                        j += 1
                        concat_token += sent[offset[j][0]:offset[j][1]]
                        if sent[offset[j][0]:offset[j][1]] not in STOP_CHARS:
                            concat_surprisal += probs[j]
                        if concat_token == words[i]:
                            surprisal += [concat_surprisal]
                            j += 1
                            break
            word_length = [len(word) for word in words]
            word_freqs, zipf_freqs = wf(sent)
            lex_feats = list(zip(words, word_freqs, zipf_freqs, surprisal, word_length))
            # s+1: shift sentence index by 1
            word_dict[textId][s+1] = lex_feats
            # TODO: Add screen ids

all_df = []
for text in word_dict:
    text_df = []
    for sentence in word_dict[text]:
        temp = pd.DataFrame(data=word_dict[text][sentence],
                            columns=['word', 'wordFreq', 'zipfFreq', 'surpBERT', 'word_length'])
        temp.insert(0, 'sentence', sentence)
        temp.insert(0, 'text', text)
        text_df.append(temp)
    text_df = pd.concat(text_df, sort=False)
    text_df['wordId'] = np.arange(1, len(text_df) + 1)
    all_df.append(text_df)
all_df = pd.concat(all_df)
all_df.reset_index(drop=True, inplace=True)
all_df.to_csv("lexical_features.csv")