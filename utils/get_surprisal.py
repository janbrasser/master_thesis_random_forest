# written by Patrick Haller, adjusted by Jan Brasser

import torch
# from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import BertTokenizerFast, BertForMaskedLM
# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
from collections import defaultdict
import glob
from wordfreq import word_frequency, zipf_frequency
import numpy as np
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesPunctNormalizer
from nltk.tokenize import sent_tokenize

STRIDE = 200
LANGUAGE = 'de'

MOSESTOKENIZER = MosesTokenizer(LANGUAGE)
MOSESDETOKENIZER = MosesDetokenizer(LANGUAGE)
MOSESNORMALIZER = MosesPunctNormalizer(LANGUAGE)


def score_ngram(sentence, model, oov_nan=False):
    # put into wikitext-103 format
    # return strange characters back to original form
    tokens = [MOSESDETOKENIZER([t]) for t in MOSESTOKENIZER(sentence)]
    tokenized_sentence = " ".join(tokens)
    spans = []
    word_start = 0
    for t in tokens:
        while sentence[word_start] != t[0]:
            word_start += 1
        spans.append((word_start, word_start+len(t)))
        word_start += len(t)
    scores = model.full_scores(tokenized_sentence, eos=False, bos=True)
    base_change = np.log10(np.exp(1))
    if oov_nan:
        return np.array([-s[0]/base_change if not s[2] else np.nan for s in scores]), spans
    return np.array([-s[0]/base_change for s in scores]), spans


def score_gpt(sentence, model, tokenizer, BOS=True):
    with torch.no_grad():
        all_log_probs = torch.tensor([], device=model.device)
        offset_mapping = []
        start_ind = 0

        while True:
            encodings = tokenizer(sentence[start_ind:], max_length=1022, truncation=True, return_offsets_mapping=True)
            if BOS:
                tensor_input = torch.tensor(
                    [[tokenizer.bos_token_id] + encodings['input_ids'] + [tokenizer.eos_token_id]], device=model.device)
            else:
                tensor_input = torch.tensor([encodings['input_ids'] + [tokenizer.eos_token_id]], device=model.device)
            output = model(tensor_input, labels=tensor_input)
            shift_logits = output['logits'][..., :-1, :].contiguous()
            shift_labels = tensor_input[..., 1:].contiguous()
            log_probs = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                          shift_labels.view(-1), reduction='none')
            assert torch.isclose(torch.exp(sum(log_probs) / len(log_probs)), torch.exp(output['loss']))
            offset = 0 if start_ind == 0 else STRIDE - 1
            all_log_probs = torch.cat([all_log_probs, log_probs[offset:-1]])
            offset_mapping.extend([(i + start_ind, j + start_ind) for i, j in encodings['offset_mapping'][offset:]])
            if encodings['offset_mapping'][-1][1] + start_ind == len(sentence):
                break
            start_ind += encodings['offset_mapping'][-STRIDE][1]
        return np.asarray(all_log_probs.cpu()), offset_mapping


def score_gpt_test(sentence):
    with torch.no_grad():
        #all_log_probs = torch.tensor([], device=model.device)
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
    # compute word frequencies without punctuation marks
    w_s = [word.strip(' !?.;””:()') for word in w_s]
    wf = [word_frequency(word, "de") for word in w_s]
    zipf = [zipf_frequency(word, "de") for word in w_s]
    return wf, zipf


MODEL = "gpt"

if MODEL == "bert":
    model = BertForMaskedLM.from_pretrained('bert-base-german-cased')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')
    score = score_bert

elif MODEL == 'gpt':
    if LANGUAGE == 'en':
        model_name = "benjamin/gerpt2-large"
    elif LANGUAGE == 'de':
        model_name = "malteos/gpt2-xl-wechsel-german"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    score = score_gpt

# TODO DOUBLE CHECK
STOP_CHARS = ['!', '?', '.', ';', '”', "”", ':']

word_dict = defaultdict(lambda: defaultdict(dict))
for sent_file in sorted(glob.glob(f'../texts_{LANGUAGE}/*')):
    print(f"working on text {sent_file}")
    all_sents = []
    if LANGUAGE == 'en':
        textId = sent_file[-14:-6].lstrip('\\') + 'L2'
    elif LANGUAGE == 'de':
        textId = sent_file[-14:-6].lstrip('\\') + 'L1'
    with open(sent_file, encoding='utf-8') as f:
        sents = sent_tokenize(f.readlines()[0])
        sents = [sent.strip() for sent in sents]
        for s in range(0, len(sents)):
            surprisal = []
            sent = sents[s]
            probs, offset = score(sent, model, tokenizer)
            words = sent.split()
            # TODO double check: sometimes tokenizer splits final punctuation marks from last token -> merge them
            if len(words)+1 == len(probs):
                probs = np.append(probs[:-2], sum(probs[-2:]))
                offset = offset[:-2] + [tuple((offset[-2][0], offset[-1][1]))]
            j = 0
            # loop through all words
            for i in range(0, len(words)):
                try:
                    # case 1: tokenized word = reference word in text
                    # print(f'{words[i]} ~ {sent[offset[j][0]:offset[j][1]]}')
                    if words[i] == sent[offset[j][0]:offset[j][1]].strip():
                        surprisal += [probs[i]]
                        j += 1
                    # case 2: tokenizer split subword tokens -> merge them and add up surprisal values until the same
                    else:
                        concat_token = sent[offset[j][0]:offset[j][1]].strip()
                        concat_surprisal = probs[j]
                        while concat_token != words[i]:
                            j += 1
                            concat_token += sent[offset[j][0]:offset[j][1]].strip()
                            if sent[offset[j][0]:offset[j][1]].strip() not in STOP_CHARS:
                                concat_surprisal += probs[j]
                            if concat_token == words[i]:
                                surprisal += [concat_surprisal]
                                j += 1
                                break
                    word_length = [len(word) for word in words]
                    word_freqs, zipf_freqs = wf(sent)
                    lex_feats = list(zip(words, word_freqs, zipf_freqs, surprisal, word_length))
                    # s+1: shift sentence index by 1
                    word_dict[textId][s + 1] = lex_feats
                except IndexError:
                    print(f'Index error in sentence: {sent}, length: {len(sent)}')
                    break


all_df = []
for text in word_dict:
    for screen in word_dict[text]:
        text_df = []
        for sentence in word_dict[text]:
            temp = pd.DataFrame(data=word_dict[text][sentence],
                                columns=['word', 'wordFreq', 'zipfFreq', 'surpGPT', 'word_length'])
            temp.insert(0, 'sentence', sentence)
            temp.insert(0, 'text', text)
            text_df.append(temp)
        text_df = pd.concat(text_df, sort=False)
        text_df['wordId'] = np.arange(1, len(text_df) + 1)
        all_df.append(text_df)
all_df = pd.concat(all_df)
all_df.reset_index(drop=True, inplace=True)
all_df.to_csv(f"lexical_features_fixed_{LANGUAGE}.csv")
