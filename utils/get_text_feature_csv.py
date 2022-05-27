import os
import pandas
import pandas as pd
import spacy
import re


def main():
    df = pandas.DataFrame()
    text_names = []
    text_lengths = []
    text_content_words = []
    text_function_words = []
    text_left_dep = []
    text_right_dep = []
    text_dep_lengths = []
    content_classes = ['ADJ', 'ADV', 'NOUN', 'NUM', 'PROPN', 'VERB']
    nlp = spacy.load('de_core_news_sm')
    nlp.tokenizer = spacy.tokenizer.Tokenizer(
        nlp.vocab, token_match=re.compile(r'\S+').match,
    )
    for file_name in os.listdir('../texts_de'):
        with open('../texts_de/' + file_name, 'r', encoding='utf-8') as f:
            text_names.append(file_name[:-6] + 'L1')
            text = f.readline()
            doc = nlp(text)
            text_lengths.append(len(doc))
            nr_content_words = 0
            nr_function_words = 0
            right_deps = 0
            left_deps = 0
            dep_lengths = 0

            for token in doc:
                if token.pos_ in content_classes:
                    nr_content_words += 1
                else:
                    nr_function_words += 1
                right_deps += token.n_rights
                left_deps += token.n_lefts
                dep_lengths += abs(token.i - token.head.i)

            text_content_words.append(round(nr_content_words / (nr_content_words + nr_function_words), 2))
            text_function_words.append(round(nr_function_words / (nr_content_words + nr_function_words), 2))
            text_left_dep.append(round(left_deps/len(doc), 2))
            text_right_dep.append(round(right_deps/len(doc), 2))
            text_dep_lengths.append((round(dep_lengths/len(doc), 2)))

    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = spacy.tokenizer.Tokenizer(
        nlp.vocab, token_match=re.compile(r'\S+').match,
    )
    for file_name in os.listdir('../texts_en'):
        with open('../texts_en/' + file_name, 'r', encoding='utf-8') as f:
            text_names.append(file_name[:-6] + 'L2')
            text = f.readline()
            doc = nlp(text)
            text_lengths.append(len(doc))
            nr_content_words = 0
            nr_function_words = 0
            right_deps = 0
            left_deps = 0
            dep_lengths = 0

            for token in doc:
                if token.pos_ in content_classes:
                    nr_content_words += 1
                else:
                    nr_function_words += 1
                right_deps += token.n_rights
                left_deps += token.n_lefts
                dep_lengths += abs(token.i - token.head.i)

            text_content_words.append(round(nr_content_words / (nr_content_words + nr_function_words), 2))
            text_function_words.append(round(nr_function_words / (nr_content_words + nr_function_words), 2))
            text_left_dep.append(round(left_deps / len(doc), 2))
            text_right_dep.append(round(right_deps / len(doc), 2))
            text_dep_lengths.append((round(dep_lengths / len(doc), 2)))

    surprisal_df = pd.read_csv('surprisals.csv')
    print(surprisal_df)
    df['text'] = text_names
    df['Text_Length'] = text_lengths
    df['prop_content_words'] = text_content_words
    df['prop_function_words'] = text_function_words
    df['left_dependencies'] = text_left_dep
    df['right_dependencies'] = text_right_dep
    df['dependency_length'] = text_dep_lengths
    df = df.merge(surprisal_df[['text', 'surpGPT']])
    df.set_index('text', inplace=True)

    df.to_csv('text_features.csv')

if __name__ == '__main__':
    main()

