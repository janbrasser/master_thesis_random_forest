# by Patrick Haller, Lena Jäger, Lena Bolliger and Jan Brasser

import numpy as np
from argparse import ArgumentParser, FileType
import pandas as pd
from typing import Dict, Collection, List, Tuple
from os.path import exists
import pickle
import pathlib

# TODO: compute character aois based on coordinates + compare with eye-link aois


class FixationDataset:

    def __init__(self, correction, language):
        self.language = language
        if self.language == 'L1':
            self.TextIds = [i for i in range(1, 14)]   # 13 L1 MECO texts (1st one training text; can be discarded later on)
            self.no_texts = 13
        elif self.language == 'L2':
            self.TextIds = [i for i in range(1, 13)]   # 12 L2 MECO texts
            self.no_texts = 12
        self.TextIds = [i for i in range(1, 14)]   # 13 MECO texts
        self.correction = correction
        self.readerIds: Collection[int]   # not true; our meco ids are ge_zh_xx
        if correction == "uncorrected":
            self.data, self.readerIds = self.read_fixations_uncorrected()

    def read_fixations_uncorrected(self):
        fixations_df: pd.DataFrame = pd.read_csv(f"../data/{self.language}/data.txt", header="infer", delimiter='\t')
        readerIds = fixations_df.RECORDING_SESSION_LABEL.unique()    # ge_zh_01, etc.

        # create an index column in the df with the subject id, the trial id and the fixation id
        fixations_df.index = [
            fixations_df.RECORDING_SESSION_LABEL,
            fixations_df.TRIAL_INDEX,
            fixations_df.CURRENT_FIX_INDEX
        ]
        fixations_df = fixations_df.drop(
            ['TRIAL_INDEX', 'CURRENT_FIX_INDEX'],
            axis=1,
        )
        return fixations_df, readerIds

    def iter_dataset(self, word_roi_dict=None):
        for reader in self.readerIds:
            for text in self.TextIds:
                try:
                    # subset of the data: one reader, one text
                    df_subset = self.data.loc[reader, text, :].sort_index()
                    if word_roi_dict:
                        word_rois = word_roi_dict[text][0]   # the aoi coordinates for the words of the current text
                        word_rois_str = word_roi_dict[text][1]   # the words of the current text
                        vocab_size = len(word_rois)
                        word_rois = self.get_word_rois(df_subset, word_rois)
                    else:
                        word_rois = None
                    print(f"processing reader {reader}, text {text}...")
                    yield Fixations(
                        fixations_df=df_subset,
                        word_rois=word_rois,
                        word_rois_str=word_rois_str,
                        vocab_size=vocab_size,
                        reader=reader,
                        text=text)
                except KeyError:
                    continue

    def get_word_rois(self, df_subset, word_rois) -> List[str]:
        xs = df_subset.CURRENT_FIX_X.to_numpy(dtype=float)
        ys = df_subset.CURRENT_FIX_Y.to_numpy(dtype=float)
        fix_rois: List = ["."] * len(xs)
        assert len(xs) == len(ys)
        # for each fixation
        for i in range(0, len(xs)):
            current_fix_x = xs[i]
            current_fix_y = ys[i]

            # loop through roi dict to find associated roi
            for j in range(len(word_rois)):
                if (
                    ((current_fix_x - word_rois[j][0][0]) >= 0)
                    & ((current_fix_x - word_rois[j][0][1]) <= 0)
                    & ((current_fix_y - word_rois[j][1][0]) >= 0)
                    & ((current_fix_y - word_rois[j][1][1]) <= 0)
                ):
                    fix_rois[i] = str(j)
                    # print(f'found roi {j} for fixation {i}')
                    break
            if fix_rois[i] == ".":
                continue
        return fix_rois

    @property
    def get_word_aoi_dict(self) -> Dict[int, Tuple[np.array, List[str]]]:
        word_XY_dict: Dict[int, Tuple[np.array, List[str]]] = dict()  # dict[text ID: tuple[coordinates, words]]
        for text in range(1, self.no_texts+1):  # 13 texts for L1, 12 for L2
            roifile = f'../aoi/{self.language}/IA_{text}.ias'
            ### Jan: added quoting=3 to deal with words containing a quote ###
            adf = pd.read_csv(roifile, delimiter='\t', header=None, quoting=3)
            word_XY_l = []  # will hold the words
            words_l = []  # will hold the cordinates
            for row in range(len(adf)):
                # add the word to the list of words of the text
                words_l.append(adf.iloc[row][6])
                # get the coordinates of the aoi rectangle (around word) and add to coordinates list
                cur_x_left = adf.iloc[row][2]
                cur_y_top = adf.iloc[row][3]
                cur_x_right = adf.iloc[row][4]
                cur_y_bot = adf.iloc[row][5]
                word_XY_l.append([[cur_x_left, cur_x_right], [cur_y_top, cur_y_bot]])
            word_XY = np.array(word_XY_l)
            word_XY_dict[text] = (word_XY, words_l)
        return word_XY_dict


class Fixations:
    def __init__(self,
                fixations_df,
                word_rois,
                word_rois_str,
                vocab_size,
                reader,
                text,
                *char_rois,):

        self.reader: int = reader
        self.text: int = text
        self.vocabsize: int = vocab_size
        self.char_rois: np.chararray = (fixations_df.CURRENT_FIX_INTEREST_AREA_INDEX.to_numpy(dtype=str))
        self.word_rois: np.chararray = word_rois
        self.word_rois_str: List = word_rois_str
        self.fixs_x: Collection[float] = fixations_df.CURRENT_FIX_X.to_numpy(dtype=float)
        self.fixs_y: Collection[float] = fixations_df.CURRENT_FIX_Y.to_numpy(dtype=float)
        self.fix_durations: Collection[float] = fixations_df.CURRENT_FIX_DURATION.to_numpy(dtype=float)
        self.fix_ias: np.chararray = fixations_df.CURRENT_FIX_INTEREST_AREA_INDEX.values  # some fixations have .
        self.reading_measures: dict = {
            i: {
                "word_index": i,
                "word_str": word_rois_str[i],
                "FFD": 0,
                "SFD": 0,
                "FD": 0,
                "FPRT": 0,
                "FRT": 0,
                "TFT": 0,
                "RRT": 0,
                "RPD_inc": 0,
                "RPD_exc": 0,
                "RBRT": 0,
                "Fix": 0,
                "FPF": 0,
                "RR": 0,
                "FPReg": 0,
                "TRC_out": 0,
                "TRC_in": 0,
                "LP": 0,
                "SL_in": 0,
                "SL_out": 0,
            }
            for i in range(0, vocab_size)
        }
        self.most_right_word: int = 0

    def get_next_roi(self, starting_point, rois):
        for j in range(starting_point + 1, len(self)):
            if rois[j] == ".":
                ### Jan: check for last fixation being outside any aoi, if so return last fixation and duration 0
                ### A bit of a hack, but it will throw a warning "next fixated word x is not in vocabulary" and handle
                ### it correctly
                if j == len(self)-1:
                    return j, 0
                continue
            next_fix_word = int(rois[j])
            next_fix_dur = self.fix_durations[j]
            return next_fix_word, next_fix_dur

    def compute_reading_measures(self, roi_level="word"):
        if roi_level == "char":
            rois = self.char_rois
        elif roi_level == "word":
            rois = self.word_rois
        # check if roi extraction matches data viewer aois
        # get index of first and last fixated roi
        first_roi = rois.index(next(roi for roi in rois if roi != "."))

        # first fixation
        cur_fix_word = int(rois[first_roi])
        cur_fix_dur = self.fix_durations[first_roi]
        cur_fix_word_char = self.char_rois[first_roi]
        next_fix_word, next_fix_dur = self.get_next_roi(first_roi, rois=rois)
        # TODO: is it okay to treat first fixation as if last fixation was on same word??
        self.compute_rm_from_triplet(
            prev_fix_word=cur_fix_word,
            cur_fix_word=cur_fix_word,
            cur_fix_dur=cur_fix_dur,
            cur_fix_word_char=cur_fix_word_char,
            next_fix_word=next_fix_word,
            next_fix_dur=next_fix_dur,
        )

        # for each fixation i
        for i in range(first_roi + 1, len(rois)-1):
            if rois[i] == ".":
                continue
            if int(rois[i]) < 0:
                continue  # TODO: Not yet implemented (if roi is in between 2 words)
            # get next fixation
            prev_fix_word = cur_fix_word
            next_fix_word, next_fix_dur = self.get_next_roi(i, rois=rois)
            cur_fix_word = int(rois[i])
            cur_fix_dur = self.fix_durations[i]
            cur_fix_word_char = self.char_rois[i]
            self.compute_rm_from_triplet(
                prev_fix_word=prev_fix_word,
                cur_fix_word=cur_fix_word,
                cur_fix_dur=cur_fix_dur,
                cur_fix_word_char=cur_fix_word_char,
                next_fix_word=next_fix_word,
                next_fix_dur=next_fix_dur,
            )
        # for last fixation, set dur = 0
        prev_fix_word = cur_fix_word
        cur_fix_word = next_fix_word
        cur_fix_dur = next_fix_dur
        next_fix_dur = 0
        next_fix_word = cur_fix_word
        cur_fix_word_char = self.char_rois[-1]
        self.compute_rm_from_triplet(
            prev_fix_word=prev_fix_word,
            cur_fix_word=cur_fix_word,
            cur_fix_dur=cur_fix_dur,
            cur_fix_word_char=cur_fix_word_char,
            next_fix_word=next_fix_word,
            next_fix_dur=next_fix_dur,
        )

        for word in self.reading_measures:
            if (
                self.reading_measures[word]["FFD"]
                == self.reading_measures[word]["FPRT"]
            ):
                self.reading_measures[word]["SFD"] = self.reading_measures[word]["FFD"]
            self.reading_measures[word]["RRT"] = (
                self.reading_measures[word]["TFT"] - self.reading_measures[word]["FPRT"]
            )
            self.reading_measures[word]["FPF"] = int(
                self.reading_measures[word]["FFD"] > 0
            )
            self.reading_measures[word]["RR"] = int(
                self.reading_measures[word]["RRT"] > 0
            )
            self.reading_measures[word]["FPReg"] = int(
                self.reading_measures[word]["RPD_exc"] > 0
            )
            self.reading_measures[word]["Fix"] = int(
                self.reading_measures[word]["TFT"] > 0
            )
            self.reading_measures[word]["RPD_inc"] = (
                self.reading_measures[word]["RPD_exc"]
                + self.reading_measures[word]["RBRT"]
            )

    def compute_rm_from_triplet(
        self,
        prev_fix_word,
        cur_fix_word,
        cur_fix_dur,
        cur_fix_word_char,
        next_fix_word,
        next_fix_dur,
    ):

        try:
            # landing position = character in word
            if self.reading_measures[cur_fix_word]["LP"] == 0:
                self.reading_measures[cur_fix_word]["LP"] = cur_fix_word_char
            # if currently fixated word is more "right" (= most advanced word in text), update variable
            if self.most_right_word < cur_fix_word:
                self.most_right_word = cur_fix_word
                self.reading_measures[cur_fix_word]["RBRT"] += int(cur_fix_dur)
                # if this word had no outgoing regressions yet
                if self.reading_measures[cur_fix_word]["TRC_out"] == 0:
                    self.reading_measures[cur_fix_word]["FPRT"] += int(cur_fix_dur)
                    # if there was a movement to the right
                    if prev_fix_word < cur_fix_word:
                        self.reading_measures[cur_fix_word]["FFD"] += int(cur_fix_dur)
            else:
                self.reading_measures[self.most_right_word]["RPD_exc"] += int(
                    cur_fix_dur
                )
            self.reading_measures[cur_fix_word]["TFT"] += int(cur_fix_dur)
            if self.reading_measures[cur_fix_word]["FD"] == 0:
                self.reading_measures[cur_fix_word]["FD"] += int(cur_fix_dur)
            if cur_fix_word < prev_fix_word:
                self.reading_measures[cur_fix_word]["TRC_in"] += 1
            if cur_fix_word > next_fix_word:
                self.reading_measures[cur_fix_word]["TRC_out"] += 1
            if self.reading_measures[cur_fix_word]["FRT"] == 0 and (
                not next_fix_word == cur_fix_word or next_fix_dur == "0"
            ):
                self.reading_measures[cur_fix_word]["FRT"] = self.reading_measures[
                    cur_fix_word
                ]["TFT"]
            if self.reading_measures[cur_fix_word]["SL_in"] == 0:
                self.reading_measures[cur_fix_word]["SL_in"] = (
                    cur_fix_word - prev_fix_word
                )
            if self.reading_measures[cur_fix_word]["SL_out"] == 0:
                self.reading_measures[cur_fix_word]["SL_out"] = (
                    next_fix_word - cur_fix_word
                )
        except KeyError:
            print(f"next fixated word {next_fix_word} is not in vocabulary")
            return

    def __len__(self) -> int:
        return len(self.fix_durations)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Extract reading measures from Data Viewer fixation report.')
    # parser.add_argument('datafile', type=FileType('r', encoding='utf-8'),
    #                     help='Data Viewer fixation report as tab-separated .txt file.')
    # parser.add_argument('--path', type='str', help='path to save the reading measure files to')
    parser.add_argument('--lang', type=str, choices=['L1', 'L2'], help='The L1 or L2 reading session')
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    Fixations_df = FixationDataset(correction="uncorrected", language=args.lang)

    if exists(f"../data/{args.lang}/xy_dict.pickle"):
        with open(f"../data/{args.lang}/xy_dict.pickle", "rb") as handle:
            word_XY_dict = pickle.load(handle)

    else:
        word_XY_dict = Fixations_df.get_word_aoi_dict
        with open(f"../data/{args.lang}/xy_dict.pickle", "wb") as handle:
            pickle.dump(word_XY_dict, handle)

    path = pathlib.Path(f"../data/{args.lang}/reading_measures")
    path.mkdir(parents=True, exist_ok=True)

    for fixations in Fixations_df.iter_dataset(word_roi_dict=word_XY_dict):
        fixations.compute_reading_measures()
        df_out = pd.DataFrame(fixations.reading_measures).T
        outputFile = f"../data/{args.lang}/reading_measures/{fixations.reader}_text_{fixations.text}_rm.txt"
        df_out.to_csv(outputFile)


if __name__ == '__main__':
    main()
