import datetime
import os
import random
def _read_data_file(file_path, train=True):
    sentences = []
    sentence = [[], [], []]
    for line in open(file_path, encoding="utf-8"):
        line = line.strip()
        if line == "":
            sentences.append(sentence)
            sentence = [[], [], []]
        else:
            idx, ejeol, ner_tag = line.split("\t")
            # idx는 0부터 시작하도록
            sentence[0].append(int(idx))
            sentence[1].append(ejeol)
            if train:
                sentence[2].append(ner_tag)
            else:
                sentence[2].append("-")

    return sentences

if __name__ == '__main__':
    DATASET_PATH = os.path.abspath("./data")
    file_path = os.path.join(DATASET_PATH, 'train', 'TrainText.txt')
    Val = _read_data_file(file_path)
    # random.shuffle(Val)
    print(Val)
