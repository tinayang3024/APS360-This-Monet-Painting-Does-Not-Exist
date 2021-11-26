

import argparse
import csv

def get_answer(cell):
    if "None of the above" in cell:
        return []
    else:
        answer = cell.split("Option ")[1:]
        return answer

def eval_answer(keys, answers):
    TN = 0
    FP = 0
    for answer in answers:
        if answer in keys:
            # real got categorized as real
            TN += 1
        else:
            # fake got categorized as real
            FP += 1
    return TN, FP

def load_csv(fp):
    TN = 0
    FP = 0
    with open(fp, 'r') as f:
        mycsv = csv.reader(f)
        first_line = True
        sec_line = False
        answer_key = []
        for line in mycsv:
            if sec_line:
                sec_line = False
                for i in range(1, len(line)-1):
                    answer_key.append(line[i].strip("\"").split("-"))
                answer_key.append(line[-1].strip("\"")[0:-2].split("-"))
                continue
            if first_line:
                first_line = False
                sec_line = True
                continue
            for i in range(1, len(line)):
                answer = get_answer(line[i])
                TN_, FP_ = eval_answer(answer_key[i-1], answer)
                TN += TN_
                FP += FP_
    print ("TN " + str(TN/(FP+TN)))
    print ("FP " + str(FP/(FP+TN)))

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)

args = parser.parse_args()
load_csv(args.file)
