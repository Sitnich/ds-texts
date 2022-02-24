import os
import pickle

from rouge import Rouge

root_dir = os.path.abspath("..")


def rogue_analysis(test_data, root_dir=root_dir):
    rouge = Rouge()

    with open(root_dir + "/reports/test_generation", "rb") as fp:
        generated_list = pickle.load(fp)

    rouge_score = rouge.get_scores(generated_list, test_data.tolist(), avg=True)
    return rouge_score


def rogue_analysis_no_finetune(test_data, root_dir=root_dir):
    rouge = Rouge()

    with open(root_dir + "/reports/test_generation_no_finetune", "rb") as fp:
        generated_list = pickle.load(fp)

    rouge_score = rouge.get_scores(generated_list, test_data.tolist(), avg=True)
    return rouge_score
