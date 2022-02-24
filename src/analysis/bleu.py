import os
import pickle

from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

import src.data.beautify_text as beau
import src.models.generate_text as gen

root_dir = os.path.abspath("..")


# считаем среднее значение метрики BLEU на тестовом датасете
def bleu_analysis(X, y, root_dir=root_dir):
    scores, generated_list = [], []
    y_list = y.tolist()
    for num, item in tqdm(enumerate(X), total=len(X)):
        reference = item
        candidate = beau.prepare_results(
            gen.text_generation_ft([y_list[num]]))[1][0][0]
        generated_list.append(candidate)
        scores.append(sentence_bleu(reference, candidate))

    with open(root_dir + "/reports/test_generation_score", "wb") as fp:
        pickle.dump(scores, fp)
    with open(root_dir + "/reports/test_generation", "wb") as fp:
        pickle.dump(generated_list, fp)

    return scores, generated_list


def bleu_analysis_no_finetune(X, y, root_dir=root_dir):
    scores, generated_list = [], []
    y_list = y.tolist()
    for num, item in tqdm(enumerate(X), total=len(X)):
        reference = item
        candidate = beau.prepare_results_no_finetune(
            gen.text_generation_no_finetune([y_list[num]]))[1][0][0]
        generated_list.append(candidate)
        scores.append(sentence_bleu(reference, candidate))

    with open(root_dir + "/reports/test_generation_score_no_finetune", "wb") as fp:
        pickle.dump(scores, fp)
    with open(root_dir + "/reports/test_generation_no_finetune", "wb") as fp:
        pickle.dump(generated_list, fp)

    return scores, generated_list
