import os
import pickle

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

root_dir = os.path.abspath("..")


# считаем среднее значение метрики BLEU на тестовом датасете

def bleu_analysis(test_data, model='ft', folder='mine-gen-func', root_dir=root_dir):
    smoothie = SmoothingFunction().method4
    scores = []
    adding = '_finetune'
    if model == 'nft':
        adding = '_no_finetune'

    with open(root_dir + "/reports/" + folder + "/test_generation" + adding, "rb") as fp:
        generated_list = pickle.load(fp)

    for num, item in enumerate(test_data):
        reference = item
        candidate = generated_list[num]
        scores.append(sentence_bleu([reference], candidate, smoothing_function=smoothie))

    with open(root_dir + "/reports/" + folder + "/test_generation_score" + adding, "wb") as fp:
        pickle.dump(scores, fp)
    return scores


def rogue_analysis(test_data, model='ft', folder='mine-gen-func', root_dir=root_dir):
    rouge = Rouge()
    adding = '_finetune'
    if model == 'nft':
        adding = '_no_finetune'
    with open(root_dir + "/reports/" + folder + "/test_generation" + adding, "rb") as fp:
        generated_list = pickle.load(fp)

    rouge_score = rouge.get_scores(generated_list, test_data.tolist(), avg=True, ignore_empty=True)
    return rouge_score
