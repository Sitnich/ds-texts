import os
import pickle
import sys

import click
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

root_dir = os.path.abspath("")
sys.path.append(root_dir)

if os.path.basename(os.path.normpath(root_dir)) == 'scripts':
    root_dir = os.path.abspath("..\..")
    sys.path.append(root_dir)

import src.data.convert as convert
import src.data.prepare as pr
import src.models.train as tr
import src.data.beautify_text as beau
import src.models.generate_text as gen
import src.analysis.rouge_an as rg


@click.group()
def cli():
    pass


@cli.command()
def convert():
    """
    конвертирует json-файл в csv, извлекая необходимую
    для обучения и тестов информацию
    """
    convert.to_csv(root_dir=root_dir)


@cli.command()
@click.option("--config", '-c', default='config', type=click.Path(),
              help='имя файла с конфигурацией')
@click.option("--path", '-p', default=root_dir + '/reports/train_info.txt', type=click.Path(),
              help='путь для сохранения вывода обучения')
def train(config, path):
    """
    обучает модель по датасету ds3.csv
    на заданной конфигурации config
    и выводит процесс обучения в /reports/train_info.txt (path)
    """
    X_train, X_test, y_train, y_test = pr.prepare_data(root_dir=root_dir)
    dataloader = pr.prepare_dataloader(X_train, y_train)
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    tr.train(model, dataloader, config_name=config, root_dir=root_dir, out_path=path)


@cli.command()
@click.option("--path_in", '-i', default=root_dir + '/data/input/items.txt', type=click.Path(),
              help='путь для чтения названий предметов')
@click.option("--path_out", '-o', default=root_dir + '/data/output/descriptions.txt', type=click.Path(),
              help='путь для сохранения описаний предметов')
@click.option("--count", '-c', default=1,
              help='количество описаний на один предмет')
def generate(path_in, path_out, count):
    """
    генерирует описания для предметов из файла data/input/items.txt (path_in)
    и записывает их в data/output/descriptions.txt (path_out)
    """

    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

    with open(path_in,'r') as f:
        data = f.read().splitlines()

    gens = gen.text_generation_ft(data, model, tokenizer, entry_count=count, root_dir=root_dir)
    labels, descs = beau.prepare_results(gens)

    if os.path.exists(path_out):
        os.remove(path_out)
    with open(path_out, 'a') as fo:
        for i in range(len(labels)):
            fo.write(f'Item:\n{labels[i]}\nDescription(s):\n')
            for j in range(len(descs[i])):
                fo.write(f'{descs[i][j]}\n')
            fo.write('\n')


@cli.command()
def analysis_results():
    """
    выводит информацию о величинах метрик bleu и rouge
    на тестовой выборке для distilgpt2 с finetune и без
    """
    _, X_test, _, _ = pr.prepare_data(root_dir=root_dir)
    rouge_score = rg.rogue_analysis(X_test, root_dir=root_dir)
    rouge_score_no_finetune = rg.rogue_analysis_no_finetune(X_test, root_dir=root_dir)
    with open("reports/test_generation_score", "rb") as fp:
        scores = pickle.load(fp)
    mean_score = np.mean(scores)

    with open("reports/test_generation_score_no_finetune", "rb") as fp:
        scores_no_finetune = pickle.load(fp)
    mean_score_no_finetune = np.mean(scores_no_finetune)

    print('BLEU scores on test dataset: \ndistilgpt2 with finetune = {} \
     \ndistilgpt2 without finetune = {}\n'.format(mean_score, mean_score_no_finetune))

    print('Rouge scores on test dataset: \ndistilgpt2 with finetune: \n{} \
     \ndistilgpt2 without finetune: \n{}'.format(rouge_score, rouge_score_no_finetune))
    with open(root_dir + "/reports/analysis_results.txt", "a") as fp:
        fp.write('BLEU scores on test dataset: \ndistilgpt2 with finetune = {} \
     \ndistilgpt2 without finetune = {}\n'.format(mean_score, mean_score_no_finetune))
        fp.write('Rouge scores on test dataset: \ndistilgpt2 with finetune: \n{} \
     \ndistilgpt2 without finetune: \n{}'.format(rouge_score, rouge_score_no_finetune))


if __name__ == "__main__":
    cli()
