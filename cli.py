import os
import pickle
import sys

import click
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

root_dir = os.path.abspath("")
sys.path.append(root_dir)

if os.path.basename(os.path.normpath(root_dir)) == 'scripts':
    root_dir = os.path.abspath("..\..")
    sys.path.append(root_dir)

import src.data.convert as convert
import src.data.prepare as pr
import src.models.train as tr
import src.models.generate_text as gen
import src.analysis.metrics as met


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
@click.option("--generator", '-g', default='mine',
              help="тип функции генерации: 'torch' или 'mine'")
@click.option("--path_in", '-i', default=root_dir + '/data/input/items.txt', type=click.Path(),
              help='путь для чтения названий предметов')
@click.option("--path_out", '-o', default=root_dir + '/data/output/descriptions.txt', type=click.Path(),
              help='путь для сохранения описаний предметов')
@click.option("--count", '-c', default=1,
              help='количество описаний на один предмет')
def generate(generator, path_in, path_out, count):
    """
    генерирует описания для предметов из файла data/input/items.txt (path_in)
    и записывает их в data/output/descriptions.txt (path_out)
    """

    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    # загружаем обученную ранее модель
    model.load_state_dict(torch.load(root_dir + f"\model\distilgpt2_ds_8.pt",
                                     map_location=torch.device('cpu')))
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

    with open(path_in, 'r') as f:
        data = f.read().splitlines()

    labels, descs = gen.text_generation(data, model, tokenizer, gen_func=generator, entry_count=count)

    if os.path.exists(path_out):
        os.remove(path_out)
    with open(path_out, 'a') as fo:
        for i in range(len(labels)):
            fo.write(f'Item:\n{labels[i]}\nDescription(s):\n')
            for j in range(len(descs[i])):
                fo.write(f'{descs[i][j]}\n')
            fo.write('\n')


@cli.command()
@click.option("--generator", '-g', default='mine',
              help="тип функции генерации: 'torch' или 'mine'")
def generate_test(generator):
    """
    генерирует описания для тестовой выборки
    на distilgpt2 с finetune и без
    """
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

    _, _, _, y_test = pr.prepare_data(root_dir=root_dir)
    y_test_list = y_test.tolist()

    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.load_state_dict(torch.load(root_dir + f"\model\distilgpt2_ds_8.pt",
                                     map_location=torch.device('cpu')))
    if generator == 'mine':
        folder = 'mine-gen-func'
    else:
        folder = 'torch-gen-func'
    ft_gen = gen.text_generation(y_test_list, model, tokenizer, gen_func=generator, entry_count=1)

    model_nft = GPT2LMHeadModel.from_pretrained('distilgpt2')
    nft_gen = gen.text_generation(y_test_list, model_nft, tokenizer, gen_func='torch', entry_count=1)

    ft_gen[1] = [desc[0] for desc in ft_gen[1]]
    nft_gen[1] = [desc[0] for desc in nft_gen[1]]

    with open(root_dir + "/reports/" + folder + "/test_generation_finetune.txt", "w+", encoding="utf-8") as fp:
        fp.write("Generated descriptions on distilgpt2+finetune:\n")
        for num in range(len(ft_gen[0])):
            fp.write(f"{ft_gen[0][num]}: {ft_gen[1][num]}\n")

    with open(root_dir + "/reports/" + folder + "/test_generation_no_finetune.txt", "w+", encoding="utf-8") as fp:
        fp.write("Generated descriptions on distilgpt2 no finetune:\n")
        for num in range(len(nft_gen[0])):
            fp.write(f"{nft_gen[0][num]}: {nft_gen[1][num]}\n")

    with open(root_dir + "/reports/" + folder + "/test_generation_finetune", "wb") as fp:
        pickle.dump(ft_gen[1], fp)
    with open(root_dir + "/reports/" + folder + "/test_generation_no_finetune", "wb") as fp:
        pickle.dump(nft_gen[1], fp)


@cli.command()
@click.option("--generator", '-g', default='mine',
              help="тип функции генерации: 'torch' или 'mine'")
def analysis_results(generator):
    """
    выводит информацию о величинах метрик bleu и rouge
    на тестовой выборке для distilgpt2 с finetune и без
    """
    if generator == 'mine':
        folder = 'mine-gen-func'
    else:
        folder = 'torch-gen-func'

    _, X_test, _, _ = pr.prepare_data(root_dir=root_dir)
    rouge_score = met.rogue_analysis(X_test, model='ft', folder=folder, root_dir=root_dir)
    rouge_score_no_finetune = met.rogue_analysis(X_test, model='nft', folder=folder, root_dir=root_dir)

    mean_score = np.mean(met.bleu_analysis(X_test, model='ft', folder=folder, root_dir=root_dir))
    mean_score_no_finetune = np.mean(met.bleu_analysis(X_test, model='nft', folder=folder, root_dir=root_dir))

    print('BLEU scores on test dataset: \ndistilgpt2 with finetune = {} \
     \ndistilgpt2 without finetune = {}\n'.format(mean_score, mean_score_no_finetune))

    print('Rouge scores on test dataset: \ndistilgpt2 with finetune: \n{} \
     \ndistilgpt2 without finetune: \n{}'.format(rouge_score, rouge_score_no_finetune))
    with open(root_dir + "/reports/analysis_results.txt", "w") as fp:
        fp.write('BLEU scores on test dataset: \ndistilgpt2 with finetune = {} \
     \ndistilgpt2 without finetune = {}\n'.format(mean_score, mean_score_no_finetune))
        fp.write('Rouge scores on test dataset: \ndistilgpt2 with finetune: \n{} \
     \ndistilgpt2 without finetune: \n{}'.format(rouge_score, rouge_score_no_finetune))


if __name__ == "__main__":
    cli()
