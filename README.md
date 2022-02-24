# Generating DS-like texts

1) Activate virtualenv ```.\venv\Scripts\activate```
2) Install requirements ```pip install -r requirements.txt```
3) Run CLI ```python cli.py```
```
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  analysis-results  выводит информацию о величинах метрик bleu и rouge на...
  convert           конвертирует json-файл в csv, извлекая необходимую...
  generate          генерирует описания для предметов из файла...
  train             обучает модель по датасету ds3.csv на заданной...
```

Before generating the descriptions you should:
- download finetuned distilgpt2 model from (https://drive.google.com/file/d/1iwRRqGQ-vJoHluMN-_3Pr-bIMsxZBBGL/view?usp=sharing) to directory **\model** 
- or train it with ```python cli.py train```