import re

# обрезание сгенерированных предложений до последней точки
# или до последней запятой, если точек нет

def prepare_results(descriptions):
    labels, descs = [], []
    for i in range(len(descriptions)):
        cur_descriptions=[]
        for j in range(len(descriptions[i][0])):
            remains = descriptions[i][0][j]
            remains = re.sub('[<|>]', '', remains)
            remains = re.sub(r'startoftext', '', remains)
            remains = re.sub(r'endoftext', '', remains)
            result = remains.replace(remains.split('.')[-1],'')
            if result == '':
                result = remains.replace(","+remains.split(',')[-1],'.')

            cur_descriptions.append(result)
        descs.append(cur_descriptions)
        labels.append(descriptions[i][1])
    return labels, descs

def prepare_results_no_finetune(descriptions):
    labels, descs = [], []
    for i in range(len(descriptions)):
        cur_descriptions=[]
        for j in range(len(descriptions[i][0])):
            remains = descriptions[i][0][j]
            cur_descriptions.append(remains)
        descs.append(cur_descriptions)
        labels.append(descriptions[i][1])
    return labels, descs