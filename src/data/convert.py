import csv
import json
import os


root_dir = os.path.abspath("..")


def to_csv(root_dir=root_dir):
    with open(root_dir + '/data/ds3.json', encoding='utf-8') as file:
        data = json.load(file)
        data_eng = data['languages']['engUS']
        data_eng.pop('containers')
        data_eng.pop('conversations')

        lines, prev_description = [], ''
        for item_type in data_eng.values():
            for item in item_type.values():
                description = item['knowledge'].replace('\n', ' ')
                description = description.partition('Skill:')[0]
                if prev_description == description or description == '':
                    continue
                idd = item['id']
                name = item['name']
                prev_description = description
                lines.append([idd, name, description])
        keys = ['id', 'name', 'description']

    with open(root_dir + '/data/ds3.csv', 'w', encoding='utf8', newline='') as output_file:
        dict_writer = csv.writer(output_file)
        dict_writer.writerow(keys)
        for d in lines:
            dict_writer.writerow(d)
