import os
import re

import torch
import torch.nn.functional as F

root_dir = os.path.abspath("..")


def generate_ft(model, tokenizer, prompt,
                entry_count=10, entry_length=35, top_p=0.8, temperature=0.8):
    model.eval()
    model = model.to('cpu')

    generated_num = 0
    generated_list = []
    with torch.no_grad():

        for idx in range(entry_count):

            description_finished = False

            # достаем поданный на вход промт и генерируем текст необходимой длины
            cur_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                # учитываем температуру
                logits = logits[:, -1, :] / temperature

                # с помощью кумулятивных сумм учитываем top-p
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_mask = cumulative_probs > top_p
                sorted_indices_mask[:, 1:] = sorted_indices_mask[:, :-1].clone()
                sorted_indices_mask[:, 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_mask]
                logits[:, indices_to_remove] = -float("Inf")

                # случайно выбираем токен из подходящих
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                cur_ids = torch.cat((cur_ids, next_token), dim=1)
                if next_token in tokenizer.encode('<|endoftext|>'):
                    description_finished = True

                # если встретили завершающий токен, значит завершаем генерацию
                # и кладем результат в generated_list
                if description_finished:
                    generated_num += 1

                    output_list = list(cur_ids.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            # если до конца текста необходимой длины генерация не завершилась,
            # завершаем принудительно
            if not description_finished:
                output_list = list(cur_ids.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<|endoftext|>"
                generated_list.append(output_text)
    return generated_list


# генерация предложений с помощью исходного distilgpt2
def generate_no_finetune(prompt_text, model, tokenizer, n_seqs=1, max_length=35, min_length = 10):
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length+len(encoded_prompt),
        min_length=min_length+len(encoded_prompt),
        temperature=0.8,
        top_p=0.8,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=n_seqs)

    # детокенизируем получившиеся последовательности в строку
    generated_list = []
    for seq in output_sequences:
        seq = seq.tolist()
        text = tokenizer.decode(seq)
        decoded_prompt = tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)
        total_sequence = (prompt_text + text[len(decoded_prompt):])
        generated_list.append(total_sequence)
    return generated_list


# генерация текста для списка названий предметов
def text_generation(test_data, model, tokenizer, gen_func='mine', entry_count=1):
    generated_descriptions = []
    for i in range(len(test_data)):
        if gen_func == 'torch':
            prompt = test_data[i] + '.'
            x = generate_no_finetune(prompt, model, tokenizer, n_seqs=entry_count)
        else:
            prompt = f'<|startoftext|>' + test_data[i] + f'\n'
            x = generate_ft(model, tokenizer, prompt, entry_count=entry_count)
        for j in range(0, entry_count):
            x[j] = x[j].replace(prompt, '')
            x[j] = x[j].replace('\n', ' ')
            x[j] = re.sub('[<|>]', '', x[j])
            x[j] = re.sub(r'startoftext', '', x[j])
            x[j] = re.sub(r'endoftext', '', x[j])
            result = x[j].replace(x[j].split('.')[-1], '')
            if len(result.split())<10:
                result = x[j].replace("," + x[j].split(',')[-1], '.')
            x[j] = result
        generated_descriptions.append(x)
    return [test_data, generated_descriptions]
