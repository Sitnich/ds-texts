import os

import torch
import torch.nn.functional as F

root_dir = os.path.abspath("..")


def generate_ft(
        model,
        tokenizer,
        prompt,
        entry_count=10,
        entry_length=35,
        top_p=0.8,
        temperature=0.8,
        model_epoch=8,
        root_dir=root_dir):
    # загружаем обученную ранее модель
    model.load_state_dict(torch.load(root_dir + f"\model\distilgpt2_ds_{model_epoch}.pt",
                                     map_location=torch.device('cpu')))


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
                # преподготавливаем logits - скоры для всех словарных токенов
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float("Inf")

                # случайно выбираем токен
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


# генерация текста для списка названий предметов
def text_generation_ft(test_data, model, tokenizer, entry_count=1, root_dir=root_dir):
    generated_descriptions = []
    for i in range(len(test_data)):
        prompt = f'<|startoftext|>' + test_data[i] + f'\n'
        x = generate_ft(model, tokenizer, prompt, entry_count=entry_count, root_dir=root_dir)
        for j in range(0, entry_count):
            x[j] = x[j].replace(prompt, '')
        generated_descriptions.append((x, test_data[i]))
    return generated_descriptions


# генерация предложений с помощью исходного distilgpt2
def generate_no_finetune(prompt_text, model, tokenizer, n_seqs=1, max_length=35):
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length,
        temperature=0.8,
        top_k=0,
        top_p=0.8,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=n_seqs
    )

    # детокенизируем получившиеся последовательности в строку
    generated_list = []
    for seq in output_sequences:
        seq = seq.tolist()
        text = tokenizer.decode(seq)
        total_sequence = (
                prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True, )):]
        )
        generated_list.append(total_sequence)
    return generated_list


def text_generation_no_finetune(test_data, model, tokenizer, entry_count=1):
    generated_descriptions = []
    for i in range(len(test_data)):
        prompt = test_data[i]
        x = generate_no_finetune(prompt, model, tokenizer, n_seqs=entry_count)
        for j in range(0, entry_count):
            x[j] = x[j].replace(prompt, '')
        generated_descriptions.append((x, test_data[i]))
    return generated_descriptions
