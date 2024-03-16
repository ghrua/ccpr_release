import sentencepiece as spm
import fire
import json
import os
import re
import os.path as path
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import random
import time
from multiprocessing import Process

from mytools.tool_utils import FileUtils, FileType, logging, StringUtils
from mytools.openai_tools import call_chat, OpenAIModel

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def extract_text(text, x):
    pattern = f"<img{x}>.*?</img{x}>"
    match = re.search(pattern, text)
    if match:
        original_text = match.group(0)[len(f"<img{x}>"):-len(f"</img{x}>")].strip()
    else:
        original_text = None
    return original_text


def split_turn(dialogue):
    start_index = dialogue.index("Human:")
    sents = dialogue[start_index:].strip().split("\n")
    turns = []
    for sent in sents:
        if sent.startswith("Human:"):
            turns.append(sent)
        elif sent.startswith("Assistant:"):
            turns.append(sent)
        else:
            if sent:
                turns[-1] += "\n" + sent
    return turns


def is_valid_turn(turns):
    for i, t in enumerate(turns):
        if t.startswith("Human:") and (i % 2) == 0:
            continue
        if t.startswith("Assistant:") and (i % 2) == 1:
            continue
        return False
    return True


def has_repeated_images(ex):
    img_tags = ["<img{}>".format(i) for i in range(len(ex['image']))]
    response = ex['response']
    for it in img_tags:
        n = len(StringUtils.find_all_indices(response, it))
        if n > 1:
            return True
    return False


def has_unseen_image(ex):
    n_image = len(ex['image'])
    pattern = r"<img(\d+)>"
    img_ids = [int(it) for it in re.findall(pattern, ex['response'])]
    for i in img_ids:
        if i >= n_image:
            return True
    return False


def remove_non_paired_img_tag(text):
    img_tags = re.findall(r'<img\d+>', text)
    # Iterate through the tags and remove them if there is no paired closing tag
    for tag in img_tags:
        closing_tag = tag.replace('<', '</')
        if closing_tag not in text:
            text = re.sub(r'\s*' + re.escape(tag), '', text)
    text = text.replace(",,", "")
    return text


def clean_tag(s):
    return re.sub('</img\d*>', '', re.sub('<img\d*>', '', s)).strip()


def clean_mim_data(data_dir, start_id=0, end_id=-1, batch_size=100, mim_edit_dist=0.1, is_single_file=False):
    data = []
    cleaned = []
    n, t_e = start_id, end_id
    if end_id < 0:
        end_id = float("inf")
    if is_single_file:
        data = FileUtils.load_file(data_dir + "/{}.{}.json".format(start_id, end_id))
    else:
        while True:
            s, e = n, n + batch_size
            file_path = data_dir + "/{}.{}.json".format(s, e)
            if FileUtils.exists(file_path) and e <= end_id:
                t_e = e
                data += FileUtils.load_file(file_path)
                n += batch_size
            else:
                break
    logging.info("{} exemples loaded".format(len(data)))
    for ex in data:
        if has_repeated_images(ex) or has_unseen_image(ex):
            continue
        caption_list = ex['image']
        response = ex['response']
        try:
            turns = split_turn(response)
        except ValueError:
            continue
        if not is_valid_turn(turns):
            continue
        turns = [remove_non_paired_img_tag(t) for t in turns]
        new_turns = []
        is_valid = True
        for sent in turns:
            caption_matched, has_caption = True, False
            for i, _ in enumerate(caption_list):
                gen_cap = extract_text(sent, i)
                if gen_cap is None:
                    continue
                has_caption = True
                clean_cap = clean_tag(caption_list[i])
                ed_score = edit_distance(gen_cap, clean_cap) / len(clean_cap)
                if ed_score > mim_edit_dist:
                    caption_matched = False
            if has_caption and not caption_matched:
                is_valid = False
                break
            new_turns.append(sent)
        if is_valid:
            ex['response'] = "\n\n".join(new_turns)
            cleaned.append(ex)
    logging.info("{} exemples left after cleaning".format(len(cleaned)))
    FileUtils.save_file(cleaned, data_dir + "/{}.{}.clean.json".format(start_id, t_e), 'json')


def clean_fewshot_pool(fewshot_pool_path, annotation_file_path, save_path, has_header=True):
    ann = FileUtils.load_file(annotation_file_path)
    fewshot_pool = FileUtils.load_file(fewshot_pool_path)
    if has_header:
        ann = ann[1:]

    def rate(x):
        if x == "very good":
            return 2
        elif x == "good":
            return 1
        else:
            return 0
    
    def category(x):
        return [c.strip() for c in x.split(",")]
    
    new_fewshot_pool = []
    for row in ann:
        if row[0] and row[2] and row[5]:
            ex_id = int(row[0])
            ex = fewshot_pool[ex_id]
            ex['rate'] = rate(row[2])
            ex['category'] = category(row[5])
            turns = split_turn(ex['response'])
            if row[3] and row[4]:
                t_s, t_e = int(row[3]), int(row[4])
            else:
                t_s, t_e = 0, len(turns)
            turns = turns[t_s:t_e+1]
            # turns = [t for t in turns]
            turns = [remove_non_paired_img_tag(t) for t in turns]
            ex['response'] = "\n\n".join(turns)
            new_fewshot_pool.append(ex)
    logging.info("Found {} non-bad cases".format(len(new_fewshot_pool)))
    FileUtils.save_file(new_fewshot_pool, save_path, 'json')

class PromptSet:
    @staticmethod
    def base_prompt():
        return "Please construct a dialogue between a human and a helpful, honest and harmless assistant.\n\nCharacteristics about the assistant:\n1. The assistant is trained to understand text, images, and their combinations.\n2. The assistant can reply the human with images and/or text.\n3. The assistant has exceptional world knowledge and commonsense reasoning capabilities.\n4. The assistant does not have access to the Internet or any other external tools.\n\nCharacteristics about the human:\n1. The human may send images and/or text to the assistant.\n2. The human may ask questions requiring visual reasoning and/or understanding the relations between multiple images.\n3. The human may ask the assistant to create new images based on his/her intention.\n4. The human may ask the assitant to do interesting things, rather than simply describing the content of the image.\n\nThe dialogue contains interleaved text and images. Each image is represented by <imgX> DESCRIPTION </imgX>, where DESCRIPTION is a textual description of the image and X is an index of the image. Please do not assume any further visual information beyond the descriptions. The constructed dialogues must and can only contain the following images:\n\n{}\n\nPlease directly give the dialogue if you understand. The number of turns of the dialogue should be less than 6. The dialogue should be self-contained. Do NOT assume any previous dialogue between the human and the assistant. Please use the same format <imgX> DESCRIPTION </imgX> to denote images in the dialogue.\n"
    
    @staticmethod
    def in_context_prompt():
        return "Please construct a dialogue between a human and a helpful, honest and harmless assistant. The dialogue contains interleaved text and images. Each image is represented by <imgX> DESCRIPTION </imgX>, where DESCRIPTION is a textual description of the image and X is an index of the image. Please do not assume any further visual information beyond the descriptions. \n\nThe constructed dialogues must and can only contain the following input images:\n\n    {}    \n\nCharacteristics about the assistant:\n1. The assistant is trained to understand text, images, and their combinations.\n2. The assistant can reply the human with images and/or text.\n3. The assistant has exceptional world knowledge and commonsense reasoning capabilities.\n4. The assistant does not have access to the Internet or any other external tools.\n5. If the assistant is asked to create an image, it can only show the image in the provided image list.\n\nCharacteristics about the human:\n1. The human may send images and/or text to the assistant.\n2. The human may ask questions requiring visual reasoning and/or understanding the relations between multiple images.\n3. The human may ask the assistant to show images based on his/her intention.\n4. The human may ask the assitant to do interesting things, rather than simply describing the content of the image.\n\n\nProperties of a bad dialogue:\n1. Simply describing or analyzing the content in the image\n2. Dialogue without a good logic.\n\nProperties of a good dialogue:\n1. Introducing extrinsic and attractive information of stuff shown in the image. \n2. Comparing multiple images in a reasonable context\n\nPlease directly give the dialogue if you understand. The number of turns of the dialogue should be less than 6. The dialogue should be self-contained. Do NOT assume any previous dialogue between the human and the assistant. Please use the same format <imgX> DESCRIPTION </imgX> to denote images in the dialogue. Note that the dialogue can stop at any time without an explicit ending.\n\nExamples:\n========\nCase 1:\nInput Images:\n\n    <img0> a group of motorbikes in the desert with a man beside them </img0>\n    <img1> heading out for the open road, on assignment </img1>\n    <img2> person and motorcycle - picture taken by philosopher on the original trip </img2>\n\nOutput Dialogue:\n\nHuman: Hello! I have a question for you. Can you tell me what these images represent and their relation to each other? <img0> a group of motorbikes in the desert with a man beside them </img0> and <img1> heading out for the open road, on assignment </img1>.\n\nAssistant: Of course! The first image shows a group of motorbikes in the desert with a man beside them. The second image symbolizes the act of departing for a journey on assignment. These two images are related in the sense that they both depict the idea of embarking on an adventure or exploration.\n\nHuman: Thank you! That's helpful. Is there any additional context or information you can provide about these images? \n\nAssistant: Yes, I can provide you with a related image that can complement the context. Here is an image that further expands on the first one: <img2> person and motorcycle - picture taken by philosopher on the original trip </img2>. This image shows a person with a motorcycle, which was taken by a philosopher during the original trip. It provides a closer look at the motorcycle and the person's association with it.\n---------\nCase 2:\nInput Images:\n\n    <img0> singer and person performs live during a concert. </img0>\n    <img1> photo of folk rock artist in concert </img1>\n\nOutput Dialogue:\n\n    Human: Hi, I have a question. Can you tell me about the different types of rock music?\n\n    Assistant: Of course! There are various types of rock music. Would you like a brief description or more details about a specific type?\n\n    Human: I'm interested in learning about folk rock and hard rock. Can you provide some information?\n\n    Assistant: Certainly! Folk rock is a subgenre of rock music that combines elements of folk and rock. It often features acoustic instruments, storytelling lyrics, and harmonies. Here's an image of a folk rock artist performing in concert.<img1> photo of folk rock artist in concert </img1>\n\n    Hard rock, on the other hand, is a more aggressive and louder subgenre of rock music. It typically features distorted electric guitars, powerful vocals, and energetic drumming. Here's an image of a hard rock band performing.<img2> hard rock artists of the band performs </img2>\n\n    Human: That's great! Can you show me an image of a singer performing live during a concert?\n\n    Assistant: Absolutely! Here's an image of a singer performing live during a concert.<img0> singer and person performs live during a concert. </img0>\n========"
    
    @staticmethod
    def in_context_prompt_v2():
        return "Please construct a dialogue between a human and a helpful, honest and harmless assistant. The dialogue contains interleaved text and images. Each image is represented by <imgX> DESCRIPTION </imgX>, where DESCRIPTION is a textual description of the image and X is an index of the image. Please do not assume any further visual information beyond the descriptions. \n\nThe constructed dialogues must and can only contain the following input images:\n\n{}\n\nCharacteristics about the assistant:\n1. The assistant is trained to understand text, images, and their combinations.\n2. The assistant can reply the human with images and/or text.\n3. The assistant has exceptional world knowledge and commonsense reasoning capabilities.\n4. The assistant does not have access to the Internet or any other external tools.\n5. If the assistant is asked to create an image, it can only show the image in the provided image list.\n6. Please do not copy the images appearing in the dialogue. The assistant should refer to the previously mentioned image by natural language.\n\nCharacteristics about the human:\n1. The human may send images and/or text to the assistant.\n2. The human may ask questions requiring visual reasoning and/or understanding the relations between multiple images.\n3. The human may ask the assistant to show images based on his/her intention.\n4. The human may ask the assitant to do interesting things, rather than simply describing the content of the image.\n\n\nProperties of a bad dialogue:\n1. Simply describing or analyzing the content in the image\n2. Dialogue without a good logic.\n\nProperties of a good dialogue:\n1. Introducing extrinsic and attractive information of stuff shown in the image. \n2. Discovering the connection between multiple images.  \n3. The dialgoue happens in a reasonable context.\n\nExamples:\n========\n{}\n========\n\nPlease directly give the dialogue if you understand. The number of turns of the dialogue should be less than 6. The dialogue should be self-contained. Do NOT assume any previous dialogue between the human and the assistant. Please use the same format <imgX> DESCRIPTION </imgX> to denote images in the dialogue and do not modify the description of the image. Note that the dialogue can stop at any time without an explicit ending."

    @staticmethod
    def fewshot_example_template():
        return 'Case {}:\n\nInput Images:\n\n{}\n\nOutput Dialogue:\n\n{}'
    

def prepare_fewshot_examples(ex_list):
    template = PromptSet.fewshot_example_template()
    fewshot_list = []
    for i, ex in enumerate(ex_list):
        case_id = i+1
        images = ["    " + img for img in ex['image']]
        images = "\n".join(images)
        dialogue = ex['response']
        dialogue = "\n\n".join(["    " + s for s in dialogue.split("\n\n")])
        fewshot_list.append(template.format(case_id, images, dialogue))
    return "\n---------\n".join(fewshot_list)


def select_fewshot_examples(categorized_fewshot_pool, k):
    if isinstance(categorized_fewshot_pool, dict):
        category_keys = list(categorized_fewshot_pool.keys())
        selected_cats = [random.choice(category_keys) for _ in range(k*len(categorized_fewshot_pool))]
        visited_ids = []
        visited_cats = []
        rates = [2, 2, 1]
        random.shuffle(rates)
        ret, ptr = [], 0
        for cat in selected_cats:
            if len(visited_ids) >= k:
                break
            rate = rates[ptr]
            sampled_ex = random.choice(categorized_fewshot_pool[cat][rate])
            eid = sampled_ex['eid']
            if eid in visited_ids or cat in visited_cats:
                continue
            visited_ids.append(eid)
            visited_cats.append(cat)
            ret.append(sampled_ex)
            ptr += 1
    elif isinstance(categorized_fewshot_pool, list) :
        ret = random.choices(categorized_fewshot_pool, k=k)
    return ret


def reorganize_fewshot_pool(fewshot_pool):
    categorized_fewshot_pool = dict()
    all_rates = set()
    all_cats = set()
    for ex in fewshot_pool:
        cats = ex['category']
        rate = ex['rate']
        all_rates.add(rate)
        for cat in cats:
            all_cats.add(cat)
    for cat in all_cats:
        categorized_fewshot_pool[cat] = {k: [] for k in all_rates}
    for eid, ex in enumerate(fewshot_pool):
        cats = ex['category']
        rate = ex['rate']
        ex['eid'] =  eid
        for cat in cats:
            categorized_fewshot_pool[cat][rate].append(ex)
    return categorized_fewshot_pool


def generate_mmmchat_sub_proc(model, worker_id, grouped_idx, caption_data, save_pref, gen_num, max_img_num, history_data, fewshot_pool):
    random.seed(10086)
    clst_ids = list(range(len(grouped_idx)))
    n_finished = 0
    if fewshot_pool:
        prompt_template = PromptSet.in_context_prompt_v2()
        if "rate" in fewshot_pool[0] and "category" in fewshot_pool[0]:
            fewshot_pool = reorganize_fewshot_pool(fewshot_pool)
    else:
        prompt_template = PromptSet.in_context_prompt()
    img_nums = list(range(2, max_img_num))
    sampled_clst_ids = random.choices(clst_ids, k=2*(gen_num + len(history_data)))
    out_data = []
    for cid in sampled_clst_ids:
        cur_example = dict()
        if (n_finished + 1) % 1000 == 0:
            logging.info("Worker {} Processed {} examples".format(worker_id, n_finished))
        if n_finished >= gen_num:
            logging.info("Worker {} Finished to generate {} cases".format(worker_id, n_finished))
            break
        sampled_img_num = random.sample(img_nums, k=1)[0]
        if len(grouped_idx[cid]) < sampled_img_num:
            continue
        sampled_img_ids = tuple([int(sp_id) for sp_id in random.sample(grouped_idx[cid], k=sampled_img_num)])
        if sampled_img_ids in history_data:
            continue
        else:
            history_data.add(sampled_img_ids)
        sampled_imgs = ["<img{}> {} </img{}>".format(i, caption_data[j]['caption'].strip(), i) for i, j in enumerate(sampled_img_ids)]
        sampled_imgs_assemble = "    " + "\n    ".join(sampled_imgs)
        if fewshot_pool:
            ex_list = select_fewshot_examples(fewshot_pool, k=3)
            # ex_list = random.choices(fewshot_pool, k=3)
            query_prompt = prompt_template.format(sampled_imgs_assemble, prepare_fewshot_examples(ex_list))
        else:
            query_prompt = prompt_template.format(sampled_imgs_assemble)
        cur_example['image'] = sampled_imgs
        cur_example['image_idx'] = sampled_img_ids
        # sampled_imgs_assemble = "\n\n    ".join(sampled_imgs)
        cur_example['url'] = [caption_data[j]['url'] for j in sampled_img_ids]
        cur_example['prompt'] = query_prompt
        out_data.append(cur_example)
        n_finished += 1
        # while True:
        #     success = False
        #     try:
        #         res = call_chat(model, [{"role": "user", "content": query_prompt}], temperature=1.0)
        #         dialogue = res['choices'][0]['message']['content']
        #         cur_example['response'] = dialogue
        #         out_data.append(cur_example)
        #         logging.info("Successfully processed example {}".format(n_finished))
        #         n_finished += 1
        #         time.sleep(10)
        #         success = True
        #     except Exception as e:
        #         logging.info("Failed to process example {}".format(n_finished))
        #         time.sleep(20)
        #     if success:
        #         break
        
    FileUtils.save_to_disk(out_data, save_pref+".{}.data.json".format(worker_id))
    FileUtils.save_to_disk(list(history_data), save_pref+".{}.hist.json".format(worker_id))


def filter_cluster(grouped_idx, match_score, alpha=30.0, min_clst_size=32):
    new_grouped_idx = []
    for clst in grouped_idx:
        new_clst = []
        for j in clst:
            if match_score[j] >= alpha:
                new_clst.append(j)
        if len(new_clst) >= min_clst_size:
            new_grouped_idx.append(new_clst)
    return new_grouped_idx


def generate_mmmchat_mp(grouped_idx_path, caption_data_path, save_pref, model=OpenAIModel.GPT3_5, gen_num=10, max_img_num=4, history_data_path="", nproc=4, match_score_path="", fewshot_pool_file=""):
    fewshot_pool = FileUtils.load_file(fewshot_pool_file) if fewshot_pool_file else []

    grouped_idx = FileUtils.load_file(grouped_idx_path)
    logging.info("Init clst number: {}".format(len(grouped_idx)))
    if match_score_path:
        match_score = FileUtils.load_file(match_score_path)
        grouped_idx = filter_cluster(grouped_idx, match_score)
        logging.info("Filtered clst number: {}".format(len(grouped_idx)))
    file_type = FileUtils.check_file_type(caption_data_path)
    caption_data = FileUtils.load_file(caption_data_path)
    if file_type == "txt":
        caption_data = [{"caption": it, 'url': None} for it in caption_data]
    elif file_type == "csv":
        caption_data = [{"caption": it[0], 'url': it[1]} for it in caption_data]
    
    n_clst_per_proc = len(grouped_idx) // nproc + 1
    history_data = set(FileUtils.load_file(history_data_path)) if history_data_path else set()
    procs = []
    gen_num_per_proc = gen_num // nproc + 1
    for i in range(nproc):
        s, e = n_clst_per_proc * i, n_clst_per_proc * (i+1)
        sub_clst = grouped_idx[s:e]
        generate_mmmchat_sub_proc(model, i, sub_clst, caption_data, save_pref, gen_num_per_proc, max_img_num, history_data, fewshot_pool)
    #     p = Process(target=generate_mmmchat_sub_proc, args=(model, i, sub_clst, caption_data, save_pref, gen_num_per_proc, max_img_num, history_data))
    #     p.start()
    #     procs.append(p)
    # for p in procs:
    #     p.join()


def generate_mmmchat(grouped_idx_path, caption_data_path, save_pref, gen_num=10, max_img_num=4, history_data_path=""):
    model = OpenAIModel.GPT3_5
    grouped_idx = FileUtils.load_file(grouped_idx_path)
    caption_data = FileUtils.load_file(caption_data_path)
    clst_ids = list(range(len(grouped_idx)))
    n_finished  = 0
    prompt_template = PromptSet.base_prompt()
    img_nums = list(range(2, max_img_num))
    history_data = set(FileUtils.load_file(history_data_path)) if history_data_path else set()
    sampled_clst_ids = random.choices(clst_ids, k=2*(gen_num + len(history_data)))
    with open(save_pref + ".dialogue.txt", 'w') as fout:
        for cid in sampled_clst_ids:
            if (n_finished + 1) % 1000 == 0:
                logging.info("Processed {} examples".format(n_finished))
            if n_finished >= gen_num:
                logging.info("Finished to generate {} cases".format(n_finished))
                break
            sampled_img_num = random.sample(img_nums, k=1)[0]
            if len(grouped_idx[cid]) < sampled_img_num:
                continue
            sampled_img_ids = tuple(random.sample(grouped_idx[cid], k=sampled_img_num))
            if sampled_img_ids in history_data:
                continue
            else:
                history_data.add(sampled_img_ids)
            sampled_imgs = ["<img{}> {} </img{}>".format(i, caption_data[j].strip(), i) for i, j in enumerate(sampled_img_ids)]
            sampled_imgs_assemble = "\n\n".join(sampled_imgs)
            query_prompt = prompt_template.format(sampled_imgs_assemble)
            try:
                res = call_chat(model, [{"role": "user", "content": query_prompt}], temperature=1.0)
                dialogue = res['choices'][0]['message']['content']
                fout.write("{}\n".format(dialogue.strip()))
            except Exception:
                continue
            n_finished += 1
    FileUtils.save_to_disk(list(history_data), save_pref+".hist.pt")


def generate_from_old(old_data_path, save_path, model=OpenAIModel.GPT4, fewshot_pool_file=""):
    old_data = FileUtils.load_from_disk(old_data_path)
    if fewshot_pool_file:
        fewshot_pool = FileUtils.load_file(fewshot_pool_file)
        prompt_template = PromptSet.in_context_prompt_v2()
    else:
        fewshot_pool = []
        prompt_template = PromptSet.in_context_prompt()
    data = []
    n = 0
    while n < len(old_data):
        it = old_data[n]
        sampled_imgs_assemble = "    " + "\n    ".join(it['image'])
        if fewshot_pool:
            ex_list = random.choices(fewshot_pool, k=3)
            query_prompt = prompt_template.format(sampled_imgs_assemble, prepare_fewshot_examples(ex_list))
        else:
            query_prompt = prompt_template.format(sampled_imgs_assemble)
        try:
            res = call_chat(model, [{"role": "user", "content": query_prompt}], temperature=1.0)
            dialogue = res['choices'][0]['message']['content']
            it['prompt'] = query_prompt
            it['response'] = dialogue
            data.append(it)
            logging.info("Successfully processed example {}".format(n))
            n += 1
            time.sleep(10)
        except Exception as e:
            logging.info("Failed to process example {}".format(n))
            time.sleep(20)
            continue
    FileUtils.save_to_disk(data, save_path)


def main():
    fire.Fire({
        "generate_mmmchat": generate_mmmchat,
        "generate_mmmchat_mp": generate_mmmchat_mp,
        "generate_from_old": generate_from_old,
        "clean_fewshot_pool": clean_fewshot_pool,
        "clean_mim_data": clean_mim_data
    })

if __name__ == "__main__":
    main()
