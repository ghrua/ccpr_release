from common_utils import TranslationTemplates, set_seed
import torch
from mytools.tool_utils import FileUtils, logging, TorchUtils
from transformers import AutoTokenizer, AutoModelForCausalLM
import fire
from tqdm import tqdm
import re
import random
from bert_score import score
from mytools.openai_tools import call_chat, OpenAIModel
from comet import download_model, load_from_checkpoint



MODEL_TYPE_MAP = {
    "llama-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama-13b": "meta-llama/Llama-2-13b-chat-hf",
    "chatgpt": OpenAIModel.GPT3_5
}

class TransMethods:
    BASELINE = "baseline"
    PHRASE = "phrase"
    CHUNK = "chunk"
    SENTENCE = "sentence"
    PHRASE_NO_GUIDELINE = "phrase_no_guideline"
    FEWSHOT_BASELINE = "fewshot_baseline"
    PHRASE_WITH_CONTEXT = "phrase_with_context"


LANG_MAP = {
    "de": "Germany",
    "en": "English",
    "cs": "Czech",
    "fi": "Finnish",
    "ro": "Romanian",
    "ru": "Russian",
    "tr": "Turkish"
}


AUX_LANG_MAP = {
    "de": ["Deutsch", "German", "Deutschland"],
    "en": ["English", "En"]
}

PHRASE_LEVEVL_MAP = {
    "phrase": 1, "chunk":2, "sent": 3
}

def extract_text_in_quotes(text):
    pattern = r"'(.*?)'|\"(.*?)\""
    matches = re.findall(pattern, text)
    extracted_texts = [match[0] or match[1] for match in matches]
    
    return extracted_texts


def prepare_alpaca_data(src_file_path, ref_file_path, save_path, src_lang="de", tgt_lang="en", method=TransMethods.BASELINE, **kwargs):
    set_seed(kwargs.get("seed", 10086))
    src_data, ref_data = FileUtils.load_file(src_file_path), FileUtils.load_file(ref_file_path)
    phrase_data_path = kwargs.get("phrase_data_path", None)
    phrase_min_dist = kwargs.get("phrase_min_dist", 0)
    phrase_topk = kwargs.get("phrase_topk", 1)
    phrase_max_num = kwargs.get("phrase_max_num", 24)
    phrase_max_num_selection_method = kwargs.get("phrase_max_num_selection_method", "rand") # e.g., `sort`, `rand`
    phrase_max_dist_percent = kwargs.get("phrase_max_dist_percent", 1.0) # e.g., 1.0, 0.75, 0.5, 0.25
    phrase_max_context_len = kwargs.get("phrase_max_context_len", 250) # number of characters, e.g., 150, 200, 250, 3000
    phrase_merge_mode = kwargs.get("phrase_merge_mode", 'none') 
    phrase_merge_level = kwargs.get("phrase_merge_level", 1) 
    data_source = kwargs.get("data_source", "")
    instruction_max_len = kwargs.get("instruction_max_len", 3796) # number of tokens
    field = kwargs.get("field", "train")
    test_subset_size = kwargs.get("test_subset_size", None)

    sent_data_path = kwargs.get("sent_data_path", None)
    sent_topk = kwargs.get("sent_topk", 1)

    
    if phrase_merge_level not in [1, 2]:
        raise ValueError("Unkown `merge_level`={}".format(phrase_merge_level))
    if phrase_merge_mode not in ["keep_max", "keep_min", "none"]:
        raise ValueError("Unknow `merge_mode`={}".format(phrase_merge_mode))
    
    if test_subset_size is not None:
        indices = list(range(len(src_data)))
        random.shuffle(indices)
        test_indices = sorted(indices[:test_subset_size])
    else:
        test_indices = list(range(len(src_data)))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE_MAP['llama-7b'], cache_dir="/project/nlp-cache/huggingface/hub")

    src_lang, tgt_lang = LANG_MAP[src_lang], LANG_MAP[tgt_lang]

    if phrase_data_path is None:
        phrase_data = None
    else:
        phrase_data = FileUtils.load_file(phrase_data_path, "json")
        phrase_data = {
            it['id']: it for it in phrase_data
        }
        if phrase_min_dist == 0 and phrase_max_dist_percent < 1.0:
            phrase_min_dist = get_phrase_min_dist(phrase_data, phrase_max_dist_percent)
            print("Setting `phrase_min_dist`={} according to `phrase_max_dist_percent`={}".format(phrase_min_dist, phrase_max_dist_percent))

    if sent_data_path is None:
        sent_data = None
    else:
        sent_data = FileUtils.load_file(sent_data_path, "json")
        sent_data = {
            it['id']: it for it in sent_data
        }


    alpaca_dataset = {field: []}
    for sid in tqdm(test_indices):
        # choose template
        sent = src_data[sid]
        if method == TransMethods.BASELINE:
            message = TranslationTemplates.plain(src_lang, tgt_lang, sent)
        elif method == TransMethods.SENTENCE:
            if sid in sent_data and len(sent_data[sid]['retrieved_sentences']) > 0:
                retrieved_sentences = sent_data[sid]['retrieved_sentences'][:sent_topk]
                message = TranslationTemplates.sentence(src_lang, tgt_lang, sent, retrieved_sentences)
            else:
                message = TranslationTemplates.plain(src_lang, tgt_lang, sent)
        elif method == TransMethods.PHRASE_WITH_CONTEXT:
            if sid not in phrase_data:
                message = TranslationTemplates.plain(src_lang, tgt_lang, sent)
            else:
                cur_phrases = phrase_filter(phrase_data[sid]['alignment'], phrase_topk, phrase_min_dist, phrase_max_context_len, phrase_max_num, phrase_max_num_selection_method, phrase_merge_mode, phrase_merge_level)
                if not cur_phrases:
                    message = TranslationTemplates.plain(src_lang, tgt_lang, sent)
                else:
                    cur_phrases = [(it[0], it[1], it[3]) for it in cur_phrases]
                    message = TranslationTemplates.phrase_with_context(src_lang, tgt_lang, sent, cur_phrases)                
        elif method == TransMethods.PHRASE:
            if sid not in phrase_data:
                message = TranslationTemplates.plain(src_lang, tgt_lang, sent)
            else:
                cur_phrases = phrase_filter(phrase_data[sid]['alignment'], phrase_topk, phrase_min_dist, phrase_max_context_len, phrase_max_num, phrase_max_num_selection_method, phrase_merge_mode, phrase_merge_level)
                if not cur_phrases:
                    message = TranslationTemplates.plain(src_lang, tgt_lang, sent)
                else:
                    cur_phrases = [(it[0], it[1]) for it in cur_phrases]
                    message = TranslationTemplates.phrase(src_lang, tgt_lang, sent, cur_phrases)
        elif method == TransMethods.PHRASE_NO_GUIDELINE:
            if sid not in phrase_data:
                message = TranslationTemplates.plain(src_lang, tgt_lang, sent)
            else:
                cur_phrases = phrase_filter(phrase_data[sid]['alignment'], phrase_topk, phrase_min_dist, phrase_max_context_len, phrase_max_num, phrase_max_num_selection_method, phrase_merge_mode, phrase_merge_level)
                if not cur_phrases:
                    message = TranslationTemplates.plain(src_lang, tgt_lang, sent)
                else:
                    cur_phrases = [(it[0], it[1]) for it in cur_phrases]
                    message = TranslationTemplates.phrase_no_guideline(src_lang, tgt_lang, sent, cur_phrases)
        elif method == TransMethods.CHUNK:
            if sid not in phrase_data:
                message = TranslationTemplates.plain(src_lang, tgt_lang, sent)
            else:
                cur_phrases = phrase_filter(phrase_data[sid]['alignment'], phrase_topk, phrase_min_dist, phrase_max_context_len, phrase_max_num, phrase_max_num_selection_method, phrase_merge_mode, phrase_merge_level)
                if not cur_phrases:
                    message = TranslationTemplates.plain(src_lang, tgt_lang, sent)
                else:
                    cur_phrases = [(it[0], it[2]) for it in cur_phrases]
                    message = TranslationTemplates.chunk(src_lang, tgt_lang, sent, cur_phrases)                
        else:
            raise ValueError("Unknow translation method {}".format(method))
        
        conversation = [{"role": "user", "content": message}]
        inst_len = tokenizer.apply_chat_template(conversation, return_tensors="pt")[0].size(0)
        if inst_len > instruction_max_len:
            continue
        
        alpaca_dataset[field].append({"input": "", "sent_id": sid, "output": ref_data[sid], "instruction": message, "data_source": data_source})
    print("{} examples in dataset".format(len(alpaca_dataset[field])))
    if test_subset_size is not None:
        FileUtils.save_file(alpaca_dataset, FileUtils.handle_file_extension(save_path, "sample.{}".format(test_subset_size)), "json")
    else:
        FileUtils.save_file(alpaca_dataset, save_path, "json")



def compare_translation(src_file_path, ref_file_path, sys1_file_path, sys2_file_path, save_path, tgt_lang="en", phrase_data_path="", topk_phrases=1):
    srcs = FileUtils.load_file(src_file_path)
    refs = FileUtils.load_file(ref_file_path)
    sys1 = {int(k): v for k, v in FileUtils.load_file(sys1_file_path).items()}
    sys2 = {int(k): v for k, v in FileUtils.load_file(sys2_file_path).items()}
    if phrase_data_path:
        phrase_data = {}

        for it in FileUtils.load_file(phrase_data_path):
            tmp_list = []
            for al in it['alignment']:
                for k in range(1, topk_phrases+1):
                    try:
                        tmp_list.append((al[0], al[k][1], al[k][2], al[k][3], al[k][-1]))
                    except Exception:
                        tmp_list.append((al[0], "EMPTY", "EMPTY", "EMPTY", 0))
            phrase_data[it['id']] = tmp_list
    else:
        phrase_data = None

    sys1_keys = set(sys1.keys())
    sys2_keys = set(sys2.keys())
    common_keys = sys1_keys.intersection(sys2_keys)
    id_map = [(j, i) for i, j in enumerate(common_keys)]
    common_refs = [refs[it[0]] for it in id_map]
    common_sys1 = [sys1[it[0]][1] for it in id_map] # (ref, trans, raw response)
    common_sys2 = [sys2[it[0]][1] for it in id_map] # (ref, trans, raw response)

    
    # Compute BERTScore for both systems
    P1, R1, F1_sys1 = score(common_sys1, common_refs, lang=tgt_lang, rescale_with_baseline=True, verbose=True)
    P2, R2, F1_sys2 = score(common_sys2, common_refs, lang=tgt_lang, rescale_with_baseline=True, verbose=True)
    
    # Comparing F1 scores for simplicity; other metrics can also be used
    results = []
    for i, (f1_sys1, f1_sys2) in enumerate(zip(F1_sys1, F1_sys2)):
        real_sent_id = id_map[i][0]
        comparison = "comparable"
        if f1_sys1 > f1_sys2:
            comparison = "System 1 is better"
        elif f1_sys2 > f1_sys1:
            comparison = "System 2 is better"
        
        results.append({
            "id": real_sent_id,
            "src": srcs[real_sent_id],
            "sys1": sys1[real_sent_id][1],
            "sys2": sys2[real_sent_id][1],
            "ref": refs[real_sent_id],
            "sys1 F1": f1_sys1.item(),
            "sys2 F1": f1_sys2.item(),
            "Comparison": comparison,
            "phrases": phrase_data[real_sent_id]
        })
    FileUtils.save_file(results, save_path)


def parse_llama_chat_data(trans_dump_file, src_data, tgt_lang, enable_safe_mode):
    trans_data = {int(k): v for k, v in FileUtils.load_file(trans_dump_file).items()}
    trans_ids, trans_sents, response_list, prompts = [], [], [], []
    for idx, val in trans_data.items():
        response = val[-1]
        if not response.strip():
            continue
        trans = extract_translation(response, src_sent=src_data[idx], tgt_lang=LANG_MAP[tgt_lang], aux_lang_list=AUX_LANG_MAP[tgt_lang])
        if trans == response and (len(response.split("\n\n")) > 1 or "apologize" in response[:30]):
            print("Sent {} met potential bad case for translation extraction:\n\n{}".format(idx, " ".join(response.split())))
            if enable_safe_mode:
                continue
        else:
            trans_ids.append(idx)
            trans_sents.append(trans)
            response_list.append(response)   
            prompts.append(val[1])
    return  trans_ids, trans_sents, response_list, prompts


def parse_alpaca_data(trans_dump_file, field):
    trans_data = FileUtils.load_file(trans_dump_file)[field]
    trans_ids, trans_sents, response_list, prompts = [], [], [], []
    for it in trans_data:
        trans_ids.append(it['sent_id'])
        trans_sents.append(it['hypothesis'])
        response_list.append(it['hypothesis'])
        prompts.append(it['instruction'])
    return  trans_ids, trans_sents, response_list, prompts


def report_score(trans_dump_file, src_file_path, ref_file_path, tgt_lang="en", no_safe_mode=False, rescale_with_baseline=True, data_format="llama_chat", **kwargs):
    save_txt_file_prefix = kwargs.get("save_txt_file_prefix", "")
    src_data = FileUtils.load_file(src_file_path)
    ref_data = FileUtils.load_file(ref_file_path)
    trans_ids, trans_sents, response_list = [], [], []
    enable_safe_mode = not no_safe_mode
    if data_format == "llama_chat":
        trans_ids, trans_sents, response_list, prompts = parse_llama_chat_data(trans_dump_file, src_data, tgt_lang, enable_safe_mode)
    elif data_format == "alpaca":
        trans_ids, trans_sents, response_list, prompts = parse_alpaca_data(trans_dump_file, kwargs.get("field", "test"))
    ref_data = [ref_data[i] for i in trans_ids]
    src_data = [src_data[i] for i in trans_ids]

    if save_txt_file_prefix:
        FileUtils.save_file(src_data, save_txt_file_prefix + ".src", 'txt')
        FileUtils.save_file(ref_data, save_txt_file_prefix + ".ref", 'txt')
        FileUtils.save_file(trans_sents, save_txt_file_prefix + ".hyp", 'txt')
    
    comet_model_path = download_model("Unbabel/wmt22-comet-da", saving_directory="../huggingface/project_cache")
    comet_model = load_from_checkpoint(comet_model_path)
    comet_data = [{"src": src, "mt": mt, "ref": ref} for src, ref, mt in zip(src_data, ref_data, trans_sents)]
    comet_model_output = comet_model.predict(comet_data, batch_size=64, gpus=1)
    
    (P1, R1, F1), hash_code = score(trans_sents, ref_data, lang=tgt_lang, rescale_with_baseline=rescale_with_baseline, verbose=True, return_hash=True, use_fast_tokenizer=True)

    print("BERTScore | Hash: {} | P {:.3f}\tR {:.3f}\tF {:.3f}".format(hash_code, P1.mean(dim=0) * 100, R1.mean(dim=0) * 100, F1.mean(dim=0) * 100))
    print("Comet | {:.3f}".format(comet_model_output.system_score * 100))

    log_data = []
    for bid, sid in enumerate(trans_ids):
        log_data.append({"id": sid, "src": src_data[bid], "ref": ref_data[bid], "hyp": trans_sents[bid], 'prompt': prompts[bid], 'response': response_list[bid], "score": (P1[bid].item(), R1[bid].item(), F1[bid].item())})

    FileUtils.save_file(log_data, FileUtils.handle_file_extension(trans_dump_file, "score.log"))


def report_multi_system_score(trans_dump_files, src_file_path, ref_file_path, tgt_lang="en", no_safe_mode=False, no_rescale_with_baseline=False):
    src_data = FileUtils.load_file(src_file_path)
    ref_data = FileUtils.load_file(ref_file_path)
    enable_safe_mode = not no_safe_mode
    multi_system_trans_ids, multi_system_trans_sents, multi_system_response_list, multi_system_prompts = [], [], [], []
    multi_system_P, multi_system_R, multi_system_F1 = [], [], []
    trans_dump_files = trans_dump_files.split(",")
    n_sys = len(trans_dump_files)
    for sys_id, trans_dump_file in enumerate(trans_dump_files):
        trans_data = {int(k): v for k, v in FileUtils.load_file(trans_dump_file).items()}
        trans_ids, trans_sents, response_list, prompts = [], {}, {}, {}
        for idx, val in trans_data.items():
            response = val[-1]
            if not response.strip():
                continue
            trans = extract_translation(response, src_sent=src_data[idx], tgt_lang=LANG_MAP[tgt_lang], aux_lang_list=AUX_LANG_MAP[tgt_lang])
            if trans == response and (len(response.split("\n\n")) > 1 or "apologize" in response[:30]):
                print("Sent {} met potential bad case for translation extraction:\n\n{}".format(idx, " ".join(response.split())))
                if enable_safe_mode:
                    continue
            else:
                trans_ids.append(idx)
                trans_sents[idx] = trans
                response_list[idx] = response
                prompts[idx] = val[1]
        multi_system_trans_ids.append(trans_ids)
        multi_system_trans_sents.append(trans_sents)
        multi_system_response_list.append(response_list)
        multi_system_prompts.append(prompts)

    common_ids = set(multi_system_trans_ids[0])
    for i in range(1, n_sys):
        common_ids = common_ids.intersection(set(multi_system_trans_ids[i]))
    common_ids = sorted(list(common_ids))
    print("{} sents got translation for all systems".format(len(common_ids)))
    
    results = []
    common_ref_data = [ref_data[i] for i in common_ids]
    for i in range(n_sys):
        trans_sents = multi_system_trans_sents[i]
        hyp_data = [trans_sents[j] for j in common_ids]
        (P, R, F1), hash_code = score(hyp_data, common_ref_data, lang=tgt_lang, rescale_with_baseline=True, verbose=True, return_hash=True, use_fast_tokenizer=True)
        multi_system_P.append(P)
        multi_system_R.append(R)
        multi_system_F1.append(F1)
        results.append("Sys: {} | Hash: {} | BERTScore: P {:.3f}\tR {:.3f}\tF {:.3f}".format(trans_dump_files[i], hash_code, P.mean(dim=0) * 100, R.mean(dim=0) * 100, F1.mean(dim=0) * 100))

    for i in range(n_sys):
        log_data = []
        trans_sents, response_list, prompts = multi_system_trans_sents[i], multi_system_response_list[i], multi_system_prompts[i]
        P, R, F1 = multi_system_P[i], multi_system_R[i], multi_system_F1[i]
        for bid, sid in enumerate(common_ids):
            log_data.append({"id": sid, "src": src_data[sid], "ref": ref_data[sid], "hyp": trans_sents[sid], 'prompt': prompts[sid], 'response': response_list[sid], "score": (P[bid].item(), R[bid].item(), F1[bid].item())})

        FileUtils.save_file(log_data, FileUtils.handle_file_extension(trans_dump_files[i], "score.log"))
    
    print("========================= Results Summary =========================")
    for it in results:
        print(it)


def extract_translation(raw_response, max_len, tgt_lang="English", aux_lang_list=None):
    # try to find the pattern "English: "
    lines = raw_response.split("\n\n")
    lang_list = [tgt_lang]
    if aux_lang_list is not None:
        lang_list += aux_lang_list
    for i, line in enumerate(lines):
        for lang in aux_lang_list:
            if line.startswith("{}:".format(lang)):
                trans = line[len(tgt_lang)+1:].strip()
                if trans:
                    return trans
                else:
                    for j in range(1+1, len(lines)):
                        if lines[j].strip():
                            return line[j].strip()
    num_lines = len(lines)
    return_tag = False
    for i in range(num_lines):
        if return_tag and lines[i].strip():
            trans = lines[i].strip().split("\n")[0]
            if trans.startswith("*") or trans.startswith("."):
                trans = trans[1:].strip()
            if trans.endswith('"') and trans.startswith('"'):
                trans = trans[1:-1]
            if trans.startswith(tgt_lang + ":"):
                trans = trans[len(tgt_lang)+1:]
            if "\"" in trans[:] and len(trans.split()) >= max_len:
                trans = extract_text_in_quotes(trans)[0]
            return trans
        if not return_tag:
            if lines[i].endswith(":"):
                return_tag = True
            else:
                return_tag = False
    if raw_response.endswith('"') and raw_response.startswith('"'):
        raw_response = raw_response[1:-1]
    if raw_response.startswith("[") or raw_response.startswith("*") or raw_response.startswith("."):
        raw_response = raw_response[1:].strip()
    if raw_response.endswith("]"):
        raw_response = raw_response[:-1].strip()
    return raw_response


def get_phrase_min_dist(phrase_data, phrase_max_dist_percent):
    dist_list = []
    for ei, ex in phrase_data.items():
        for it in ex['alignment']:
            if len(it) > 1:
                dist_list.append(it[1][-1])

    dist_list = sorted(dist_list, key=lambda x: -x)
    if 0.0 <= phrase_max_dist_percent <= 1.0:
        num_dist_list = int(len(dist_list) * phrase_max_dist_percent)
    else:
        raise ValueError("`phrase_max_dist_percent` should be in [0, 1], but it is {} now".format(phrase_max_dist_percent))
    return dist_list[num_dist_list]


def handle_overlength(aligned_phrase_ctx, phrase_max_context_len, marker_start="[[", marker_end="]]"):
    words = aligned_phrase_ctx.split(" ")
    n_words = len(words)
    half_phrase_max_context_len = phrase_max_context_len // 2
    left_words, right_words = [], []
    for i in range(n_words):
        if words[i].startswith(marker_start):
            n_left_contxt_len = 0
            for j in range(i, -1, -1):
                n_left_contxt_len += len(words[j]) + 1
                if n_left_contxt_len >= half_phrase_max_context_len:
                    break
                left_words.append(words[j])
            left_words = left_words[::-1]
            n_right_contxt_len = 0
            for j in range(i+1, n_words):
                n_right_contxt_len += len(words[j]) + 1
                if n_right_contxt_len >= half_phrase_max_context_len:
                    break
                right_words.append(words[j])
            break
    left_ctx = " ".join(left_words)
    right_ctx = " ".join(right_words)
    return "... {} {} ...".format(left_ctx, right_ctx)


def clean_phrases(phrases):
    new_phrases = []
    s, prev_src_p = 0, ""
    for it in phrases:
        if it[0] != prev_src_p:
            prev_src_p = it[0]
            new_phrases.append(it)
        else:
            # there are top-k candidates sharing the same src phrase
            new_phrases.append(it)


def dfs(src_phrase_dict, s, e, ans):
    if (s, e) in src_phrase_dict:
        ans.append([(s, e)])
    for j in range(s, e):
        split_a_ans, split_b_ans = [], []
        dfs(src_phrase_dict, s, j, split_a_ans)
        dfs(src_phrase_dict, j+1, e, split_b_ans)
        for it in split_a_ans:
            for jt in split_b_ans:
                ans.append(it + jt)
    return


def phrase_merge(phrase_list, merge_mode="keep_max", merge_level=3):
    src_phrase_dict = {(it[0][1], it[0][2]): it[0][0] for it in phrase_list}
    trans_phrase_dict = {(it[0][1], it[0][2]): it[1][1] for it in phrase_list if len(it) > 1}
    src_phrase_list = sorted(list(src_phrase_dict.keys()), key=lambda x: x[0]-x[1])

    remove_set = set()
    for s, e in src_phrase_list:
        pos = (s, e)
        cur_phrase = src_phrase_dict[pos]
        cur_trans = trans_phrase_dict.get(pos, "")
        if pos in remove_set:
            continue
        ans = []
        dfs(src_phrase_dict, s, e, ans)
        if len(ans) == 1:
            continue
        elif merge_mode == "keep_max":
            for it in ans:
                if len(it) > 1:
                    found_phrase = " ".join([src_phrase_dict[found_pos] for found_pos in it])
                    found_trans = " ".join([trans_phrase_dict.get(found_pos, "")  for found_pos in it])
                    if merge_level == 1:
                        remove_set.update(it)
                    elif merge_level == 2 and found_phrase == cur_phrase and cur_trans == found_trans:
                        remove_set.update(it)
        elif merge_mode == "keep_min":
            for it in ans:
                if len(it) > 1:
                    found_phrase = " ".join([src_phrase_dict[found_pos] for found_pos in it])
                    found_trans = " ".join([trans_phrase_dict.get(found_pos, "") for found_pos in it])
                    if merge_level == 1:
                        remove_set.add(pos)
                    elif merge_level == 2 and found_phrase == cur_phrase and cur_trans == found_trans:
                        remove_set.add(pos)
            
    new_phrase_list = []
    for it in phrase_list:
        pos = (it[0][1], it[0][2])
        if pos not in remove_set:
            new_phrase_list.append(it)
    return new_phrase_list


def phrase_filter(phrase_list, phrase_topk, phrase_min_dist, phrase_max_context_len, phrase_max_num, phrase_max_num_selection_method, merge_mode="none", merge_level=3):
    
    cur_phrases = []
    if merge_mode != "none":
        phrase_list = phrase_merge(phrase_list, merge_mode, merge_level)
    for it in phrase_list:
        for jt in it[1:1+phrase_topk]:
            aligned_phrase, aligned_chunk, aligned_phrase_ctx , dist = jt[1], jt[2], jt[3], jt[-1]
            if dist >= phrase_min_dist:
                if len(aligned_phrase_ctx) >= phrase_max_context_len:
                    aligned_phrase_ctx = handle_overlength(aligned_phrase_ctx, phrase_max_context_len)
                cur_phrases.append((it[0][0], aligned_phrase, aligned_chunk, aligned_phrase_ctx, dist))
    
    if len(cur_phrases) > phrase_max_num:
        if phrase_max_num_selection_method == "rand":
            cur_phrases = random.sample(cur_phrases, k=phrase_max_num)
        elif phrase_max_num_selection_method == "sort":
            cur_phrases = sorted(cur_phrases, key=lambda x: -x[-1])[:phrase_max_num]
        else:
            raise ValueError("Unkown `phrase_max_num_selection_method={}`, which should be `rand` or `sort`.".format(phrase_max_num_selection_method))
    return cur_phrases


def translate(input_file_path, save_path, model_type="llama-7b", src_lang="de", tgt_lang="en", **kwargs):
    model_type = MODEL_TYPE_MAP[model_type]
    if model_type != OpenAIModel.GPT3_5:
        tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir="/project/nlp-cache/huggingface/hub")
        model = AutoModelForCausalLM.from_pretrained(model_type, cache_dir="/project/nlp-cache/huggingface/hub")
        model = model.eval()
        model = model.half().to("cuda")
    data = FileUtils.load_file(input_file_path)
    """Get arguments"""
    batch_size = kwargs.get("batch_size", 1)
    if model_type == OpenAIModel.GPT3_5:
        assert batch_size == 1

    field=kwargs.get("field", "validation")

    src_lang_abb, tgt_lang_abb = src_lang, tgt_lang
    src_lang, tgt_lang = LANG_MAP[src_lang], LANG_MAP[tgt_lang]

    generate_kwargs = dict(
        max_new_tokens=kwargs.get('max_new_tokens', 128),
        do_sample=not kwargs.get('greedy_search', False),
        top_p=kwargs.get('top_p', 0.9),
        top_k=kwargs.get('top_k', 50),
        temperature=kwargs.get('temperature', 0.6),
        num_beams=kwargs.get('num_beams', 1),
        repetition_penalty=kwargs.get('repetition_penalty', 1.2),
    )
    batch_sents, batch_case_ids = [], []
    num_test_examples = len(data[field])

    for i in tqdm(range(num_test_examples)):
        if len(batch_sents) >= batch_size or (i+1) == num_test_examples:
            sequences = []
            if model_type != OpenAIModel.GPT3_5:
                for k in range(len(batch_sents)):
                    conversation = [{"role": "user", "content": batch_sents[k]}]
                    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
                    sequences.append(input_ids[0])
                batch_input_ids, attention_mask = TorchUtils.batchfy(sequences, tokenizer.eos_token_id)
                generate_kwargs['input_ids'] = batch_input_ids
                generate_kwargs['attention_mask'] = attention_mask
                outputs = model.generate(**generate_kwargs)                    
            else:
                outputs = call_chat(model_type, messages=conversation)
            # decode
            for k in range(len(batch_sents)):
                case_id = batch_case_ids[k]
                if model_type != OpenAIModel.GPT3_5:
                    response = tokenizer.decode(outputs[k], skip_special_tokens=True)
                    response = response.split("[/INST]")[-1].strip()
                    extracted_trans = extract_translation(response, max_len=128, tgt_lang=tgt_lang, aux_lang_list=AUX_LANG_MAP[tgt_lang_abb])
                else:
                    response = outputs.choices[0].message.content
                    extracted_trans = outputs.choices[0].message.content
                
                data[field][case_id]['response'] = response
                data[field][case_id]['hypothesis'] = extracted_trans
            batch_sents, batch_case_ids = [data[field][i]['instruction']], [i]
        else:
            batch_sents.append(data[field][i]['instruction'])
            batch_case_ids.append(i)
    FileUtils.save_file(data, save_path, "json")    


if __name__ == "__main__":
    fire.Fire({
        "translate": translate,
        "compare_translation": compare_translation,
        "report_score": report_score,
        "prepare_alpaca_data": prepare_alpaca_data,
        "report_multi_system_score": report_multi_system_score
    })