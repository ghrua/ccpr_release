from mytools.tool_utils import StringUtils
import numpy as np
from typing import List


def overlap_score(text: str, sub_str_list: List[str], lower=False, remove_space=True, verbose=False):
    
    def get_verbose_str(visited, text):
        verbose_str = ""
        for ch, v in zip(text, visited):
            if ch == " ": verbose_str += ' '
            elif v > 0: verbose_str += '*'
            else:
                verbose_str += '_'
        return verbose_str
    text_len = len(text)
    visited = np.zeros(text_len)
    if not text or not sub_str_list:
        if verbose:
            return get_verbose_str(visited, text), 0
        else:
            return 0
    if lower:
        text = text.lower()
        sub_str_list = [it.lower() for it in sub_str_list]
    for sub_str in sub_str_list:
        sub_str_len = len(sub_str)
        indices = StringUtils.find_all_indices(text, sub_str)
        if not indices:
            continue
        max_n_left = 0
        best_idx = -1
        for idx in indices:
            n_left = sub_str_len - sum(visited[idx:idx+sub_str_len])
            if n_left > max_n_left:
                max_n_left = n_left
                best_idx = idx
        if best_idx >= 0:
            visited[best_idx:best_idx+sub_str_len] = 1
    if remove_space:
        for i, ch in enumerate(text):
            if ch == " ":
                visited[i] = 0
                text_len -= 1
    if text_len <= 0:
        if verbose:
            return get_verbose_str(visited, text), 0
        else:
            return 0
    if verbose:
        # print("=================================================================")
        # print("------ Input Text ------")
        # print(text)
        # print("------ Phrase List ------")
        # print(sub_str_list)
        # print("------ Match Results (`x` for match and `o` for non-match) ------")
        return get_verbose_str(visited, text), sum(visited) / text_len
        # print("".join(["x"  if it > 0 else "o" for it in visited]))
        # print(text)
    else:
        return sum(visited) / text_len