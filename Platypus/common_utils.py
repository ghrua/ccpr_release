import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TranslationTemplates:
    @staticmethod
    def plain(src_lang, tgt_lang, src_text):
        input_template = "{}:\n{}\n\n{}:"
        input_text = input_template.format(src_lang, src_text, tgt_lang)
        return "Please faithfully translate the following sentence from {} into {}, and do not alter its meaning:\n\n{}".format(src_lang, tgt_lang, input_text)

    @staticmethod
    def sentence(src_lang, tgt_lang, src_text, ret_sents):
        fewshot_template = "Related {} Sentence: {}"
        sent_examples = [fewshot_template.format(tgt_lang, sent) for sent in ret_sents]
        input_template = "{}:\n{}\n\n{}:"
        input_text = input_template.format(src_lang, src_text, tgt_lang)
        return "Please help me translate a sentence from {} into {}.\n\nAs shown below, there is a related sentence that may be helpfu:\n\n------------------\n{}\n------------------\n\nGuidelines for using the related sentence:\n1. A related sentence may be noisy, so you can discard the uesless part.\n2. If any part in the related sentence is useful, you can flexibly use the synonyms, variants, etc.\n\nBased on the provided related sentence, please faithfully translate the following sentence from {} into {}, and do not alter its meaning:\n\n{}".format(src_lang, tgt_lang, "\n\n".join(sent_examples), src_lang, tgt_lang, input_text)

    @staticmethod
    def fewshot_plain(src_lang, tgt_lang, src_text, fewshot_examples):
        fewshot_template = "{}:\n{}\n{}:\n{}"
        fewshot_examples = [fewshot_template.format(src_lang, ex_src, tgt_lang, ex_tgt) for ex_src, ex_tgt in fewshot_examples]
        input_template = "{}:\n{}\n\n{}:"
        input_text = input_template.format(src_lang, src_text, tgt_lang)
        return "Please translate the following sentence from {} into {}.\n\n{}\n\n{}".format(src_lang, tgt_lang, "\n\n".join(fewshot_examples), input_text)

    @staticmethod
    def phrase(src_lang, tgt_lang, src_text, phrase_examples):
        fewshot_template = "{} Phrase: {}\nPotential Translation: {}"
        phrase_examples = [fewshot_template.format(src_lang, src_phrase, "\t".join(tgt_phrases)) for src_phrase, tgt_phrases in phrase_examples]
        input_template = "{}:\n{}\n\n{}:"
        input_text = input_template.format(src_lang, src_text, tgt_lang)
        return "Please help me translate a sentence from {} into {}.\n\nThere are some potential {} translations of {} phrases that may be helpfu:\n\n------------------\n{}\n------------------\n\nGuidelines for using those phrases:\n1. Some phrase pairs are noisy, so you can discard those phrases whose translations are NOT appropriate.\n2. If a translation is useful, you can flexibly use its synonyms, variants, etc.\n\nPlease translate the following sentence from {} into {}, based on the provided phrases, wihtout any explanation:\n\n{}".format(src_lang, tgt_lang, tgt_lang, src_lang, "\n\n".join(phrase_examples), src_lang, tgt_lang, input_text)

    @staticmethod
    def phrase_with_context(src_lang, tgt_lang, src_text, phrase_examples):
        fewshot_template = "{} Phrase: {}\nPotential Translation: {}\nContext: {}"
        phrase_examples = [fewshot_template.format(src_lang, src_phrase, tgt_phrase, phrase_ctx) for src_phrase, tgt_phrase, phrase_ctx in phrase_examples]
        input_template = "{}:\n{}\n\n{}:"
        input_text = input_template.format(src_lang, src_text, tgt_lang)
        return "Please help me translate a sentence from {} into {}.\n\nAs shown below, there are some potential {} translations of {} phrases that may be helpfu. Note that each phrase translation is paired with a context sentence and is marked by `[[]]`, and unfinished contexts are represented by `...`:\n\n------------------\n{}\n------------------\n\nGuidelines for using those phrases:\n1. Some phrase pairs are noisy, so you can discard those phrases whose translations are NOT appropriate.\n2. If a translation is useful, you can flexibly use its synonyms, variants, etc.\n3. Do not use the `[[]]` mark in your translation.\n\nBased on the provided information of phrase translation, please faithfully translate the following sentence from {} into {}, and do not alter its meaning:\n\n{}".format(src_lang, tgt_lang, tgt_lang, src_lang, "\n\n".join(phrase_examples), src_lang, tgt_lang, input_text)


    @staticmethod
    def phrase_no_guideline(src_lang, tgt_lang, src_text, phrase_examples):
        fewshot_template = "{} Phrase: {}\nPotential Translation: {}"
        phrase_examples = [fewshot_template.format(src_lang, src_phrase, "\t".join(tgt_phrases)) for src_phrase, tgt_phrases in phrase_examples]
        input_template = "{}:\n{}\n\n{}:"
        input_text = input_template.format(src_lang, src_text, tgt_lang)
        return "Please help me translate a sentence from {} into {}.\n\nThere are some potential {} translations of {} phrases that may be helpfu:\n\n------------------\n{}\n------------------\n\nPlease translate the following sentence from {} into {} based on the provided phrases:\n\n{}".format(src_lang, tgt_lang, tgt_lang, src_lang, "\n\n".join(phrase_examples), src_lang, tgt_lang, input_text)
    
    @staticmethod
    def chunk(src_lang, tgt_lang, src_text, chunk_examples):
        new_chunk_examples = []
        for it in chunk_examples:
            new_chunk_list = []
            for j, chunk in enumerate(it[1]):
                new_chunk_list.append("{}. {}".format(j+1, chunk))
            new_chunk_examples.append((it[0], tuple(new_chunk_list)))
        chunk_examples = new_chunk_examples
        fewshot_template = "{} Phrase:\t{}\n{} Snippets:\n{}"
        chunk_examples = [fewshot_template.format(src_lang, src_phrase, tgt_lang, "\n".join(tgt_chunks)) for src_phrase, tgt_chunks in chunk_examples]
        input_template = "{}:\n{}\n\n{}:"
        input_text = input_template.format(src_lang, src_text, tgt_lang)
        return "Please help me translate a sentence from {} into {}.\n\nThere are some potential {} translations of {} phrases that may be helpfu:\n\n------------------\n{}\n------------------\n\nGuidelines for using those snippets:\n1. The information in snippets might be noisy, so you need to carefully examine whether they are appropriate.\n2. If any provided translation snippet is beneficial for the translation, you can use it flexibly, including adopting sentence structures, certain words, or variants of some words from the translation snippet.\n\nPlease complete the following translation from {} into {}, based on the provided snippets, wihtout any explanation:\n\n{}".format(src_lang, tgt_lang, tgt_lang, src_lang, "\n\n".join(chunk_examples), src_lang, tgt_lang, input_text)
    


if __name__ == "__main__":
    src_lang = "Germany"
    tgt_lang = "English"
    src_sent = "Die Premierminister Indiens und Japans trafen sich in Tokio."
    fewshot_examples = []
    for i in range(3):
        fewshot_examples.append((src_sent, src_sent[::-1]))

    phrases = [("Premierminister", ("Prime Ministers","Prime Ministers")), ("Indiens", ("India","Prime Ministers"))]
    chunks = [("Wahlsieg im Mai", ("from 1 votes July from the", "before the handover in 1997, when")), ("Auslandsbesuch", ("a rare overseas trip, are", "the last foreign trips for the"))]
    phrase_with_context = [("Premierminister", "Prime Ministers", "The [[Prime Ministers]] of India and Pakistan recently met in Pakistan to discuss the question."), ("Indiens", "India", "The Prime Ministers of [[India]] and Pakistan recently met in Pakistan to discuss the question.")]
    retrieved_sentences = ["The Prime Ministers of India and Pakistan recently met in Pakistan to discuss the question."]
    print(TranslationTemplates.plain(src_lang, tgt_lang, src_sent))
    print("-----------------------------------------")
    print(TranslationTemplates.fewshot_plain(src_lang, tgt_lang, src_sent, fewshot_examples))
    print("-----------------------------------------")
    print(TranslationTemplates.phrase(src_lang, tgt_lang, src_sent, phrases))
    print("-----------------------------------------")
    print(TranslationTemplates.phrase_no_guideline(src_lang, tgt_lang, src_sent, phrases))
    print("-----------------------------------------")
    print(TranslationTemplates.chunk(src_lang, tgt_lang, src_sent, chunks))
    print("-----------------------------------------")
    print(TranslationTemplates.phrase_with_context(src_lang, tgt_lang, src_sent, phrase_with_context))
    print("-----------------------------------------")
    print(TranslationTemplates.sentence(src_lang, tgt_lang, src_sent, retrieved_sentences))
