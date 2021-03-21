import string

from nltk.tokenize import TweetTokenizer
from nltk.translate import bleu_score


def tokenize(text):
    tokenizer = TweetTokenizer(preserve_case=False)
    return tokenizer.tokenize(text)

def untokenize(tokens):
    # check if a blank should be added before a word.
    # when word is punctuation or a part of shortened form,
    # space is not needed.
    is_space = lambda token: (token not in string.punctuation) and (not token.startswith("'"))
    text = "".join([
        (" " + token) if is_space(token) else token for token in tokens
    ])
    text = text.strip()

    return text

def calc_bleu_score_for_seqs(hypo_seq, ref_seqs):
    sf = bleu_score.SmoothingFunction()
    return bleu_score.sentence_bleu(
        references=ref_seqs,
        hypothesis=hypo_seq,
        smoothing_function=sf.method1,
        weights=(0.5, 0.5),
    )

def calc_bleu_score_for_seq(hypo_seq, ref_seq):
    return calc_bleu_score_for_seqs(hypo_seq, [ref_seq])
