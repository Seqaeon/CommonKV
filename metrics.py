"""
LongBench evaluation metrics — standalone root-level copy.
Identical to PyramidKV/metrics.py but with lazy jieba import so that
English-only evaluation works even if jieba / fuzzywuzzy are not installed.
"""
import re
import string
from collections import Counter

# --- Optional heavy deps (Chinese tasks only) ---
try:
    import jieba as _jieba
    _JIEBA_OK = True
except ImportError:
    _jieba = None
    _JIEBA_OK = False

try:
    from fuzzywuzzy import fuzz as _fuzz
    _FUZZ_OK = True
except ImportError:
    try:
        from rapidfuzz import fuzz as _fuzz  # drop-in replacement
        _FUZZ_OK = True
    except ImportError:
        _fuzz = None
        _FUZZ_OK = False

try:
    from rouge import Rouge as _Rouge
    _ROUGE_OK = True
except ImportError:
    _Rouge = None
    _ROUGE_OK = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    def white_space_fix(text):
        return "".join(text.split())
    def remove_punc(text):
        cn_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        all_punc = set(string.punctuation + cn_punc)
        return "".join(ch for ch in text if ch not in all_punc)
    return white_space_fix(remove_punc(s.lower()))


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def qa_f1_score(prediction, ground_truth, **kwargs):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    if not _JIEBA_OK:
        return qa_f1_score(prediction, ground_truth)  # graceful fallback
    prediction_tokens = [normalize_zh_answer(t) for t in _jieba.cut(prediction, cut_all=False)]
    ground_truth_tokens = [normalize_zh_answer(t) for t in _jieba.cut(ground_truth, cut_all=False)]
    prediction_tokens = [t for t in prediction_tokens if len(t) > 0]
    ground_truth_tokens = [t for t in ground_truth_tokens if len(t) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


def rouge_score(prediction, ground_truth, **kwargs):
    if not _ROUGE_OK:
        # Lightweight fallback: unigram F1 as a proxy for ROUGE-L
        return qa_f1_score(prediction, ground_truth)
    rouge = _Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        return scores["rouge-l"]["f"]
    except Exception:
        return 0.0


def rouge_zh_score(prediction, ground_truth, **kwargs):
    if not _JIEBA_OK:
        return rouge_score(prediction, ground_truth)
    prediction = " ".join(list(_jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(_jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs.get("all_classes", []) or []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0] if matches else ""
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if n == ground_truth_id)
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0] if matches else ""
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if n == ground_truth_id)
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if n == str(ground_truth))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    pred_line = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            pred_line = line
            break
    if not _FUZZ_OK:
        # Fallback: character-level overlap
        common = sum(1 for a, b in zip(pred_line, ground_truth) if a == b)
        max_len = max(len(pred_line), len(ground_truth), 1)
        return common / max_len
    return _fuzz.ratio(pred_line, ground_truth) / 100


def string_match_all(preds, refs):
    """Evaluation metric for RULER."""
    score = sum(
        sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)
        for pred, ref in zip(preds, refs)
    ) / len(preds) * 100
    return round(score, 2)