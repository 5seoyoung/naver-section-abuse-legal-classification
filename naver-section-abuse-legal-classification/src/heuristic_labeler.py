"""
Rule-based heuristic labeler for legal-type abusive comments.
"""
import re
from typing import Dict, List, Tuple, Optional

from src.labeling_rules import LABEL_PRIORITY, get_label_name
from src.preprocessing import preprocess_text


RuleMatch = Tuple[str, str]  # (match_type, pattern)


HEURISTIC_RULES: Dict[str, Dict[str, List[str]]] = {
    "D": {  # 협박·위협형
        "keywords": [
            "죽여", "죽인다", "죽여버", "살려둘", "테러", "폭탄", "박살", "없애버",
            "쳐 죽", "죽창", "배때지", "불태워", "총쏴", "때려죽",
        ],
        "regex": [
            r"(죽|살)여\s?버리", r"목\s?을\s?(따|자)", r"불\s?태워",
            r"(살해|살인)\w*", r"(피해|응징)\s?해", r"조져버",
        ],
    },
    "E": {  # 성희롱·성폭력형
        "keywords": [
            "몸매", "야해", "ㄸㅅ", "ㅅㅅ", "성추행", "성폭", "입술", "가슴",
            "골반", "음란", "야동", "야사", "벗겨", "침대", "키스나", "섹스",
            "변태", "야한", "음탕", "성욕", "스킨십", "희롱", "추행",
        ],
        "regex": [
            r"(성\s?(추행|폭력|희롱))", r"야한\s?\w+", r"(벗겨|벗기)",
            r"(강간|성폭행)", r"(ㅅㅅ|ㅈㅈ|ㅂㅂ)\w*", r"찢어버리.*(하|겠)",
        ],
    },
    "A": {  # 명예훼손형
        "keywords": [
            "사기꾼", "비리", "부패", "무능", "거짓말쟁이", "허위", "조작",
            "매국노", "도둑", "범죄자", "깡패", "불법", "탈세", "횡령", "배임",
            "위선자", "추문", "음해", "날조",
        ],
        "regex": [
            r"(비리|부패)\w+", r"(허위|날조)\w+", r"(조작|조작된)",
            r"(거짓말|거짓)\w*", r"범죄\s?(자|집단)",
        ],
    },
    "C": {  # 혐오표현형
        "keywords": [
            "김치녀", "된장녀", "한남", "여혐", "남혐", "틀딱", "잼민",
            "맘충", "빨갱이", "개돼지", "흑형", "짱깨", "쪽바리", "버러지",
            "동성애자", "게이", "레즈", "동양인", "외국인",
        ],
        "regex": [
            r"(여자|남자|노인|학생)\w*(들)?\s?(은|들)은\s?(다|모두)\s?(나쁘|혐오)",
            r"(지역|종교|성별)\w*\s?(차별|혐오)",
            r"(불법체류자|이주민)\s?퇴출",
        ],
    },
    "B": {  # 모욕형
        "keywords": [
            "바보", "멍청", "미친", "정신병자", "쓰레기", "바퀴벌레",
            "저능아", "찌질", "재수없", "꼴값", "쓸모없", "한심", "역겹",
            "저질", "거지같", "또라이", "미친놈", "개같", "병신", "ㅂㅅ", "ㅅㅂ",
        ],
        "regex": [
            r"(바보|멍청|한심)\w*", r"(미친|정신)\w*", r"개(새끼|같은)",
            r"(병신|븅신|ㅂㅅ)", r"(쓰레기|찌질)\w*", r"(역겹|징그러)",
        ],
    },
}


def _match_keywords(text: str, keywords: List[str]) -> List[RuleMatch]:
    matches = []
    for keyword in keywords:
        if keyword.lower() in text:
            matches.append(("keyword", keyword))
    return matches


def _match_regex(text: str, patterns: List[str]) -> List[RuleMatch]:
    matches = []
    for pattern in patterns:
        if re.search(pattern, text):
            matches.append(("regex", pattern))
    return matches


def score_comment(text: str) -> Dict[str, Dict[str, List[RuleMatch]]]:
    """
    Evaluate text against heuristic rules and return matches per label.
    """
    normalized = preprocess_text(text, normalize=True, replace_vulgar=True, tokenize=False)
    normalized = normalized.lower()

    label_matches: Dict[str, Dict[str, List[RuleMatch]]] = {}
    for label, rules in HEURISTIC_RULES.items():
        matches: List[RuleMatch] = []
        matches.extend(_match_keywords(normalized, rules["keywords"]))
        matches.extend(_match_regex(normalized, rules["regex"]))
        if matches:
            label_matches[label] = {"matches": matches}
    return label_matches


def aggregate_scores(label_matches: Dict[str, Dict[str, List[RuleMatch]]]) -> Dict[str, float]:
    """
    Convert matches to confidence scores.
    """
    scores = {}
    for label, data in label_matches.items():
        match_count = len(data["matches"])
        scores[label] = min(1.0, 0.4 + 0.15 * match_count)
    return scores


def heuristic_label(text: str) -> Tuple[str, float, List[RuleMatch]]:
    """
    Assign a heuristic label to text.
    Returns (label, confidence, matches)
    """
    matches = score_comment(text)
    if not matches:
        return "F", 0.1, []

    scores = aggregate_scores(matches)
    best_label = max(scores.items(), key=lambda item: (item[1], LABEL_PRIORITY.get(item[0], 0)))[0]
    return best_label, scores[best_label], matches[best_label]["matches"]


def explain_matches(matches: List[RuleMatch]) -> str:
    """
    Format matched rules for logging/export.
    """
    return "; ".join([f"{match_type}:{pattern}" for match_type, pattern in matches])


def label_comment(text: str) -> Dict[str, str]:
    """
    Convenience function that returns label info for a comment.
    """
    label, confidence, matches = heuristic_label(text)
    return {
        "predicted_label": label,
        "predicted_label_name": get_label_name(label),
        "confidence": round(confidence, 3),
        "rule_matches": explain_matches(matches),
    }

