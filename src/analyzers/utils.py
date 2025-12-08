from typing import List, Tuple

try:
    from konlpy.tag import Okt
except ImportError:
    print("Warning: konlpy is not installed. Please install it using 'pip install konlpy' for full functionality.")
    Okt = None

okt = Okt() if Okt else None

def pos_tag(text: str) -> List[Tuple[str, str]]:
    """
    주어진 텍스트에 대해 형태소 분석 및 품사 태깅을 수행합니다.
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        (형태소, 품사) 튜플의 리스트
    """
    if not okt:
        raise ImportError("Okt tagger is not available. Please install konlpy.")
    return okt.pos(text, stem=True)

def get_morphs(text: str) -> List[str]:
    """
    주어진 텍스트에서 형태소만 추출합니다.
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        형태소 문자열의 리스트
    """
    if not okt:
        raise ImportError("Okt tagger is not available. Please install konlpy.")
    return okt.morphs(text, stem=True)

def get_pos(text: str) -> List[str]:
    """
    주어진 텍스트에서 품사만 추출합니다.
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        품사 문자열의 리스트
    """
    if not okt:
        raise ImportError("Okt tagger is not available. Please install konlpy.")
    return [tag for _, tag in okt.pos(text, stem=True)]