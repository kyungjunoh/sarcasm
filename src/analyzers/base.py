from abc import ABC, abstractmethod
from typing import List, Any, Dict
import pandas as pd


class BaseAnalyzer(ABC):
    """분석기 기본 클래스"""

    @abstractmethod
    def analyze(self, *args, **kwargs):
        """데이터를 분석하는 메서드"""
        pass

    @abstractmethod
    def visualize(self, *args, **kwargs):
        """분석 결과를 시각화하는 메서드"""
        pass

    def get_statistics(self, *args, **kwargs) -> Dict:
        """분석 결과에 대한 통계를 반환하는 메서드"""
        return {}

    def get_distinctive_features(self, *args, **kwargs) -> pd.DataFrame:
        """분석 결과에서 특징적인 피처를 데이터프레임으로 반환하는 메서드"""
        return pd.DataFrame()