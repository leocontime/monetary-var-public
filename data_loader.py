import pandas as pd
import numpy as np
import requests 
from datetime import datetime

# ==============================================================================
# 전역 상수 정의 (데이터 관련)
# ==============================================================================
FRED_API_KEY = '8e306e0c6fb0066a4f1922508eb5f0f4' 
START_DATE = '2008-01-01' 

# FRED Code와 전처리 방식 매핑
variable_map = {
    'GDPC1': 'log_diff', 'CPIAUCSL': 'log_diff', 'FEDFUNDS': 'log_diff', 'M2SL': 'log_diff',
}
fred_codes = list(variable_map.keys())

# 변수 이름 매핑 (보고서용)
name_mapping = {
    'GDPC1': 'Y_t', 'CPIAUCSL': 'pi_t', 'FEDFUNDS': 'r_t', 'M2SL': 'M_t',
}

# VAR 변수 순서 (데이터프레임 순서 지정 및 Cholesky 순서)
VAR_ORDER = ['M_t', 'Y_t', 'pi_t', 'r_t']


# ==============================================================================
# 데이터 로딩/전처리 함수
# ==============================================================================
def get_fred_data(series_id, start_date):
    """FRED API를 통해 데이터를 분기별로 조회합니다."""
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id, 'api_key': FRED_API_KEY, 'file_type': 'json',
        'observation_start': start_date, 'frequency': 'q', 'sort_order': 'asc'
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"API Request Failed for {series_id} (Status: {response.status_code})")
    
    data = response.json()
    if 'observations' not in data:
         return pd.Series([], dtype=float, name=series_id) 
    
    dates = [obs['date'] for obs in data['observations'] if obs['value'] != '.']
    values = [float(obs['value']) for obs in data['observations'] if obs['value'] != '.']
    
    return pd.Series(values, index=pd.to_datetime(dates), name=series_id)


def load_and_preprocess_data():
    """FRED 데이터를 직접 API 호출로 불러오고 전처리합니다."""
    print("✅ 1. 데이터 수집 및 전처리 시작...")
    
    all_data = {}
    for code in fred_codes:
        try:
            all_data[code] = get_fred_data(code, START_DATE)
        except Exception as e:
            print(f"❌ 데이터 로드 실패 for {code}: {e}")
            raise 

    data = pd.DataFrame(all_data)
    data_resampled = data.resample('QE').last().ffill().bfill() 
    data = data_resampled.copy()
    final_var_data = pd.DataFrame(index=data.index)
    
    for code, method in variable_map.items():
        series = data.get(code)
        if series is None: continue
            
        var_col_name = name_mapping.get(code, code)
            
        if method == 'log_diff':
            final_var_data[var_col_name] = np.log(series).diff() * 100 
        elif method == 'level':
            final_var_data[var_col_name] = series
            
    return final_var_data.dropna()

if __name__ == "__main__":
    # 테스트 목적
    var_data = load_and_preprocess_data()
    print(var_data.tail())
