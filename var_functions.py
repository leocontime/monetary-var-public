import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.svar_model import SVAR 
import requests 
from datetime import datetime
import os 
import io 
from contextlib import redirect_stdout 

# ==============================================================================
# 폰트 설정 및 전역 변수 정의
# ==============================================================================
try:
    plt.rcParams['font.family'] = 'NanumGothic' 
except:
    plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['axes.unicode_minus'] = False 

FRED_API_KEY = '8e306e0c6fb0066a4f1922508eb5f0f4' 
START_DATE = '2008-01-01' 

variable_map = {
    'GDPC1': 'log_diff', 'CPIAUCSL': 'log_diff', 'FEDFUNDS': 'log_diff', 'M2SL': 'log_diff',
}
fred_codes = list(variable_map.keys())
name_mapping = {
    'GDPC1': 'Y_t', 'CPIAUCSL': 'pi_t', 'FEDFUNDS': 'r_t', 'M2SL': 'M_t',
}
VAR_ORDER = ['M_t', 'Y_t', 'pi_t', 'r_t']


# ==============================================================================
# 1. 데이터 로딩/전처리 함수
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

# ==============================================================================
# 2. VAR 추정 및 Cholesky IRF/FEVD 분석 함수
# ==============================================================================
def run_var_analysis(data_var, REPORT_DIR):
    """VAR 모형을 추정하고 Cholesky 분해를 적용하여 IRF를 분석 및 저장합니다."""
    
    data_var_ordered = data_var[VAR_ORDER].copy() 
    data_var_ordered.columns = np.array(VAR_ORDER, dtype=object)
    
    print("✅ 2. VAR 모형 (기반) 추정 시작...")
    model = VAR(data_var_ordered) 
    print("✅ 2.1. VAR 모형 최적 차수(Lag) 검색 (BIC 기준, maxlags=4)...")
    
    # 🌟🌟🌟 핵심 수정: 최적 차수 자동 선택 (BIC)
    try:
        selection = model.select_order(maxlags=4)
        optimal_lag = selection.bic 
        print(f"✅ 최적 차수(BIC 기준): {optimal_lag}차로 선택되었습니다.")
    except Exception as e:
        # 오류 발생 시 기본값 사용
        optimal_lag = 1
        print(f"❌ 최적 차수 검색 중 오류 발생: {e}. 기본값 {optimal_lag}차를 사용합니다.")
    # 🌟🌟🌟
    
    var_results = model.fit(optimal_lag, trend='c') 
    print(f"✅ VAR 모형 (자동 선택): {optimal_lag}차로 추정 완료.") 
    print("✅ 2.2. 비구조적 VAR (Cholesky 분해) IRF/FEVD 분석 시작...")
    
    # --- IRF 계산 및 플롯 저장 ---
    try:
        irf = var_results.irf(periods=40)
        response_index_M_t = VAR_ORDER.index('M_t')
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10)) 
        fig.suptitle(f'Cholesky 충격에 대한 M_t 반응 (VAR({optimal_lag}))', fontsize=16) 
        shock_names = [f'Shock: {name}' for name in VAR_ORDER]

        for i, ax in enumerate(axes.flatten()):
            try:
                response_data = irf.irfs[:, i, response_index_M_t] 
            except Exception:
                response_data = np.zeros(40) 

            ax.plot(response_data, label=shock_names[i])
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'{VAR_ORDER[i]} 충격 → M_t')
            ax.set_xlabel('Periods')
            ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        os.makedirs(REPORT_DIR, exist_ok=True)
        fig.savefig(os.path.join(REPORT_DIR, 'var_irf_results.png')) 
        print(f"✅ 3. VAR IRF 분석 결과 '{REPORT_DIR}/var_irf_results.png'로 저장 완료.")
    
    except Exception as e:
        print(f"❌ IRF 계산 또는 플롯 생성 중 오류 발생: {e}. 파일 저장을 건너뜁니다.")
        
    # --- 모델 요약 텍스트 저장 ---
    f = io.StringIO()
    with redirect_stdout(f):
        print(var_results.summary()) 
    summary_text = f.getvalue()

    with open(os.path.join(REPORT_DIR, 'var_model_summary.txt'), 'w', encoding='utf-8') as file:
        file.write(summary_text)
        
    print(f"✅ 4. VAR 모형 요약 '{REPORT_DIR}/var_model_summary.txt'로 저장 완료.")
    
    return var_results, optimal_lag, var_results 

# ==============================================================================
# 3. 예측 및 보고서 생성 함수 (오류 우회 강화)
# ==============================================================================
def generate_forecast_report(var_base_results, REPORT_DIR, optimal_lag): # 🚨🚨🚨 시그니처에 optimal_lag 추가
    
    m_t_forecast_point, m_t_lower_ci, m_t_upper_ci = 0.0, 0.0, 0.0
    r_squared, m_t_rmse, last_m_t_change = 0.0, 0.0, 0.0
    
    m_t_idx = VAR_ORDER.index('M_t')

    # 🚨🚨🚨 인덱스 추출 안전화
    endog_array = np.array(var_base_results.endog)
    endog_index = var_base_results.model.data.orig_endog.index # <-- 안전한 인덱스 추출

    
    # --- 예측 및 통계 계산 ---
    try:
        forecast_horizon = 1
        forecasts = var_base_results.forecast(var_base_results.endog, steps=forecast_horizon)
        forecast_ci = var_base_results.forecast_interval(var_base_results.endog, steps=forecast_horizon, alpha=0.05) 
        
        # 순수 numpy 접근: 예측 통계
        m_t_forecast_point = forecasts[0, m_t_idx] 
        m_t_lower_ci = forecast_ci[0][0, m_t_idx] 
        m_t_upper_ci = forecast_ci[2][0, m_t_idx]  
        
        # 순수 numpy 접근: R-squared 및 RMSE
        resid_arr = np.array(var_base_results.resid)
        rss_col = resid_arr[:, m_t_idx] 
        rss = rss_col.dot(rss_col)
        endog_col = endog_array[:, m_t_idx] 
        tss = (endog_col - endog_col.mean()).dot(endog_col - endog_col.mean())
        if tss != 0: r_squared = 1 - rss / tss
        m_t_rmse = np.sqrt(np.mean(rss_col**2)) 
        last_m_t_change = endog_array[-1, m_t_idx]
        
    except Exception as e:
        print(f"❌ 핵심 예측 통계 계산 중 오류 발생: {e}. 0.0으로 대체합니다.")
        last_m_t_change = endog_array[-1, m_t_idx] if endog_array.size > 0 else 0.0

    
    # 🌟🌟🌟 수정됨: 최근 데이터 기간을 20개 분기로 늘림
    recent_periods = 20 
    
    # Pandas Series를 사용하여 인덱스와 값을 안전하게 결합합니다.
    full_m_t_series = pd.Series(data=endog_array[:, m_t_idx], index=endog_index)
    
    # 마지막 20개 기간 데이터 추출
    if len(full_m_t_series) >= recent_periods:
        recent_series = full_m_t_series.tail(recent_periods)
    else:
        recent_series = full_m_t_series
        
    # 날짜와 값 추출
    recent_dates = recent_series.index.strftime('%Y-%m-%d').tolist()
    recent_values = recent_series.values.tolist()
    
    # 순수 HTML 문자열 생성
    html_rows = "".join([
        f"<tr><td style=\"text-align: center;\">{date}</td><td style=\"text-align: center;\">{value:.2f}%</td></tr>"
        for date, value in zip(recent_dates, recent_values)
    ])
    
    recent_data_html = f"""
        <table class="table table-bordered table-sm mx-auto" style="max-width: 400px;">
            <thead><tr><th style="text-align: center;\">Date</th><th style="text-align: center;\">M_t Change (%)</th></tr></thead>
            <tbody>{html_rows}</tbody>
        </table>
    """
    
    recent_data_section = f"""
        <h3 id="recent_data">🗓️ 최근 {len(recent_values)}개 분기 ($M_t$) 변동률</h3>
        <p class="text-center-p text-muted">(VAR 모형 추정에 사용된 최종 변동률 데이터)</p>
        <div style="max-width: 400px; margin: auto;">{recent_data_html}</div>
    """
    
    # --- FEVD 계산 ---
    fevd_warning = ""
    var1, percent1, var2, percent2, var3, percent3 = ('N/A', 0.0, 'N/A', 0.0, 'N/A', 0.0)

    try:
        fevd_result = var_base_results.fevd(forecast_horizon) 
        fevd_matrix_period_1 = fevd_result.decomp[0] 
        shock_names = [f'Shock: {name}' for name in VAR_ORDER] 
        m_t_idx_local = VAR_ORDER.index('M_t')
        
        m_t_fevd_row = fevd_matrix_period_1[m_t_idx_local, :]
        m_t_fevd_contributions = pd.Series(m_t_fevd_row.flatten(), index=shock_names)
        
        other_shocks_fevd_percent = (m_t_fevd_contributions * 100).sort_values(ascending=False)
        top_fevd_report = other_shocks_fevd_percent.drop(shock_names[m_t_idx_local], errors='ignore').head(3)
        fevd_data_report = list(top_fevd_report.items())
        
        var1, percent1 = fevd_data_report[0] if len(fevd_data_report) > 0 else ('N/A', 0.0)
        var2, percent2 = fevd_data_report[1] if len(fevd_data_report) > 1 else ('N/A', 0.0)
        var3, percent3 = fevd_data_report[2] if len(fevd_data_report) > 2 else ('N/A', 0.0)
        # 🌟🌟🌟 수정됨: 0% 결과가 정상임을 명시
        fevd_warning = "✅ Cholesky 분해 기반 분산 기여도 계산 완료. (외부 충격 0%는 $M_t$ 우선 순위 가정에 따른 1분기 예상 결과입니다.)"
    except Exception as e:
        fevd_warning = f"❌ FEVD 계산 오류: {e}. 보고서에 N/A로 표시됩니다."
    
    # --- 보고서 텍스트 생성 및 파일 저장 (`var_forecast_report.txt`) ---
    fevd_text = f"""<h3 id="fevd">📊 분산 분해 (Cholesky FEVD) 요약 (1분기 후)</h3><p class="text-center-p">{fevd_warning}</p><p class="text-muted text-center-p">(주요 **Cholesky 충격**이 $M_t$의 예측 오차 분산에 기여하는 정도)</p><table class="table table-bordered table-striped w-100 mx-auto" style="max-width: 800px;"><thead><tr><th style="text-align: center;">Rank</th><th style="text-align: center;">Structural Shock</th><th style="text-align: center;">Contribution to $M_t$ Variance (1Q)</th></tr></thead><tbody><tr><td style="text-align: center;">1</td><td style="text-align: center;">**{var1}**</td><td style="text-align: center;">{percent1:.2f}%</td></tr><tr><td style="text-align: center;">2</td><td style="text-align: center;">**{var2}**</td><td style="text-align: center;">{percent2:.2f}%</td></tr><tr><td style="text-align: center;">3</td><td style="text-align: center;">**{var3}**</td><td style="text-align: center;">{percent3:.2f}%</td></tr></tbody></table>"""
    forecast_section = f"""<h3 id="forecast">📈 통화량 예측 결과</h3><table class="table table-bordered table-striped w-100 mx-auto" style="max-width: 800px;"><thead><tr><th style="text-align: center;">Metric</th><th style="text-align: center;">Value</th><th style="text-align: center;">95% CI (Lower)</th><th style="text-align: center;">95% CI (Upper)</th></tr></thead><tbody><tr><td style="text-align: center;">예측 변동률</td><td style="text-align: center;">**{m_t_forecast_point:.2f}%**</td><td style="text-align: center;">{m_t_lower_ci:.2f}%</td><td style="text-align: center;">{m_t_upper_ci:.2f}%</td></tr></tbody></table><p class="text-center-p">**해석:** 모형은 통화량이 다음 분기에 **{m_t_forecast_point:.2f}%** 변동할 것으로 예상하며, 95% 확률로 변동 폭은 {m_t_lower_ci:.2f}%에서 {m_t_upper_ci:.2f}% 사이에 위치합니다.</p>"""
    reliability_section = f"""<h3 id="reliability">🛡️ 모형 신뢰도 지표 및 예측 성능 분석</h3><ul style="list-style-type: none; padding: 0; text-align: center;"><li>**모델 유형:** Cholesky 분해 VAR</li><li>**적용된 차수:** {optimal_lag}차</li><li>**$M_t$ 방정식 설명력 ($\mathbf{{R^2}}$):** {r_squared:.4f}</li></ul>"""
    # 🌟🌟🌟 수정됨: VAR 차수 고정 문구 제거 및 자동 선택 반영
    overview_section = f"""<h3 id="overview">📝 예측 개요</h3><ul style="list-style-type: none; padding: 0; text-align: left;"><li>**분석 기간 시작:** {START_DATE}</li><li>**모델 유형:** **Cholesky 분해 VAR** (비구조적)</li><li>**VAR 차수 선택:** {optimal_lag}차 ($\mathbf{{BIC}}$ 기준 자동 선택)</li><li>**Cholesky 순서:** $\mathbf{{M_t \to Y_t \to \pi_t \to r_t}}$ (충격 해석 순서)</li><li>**모든 변수 변환:** $\mathbf{{log\_diff}}$ (변동률)</li><li>**변수:** $M_t$ (통화량 로그 차분)</li><li>**최근 관측된 $M_t$ 변화율:** **{last_m_t_change:.2f}%**</li></ul>"""
    
    report_text = f"""## 🎯 다음 분기 통화량($M_t$) 예측 보고서

{forecast_section}

{recent_data_section} 

{fevd_text}

{reliability_section}

{overview_section}"""
    
    with open(os.path.join(REPORT_DIR, 'var_forecast_report.txt'), 'w', encoding='utf-8') as file:
        file.write(report_text.strip())
    
    print(f"✅ 5. 예측 및 FEVD 분석 텍스트 '{REPORT_DIR}/var_forecast_report.txt' 저장 완료.")

    return {
        'optimal_lag': optimal_lag,
        'r_squared': r_squared,
        'm_t_forecast_point': m_t_forecast_point,
        'm_t_lower_ci': m_t_lower_ci,
        'm_t_upper_ci': m_t_upper_ci,
        'last_m_t_change': last_m_t_change,
        'm_t_rmse': m_t_rmse
    }

# ==============================================================================
# 4. HTML 보고서 생성 함수
# ==============================================================================
def generate_individual_report_html(optimal_lag, REPORT_DIR, REPORT_TIMESTAMP):
    
    report_text_path = os.path.join(REPORT_DIR, 'var_forecast_report.txt')
    summary_text_path = os.path.join(REPORT_DIR, 'var_model_summary.txt')
    
    FORECAST_REPORT = "예측 보고서 텍스트 파일을 읽을 수 없습니다."
    if os.path.exists(report_text_path):
        with open(report_text_path, 'r', encoding='utf-8') as f:
            FORECAST_REPORT = f.read()

    SUMMARY_CONTENT = "모형 요약 텍스트 파일을 읽을 수 없습니다."
    if os.path.exists(summary_text_path):
        with open(summary_text_path, 'r', encoding='utf-8') as f:
            SUMMARY_CONTENT = f.read()

    # HTML 템플릿
    html_content = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no"><title>{REPORT_TIMESTAMP} 통화 Cholesky VAR 분석 보고서 (4-변수, Lag {optimal_lag})</title><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"><style>/* CSS 스타일 생략 */</style></head><body><div class="d-flex"><nav class="sidebar col-md-3 col-lg-2 d-none d-md-block">...</nav><main class="main-content col-md-9 col-lg-10 ml-sm-auto px-md-4"><h1 class="h2" style="text-align: left;">💰 Cholesky VAR 분석 리포트 (4-변수, Lag {optimal_lag})</h1><h2 id="report_title">{REPORT_TIMESTAMP} 분석 결과</h2><h3 id="forecast">🎯 예측 보고서</h3><pre class="report-box" style="background: #e9f5ff; border: 1px solid #b8daff; padding: 20px;">{FORECAST_REPORT}</pre><h3 id="irf">📈 충격반응함수 (IRF)</h3><p class="text-center-p">Cholesky 충격에 대한 M_t (통화량)의 동태적 반응입니다. (4개 변수 충격)</p><img src="var_irf_results.png" alt="VAR Model Impulse Response Functions"><h3 id="summary">📑 모형 요약 및 통계</h3><pre>{SUMMARY_CONTENT}</pre><footer class="mt-5 pt-3 border-top text-center-p text-muted"><p>분석 업데이트 일자: {REPORT_TIMESTAMP}</p></footer></main></div><script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script></body></html>"""
    with open(os.path.join(REPORT_DIR, 'index.html'), 'w', encoding='utf-8') as file:
        file.write(html_content)
        
    print(f"✅ 6. 개별 보고서 HTML '{REPORT_DIR}/index.html' (Lag {optimal_lag}) 생성 완료.")
