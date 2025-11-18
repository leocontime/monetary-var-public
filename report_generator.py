import pandas as pd
import numpy as np
import os 
import io 
from contextlib import redirect_stdout 

# ==============================================================================
# 전역 상수 정의 (보고서 관련)
# ==============================================================================
START_DATE = '2008-01-01' 
VAR_ORDER = ['M_t', 'Y_t', 'pi_t', 'r_t']
RECENT_PERIODS = 20 # 최근 데이터 표시 기간 (20개 분기로 확대됨)

# ==============================================================================
# 예측 및 보고서 생성 함수
# ==============================================================================
def generate_forecast_report(var_base_results, REPORT_DIR, optimal_lag):
    
    m_t_forecast_point, m_t_lower_ci, m_t_upper_ci = 0.0, 0.0, 0.0
    r_squared, m_t_rmse, last_m_t_change = 0.0, 0.0, 0.0
    
    m_t_idx = VAR_ORDER.index('M_t')

    # 인덱스 추출 안전화
    endog_array = np.array(var_base_results.endog)
    endog_index = var_base_results.model.data.orig_endog.index 
    
    # --- 예측 및 통계 계산 ---
    try:
        forecast_horizon = 1
        forecasts = var_base_results.forecast(var_base_results.endog, steps=forecast_horizon)
        forecast_ci = var_base_results.forecast_interval(var_base_results.endog, steps=forecast_horizon, alpha=0.05) 
        
        m_t_forecast_point = forecasts[0, m_t_idx] 
        m_t_lower_ci = forecast_ci[0][0, m_t_idx] 
        m_t_upper_ci = forecast_ci[2][0, m_t_idx]  
        
        # R-squared 계산
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

    
    # --- 최근 데이터 표시 (20개 분기) ---
    full_m_t_series = pd.Series(data=endog_array[:, m_t_idx], index=endog_index)
    
    if len(full_m_t_series) >= RECENT_PERIODS:
        recent_series = full_m_t_series.tail(RECENT_PERIODS)
    else:
        recent_series = full_m_t_series
        
    recent_dates = recent_series.index.strftime('%Y-%m-%d').tolist()
    recent_values = recent_series.values.tolist()
    
    html_rows = "".join([
        f"<tr><td style=\"text-align: center;\">{date}</td><td style=\"text-align: center;\">{value:.2f}%</td></tr>"
        for date, value in zip(recent_dates, recent_values)
    ])
    
    # 🚨 모바일 반응형: table-responsive 클래스 적용
    recent_data_html = f"""
        <div class="table-responsive">
            <table class="table table-bordered table-sm mx-auto" style="max-width: 400px;">
                <thead><tr><th style="text-align: center;\">Date</th><th style="text-align: center;\">M_t Change (%)</th></tr></thead>
                <tbody>{html_rows}</tbody>
            </table>
        </div>
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
        forecast_horizon = 1
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
        
        fevd_warning = "✅ Cholesky 분해 기반 분산 기여도 계산 완료. (외부 충격 0%는 $M_t$ 우선 순위 가정에 따른 1분기 예상 결과입니다.)"
    except Exception as e:
        fevd_warning = f"❌ FEVD 계산 오류: {e}. 보고서에 N/A로 표시됩니다."
    
    # --- 보고서 텍스트 생성 ---
    # 🚨 모바일 반응형: table-responsive 클래스 적용
    fevd_text = f"""<h3 id="fevd">📊 분산 분해 (Cholesky FEVD) 요약 (1분기 후)</h3><p class="text-center-p">{fevd_warning}</p><p class="text-muted text-center-p">(주요 **Cholesky 충격**이 $M_t$의 예측 오차 분산에 기여하는 정도)</p><div class="table-responsive"><table class="table table-bordered table-striped w-100 mx-auto" style="max-width: 800px;"><thead><tr><th style="text-align: center;">Rank</th><th style="text-align: center;">Structural Shock</th><th style="text-align: center;">Contribution to $M_t$ Variance (1Q)</th></tr></thead><tbody><tr><td style="text-align: center;">1</td><td style="text-align: center;">**{var1}**</td><td style="text-align: center;">{percent1:.2f}%</td></tr><tr><td style="text-align: center;">2</td><td style="text-align: center;">**{var2}**</td><td style="text-align: center;">{percent2:.2f}%</td></tr><tr><td style="text-align: center;">3</td><td style="text-align: center;">**{var3}**</td><td style="text-align: center;">{percent3:.2f}%</td></tr></tbody></table></div>"""
    forecast_section = f"""<h3 id="forecast">📈 통화량 예측 결과</h3><div class="table-responsive"><table class="table table-bordered table-striped w-100 mx-auto" style="max-width: 800px;"><thead><tr><th style="text-align: center;">Metric</th><th style="text-align: center;">Value</th><th style="text-align: center;">95% CI (Lower)</th><th style="text-align: center;">95% CI (Upper)</th></tr></thead><tbody><tr><td style="text-align: center;">예측 변동률</td><td style="text-align: center;">**{m_t_forecast_point:.2f}%**</td><td style="text-align: center;">{m_t_lower_ci:.2f}%</td><td style="text-align: center;">{m_t_upper_ci:.2f}%</td></tr></tbody></table></div><p class="text-center-p">**해석:** 모형은 통화량이 다음 분기에 **{m_t_forecast_point:.2f}%** 변동할 것으로 예상하며, 95% 확률로 변동 폭은 {m_t_lower_ci:.2f}%에서 {m_t_upper_ci:.2f}% 사이에 위치합니다.</p>"""
    reliability_section = f"""<h3 id="reliability">🛡️ 모형 신뢰도 지표 및 예측 성능 분석</h3><ul style="list-style-type: none; padding: 0; text-align: center;"><li>**모델 유형:** Cholesky 분해 VAR</li><li>**적용된 차수:** {optimal_lag}차</li><li>**$M_t$ 방정식 설명력 ($\mathbf{{R^2}}$):** {r_squared:.4f}</li></ul>"""
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


def generate_individual_report_html(optimal_lag, REPORT_DIR, REPORT_TIMESTAMP):
    """최종 HTML 파일을 생성합니다."""
    
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

    # 🚨 HTML 템플릿: 모바일 대응 CSS 클래스와 메뉴 링크 적용
    html_content = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no"><title>{REPORT_TIMESTAMP} 통화 Cholesky VAR 분석 보고서 (4-변수, Lag {optimal_lag})</title><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"><style>/* CSS 스타일 생략 */</style></head><body><div class="container-fluid"><main class="main-content col-12 px-md-4"><p class="my-3"><a href="../index.html" class="btn btn-sm btn-outline-secondary">⬅️ 전체 대시보드로 돌아가기</a></p><h1 class="h2 text-left">💰 Cholesky VAR 분석 리포트 (4-변수, Lag {optimal_lag})</h1><h2 id="report_title" class="mb-4">{REPORT_TIMESTAMP} 분석 결과</h2><h3 id="forecast">🎯 예측 보고서</h3><div class="table-responsive"><pre class="report-box" style="white-space: pre-wrap; word-wrap: break-word; background: #e9f5ff; border: 1px solid #b8daff; padding: 20px;">{FORECAST_REPORT}</pre></div><h3 id="irf">📈 충격반응함수 (IRF)</h3><p class="text-center-p">Cholesky 충격에 대한 M_t (통화량)의 동태적 반응입니다. (4개 변수 충격)</p><div class="text-center"><img src="var_irf_results.png" alt="VAR Model Impulse Response Functions" class="img-fluid" style="max-width: 100%; height: auto;"></div><h3 id="summary">📑 모형 요약 및 통계</h3><div class="table-responsive"><pre style="white-space: pre-wrap; word-wrap: break-word;">{SUMMARY_CONTENT}</pre></div><footer class="mt-5 pt-3 border-top text-center-p text-muted"><p>분석 업데이트 일자: {REPORT_TIMESTAMP}</p></footer></main></div><script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script></body></html>"""
    with open(os.path.join(REPORT_DIR, 'index.html'), 'w', encoding='utf-8') as file:
        file.write(html_content)
        
    print(f"✅ 6. 개별 보고서 HTML '{REPORT_DIR}/index.html' (Lag {optimal_lag}) 생성 완료.")
