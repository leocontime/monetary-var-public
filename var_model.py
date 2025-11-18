import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import os 
import io 
from contextlib import redirect_stdout 

# ==============================================================================
# 전역 상수 정의 (모델링 관련)
# VAR_ORDER는 data_loader.py에서 정의된 것과 동일하게 유지
# ==============================================================================
VAR_ORDER = ['M_t', 'Y_t', 'pi_t', 'r_t']

# 폰트 설정 (matplotlib 사용을 위해 필요)
try:
    plt.rcParams['font.family'] = 'NanumGothic' 
except:
    plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['axes.unicode_minus'] = False 


# ==============================================================================
# VAR 추정 및 Cholesky IRF/FEVD 분석 함수
# ==============================================================================
def run_var_analysis(data_var, REPORT_DIR):
    """VAR 모형을 추정하고 Cholesky 분해를 적용하여 IRF를 분석 및 저장합니다."""
    
    # 순서에 따라 데이터프레임 정렬
    data_var_ordered = data_var[VAR_ORDER].copy() 
    data_var_ordered.columns = np.array(VAR_ORDER, dtype=object)
    
    print("✅ 2. VAR 모형 (기반) 추정 시작...")
    model = VAR(data_var_ordered) 
    print("✅ 2.1. VAR 모형 최적 차수(Lag) 검색 (BIC 기준, maxlags=4)...")
    
    # 최적 차수 자동 선택 (BIC)
    try:
        selection = model.select_order(maxlags=4)
        optimal_lag = selection.bic 
        print(f"✅ 최적 차수(BIC 기준): {optimal_lag}차로 선택되었습니다.")
    except Exception as e:
        optimal_lag = 1
        print(f"❌ 최적 차수 검색 중 오류 발생: {e}. 기본값 {optimal_lag}차를 사용합니다.")
    
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
    
    return var_results, optimal_lag
