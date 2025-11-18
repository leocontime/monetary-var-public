import pandas as pd
import numpy as np
from datetime import datetime
import os 

# 🚨 분리된 함수 파일을 가져옵니다.
from data_loader import load_and_preprocess_data 
from var_model import run_var_analysis
from report_generator import generate_forecast_report, generate_individual_report_html 

# ==============================================================================
# 메인 실행 블록
# ==============================================================================
if __name__ == "__main__":
    
    # 🌟 메인 파일에서 환경 변수 설정
    REPORT_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_VAR')
    REPORT_DIR = f'./dist/reports/{REPORT_TIMESTAMP}'
    DATA_ARCHIVE_DIR = './dist/data_archive'
    SUMMARY_CSV_PATH = './dist/report_summaries_VAR.csv' 
    
    try:
        os.makedirs('./dist', exist_ok=True)
        os.makedirs(REPORT_DIR, exist_ok=True) 
        os.makedirs(DATA_ARCHIVE_DIR, exist_ok=True)
        
        # 1. 데이터 로드 및 전처리
        data_for_var = load_and_preprocess_data() 
        
        if data_for_var.empty:
            raise ValueError("❌ 데이터 수집 후 유효한 관측치가 없습니다.")

        last_date_for_report = data_for_var.index[-1]
        
        DATA_ARCHIVE_PATH = f'{DATA_ARCHIVE_DIR}/{REPORT_TIMESTAMP}_data.csv'
        data_for_var.to_csv(DATA_ARCHIVE_PATH, index=True)
        print(f"✅ 1.1. 전처리된 데이터 '{DATA_ARCHIVE_PATH}' 저장 완료.")

        
        # 2. VAR 모델 추정 및 IRF/FEVD 분석
        # var_model.py의 run_var_analysis는 var_results, optimal_lag 2개만 반환합니다.
        var_base_results, optimal_lag = run_var_analysis(data_for_var, REPORT_DIR)
        
        # 3. 예측 및 텍스트 보고서 생성
        # 🚨 수정됨: optimal_lag을 report_generator.py로 전달
        metrics = generate_forecast_report(var_base_results, REPORT_DIR, optimal_lag) 
        
        # 4. HTML 보고서 생성
        generate_individual_report_html(optimal_lag, REPORT_DIR, REPORT_TIMESTAMP)

        
        # 5. CSV 요약 저장
        summary_data = {
            'report_date': [REPORT_TIMESTAMP], 'base_date': [last_date_for_report.strftime('%Y-%m-%d')], 'optimal_lag': [optimal_lag], 
            'last_m_t_obs': [metrics['last_m_t_change']], 'forecast_m_t_change': [metrics['m_t_forecast_point']], 
            'r_squared': [metrics['r_squared']], 'rmse_mt': [metrics['m_t_rmse']], 
            'model_type': ['VAR_4_vars_Cholesky_2008_AutoLag'] # 모델 타입 이름도 자동 선택에 맞게 변경
        }
        summary_df = pd.DataFrame(summary_data)
        
        if not os.path.exists(SUMMARY_CSV_PATH):
            summary_df.to_csv(SUMMARY_CSV_PATH, index=False, encoding='utf-8-sig') 
        else:
            try:
                existing_df = pd.read_csv(SUMMARY_CSV_PATH, encoding='utf-8-sig')
                if not existing_df['report_date'].eq(REPORT_TIMESTAMP).any():
                    summary_df.to_csv(SUMMARY_CSV_PATH, mode='a', header=False, index=False, encoding='utf-8-sig')
            except:
                 summary_df.to_csv(SUMMARY_CSV_PATH, index=False, encoding='utf-8-sig') 


        print(f"\n🎉 모든 VAR 분석 및 보고서 생성이 완료되었습니다. dist/reports/{REPORT_TIMESTAMP} 폴더를 확인해주세요.")

    except Exception as e:
        print(f"\n❌ 치명적인 오류가 발생했습니다: {e}")
