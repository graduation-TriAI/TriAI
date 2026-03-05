import pandas as pd
import numpy as np

def upsampling(df, east_col, north_col, up_col, time_col="Date/Time", target_fs=100):
    # 필요한 열만 추출, 숫자/시간형 변환
    df_processed = df[[time_col, east_col, north_col, up_col]].copy()
    df_processed[time_col] = pd.to_datetime(df_processed[time_col], errors='coerce')

    for col in [east_col, north_col, up_col]:
        df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
        
    df_processed = df_processed.dropna()
    
    if len(df_processed) < 2:
        return np.empty((0, 3), dtype=np.float32)
    
    # 시간을 인덱스로 설정
    df_processed = df_processed.set_index(time_col)
    df_processed = df_processed[~df_processed.index.duplicated(keep='first')]

    # 100Hz 간격으로 업샘플링 및 cubic 보간
    resampling = f"{int(1000/target_fs)}ms"
    df_upsampled = df_processed.resample(resampling).interpolate(method='cubic').dropna()

    # (N, 3) 데이터 배열만 추출해서 반환
    data = df_upsampled[[east_col, north_col, up_col]].to_numpy(dtype=np.float32)

    return data
