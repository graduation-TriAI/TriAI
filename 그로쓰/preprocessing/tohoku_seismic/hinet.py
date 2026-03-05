import os
import glob
from HinetPy import win32
import obspy
import tempfile
tempfile.tempdir = 

# 1. 경로 설정 (사용자 지정 폴더)
base_dir = r"C:\Github\TriAI\hi-net_data"
ch_file = os.path.join(base_dir, "01_01_20110311.sjis.ch")  # 해독기 파일
temp_sac_dir = os.path.join(base_dir, "temp_sac")
output_mseed = os.path.join(base_dir, "tohoku_merged_45min.mseed")

# 2. .cnt 파일 목록 확보
cnt_files = sorted(glob.glob(os.path.join(base_dir, "*.cnt")))
print(f"총 {len(cnt_files)}개의 파일을 찾았습니다. 변환을 시작합니다.")

if not os.path.exists(temp_sac_dir):
    os.makedirs(temp_sac_dir)

# 3. 루프를 돌며 SAC로 변환
for cnt in cnt_files:
    print(f"처리 중: {os.path.basename(cnt)}")
    # 주의: 윈도우에 win32tools가 설치되어 있어야 실행
    win32.extract_sac(cnt, ch_file, outdir=temp_sac_dir)

# 4. 추출된 모든 SAC 파일을 하나로 읽기
print("데이터를 하나로 합치는 중입니다...")
st = obspy.read(os.path.join(temp_sac_dir, "*"))

# 5. 시간 순서대로 병합 (Merge)
# 1분씩 끊긴 파형을 하나의 긴 선으로 이어붙입니다.
st.merge(method=1, fill_value='interpolate')

# 6. 최종 mseed 파일로 저장
st.write(output_mseed, format="MSEED")

print("-" * 30)
print(f"완료! 통합 파일이 저장되었습니다: \n{output_mseed}")