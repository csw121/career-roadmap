import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="2025 개발자 커리어 로드맵 & AI 분석",
    page_icon="🧭",
    layout="wide"
)

# ---------------------------------------------------------
# 2. 한글 폰트 설정 (그래프 깨짐 방지)
# ---------------------------------------------------------
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif system_name == 'Darwin': # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
else: # Linux (Colab, Docker 등)
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 3. 데이터 로드 (CSV 모드)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # CSV 파일 로드 (인코딩 문제시 encoding='cp949' 추가)
        df = pd.read_csv('2025survey_results_public.csv')
        return df
    except Exception as e:
        return None

df = load_data()

# ---------------------------------------------------------
# 4. 사이드바: 데이터 확인 및 직무 선택
# ---------------------------------------------------------
with st.sidebar:
    st.header("📂 설정 및 선택")

    # 파일이 없을 경우 업로드 버튼 제공
    if df is None:
        st.warning("파일을 찾을 수 없습니다.")
        uploaded_file = st.file_uploader("CSV 파일(.csv)을 업로드해주세요", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.stop() # 파일 없으면 여기서 멈춤
    else:
        st.success("✅ 데이터 로드 성공! (CSV)")

    st.divider()
    st.subheader("🎯 직무 선택")

    # 직무 목록 만들기 (DevType 컬럼 분리)
    if 'DevType' in df.columns:
        all_jobs = df['DevType'].dropna().astype(str).str.split(';').explode().str.strip().unique()
        all_jobs = sorted([job for job in all_jobs if job.lower() != 'nan'])
    else:
        st.error("'DevType' 컬럼이 없습니다.")
        st.stop()

    # 직무 선택 박스
    # 기본값으로 'Developer, back-end'가 있으면 선택
    default_index = all_jobs.index('Developer, back-end') if 'Developer, back-end' in all_jobs else 0
    target_job = st.selectbox(
        "분석할 직무를 고르세요:",
        all_jobs,
        index=default_index
    )

    # 선택된 직무로 데이터 필터링
    job_df = df[df['DevType'].astype(str).str.contains(target_job, case=False, na=False, regex=False)]
    respondents = len(job_df)

    st.markdown(f"--- \n👥 분석 대상: **{respondents:,}명**")

# ---------------------------------------------------------
# 5. 메인 화면 구성 (탭으로 기능 분리)
# ---------------------------------------------------------
st.title(f"🧭 [{target_job}] 커리어 인사이트")

# 3개의 탭 생성
tab1, tab2, tab3 = st.tabs(["📊 기술 트렌드 (로드맵)", "🤖 AI 인식 (위협/불만)", "🧠 ML 심화 분석 (군집/추천)"])

# =========================================================
# [TAB 1] 기술 스택 트렌드 (Current vs Future)
# =========================================================
with tab1:
    st.markdown("### 1️⃣ 기술 스택 로드맵 (현재 vs 미래)")

    tech_cols = {
        '💻 언어': ('LanguageHaveWorkedWith', 'LanguageWantToWorkWith'),
        '🗄️ DB': ('DatabaseHaveWorkedWith', 'DatabaseWantToWorkWith'),
        '☁️ 플랫폼': ('PlatformHaveWorkedWith', 'PlatformWantToWorkWith'),
        '🤖 AI 모델': ('AIModelsHaveWorkedWith', 'AIModelsWantToWorkWith')
    }

    # 상위 N개 추출 함수
    def get_top_skills(data, col, n=7):
        if col not in data.columns: return pd.Series()
        return data[col].dropna().astype(str).str.split(';').explode().str.strip().value_counts().head(n)

    for i, (name, (curr, want)) in enumerate(tech_cols.items()):
        c1, c2 = st.columns(2)

        # 왼쪽: 현재 (파란색)
        with c1:
            st.markdown(f"#### 🟦 현재 (필수 기술) - {name}")
            top_curr = get_top_skills(job_df, curr)
            if not top_curr.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=top_curr.values, y=top_curr.index, ax=ax, palette='Blues_r')
                ax.set_xlabel("사용자 수")
                ax.bar_label(ax.containers[0], fmt='%d', padding=3)
                st.pyplot(fig)
            else:
                st.info("데이터 없음")

        # 오른쪽: 미래 (초록색)
        with c2:
            st.markdown(f"#### 🟩 미래 (성장 기술) - {name}")
            top_want = get_top_skills(job_df, want)
            if not top_want.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=top_want.values, y=top_want.index, ax=ax, palette='Greens_r')
                ax.set_xlabel("희망자 수")
                ax.bar_label(ax.containers[0], fmt='%d', padding=3)
                st.pyplot(fig)
            else:
                st.info("데이터 없음")
        st.divider()

# =========================================================
# [TAB 2] AI 수용 태도 (위협 인식 & 불만 사항)
# =========================================================
with tab2:
    st.markdown("### 2️⃣ 이 직무의 AI 수용 태도")
    c1, c2 = st.columns(2)

    # (1) AI 위협 인식 (파이 차트)
    with c1:
        st.markdown("##### 😨 AI를 위협으로 느끼나요?")
        if 'AIThreat' in job_df.columns:
            threat = job_df['AIThreat'].value_counts()
            if not threat.empty:
                fig, ax = plt.subplots(figsize=(5, 5))
                colors = {'Yes': '#ff9999', "I'm not sure": '#d3d3d3', 'No': '#99ff99'}
                pie_cols = [colors.get(x, '#abcdef') for x in threat.index]

                ax.pie(threat, labels=threat.index, autopct='%1.1f%%', startangle=90, colors=pie_cols)
                st.pyplot(fig)
            else:
                st.info("데이터 없음")
        else:
            st.warning("'AIThreat' 컬럼이 없습니다.")

    # (2) AI 불만 사항 (막대 차트)
    with c2:
        st.markdown("##### 😤 AI 도구의 가장 큰 불만은?")
        if 'AIFrustration' in job_df.columns:
            frust = job_df['AIFrustration'].dropna().astype(str).value_counts().head(5)
            if not frust.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=frust.values, y=frust.index, ax=ax, palette='Reds_r')
                ax.set_xlabel("응답 수")
                st.pyplot(fig)
            else:
                st.info("데이터 없음")
        else:
            st.warning("'AIFrustration' 컬럼이 없습니다.")

# =========================================================
# [TAB 3] ML 심화 분석 (군집화 & 연관 추천)
# =========================================================
with tab3:
    st.markdown("### 3️⃣ 머신러닝 심화 분석")

    # 1. K-Means 군집 분석 (연봉 vs 경력)
    st.subheader("📊 개발자 유형 군집화 (K-Means)")
    st.info("💡 경력과 연봉 데이터를 기반으로 개발자들을 **3가지 그룹**으로 자동 분류합니다.")

    # 데이터 준비
    ml_data = job_df[['YearsCode', 'ConvertedCompYearly']].dropna().copy()

    # 경력 문자열 -> 숫자 변환
    def clean_years(x):
        if x == 'Less than 1 year': return 0.5
        if x == 'More than 50 years': return 50
        try: return float(x)
        except: return 0

    ml_data['YearsCode_Num'] = ml_data['YearsCode'].apply(clean_years)
    ml_data = ml_data[ml_data['ConvertedCompYearly'] < 300000] # 이상치 제거

    if len(ml_data) > 30:
        # 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(ml_data[['YearsCode_Num', 'ConvertedCompYearly']])

        # 모델 학습 (K=3)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        ml_data['Cluster'] = kmeans.fit_predict(X_scaled)

        # 시각화
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=ml_data, x='YearsCode_Num', y='ConvertedCompYearly',
            hue='Cluster', palette='viridis', s=60, ax=ax
        )
        ax.set_title(f"[{target_job}] 개발자 그룹 분포", fontsize=15)
        ax.set_xlabel("경력 (년)")
        ax.set_ylabel("연봉 (USD)")
        st.pyplot(fig)
        st.caption("* 색깔이 다른 점들은 AI가 분류한 '비슷한 성향의 그룹'입니다.")
    else:
        st.warning("데이터가 부족하여 군집 분석을 할 수 없습니다.")

    st.divider()

    # 2. 기술 연관 분석
    st.subheader("🔗 기술 추천 (Association Analysis)")

    # 언어 데이터 추출
    langs = job_df['LanguageHaveWorkedWith'].dropna().astype(str).str.split(';')
    all_langs = sorted(list(set([l for sublist in langs for l in sublist])))

    # 사용자 입력
    selected_lang = st.selectbox("어떤 언어를 주로 사용하시나요?", all_langs, index=0 if not all_langs else 0)

    # 연관 기술 찾기
    related_skills = {}
    for user_skills in langs:
        if selected_lang in user_skills:
            for skill in user_skills:
                if skill != selected_lang:
                    related_skills[skill] = related_skills.get(skill, 0) + 1

    # Top 5 시각화
    if related_skills:
        sorted_skills = sorted(related_skills.items(), key=lambda x: x[1], reverse=True)[:5]
        skill_names = [x[0] for x in sorted_skills]
        skill_counts = [x[1] for x in sorted_skills]

        fig, ax = plt.subplots(figsize=(8, 3))
        sns.barplot(x=skill_counts, y=skill_names, palette='magma')
        ax.set_title(f"'{selected_lang}' 사용자가 함께 쓰는 기술 Top 5")
        st.pyplot(fig)
    else:
        st.info("연관된 데이터가 없습니다.")

# ---------------------------------------------------------
# 6. 마무리 조언
# ---------------------------------------------------------
st.divider()
st.success(f"🎓 **{target_job} 취업 전략:** 파란색 그래프(현재)로 기본기를 다지고, 초록색 그래프(미래) 기술을 익혀 경쟁력을 확보하세요!")