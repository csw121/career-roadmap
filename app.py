import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import numpy as np
import time
import os
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# [설정] 0. 데이터베이스 정의 (DevNavi 로직용)
# ---------------------------------------------------------
RECOMMENDATION_DB = {
    "interests": {
        "1": {"key": "web", "label": "웹 개발 (Full Stack)", "base_lang": "JavaScript / TypeScript", "desc": "브라우저와 서버를 오가는 만능 개발자가 되는 길입니다."},
        "2": {"key": "ai", "label": "AI / 데이터 사이언스", "base_lang": "Python", "desc": "데이터에서 가치를 창출하고 지능형 모델을 만듭니다."},
        "3": {"key": "mobile", "label": "모바일 앱 개발", "base_lang": "Dart (Flutter) / Swift", "desc": "손 안의 세상을 만드는 앱 개발자입니다."},
        "4": {"key": "system", "label": "시스템 / 백엔드 최적화", "base_lang": "Go / Rust", "desc": "고성능 서버와 인프라를 구축합니다."}
    },
    "goals": {
        "1": {"key": "employment", "label": "취업 (대기업/IT기업)", "bonus": ["알고리즘(Coding Test)", "CS 지식", "대규모 트래픽 처리"]},
        "2": {"key": "startup", "label": "창업 / 서비스 런칭", "bonus": ["빠른 프로토타이핑", "클라우드 배포(AWS)", "마케팅 감각"]},
        "3": {"key": "research", "label": "대학원 / 연구", "bonus": ["논문 리딩", "수학/통계", "영어"]}
    }
}

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="2025 개발자 커리어 로드맵 & AI 분석",
    page_icon="🧭",
    layout="wide"
)

# ---------------------------------------------------------
# 2. 한글 폰트 설정 (깨짐 방지 강화 버전)
# ---------------------------------------------------------
def set_korean_font():
    """OS에 맞는 한글 폰트를 자동으로 찾아서 설정합니다."""
    system_name = platform.system()
    font_path = None

    # 1. OS별 기본 폰트 경로 시도
    if system_name == 'Windows':
        font_path = "C:/Windows/Fonts/malgun.ttf"
    elif system_name == 'Darwin': # Mac
        font_path = "/System/Library/Fonts/AppleGothic.ttf"
    else: # Linux (Streamlit Cloud, Ubuntu 등)
        # 나눔폰트가 설치된 일반적인 경로들 탐색
        possible_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                font_path = p
                break

    # 2. 경로에 폰트가 없으면 시스템 폰트 리스트에서 탐색 (2차 시도)
    if not font_path or not os.path.exists(font_path):
        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        for f in font_list:
            # 파일명이나 경로에 한글 폰트 관련 키워드가 있는지 확인
            if 'Nanum' in f or 'Gothic' in f or 'Batang' in f:
                font_path = f
                break

    # 3. 폰트 적용
    if font_path and os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
        return True, font_name
    else:
        return False, None

# 폰트 설정 실행
font_found, font_name_used = set_korean_font()

# ---------------------------------------------------------
# 3. 데이터 로드 (CSV)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # [주의] 로컬에 '2025survey_results_public.zip' 또는 '.csv' 파일이 있어야 합니다.
        df = pd.read_csv('2025survey_results_public.zip')
        return df
    except Exception as e:
        return None

df = load_data()

# ---------------------------------------------------------
# 4. 사이드바: 데이터 확인 및 직무 선택
# ---------------------------------------------------------
with st.sidebar:
    st.header("📂 설정 및 선택")
    
    # 폰트 디버깅 정보 (문제 발생 시 확인용)
    if font_found:
        st.caption(f"✅ 한글 폰트 적용됨: {font_name_used}")
    else:
        st.error("⚠️ 한글 폰트를 찾지 못했습니다.")
        if platform.system() == 'Linux':
            st.info("리눅스 환경입니다. 'fonts-nanum' 패키지를 설치해주세요.")

    if df is None:
        st.warning("기본 데이터 파일을 찾을 수 없습니다.")
        uploaded_file = st.file_uploader("CSV 파일(.csv)을 업로드해주세요", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("데이터 파일이 없어도 [Tab 4: 맞춤 커리어 추천]은 사용 가능합니다.")
            
    # 직무 선택 로직 (데이터가 있을 때만 활성화)
    target_job = "Developer, back-end" # 기본값
    job_df = None
    
    if df is not None:
        st.success("✅ 데이터 로드 성공!")
        st.divider()
        st.subheader("🎯 분석할 직무 선택")

        if 'DevType' in df.columns:
            all_jobs = df['DevType'].dropna().astype(str).str.split(';').explode().str.strip().unique()
            all_jobs = sorted([job for job in all_jobs if job.lower() != 'nan'])
            
            default_index = all_jobs.index('Developer, back-end') if 'Developer, back-end' in all_jobs else 0
            target_job = st.selectbox("직무:", all_jobs, index=default_index)

            # 데이터 필터링
            job_df = df[df['DevType'].astype(str).str.contains(target_job, case=False, na=False, regex=False)]
            respondents = len(job_df)
            st.markdown(f"--- \n👥 분석 대상: **{respondents:,}명**")
        else:
            st.error("'DevType' 컬럼이 없습니다.")

# ---------------------------------------------------------
# 5. 메인 화면 구성 (4개 탭)
# ---------------------------------------------------------
st.title(f"🧭 2025 개발자 커리어 인사이트")

# ★★★ 탭 4개 생성 (DevNavi 추가됨) ★★★
tab1, tab2, tab3, tab4 = st.tabs(["📊 기술 트렌드", "🤖 AI 인식", "🧠 ML 심화 분석", "🧭 맞춤 커리어 추천"])

# =========================================================
# [TAB 1~3] 데이터 분석 (CSV 파일 필요)
# =========================================================
if job_df is not None:
    # [TAB 1] 기술 스택 트렌드
    with tab1:
        st.markdown(f"### 1️⃣ [{target_job}] 기술 스택 로드맵")
        tech_cols = {
            '💻 언어': ('LanguageHaveWorkedWith', 'LanguageWantToWorkWith'),
            '🗄️ DB': ('DatabaseHaveWorkedWith', 'DatabaseWantToWorkWith'),
            '☁️ 플랫폼': ('PlatformHaveWorkedWith', 'PlatformWantToWorkWith'),
            '🤖 AI 모델': ('AIModelsHaveWorkedWith', 'AIModelsWantToWorkWith')
        }
        
        def get_top_skills(data, col, n=7):
            if col not in data.columns: return pd.Series(dtype='float64')
            return data[col].dropna().astype(str).str.split(';').explode().str.strip().value_counts().head(n)

        for i, (name, (curr, want)) in enumerate(tech_cols.items()):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### 🟦 현재 (필수) - {name}")
                top_curr = get_top_skills(job_df, curr)
                if not top_curr.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=top_curr.values, y=top_curr.index, ax=ax, palette='Blues_r')
                    ax.bar_label(ax.containers[0], fmt='%d')
                    st.pyplot(fig)
            with c2:
                st.markdown(f"#### 🟩 미래 (성장) - {name}")
                top_want = get_top_skills(job_df, want)
                if not top_want.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=top_want.values, y=top_want.index, ax=ax, palette='Greens_r')
                    ax.bar_label(ax.containers[0], fmt='%d')
                    st.pyplot(fig)
            st.divider()

    # [TAB 2] AI 수용 태도
    with tab2:
        st.markdown(f"### 2️⃣ [{target_job}]의 AI 수용 태도")
        c1, c2 = st.columns(2)

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

        with c2:
            st.markdown("##### 😤 AI 도구 불만사항 (Top 5)")
            if 'AIFrustration' in job_df.columns:
                frust = job_df['AIFrustration'].dropna().astype(str).value_counts().head(5)
                if not frust.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=frust.values, y=frust.index, ax=ax, palette='Reds_r')
                    st.pyplot(fig)

    # [TAB 3] ML 심화 분석
    with tab3:
        st.markdown("### 3️⃣ 머신러닝 심화 분석")

        # 1. 군집 분석
        st.subheader("📊 개발자 성향 군집화 (Cluster Analysis)")
        ml_data = job_df[['YearsCode', 'ConvertedCompYearly']].dropna().copy()
        
        def clean_years(x):
            if x == 'Less than 1 year': return 0.5
            if x == 'More than 50 years': return 50
            try: return float(x)
            except: return 0

        ml_data['YearsCode_Num'] = ml_data['YearsCode'].apply(clean_years)
        ml_data = ml_data[ml_data['ConvertedCompYearly'] < 300000] # 이상치 제거

        if len(ml_data) > 30:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(ml_data[['YearsCode_Num', 'ConvertedCompYearly']])
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            ml_data['Cluster'] = kmeans.fit_predict(X_scaled)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=ml_data, x='YearsCode_Num', y='ConvertedCompYearly', hue='Cluster', palette='viridis', ax=ax)
            ax.set_title("개발자 그룹 분포 (연봉 vs 경력)")
            ax.set_xlabel("경력 (년)")
            ax.set_ylabel("연봉 (USD)")
            st.pyplot(fig)
        else:
            st.warning("데이터가 부족하여 군집 분석을 수행할 수 없습니다.")

        st.divider()

        # 2. 연관 분석
        st.subheader("🔗 기술 연관 분석 (Association Analysis)")
        langs = job_df['LanguageHaveWorkedWith'].dropna().astype(str).str.split(';')
        all_langs = sorted(list(set([l for sublist in langs for l in sublist])))
        
        selected_lang = st.selectbox("기준 언어를 선택하세요:", all_langs, index=0 if all_langs else 0)

        related_skills = {}
        for user_skills in langs:
            if selected_lang in user_skills:
                for skill in user_skills:
                    if skill != selected_lang:
                        related_skills[skill] = related_skills.get(skill, 0) + 1

        if related_skills:
            sorted_skills = sorted(related_skills.items(), key=lambda x: x[1], reverse=True)[:5]
            names = [x[0] for x in sorted_skills]
            counts = [x[1] for x in sorted_skills]

            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(x=counts, y=names, palette='magma')
            ax.set_title(f"'{selected_lang}' 사용자가 함께 쓰는 기술 Top 5")
            st.pyplot(fig)
        else:
            st.info("데이터 없음")

else:
    # 데이터가 없을 경우 안내 문구
    msg = "👈 왼쪽 사이드바에서 CSV 파일을 업로드하면 데이터 분석 결과가 표시됩니다."
    with tab1: st.info(msg)
    with tab2: st.info(msg)
    with tab3: st.info(msg)


# =========================================================
# [TAB 4] DevNavi - 맞춤 커리어 추천 (통합됨)
# =========================================================
with tab4:
    st.header("🧭 DevNavi - 신규 개발자 커리어 추천 AI")
    st.markdown("데이터 분석 결과가 없어도 이용할 수 있습니다. **개인의 성향과 목표**에 맞춰 커리어 로드맵을 설계해 드립니다.")
    st.divider()

    # 입력 폼 생성 (Streamlit 위젯 사용)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Q1. 관심 분야")
        interest_options = {v['label']: k for k, v in RECOMMENDATION_DB['interests'].items()}
        selected_interest_label = st.radio("가장 흥미로운 분야는?", list(interest_options.keys()))
        selected_interest_key = interest_options[selected_interest_label]
        user_interest = RECOMMENDATION_DB['interests'][selected_interest_key]

    with col2:
        st.subheader("Q2. 학습 목표")
        goal_options = {v['label']: k for k, v in RECOMMENDATION_DB['goals'].items()}
        selected_goal_label = st.radio("주된 목표는 무엇인가요?", list(goal_options.keys()))
        selected_goal_key = goal_options[selected_goal_label]
        user_goal = RECOMMENDATION_DB['goals'][selected_goal_key]

    st.subheader("Q3. 현재 실력")
    level_choice = st.select_slider(
        "본인의 코딩 실력은?",
        options=["입문 (코드 처음 봄)", "초급 (문법은 뗌)", "중급 (프로젝트 경험 있음)"]
    )
    
    level_map = {"입문 (코드 처음 봄)": "1", "초급 (문법은 뗌)": "2", "중급 (프로젝트 경험 있음)": "3"}
    user_level = level_map[level_choice]

    st.markdown("---")
    
    # 분석 버튼
    if st.button("🚀 나만의 커리어 로드맵 분석하기", type="primary", use_container_width=True):
        
        # 분석 시뮬레이션 효과
        with st.spinner('🔍 데이터를 분석하고 채용 트렌드와 매칭 중입니다...'):
            time.sleep(1.2)
        
        # --- 추천 로직 (DevNavi 알고리즘) ---
        framework = ""
        ai_tools = ["GitHub Copilot"]

        if user_interest["key"] == 'web':
            framework = "Next.js + Supabase" if user_goal["key"] == 'startup' else "React + Spring Boot"
        elif user_interest["key"] == 'ai':
            framework = "PyTorch" if user_goal["key"] == 'research' else "TensorFlow + FastAPI"
            ai_tools.append("Hugging Face")
        elif user_interest["key"] == 'mobile':
            framework = "Flutter"
        else:
            framework = "Kubernetes + Docker"

        if user_goal["key"] == 'startup':
            ai_tools.extend(["Cursor IDE", "v0.dev"])
        elif user_goal["key"] == 'employment':
            ai_tools.append("LeetCode (AI Help)")

        if user_level == "1":
            ai_tools.append("ChatGPT (튜터링용)")

        # --- 결과 출력 ---
        st.success("🎉 분석이 완료되었습니다!")
        
        st.markdown(f"### 📌 추천 트랙: **{user_interest['label']} 전문가 과정**")
        st.info(f"💡 {user_interest['desc']}")

        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("#### [1] 1순위 추천 언어")
            st.code(f"{user_interest['base_lang']}")
            
            st.markdown("#### [2] 필수 프레임워크")
            st.code(f"{framework}")

        with res_col2:
            st.markdown("#### [3] AI 생산성 도구")
            st.write(f"👉 {', '.join(ai_tools)}")

        st.markdown("#### [4] 학습 로드맵")
        steps = [
            f"**1단계:** {user_interest['base_lang'].split('/')[0]} 기초 문법 완벽 이해",
            f"**2단계:** {framework} 공식 문서 따라하며 'Hello World' 찍기",
            f"**3단계:** (보너스) {', '.join(user_goal['bonus'])} 학습",
            "**4단계:** 나만의 실전 프로젝트 배포하기"
        ]
        for step in steps:
            st.write(step)
