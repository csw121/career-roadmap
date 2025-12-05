import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time

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
# 3. 데이터 로드 (ZIP 파일 지원)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # 깃허브 용량 문제 해결을 위해 zip 파일을 읽습니다.
        # pandas는 zip 내부의 csv를 자동으로 찾아 읽어줍니다.
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

    if df is None:
        st.warning("데이터 파일을 찾을 수 없습니다.")
        st.info("2025survey_results_public.zip 파일이 같은 폴더에 있는지 확인해주세요.")
        uploaded_file = st.file_uploader("또는 CSV/ZIP 파일을 직접 업로드하세요", type=['csv', 'zip'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.stop()
    else:
        st.success(f"✅ 데이터 로드 성공! ({len(df):,}행)")

    st.divider()
    st.subheader("🎯 직무 선택")

    if 'DevType' in df.columns:
        all_jobs = df['DevType'].dropna().astype(str).str.split(';').explode().str.strip().unique()
        all_jobs = sorted([job for job in all_jobs if job.lower() != 'nan'])
    else:
        st.error("'DevType' 컬럼이 없습니다.")
        st.stop()

    default_index = all_jobs.index('Developer, back-end') if 'Developer, back-end' in all_jobs else 0
    target_job = st.selectbox(
        "분석할 직무를 고르세요:",
        all_jobs,
        index=default_index
    )

    job_df = df[df['DevType'].astype(str).str.contains(target_job, case=False, na=False, regex=False)]
    respondents = len(job_df)

    st.markdown(f"--- \n👥 분석 대상: **{respondents:,}명**")

# ---------------------------------------------------------
# 5. 메인 화면 구성 (4개의 탭)
# ---------------------------------------------------------
st.title(f"🧭 [{target_job}] 커리어 인사이트 & 로드맵")

# 탭 4개 생성 (마지막 탭 추가됨)
tab1, tab2, tab3, tab4 = st.tabs(["📊 기술 트렌드", "🤖 AI 인식", "🧠 ML 심화 분석", "🎓 커리어 컨설팅"])

# =========================================================
# [TAB 1] 기술 스택 트렌드
# =========================================================
with tab1:
    st.markdown("### 1️⃣ 기술 스택 로드맵 (현재 vs 미래)")

    tech_cols = {
        '💻 언어': ('LanguageHaveWorkedWith', 'LanguageWantToWorkWith'),
        '🗄️ DB': ('DatabaseHaveWorkedWith', 'DatabaseWantToWorkWith'),
        '☁️ 플랫폼': ('PlatformHaveWorkedWith', 'PlatformWantToWorkWith'),
        '🤖 AI 모델': ('AIModelsHaveWorkedWith', 'AIModelsWantToWorkWith')
    }

    def get_top_skills(data, col, n=7):
        if col not in data.columns: return pd.Series()
        return data[col].dropna().astype(str).str.split(';').explode().str.strip().value_counts().head(n)

    for i, (name, (curr, want)) in enumerate(tech_cols.items()):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### 🟦 현재 (필수 기술) - {name}")
            top_curr = get_top_skills(job_df, curr)
            if not top_curr.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=top_curr.values, y=top_curr.index, ax=ax, palette='Blues_r')
                ax.bar_label(ax.containers[0], fmt='%d', padding=3)
                st.pyplot(fig)
        with c2:
            st.markdown(f"#### 🟩 미래 (성장 기술) - {name}")
            top_want = get_top_skills(job_df, want)
            if not top_want.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=top_want.values, y=top_want.index, ax=ax, palette='Greens_r')
                ax.bar_label(ax.containers[0], fmt='%d', padding=3)
                st.pyplot(fig)
        st.divider()

# =========================================================
# [TAB 2] AI 수용 태도
# =========================================================
with tab2:
    st.markdown("### 2️⃣ 이 직무의 AI 수용 태도")
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
        st.markdown("##### 😤 AI 도구의 가장 큰 불만은?")
        if 'AIFrustration' in job_df.columns:
            frust = job_df['AIFrustration'].dropna().astype(str).value_counts().head(5)
            if not frust.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=frust.values, y=frust.index, ax=ax, palette='Reds_r')
                st.pyplot(fig)

# =========================================================
# [TAB 3] ML 심화 분석
# =========================================================
with tab3:
    st.markdown("### 3️⃣ 머신러닝 심화 분석")

    st.subheader("📊 개발자 유형 군집화 (K-Means)")
    st.info("💡 경력과 연봉 데이터를 기반으로 개발자들을 **3가지 그룹**으로 자동 분류합니다.")

    ml_data = job_df[['YearsCode', 'ConvertedCompYearly']].dropna().copy()

    def clean_years(x):
        if x == 'Less than 1 year': return 0.5
        if x == 'More than 50 years': return 50
        try: return float(x)
        except: return 0

    ml_data['YearsCode_Num'] = ml_data['YearsCode'].apply(clean_years)
    ml_data = ml_data[ml_data['ConvertedCompYearly'] < 300000]

    if len(ml_data) > 30:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(ml_data[['YearsCode_Num', 'ConvertedCompYearly']])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        ml_data['Cluster'] = kmeans.fit_predict(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=ml_data, x='YearsCode_Num', y='ConvertedCompYearly',
            hue='Cluster', palette='viridis', s=60, ax=ax
        )
        ax.set_title(f"[{target_job}] 개발자 그룹 분포")
        ax.set_xlabel("경력 (년)")
        ax.set_ylabel("연봉 (USD)")
        st.pyplot(fig)
    else:
        st.warning("데이터가 부족하여 군집 분석을 할 수 없습니다.")

    st.divider()

    st.subheader("🔗 기술 추천 (Association Analysis)")
    langs = job_df['LanguageHaveWorkedWith'].dropna().astype(str).str.split(';')
    all_langs = sorted(list(set([l for sublist in langs for l in sublist])))

    selected_lang = st.selectbox("어떤 언어를 주로 사용하시나요?", all_langs, index=0 if not all_langs else 0)

    related_skills = {}
    for user_skills in langs:
        if selected_lang in user_skills:
            for skill in user_skills:
                if skill != selected_lang:
                    related_skills[skill] = related_skills.get(skill, 0) + 1

    if related_skills:
        sorted_skills = sorted(related_skills.items(), key=lambda x: x[1], reverse=True)[:5]
        skill_names = [x[0] for x in sorted_skills]
        skill_counts = [x[1] for x in sorted_skills]

        fig, ax = plt.subplots(figsize=(8, 3))
        sns.barplot(x=skill_counts, y=skill_names, palette='magma')
        ax.set_title(f"'{selected_lang}' 사용자가 함께 쓰는 기술 Top 5")
        st.pyplot(fig)

# =========================================================
# [TAB 4] 커리어 컨설팅 (새로 추가된 기능)
# =========================================================
with tab4:
    st.markdown("### 🎓 AI 커리어 맞춤 컨설팅")
    st.info("💡 몇 가지 질문에 답하면, 2025년 트렌드에 맞는 학습 로드맵을 설계해 드립니다.")

    # --- 데이터베이스 ---
    RECOMMENDATION_DB = {
        "interests": {
            "web": {"label": "웹 개발 (Full Stack)", "base_lang": "JavaScript / TypeScript", "desc": "브라우저와 서버를 오가는 만능 개발자"},
            "ai": {"label": "AI / 데이터 사이언스", "base_lang": "Python", "desc": "데이터에서 가치를 창출하는 모델 개발"},
            "mobile": {"label": "모바일 앱 개발", "base_lang": "Dart (Flutter) / Swift", "desc": "iOS/Android 앱을 만드는 크리에이터"},
            "system": {"label": "시스템 / 백엔드 최적화", "base_lang": "Go / Rust", "desc": "고성능 서버와 인프라 구축"}
        },
        "goals": {
            "employment": {"label": "취업 (대기업/IT기업)", "bonus": ["알고리즘(Coding Test)", "CS 지식", "대규모 트래픽 처리"]},
            "startup": {"label": "창업 / 서비스 런칭", "bonus": ["빠른 프로토타이핑", "클라우드 배포(AWS)", "마케팅 감각"]},
            "research": {"label": "대학원 / 연구", "bonus": ["논문 리딩", "수학/통계", "영어"]}
        }
    }

    # --- 사용자 입력 ---
    col1, col2 = st.columns(2)
    with col1:
        user_interest_key = st.selectbox(
            "Q1. 가장 관심 있는 분야는?",
            options=list(RECOMMENDATION_DB["interests"].keys()),
            format_func=lambda x: RECOMMENDATION_DB["interests"][x]["label"]
        )
    with col2:
        user_goal_key = st.selectbox(
            "Q2. 학습의 주된 목표는?",
            options=list(RECOMMENDATION_DB["goals"].keys()),
            format_func=lambda x: RECOMMENDATION_DB["goals"][x]["label"]
        )

    user_level = st.radio("Q3. 현재 코딩 실력은?", ["입문 (코드 처음 봄)", "초급 (문법은 뗌)", "중급 (프로젝트 경험 있음)"], horizontal=True)

    # --- 분석 버튼 ---
    if st.button("🚀 나만의 로드맵 생성하기", type="primary"):
        with st.spinner("🔍 AI가 데이터를 분석하고 있습니다..."):
            time.sleep(1.5)  # 로딩 효과

            # 선택된 데이터 매핑
            interest_data = RECOMMENDATION_DB["interests"][user_interest_key]
            goal_data = RECOMMENDATION_DB["goals"][user_goal_key]

            # 추천 로직
            framework = ""
            ai_tools = ["GitHub Copilot"]

            if user_interest_key == 'web':
                framework = "Next.js + Supabase" if user_goal_key == 'startup' else "React + Spring Boot"
            elif user_interest_key == 'ai':
                framework = "PyTorch" if user_goal_key == 'research' else "TensorFlow + FastAPI"
                ai_tools.append("Hugging Face")
            elif user_interest_key == 'mobile':
                framework = "Flutter"
            else:
                framework = "Kubernetes + Docker"

            if user_goal_key == 'startup':
                ai_tools.extend(["Cursor IDE", "v0.dev"])
            elif user_goal_key == 'employment':
                ai_tools.append("LeetCode (AI Help)")

            if "입문" in user_level:
                ai_tools.append("ChatGPT (튜터링용)")

            # --- 결과 출력 ---
            st.divider()
            st.success(f"🎉 **{interest_data['label']} 전문가 과정**이 설계되었습니다!")

            r1, r2, r3 = st.columns(3)
            r1.metric("1순위 언어", interest_data['base_lang'])
            r2.metric("필수 프레임워크", framework)
            r3.metric("추천 AI 도구", ai_tools[-1])

            st.markdown(f"""
            #### 📝 상세 학습 로드맵
            1. **기초 다지기**: {interest_data['base_lang'].split('/')[0]} 문법 완벽 이해
            2. **실전 기술**: {framework} 공식 문서로 'To-Do 리스트' 만들어보기
            3. **스펙 업**: {', '.join(goal_data['bonus'])} 집중 학습
            4. **AI 활용**: {', '.join(ai_tools)} 설치 및 사용법 익히기
            """)

            if user_goal_key == 'startup':
                st.caption("🚀 팁: 창업이 목표라면 완벽한 코드보다 '실행되는 서비스'를 먼저 만드세요!")
            elif user_goal_key == 'research':
                st.caption("📚 팁: 최신 논문(ArXiv)을 요약해주는 AI 서비스를 적극 활용하세요.")

# ---------------------------------------------------------
# 6. 마무리 (푸터)
# ---------------------------------------------------------
st.markdown("---")
st.caption("© 2025 Developer Roadmap Service | Data Source: Stack Overflow Survey")

