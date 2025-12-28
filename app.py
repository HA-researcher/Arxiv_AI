import streamlit as st
import arxiv
import pandas as pd
import google.generativeai as genai
from typing import List, Dict
import os

# --- 設定と定数 ---
PAGE_TITLE = "Arxiv Multi-View Reviewer 📚"
PAGE_ICON = "🤖"

# ページ設定（ブラウザのタブ名などを設定）
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- クラス定義: データ取得層 (For Data Science Appeal) ---
class ArxivFetcher:
    """
    Arxiv APIを使用して論文データを取得・整形するクラス。
    データエンジニアリングのスキルセット（ETLプロセス）を意識しています。
    """
    def __init__(self):
        self.client = arxiv.Client()

    @st.cache_data(ttl=3600) # 1時間キャッシュ（API制限対策と高速化のアピール）
    def search_papers(_self, query: str, max_results: int = 5) -> List[Dict]:
        """
        キーワードに基づいて論文を検索し、辞書リストで返す
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers = []
        try:
            for result in _self.client.results(search):
                papers.append({
                    "title": result.title,
                    "summary": result.summary,
                    "url": result.entry_id,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "authors": ", ".join([author.name for author in result.authors])
                })
        except Exception as e:
            st.error(f"Arxiv API Error: {e}")
            return []
        
        return papers

# --- クラス定義: AIロジック層 (For ML/Web Appeal) ---
class GeminiProcessor:
    """
    Google Gemini APIを操作するクラス。
    マルチペルソナ（役割の切り替え）と温度パラメータの制御を担当。
    """
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model_name = "gemini-1.5-flash" # 高速でコスト効率が良いモデル

    def generate_review(self, text: str, persona: str) -> str:
        """
        ペルソナに応じたプロンプトでレビューを生成する
        """
        model = genai.GenerativeModel(self.model_name)
        
        # ペルソナごとのプロンプト定義（プロンプトエンジニアリング）
        prompts = {
            "expert": """
            あなたは熟練のデータサイエンティストです。以下の論文アブストラクトを読み、技術的な専門家の視点で分析してください。
            出力形式:
            - **技術的アプローチ**: 手法やアルゴリズムの核心
            - **新規性**: 既存研究との違い
            - **課題**: 考えられる懸念点
            (Temperature: 0.0 - 事実に忠実)
            """,
            "beginner": """
            あなたは好奇心旺盛なテックブロガーです。以下の論文アブストラクトを読み、AI初心者やビジネスマンに向けてわかりやすく解説してください。
            出力形式:
            - **ひとことで言うと**: キャッチーなタイトル
            - **何がすごいの？**: 比喩を使ったわかりやすい解説
            - **未来はどうなる？**: この技術が社会実装された時のワクワクする未来
            (Temperature: 0.7 - 創造的)
            """
        }

        # パラメータの動的制御 (Temperature Control)
        # 専門家は正確性重視(0.0)、初心者は創造性重視(0.7)
        generation_config = genai.types.GenerationConfig(
            temperature=0.0 if persona == "expert" else 0.7
        )

        full_prompt = f"{prompts[persona]}\n\nTarget Abstract:\n{text}"
        
        try:
            response = model.generate_content(
                full_prompt, 
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            return f"Error generation review: {e}"

# --- UI構築 (Main Application) ---
def main():
    st.title(f"{PAGE_TITLE}")
    st.markdown("""
    **GMOインターンシップ応募用ポートフォリオ** 最新のArxiv論文を検索し、**「専門家視点」**と**「初心者視点」**の2つのAIエージェントが同時にレビューします。
    データソースへのアクセスとLLMのハンドリング技術を実証するデモアプリです。
    """)

    # --- サイドバー: 設定 ---
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("Gemini API Key", type="password", help="Google AI Studioで取得したキーを入力")
        
        st.divider()
        st.header("🔎 Search Filter")
        query = st.text_input("Keyword", value="LLM Agents")
        max_results = st.slider("Max Papers", 1, 10, 3)
        
        st.info("💡 Tip: Try keywords like 'RAG', 'Time Series', 'Transformer'")

    # --- メインロジック ---
    if not api_key:
        st.warning("👈 サイドバーにGemini API Keyを入力してスタートしてください。")
        return

    # インスタンス化
    fetcher = ArxivFetcher()
    processor = GeminiProcessor(api_key)

    if st.button("Search & Analyze 🚀", type="primary"):
        with st.spinner("Fetching papers from Arxiv..."):
            papers = fetcher.search_papers(query, max_results)

        if not papers:
            st.warning("論文が見つかりませんでした。キーワードを変えて試してください。")
            return

        st.success(f"{len(papers)} 件の論文が見つかりました。AI分析を開始します...")

        # 各論文の表示と分析
        for i, paper in enumerate(papers):
            with st.container():
                st.markdown(f"### {i+1}. {paper['title']}")
                st.caption(f"Authors: {paper['authors']} | Published: {paper['published']}")
                
                # 生データの確認用（アコーディオン）
                with st.expander("Show Original Abstract"):
                    st.write(paper['summary'])
                
                # 2カラムレイアウトでマルチビュー表示
                col1, col2 = st.columns(2)
                
                # 左カラム：専門家視点
                with col1:
                    st.markdown("#### 👓 Expert View (Data Scientist)")
                    with st.spinner("Analyzing (Expert)..."):
                        review_expert = processor.generate_review(paper['summary'], "expert")
                        st.info(review_expert)

                # 右カラム：初心者視点
                with col2:
                    st.markdown("#### 💡 Beginner View (Web/Biz)")
                    with st.spinner("Analyzing (Beginner)..."):
                        review_beginner = processor.generate_review(paper['summary'], "beginner")
                        st.success(review_beginner)
                
                st.divider()

if __name__ == "__main__":
    main()
