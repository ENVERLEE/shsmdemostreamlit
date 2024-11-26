import streamlit as st
import os
from main import ResearchAssistant, MainFunction
import asyncio
from config.settings import LLM_CONFIG, EMBEDDING_CONFIG

def initialize_session_state():
    if 'assistant' not in st.session_state:
        st.session_state.assistant = ResearchAssistant()
    if 'api_keys_set' not in st.session_state:
        st.session_state.api_keys_set = False

def save_api_keys(llm_key: str, voyage_key: str):
    """API 키를 설정에 저장"""
    LLM_CONFIG["api_key"] = llm_key
    EMBEDDING_CONFIG["api_key"] = voyage_key
    st.session_state.api_keys_set = True

def api_keys_sidebar():
    """API 키 입력을 위한 사이드바"""
    with st.sidebar:
        st.header("API 키 설정")
        
        # 현재 설정된 API 키 상태 표시
        if st.session_state.api_keys_set:
            st.success("API 키가 설정되었습니다.")
        
        # API 키 입력 폼
        with st.form("api_keys_form"):
            llm_key = st.text_input(
                "LLM API 키",
                type="password",
                value=LLM_CONFIG["api_key"] if LLM_CONFIG["api_key"] != "place-holder" else ""
            )
            voyage_key = st.text_input(
                "VoyageAI API 키",
                type="password",
                value=EMBEDDING_CONFIG["api_key"] if EMBEDDING_CONFIG["api_key"] != "place-holder" else ""
            )
            
            if st.form_submit_button("API 키 저장"):
                if llm_key and voyage_key:
                    save_api_keys(llm_key, voyage_key)
                    st.success("API 키가 성공적으로 저장되었습니다.")
                else:
                    st.error("모든 API 키를 입력해주세요.")

def main():
    st.title('SUHANGSSALMUKDEMO')
    st.header("Welcome to Research Assistant CLI!")
    st.write("This tool will help you conduct research and analysis.")

    initialize_session_state()
    api_keys_sidebar()  # API 키 설정 사이드바 추가

    # API 키가 설정되지 않은 경우 경고 표시
    if not st.session_state.api_keys_set:
        st.warning("시작하기 전에 사이드바에서 API 키를 설정해주세요.")
        return

    with st.form(key='research_form'):
        query = st.text_input(label='연구주제')
        evaluation_criteria = st.text_area(label='연구평가기준')
        context = st.text_area(label='연구설명')
        research_direction = st.text_area(label='연구방향성')
        submit_button = st.form_submit_button(label='제출')
        
        if submit_button:
            if query:
                # API 키가 설정된 경우에만 실행
                if st.session_state.api_keys_set:
                    result = asyncio.run(MainFunction.process_and_display_results(
                        st.session_state.assistant,
                        query,
                        evaluation_criteria,
                        context,
                        research_direction
                    ))
                    if result:
                        st.session_state.last_result = result
                else:
                    st.error("API 키를 먼저 설정해주세요.")
            else:
                st.warning("연구주제를 입력해주세요.")

if __name__ == "__main__":
    main()
