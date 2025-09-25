# app.py
# -*- coding: utf-8 -*-
# Streamlit: trouble maker —— 问题制造器
# 需求覆盖：
# 1) Streamlit 界面
# 2) “以xxxx为主题生成若干问题”，数量可调
# 3) GPT API 调用（OpenAI 新旧 SDK 兼容；可自定义 Base URL）
# 4) 两列表格展示（提示词｜问题）
# 5) Excel 下载

import os
import io
import time
import pandas as pd
import streamlit as st

APP_TITLE = "trouble maker"

st.set_page_config(page_title=APP_TITLE, page_icon="🧨", layout="centered")
st.title(APP_TITLE)

# ----------------------
# 侧边栏：API & 生成参数
# ----------------------
with st.sidebar:
    st.subheader("🔐 API 设置")
    default_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", value=default_key, type="password")
    base_url = st.text_input(
        "Base URL（可选）",
        value=os.getenv("OPENAI_BASE_URL", ""),
        help="留空使用官方；或填写自建/代理的 OpenAI 兼容网关"
    )
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 1.0, 0.05)
    st.caption("仅生成**问题**，不返回答案。")

st.markdown("### 🧩 生成设置")
col1, col2 = st.columns([2, 1])
with col1:
    subject = st.text_input(
        "主题（建议使用“以xxxx为主题生成若干问题”的句式；也可只填主题关键词）",
        value="以人体免疫系统为主题生成若干问题"
    )
with col2:
    n_questions = st.number_input("生成数量", min_value=1, max_value=200, value=10, step=1)

def normalize_prompt(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "以通用思维为主题生成若干问题"
    if t.startswith("以") and "为主题" in t:
        return t
    return f"以{t}为主题生成若干问题"

normalized_prompt = normalize_prompt(subject)

go = st.button("开始生成", type="primary", use_container_width=True)

# ----------------------
# GPT 调用（新旧 SDK 兼容）
# ----------------------
def chat_complete_compatible(messages, model, temperature, top_p, api_key, base_url=None):
    """
    优先使用 openai>=1.x 的新式客户端：
        from openai import OpenAI
        OpenAI(...).chat.completions.create(...)
    若不可用，回退到旧版：
        import openai
        openai.ChatCompletion.create(...)
    """
    # 尝试新 SDK
    try:
        from openai import OpenAI
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p
        )
        return resp.choices[0].message.content
    except Exception:
        # 回退旧 SDK
        try:
            import openai as openai_legacy
            openai_legacy.api_key = api_key
            if base_url:
                # 旧版使用 api_base
                openai_legacy.api_base = base_url
            resp = openai_legacy.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p
            )
            return resp["choices"][0]["message"]["content"]
        except Exception as e2:
            raise RuntimeError(f"API 调用失败（新旧 SDK 均不可用）：{e2}")

@st.cache_data(show_spinner=False)
def generate_questions_cached(api_key, base_url, model, temperature, top_p, prompt_text, n):
    if not api_key:
        raise ValueError("API Key 未设置。")
    system = (
        "你是一个问题生成器。根据用户提示词，只输出问题列表，不要解释。"
        "每个问题独立一行，语言与提示词一致，尽量多样且具体。"
    )
    user = f"提示词：{prompt_text}\n请直接输出 {n} 个不同的问题，每行一个，不要编号和多余前后缀。"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    text = chat_complete_compatible(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        base_url=base_url
    )

    # 解析为行；清理常见编号符号
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    cleaned = []
    for ln in lines:
        s = ln
        # 去除常见编号前缀（1. 1) 1）- • — 等）
        s = s.lstrip("•-—").lstrip()
        while s and (s[0].isdigit() or s[0] in ".)）、"):
            s = s[1:].lstrip()
        cleaned.append(s)

    # 截断到 n；若少于 n，不强行补齐
    if len(cleaned) > n:
        cleaned = cleaned[:n]
    return cleaned

# ----------------------
# 主流程
# ----------------------
df_result = None
if go:
    if not api_key:
        st.error("请先在左侧输入 **OpenAI API Key**。")
    else:
        with st.spinner("正在调用 GPT 生成问题..."):
            try:
                questions = generate_questions_cached(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_text=normalized_prompt,
                    n=n_questions
                )
                if not questions:
                    st.warning("未生成到问题，请更换主题或调整参数后再试。")
                else:
                    df_result = pd.DataFrame({
                        "提示词": [normalized_prompt] * len(questions),
                        "问题": questions
                    })
                    st.success(f"生成完成：{len(questions)} 条。")
            except Exception as e:
                st.error(f"发生错误：{e}")

# ----------------------
# 展示 & Excel 下载
# ----------------------
if df_result is not None and not df_result.empty:
    st.markdown("### 📋 结果")
    st.dataframe(df_result, use_container_width=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_result.to_excel(writer, sheet_name="questions", index=False)
    output.seek(0)

    ts = time.strftime("%Y%m%d-%H%M%S")
    filename = f"trouble_maker_{ts}.xlsx"
    st.download_button(
        label="📥 下载为 Excel",
        data=output,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with st.expander("使用说明 / Tips"):
    st.markdown("""
- **提示词格式**：推荐使用“**以xxxx为主题生成若干问题**”；若只输入关键词，程序会自动补全为该格式。
- **模型兼容**：优先走 **openai>=1.x** 的 `chat.completions`；若环境较旧会自动回退到 `openai.ChatCompletion`。
- **Base URL**：如使用代理/自建网关，请在侧边栏填写 **Base URL**（示例：https://your-proxy/v1）。
- **导出**：结果支持一键下载为 **Excel**（两列：提示词｜问题）。
""")
