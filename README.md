# AutoIntel-Agents 🧠
# Intelligent Market Research and Competitor Analysis

An AI-powered multi-agent system to automate industry-specific market research, track AI/ML trends, perform competitor sentiment analysis, generate use cases, and find relevant datasets — streamlining the entire AI adoption journey for businesses.

---

## 📌 Overview

Trendlytic eliminates the need for manual expertise in AI adoption by automating:

- Industry research
- Competitor & sentiment analysis
- Generative AI use-case creation
- Dataset discovery

This intelligent system enables companies to identify high-impact AI solutions, align them with business goals, and kick-start AI/ML projects with minimal effort.

---

## 🚀 Features

✅ **Market Research Agent**  
Fetches AI/ML trends, industry reports, and company-specific data using Tavily & Serper APIs.

✅ **Competitor Sentiment Analysis**  
Performs real-time multilingual sentiment analysis using BERT to analyze competitor perception and strategy.

✅ **Use Case Generation Agent**  
Uses the Groq API (LLaMA model) to create tailored AI/ML/GenAI use cases based on gathered insights.

✅ **Dataset Collector Agent**  
Automatically sources datasets from platforms like Kaggle, Hugging Face, and GitHub for each use case.

✅ **Streamlit UI**  
Simple frontend interface to enter industry/company and view/export analysis results.

✅ **Export to CSV**  
All outputs — use cases, sentiment analysis, dataset links — are saved as CSV files.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit + Custom CSS
- **Backend**: Python + Modular Agent Architecture
- **APIs**:
  - [Tavily](https://docs.tavily.com/) (market research)
  - [Serper](https://serper.dev/) (web search)
  - [Groq](https://groq.com/) (use case generation)
- **ML Model**: `nlptown/bert-base-multilingual-uncased-sentiment` (Hugging Face)
- **Other Libraries**: `transformers`, `dotenv`, `torch`, `pandas`, `requests`, etc.

---

## 📁 Project Structure
Trendlytic/
│
├── app.py # Streamlit entry point
├── agents/
│ ├── market_research.py
│ ├── sentiment_analysis.py
│ ├── use_case_generator.py
│ └── dataset_collector.py
├── outputs/ # All CSV results saved here
├── .env # Stores API keys securely
├── requirements.txt # Python dependencies
└── README.md




