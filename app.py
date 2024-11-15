import requests
from tavily import TavilyClient
import json
import pandas as pd 
import os
from groq import Groq
import streamlit as st
import tempfile
import re
from transformers import pipeline
import torch

# API Keys (replace with your actual API keys)
TAVILY_API_KEY = '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'
SERPER_API_KEY = 'e622090f73b6e3ca46137af435b760fdda23ecfe'
GROQ_API_KEY = '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'  # Replace with your actual Groq API key
os.environ['GROQ_API_KEY'] = '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'

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        /* Global Styles */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        /* Container Styling */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
            padding: 1rem;  /* Reduced padding */
        }

        /* Header and Label Styling */
        h1, h2, h3, h4, h5, h6, .stTextInput > label {
            font-family: 'Poppins', sans-serif !important;
            color: #2c3e50 !important;
            font-weight: 600 !important;
            margin-bottom: 0.3rem !important;  /* Reduced margin */
        }

        /* Main Title Styling */
        .stApp h1 {
            font-size: 2rem !important;  /* Reduced font size */
            margin-bottom: 1rem !important;  /* Reduced margin */
            color: #2c3e50 !important;
            padding-top: 1rem !important;  /* Added padding top */
        }

        /* Input Container Styling */
        .stTextInput {
            margin-bottom: 1rem !important;  /* Reduced margin */
        }

        /* Input Field Styling */
        .stTextInput > div > div > input {
            font-family: 'Roboto', sans-serif !important;
            color: #2c3e50 !important;
            background-color: white !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 4px !important;
            padding: 8px !important;  /* Reduced padding */
            font-size: 0.9rem !important;  /* Reduced font size */
            transition: all 0.3s ease !important;
        }

        /* Selectbox Styling */
        .stSelectbox > div > div {
            background-color: white !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 4px !important;
            margin-bottom: 1rem !important;
            color: #2c3e50 !important;
        }

        /* Button Styling */
        .stButton > button {
            font-family: 'Poppins', sans-serif !important;
            background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%) !important;
            color: white !important;
            padding: 8px 16px !important;  /* Reduced padding */
            border-radius: 4px !important;
            border: none !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;  /* Reduced font size */
            letter-spacing: 0.5px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            transition: all 0.3s ease !important;
            margin-top: 0.5rem !important;  /* Added margin top */
            width: auto !important;  /* Auto width */
            min-width: 150px !important;  /* Minimum width */
        }

        .stButton > button:hover {
            transform: translateY(-1px) !important;  /* Reduced transform */
            box-shadow: 0 3px 6px rgba(0,0,0,0.15) !important;
        }

        /* Description Text Styling */
        .stApp p {
            font-family: 'Roboto', sans-serif !important;
            color: #5a6c7d !important;
            font-size: 0.95rem !important;  /* Reduced font size */
            line-height: 1.4 !important;
            margin-bottom: 1rem !important;  /* Reduced margin */
        }

        /* Help Text Styling */
        .stTextInput > div > div > div small {
            color: #718096 !important;
            font-size: 0.8rem !important;  /* Reduced font size */
            margin-top: 2px !important;  /* Reduced margin */
        }
    </style>
""", unsafe_allow_html=True)

# Initialize Tavily client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

class MarketResearchAgent:
    def __init__(self, industry=None, company=None, specific_market=None):
        self.industry = industry if industry else "Retail"
        self.company = company if company else "Reliance Digital"
        self.specific_market = specific_market if specific_market else "Customer Satisfaction"
        dev = 0 if torch.cuda.is_available() else -1
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=dev)
    def search_tavily_market(self):
        query = f"Market Research in {self.industry} industry for {self.company} Company focusing on {self.specific_market}"
        response = tavily_client.search(query, search_depth="advanced", max_results=10)
        results = []
        for item in response.get("results", []):
            if item['score'] > 0.9:
                result = {
                    "content": item['content'],
                    "link": item['url']               
                }
                results.append(result)
        return results
    
    def search_tavily_competitor(self):
        query = f"Competitor Analysis in {self.industry} industry for {self.company} Company focusing on {self.specific_market}"
        response = tavily_client.search(query, search_depth="advanced")
        results = []
        for item in response.get("results", []):
            if item['score'] > 0.9:
                result = {
                    "content": item['content'],
                    "link": item['url']
                }
                results.append(result)
        return results
    
    def search_tavily_ml(self):
        query = f"AI/ML Trends in {self.industry} industry for {self.company} Company focusing on {self.specific_market}"
        response = tavily_client.search(query, search_depth="advanced")
        results = []
        for item in response.get("results", []):
            if item['score'] > 0.9:
                result = {
                    "content": item['content'],
                    "link": item['url']
                }
                results.append(result)
        return results
    
    def search_serper(self, query):
        """Search for competitor analysis using Serper's API."""
        url = "https://google.serper.dev/search"
        headers = {'X-API-KEY': SERPER_API_KEY,'Content-Type': 'application/json'}
        # query = f"Competitor analysis in {self.industry} industry for {self.company} Company"
        payload=json.dumps({"q": query,"gl":"in"})
        response = requests.request("POST",url,headers=headers, data=payload).json()
        results=[]
        for item in response.get("organic", []):
            result={
                "title":item['title'],
                "link":item['link'],
                "snippet":item['snippet']
            }
            results.append(result)
        return results

    def split_text_into_chunks(text, chunk_size=200):
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks, current_chunk = [], ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def analyze_sentiment(self, competitor_text):
        # Ensure competitor_text is a string
        if not isinstance(competitor_text, str):
            raise TypeError("Expected a string or bytes-like object for competitor_text")

        # Split the text into manageable chunks for analysis
        text_chunks = [competitor_text[i:i+512] for i in range(0, len(competitor_text), 512)]

        # Map star ratings to sentiment labels
        star_to_label = {
            1: "Very Negative",
            2: "Negative",
            3: "Neutral",
            4: "Positive",
            5: "Very Positive"
        }

        # Analyze sentiment for each chunk
        sentiment_results = []
        for chunk in text_chunks:
            sentiment = self.sentiment_analyzer(chunk)[0]
            label = sentiment['label']
            score = sentiment['score']
            # Map the sentiment label (stars) to descriptive labels
            star_rating = int(label.split()[0])  # Extracts the star rating
            sentiment_label = star_to_label.get(star_rating, "Unknown")
            sentiment_results.append({
                "Text": chunk,
                "Stars": star_rating,
                "Sentiment Label": sentiment_label,
                "Score": score
            })
        sentiment_results_df = pd.DataFrame(sentiment_results)
        sentiment_results_df.to_csv("sentiment_results.csv", index=False)
        return sentiment_results_df
    
    def perform_search(self):
        # Define queries for Serper API
        market_query = f"Market trends in {self.industry} industry for {self.company}"
        competitor_query = f"Competitor analysis in {self.industry} industry for {self.company}"
        ai_ml_query = f"AI and ML trends in {self.industry} industry for {self.company}"

        # Collecting search results from Tavily API
        market_trends = self.search_tavily_market()
        competitor_analysis = self.search_tavily_competitor()
        ai_ml_trends = self.search_tavily_ml()

        # Collecting search results from Serper API
        market_trends_serper = self.search_serper(market_query)
        competitor_analysis_serper = self.search_serper(competitor_query)
        ai_ml_trends_serper = self.search_serper(ai_ml_query)

        # Combining Tavily API results into a single text
        combined_text_tavily = (
            "Market Trends (Tavily):\n" + "\n".join([item["content"] for item in market_trends]) + "\n\n" +
            "Competitor Analysis (Tavily):\n" + "\n".join([item["content"] for item in competitor_analysis]) + "\n\n" +
            "AI/ML Trends (Tavily):\n" + "\n".join([item["content"] for item in ai_ml_trends])
        )

        # Combining Serper API results into a single text
        # combined_text_serper = (
        #     "Market Trends (Serper):\n" + "\n".join([f"{item['snippet']} (Link: {item['link']})" for item in market_trends_serper]) + "\n\n" +
        #     "Competitor Analysis (Serper):\n" + "\n".join([f"{item['snippet']} (Link: {item['link']})" for item in competitor_analysis_serper]) + "\n\n" +
        #     "AI/ML Trends (Serper):\n" + "\n".join([f"{item['snippet']} (Link: {item['link']})" for item in ai_ml_trends_serper])
        # )
         # Create a DataFrame for Serper results
        serper_data = {
            "title": [item['title'] for item in market_trends_serper + competitor_analysis_serper + ai_ml_trends_serper],
            "snippet": [item['snippet'] for item in market_trends_serper + competitor_analysis_serper + ai_ml_trends_serper],
            "link": [item['link'] for item in market_trends_serper + competitor_analysis_serper + ai_ml_trends_serper]
        }
        serper_df = pd.DataFrame(serper_data)

        # Save the DataFrame to a CSV file
        serper_df.to_csv("market_research.csv", index=False)

        return combined_text_tavily

    def run(self):
        combined_text_tavily = self.perform_search()
        if not combined_text_tavily.strip() :
            return "No data found for the given industry and company."
        
        # Display the combined text from both APIs
        print("Tavily Output:\n", combined_text_tavily)
        print("\n" + "-"*50 + "\n")

        return combined_text_tavily

class UseCaseGenerationAgent:
    def __init__(self, research_output, emphasis=None):
        self.research_output = research_output
        self.emphasis = emphasis
        self.client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    def generate_use_cases(self):
        prompt = (
        f"Based on the following Market research output, propose relevant use caseshere the company can leverage GenAI, LLMs, and"
        f"ML technologies along with their applications and cross-functional benefits. "
        f"For each use case, provide the following details:\n"
        f"1. Use Case: A brief description of the use case.\n"
        f"2. Application: How the use case can be applied in the company.\n"
        f"3. Cross-Functional Benefits: The benefits of the use case across different functions of the company.\n\n"
        f"Market Research Output:\n{self.research_output}\n\n"
        )
        
        # Payload for the Groq API
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="llama-3.2-90b-vision-preview",
        )

        if chat_completion:
            use_cases = [choice.message.content.strip() for choice in chat_completion.choices]
            with open("use_cases.txt", "w") as file:
                for use_case in use_cases:
                    file.write(use_case + "\n\n")
            return use_cases
        else:
            print("Error in generating use cases from Groq API")
            return ["Error in generating use cases from Groq API"]

class ResourceAssetCollectionAgent:
    def __init__(self,use_cases, use_case_file="use_cases.txt"):
        self.use_case_file = use_case_file
        self.dataset_links = []
        self.use_cases = use_cases
        self.use_cases = self.load_use_cases()

    def load_use_cases(self):
        """
        Loads use cases from the specified file.
        """
        use_cases = []
        try:
            with open(self.use_case_file, "r") as file:
                # Read lines and extract each use case title from the file
                for line in file:
                    if line.startswith("**Use Case"):  # Extract only use case titles
                        title = line.split(": ")[1].strip()
                        use_cases.append(title)
        except FileNotFoundError:
            print(f"Error: {self.use_case_file} not found.")
        return use_cases

    def search_datasets_serper(self, query):
        """
        Searches for datasets using the Serper API with the provided use case title as the query.
        """
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        
        # Define payload with use case title as query
        payload = json.dumps({
            "q": query + " dataset",
            "gl": "in"
        })
        
        response = requests.post(url, headers=headers, data=payload).json()
        
        # Process search results to extract dataset links
        results = []
        for item in response.get("organic", []):
            results.append({
                "title": item['title'],
                "snippet": item['snippet'],
                "link": item['link']
            })
        return results

    def collect_datasets(self):
        """
        Searches for datasets based on each use case title.
        """
        for use_case in self.use_cases:
            # Search datasets for each use case title using Serper
            results = self.search_datasets_serper(use_case)
            for result in results:
                self.dataset_links.append({
                    "use_case": use_case,
                    "title": result["title"],
                    "snippet": result["snippet"],
                    "link": result["link"]
                })

    def save_to_csv(self, filename="Resource_Links.csv"):
        """
        Save collected dataset links and use cases to a CSV file.
        """
        df = pd.DataFrame(self.dataset_links)
        df.to_csv(filename, index=False)

    def curate_resources(self):
        """
        Run dataset search and save results to CSV.
        """
        self.collect_datasets()
        self.save_to_csv()

def main():
    st.title("📊 Market Research & AI Use Case Generator")
    st.write("This application conducts market research and generates AI/ML use cases for specific industries and companies.")

    # User input fields for industry, company, and market focus
    industry = st.text_input("Industry", value="Retail", key="industry", help="Enter the industry to research")
    company = st.text_input("Company Name", value="Reliance Digital", key="company", help="Enter the company for research")
    specific_market = st.text_input("Specific Market Focus Area", value="Customer Satisfaction", key="specific_market", help="Enter the market focus area")

    
    # Dropdown for selecting the type of market research
    research_type = st.selectbox(
        "Select Market Research Type",
        ("Market Trends", "Competitor Analysis", "AI-ML Trends")
    )
    
    # Initialize serper_output in case no valid research_type is selected
    serper_output = []

    if st.button("⏳  Run Market Research"):
        # Instantiate the Market Research Agent
        market_agent = MarketResearchAgent(industry=industry, company=company, specific_market=specific_market)
        
        with st.spinner("🔍 Conducting market research..."):
            # Perform search based on selected research type
            if research_type == "Market Trends":
                research_output = market_agent.search_tavily_market()
                serper_output = market_agent.search_serper(f"Market trends in {industry} industry for {company}")
                backend_output=market_agent.perform_search()

            elif research_type == "Competitor Analysis":
                research_output = market_agent.search_tavily_competitor()
                serper_output = market_agent.search_serper(f"Competitor analysis in {industry} industry for {company}")
                backend_output=market_agent.perform_search()
                competitor_text=backend_output
                sentiment_results=market_agent.analyze_sentiment(competitor_text)
                st.subheader("Sentiment Analysis Results")
                st.write(pd.read_csv("sentiment_results.csv"))
                
            elif research_type == "AI-ML Trends":
                research_output = market_agent.search_tavily_ml()
                serper_output = market_agent.search_serper(f"AI and ML trends in {industry} industry for {company}")
                backend_output=market_agent.perform_search()
            progress_bar=st.progress(100)
            for percent_complete in range(100):
                progress_bar.progress(percent_complete + 1)
            progress_bar.empty() 
        #     # Save Tavily research results to a CSV file
        #     research_data = {
        #         "content": [item['content'] for item in research_output],
        #         "link": [item['link'] for item in research_output]
        #     }
        #     research_df = pd.DataFrame(research_data)
        #     research_df.to_csv("market_research.csv", index=False)
        
        # st.subheader("Market Research Output")
        # st.write(pd.read_csv("market_research.csv"))
        
        # Display serper_output if it has data
        if serper_output:
            serper_data = {
                "title": [item['title'] for item in serper_output],
                "snippet": [item['snippet'] for item in serper_output],
                "link": [item['link'] for item in serper_output]
            }
            serper_df = pd.DataFrame(serper_data)

            st.subheader(f"{research_type} Output")
            st.write(serper_df)
            
            # Pass serper_output to the use case generation agent
            research_output_for_use_case = serper_output
        else:
            research_output_for_use_case = []  # Default to empty if no serper output available
    
        # Generate use cases based on the serper_output
        if research_output_for_use_case:  # Only proceed if there's data to work with
            use_case_agent = UseCaseGenerationAgent(research_output=research_output_for_use_case, emphasis=specific_market)
            with st.spinner("Generating use cases..."):
                use_case_output = use_case_agent.generate_use_cases()
            st.subheader("Generated Use Cases")
            st.write("\n\n".join(use_case_output))
        
        # Save use_case_output to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as temp_file:
            temp_file.writelines("\n".join(use_case_output))
            temp_file_path = temp_file.name

        if use_case_output:
            # Resource Asset Collection based on generated use cases
            resource_agent = ResourceAssetCollectionAgent(use_case_output, use_case_file=temp_file_path)
            with st.spinner("Collecting dataset resources..."):
                resource_agent.curate_resources()
            st.subheader("Resource Links")
            st.write(pd.read_csv("Resource_Links.csv"))
            st.success("Resource links saved to Resource_Links.csv")

if __name__ == "__main__":
    main()