import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from datetime import datetime, timedelta
import re
from collections import Counter
import base64
from io import BytesIO

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="LinkedIn Job Analytics Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #0077B5;
        --secondary-color: #00A0DC;
        --background-color: #F3F6F8;
        --card-background: #FFFFFF;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2.5em;
        font-weight: bold;
    }
    
    .metric-card p {
        margin: 5px 0 0 0;
        font-size: 1em;
        opacity: 0.9;
    }
    
    /* Sentiment badges */
    .sentiment-positive {
        background-color: #10b981;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .sentiment-negative {
        background-color: #ef4444;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .sentiment-neutral {
        background-color: #f59e0b;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(90deg, #0077B5 0%, #00A0DC 100%);
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,119,181,0.3);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #0077B5 0%, #00A0DC 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 1.5em;
        font-weight: bold;
    }
    
    /* Data table styling */
    .dataframe {
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Info box */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Success box */
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_linkedin_job_listings_cleaned.csv')
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# ==================== HELPER FUNCTIONS ====================
def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    if pd.isna(text):
        return 0
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity

def extract_skills(text):
    """Extract common tech skills from job descriptions"""
    if pd.isna(text):
        return []
    
    common_skills = [
        'python', 'java', 'javascript', 'sql', 'aws', 'azure', 'docker',
        'kubernetes', 'react', 'angular', 'node.js', 'machine learning',
        'data analysis', 'excel', 'powerbi', 'tableau', 'git', 'agile',
        'scrum', 'leadership', 'communication', 'teamwork', 'project management'
    ]
    
    text_lower = str(text).lower()
    found_skills = [skill for skill in common_skills if skill in text_lower]
    return found_skills

def create_metric_card(value, label, icon=""):
    """Create a styled metric card"""
    return f"""
    <div class="metric-card">
        <h3>{icon} {value}</h3>
        <p>{label}</p>
    </div>
    """

def get_sentiment_badge(sentiment):
    """Return HTML for sentiment badge"""
    if sentiment == 'positive':
        return '<span class="sentiment-positive">😊 Positive</span>'
    elif sentiment == 'negative':
        return '<span class="sentiment-negative">😞 Negative</span>'
    else:
        return '<span class="sentiment-neutral">😐 Neutral</span>'

def export_to_csv(df):
    """Export dataframe to CSV"""
    return df.to_csv(index=False).encode('utf-8')

# ==================== ML TRAINING ====================
@st.cache_resource
def train_models(df, targets):
    """Train ML models for job predictions"""
    models = {}
    classification_reports = {}

    for target in targets:
        try:
            X = df['job_title'].fillna('')
            y = df[target].fillna('Unknown')
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = make_pipeline(CountVectorizer(), MultinomialNB())
            model.fit(X_train, y_train)
            models[target] = model
            
            predictions = model.predict(X_test)
            report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
            classification_reports[target] = report
        except Exception as e:
            st.warning(f"Could not train model for {target}: {e}")
            
    return models, classification_reports

# ==================== LOAD DATA ====================
df = load_data()

if df is not None:
    # Preprocess data
    df['sentiment_score'] = df['job_summary'].apply(analyze_sentiment)
    df['sentiment'] = df['sentiment_score'].apply(
        lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
    )
    
    # Extract skills if job_summary exists
    if 'job_summary' in df.columns:
        df['skills'] = df['job_summary'].apply(extract_skills)
    
    # Convert job_posted_time to datetime if possible
    if 'job_posted_time' in df.columns:
        try:
            df['posted_date'] = pd.to_datetime(df['job_posted_time'], errors='coerce')
        except:
            pass

# ==================== SIDEBAR ====================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png", width=100)
st.sidebar.title("🎯 Navigation")

# Main menu
menu_option = st.sidebar.radio(
    "Select Dashboard",
    ["🏠 Home", "📊 Sentiment Analysis", "🤖 Job Predictions", 
     "🔍 Advanced Search", "📈 Analytics", "💼 Recommendations"]
)

st.sidebar.markdown("---")

# Filters
st.sidebar.subheader("🔧 Filters")

if df is not None:
    # Location filter
    locations = ['All'] + sorted(df['job_location'].dropna().unique().tolist())
    selected_location = st.sidebar.selectbox("📍 Location", locations)
    
    # Company filter
    companies = ['All'] + sorted(df['company_name'].dropna().unique().tolist())
    selected_company = st.sidebar.selectbox("🏢 Company", companies)
    
    # Employment type filter
    if 'job_employment_type' in df.columns:
        emp_types = ['All'] + sorted(df['job_employment_type'].dropna().unique().tolist())
        selected_emp_type = st.sidebar.selectbox("💼 Employment Type", emp_types)
    else:
        selected_emp_type = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['job_location'] == selected_location]
    if selected_company != 'All':
        filtered_df = filtered_df[filtered_df['company_name'] == selected_company]
    if selected_emp_type != 'All' and 'job_employment_type' in df.columns:
        filtered_df = filtered_df[filtered_df['job_employment_type'] == selected_emp_type]

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use filters to narrow down results")

# ==================== MAIN CONTENT ====================

# HOME DASHBOARD
if menu_option == "🏠 Home":
    st.title("💼 LinkedIn Job Analytics Pro")
    st.markdown("### Welcome to your comprehensive job market analysis platform")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            f"{len(filtered_df):,}", 
            "Total Jobs", 
            "📋"
        ), unsafe_allow_html=True)
    
    with col2:
        avg_sentiment = filtered_df['sentiment_score'].mean()
        st.markdown(create_metric_card(
            f"{avg_sentiment:.2f}", 
            "Avg Sentiment", 
            "😊"
        ), unsafe_allow_html=True)
    
    with col3:
        positive_pct = (filtered_df['sentiment'] == 'positive').sum() / len(filtered_df) * 100
        st.markdown(create_metric_card(
            f"{positive_pct:.1f}%", 
            "Positive Jobs", 
            "✅"
        ), unsafe_allow_html=True)
    
    with col4:
        companies_count = filtered_df['company_name'].nunique()
        st.markdown(create_metric_card(
            f"{companies_count:,}", 
            "Companies", 
            "🏢"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Overview Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📊 Sentiment Distribution</div>', unsafe_allow_html=True)
        sentiment_counts = filtered_df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'},
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">🏆 Top 10 Companies</div>', unsafe_allow_html=True)
        top_companies = filtered_df['company_name'].value_counts().head(10)
        fig = px.bar(
            x=top_companies.values,
            y=top_companies.index,
            orientation='h',
            color=top_companies.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            xaxis_title="Number of Jobs",
            yaxis_title="Company",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent jobs table
    st.markdown('<div class="section-header">🆕 Recent Job Postings</div>', unsafe_allow_html=True)
    recent_jobs = filtered_df.head(10)[['job_title', 'company_name', 'job_location', 'sentiment', 'job_num_applicants']]
    st.dataframe(recent_jobs, use_container_width=True)

# SENTIMENT ANALYSIS
elif menu_option == "📊 Sentiment Analysis":
    st.title("📊 Sentiment Analysis Dashboard")
    
    # Search section
    st.markdown('<div class="section-header">🔍 Job Search</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_query = st.text_input("🔎 Enter job title keyword:", placeholder="e.g., Data Scientist, Software Engineer")
    with col2:
        search_button = st.button("🔍 Search Jobs", use_container_width=True)
    
    if search_button and search_query:
        with st.spinner("Searching..."):
            search_results = filtered_df[
                filtered_df['job_title'].str.contains(search_query, case=False, na=False)
            ]
            
            if not search_results.empty:
                st.success(f"✅ Found {len(search_results)} jobs matching '{search_query}'")
                
                # Display results
                for idx, row in search_results.head(20).iterrows():
                    with st.expander(f"**{row['job_title']}** at {row['company_name']}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"📍 **Location:** {row['job_location']}")
                        with col2:
                            st.write(f"👥 **Applicants:** {row['job_num_applicants']}")
                        with col3:
                            st.markdown(f"**Sentiment:** {get_sentiment_badge(row['sentiment'])}", unsafe_allow_html=True)
                        
                        if 'job_summary' in row and pd.notna(row['job_summary']):
                            st.write(f"📝 **Summary:** {str(row['job_summary'])[:300]}...")
                        
                        if 'apply_link' in row and pd.notna(row['apply_link']):
                            st.markdown(f"[🔗 Apply Now]({row['apply_link']})")
                
                # Export option
                csv = export_to_csv(search_results)
                st.download_button(
                    label="📥 Download Search Results (CSV)",
                    data=csv,
                    file_name=f"job_search_{search_query}.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"No jobs found for '{search_query}'. Try different keywords.")
    
    st.markdown("---")
    
    # Visualization section
    st.markdown('<div class="section-header">📈 Sentiment Visualizations</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox("Select Chart Type", 
                                  ["Bar Chart", "Pie Chart", "Donut Chart", "Treemap"])
    
    with col2:
        sentiment_filter = st.multiselect("Filter by Sentiment", 
                                         ['positive', 'negative', 'neutral'],
                                         default=['positive', 'negative', 'neutral'])
    
    viz_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]
    
    if chart_type == "Bar Chart":
        fig = px.bar(
            viz_df['sentiment'].value_counts(),
            color=viz_df['sentiment'].value_counts().index,
            color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'}
        )
        fig.update_layout(showlegend=False, xaxis_title="Sentiment", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Pie Chart":
        fig = px.pie(
            values=viz_df['sentiment'].value_counts().values,
            names=viz_df['sentiment'].value_counts().index,
            color=viz_df['sentiment'].value_counts().index,
            color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Donut Chart":
        fig = px.pie(
            values=viz_df['sentiment'].value_counts().values,
            names=viz_df['sentiment'].value_counts().index,
            hole=0.4,
            color=viz_df['sentiment'].value_counts().index,
            color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Treemap":
        fig = px.treemap(
            viz_df,
            path=['sentiment', 'company_name'],
            color='sentiment',
            color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'}
        )
        st.plotly_chart(fig, use_container_width=True)

# JOB PREDICTIONS
elif menu_option == "🤖 Job Predictions":
    st.title("🤖 AI-Powered Job Predictions")
    st.markdown("### Predict job attributes based on job title")
    
    required_columns = [
        'job_title', 'company_name', 'job_location',
        'job_employment_type', 'job_function',
        'job_industries', 'job_seniority_level'
    ]
    
    # Check if required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
    else:
        targets = [
            'company_name', 'job_location', 'job_employment_type',
            'job_function', 'job_industries', 'job_seniority_level'
        ]
        
        with st.spinner("🤖 Training AI models..."):
            models, classification_reports = train_models(df, targets)
        
        if models:
            st.success("✅ Models trained successfully!")
            
            # Model performance
            st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
            
            selected_target = st.selectbox("Select Attribute for Performance Metrics:", targets)
            
            if selected_target in classification_reports:
                report = classification_reports[selected_target]
                df_report = pd.DataFrame(report).loc[['precision', 'recall', 'f1-score']].T.reset_index()
                df_report.columns = ['class', 'precision', 'recall', 'f1-score']
                df_report = df_report[df_report['class'].str.contains('accuracy|macro|weighted') == False]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Precision', x=df_report['class'], y=df_report['precision']))
                fig.add_trace(go.Bar(name='Recall', x=df_report['class'], y=df_report['recall']))
                fig.add_trace(go.Bar(name='F1-Score', x=df_report['class'], y=df_report['f1-score']))
                
                fig.update_layout(
                    title=f'Model Performance: {selected_target.replace("_", " ").title()}',
                    barmode='group',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Prediction section
            st.markdown('<div class="section-header">🎯 Make Predictions</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                job_title_input = st.selectbox(
                    "Select or enter a job title:",
                    options=df['job_title'].unique()
                )
            with col2:
                predict_button = st.button("🔮 Predict", use_container_width=True)
            
            if predict_button and job_title_input:
                with st.spinner("Making predictions..."):
                    predictions = {}
                    for target, model in models.items():
                        try:
                            predictions[target] = model.predict([job_title_input])[0]
                        except:
                            predictions[target] = "N/A"
                    
                    st.markdown('<div class="success-box">✅ Predictions completed!</div>', unsafe_allow_html=True)
                    
                    # Display predictions
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Predicted Attributes")
                        predictions_df = pd.DataFrame(
                            predictions.items(),
                            columns=['Attribute', 'Predicted Value']
                        )
                        predictions_df['Attribute'] = predictions_df['Attribute'].str.replace('_', ' ').str.title()
                        st.dataframe(predictions_df, use_container_width=True)
                    
                    with col2:
                        st.subheader("📊 Visual Representation")
                        fig = px.bar(
                            predictions_df,
                            x='Attribute',
                            y=[1]*len(predictions_df),
                            text='Predicted Value',
                            color='Attribute'
                        )
                        fig.update_traces(textposition='inside')
                        fig.update_layout(showlegend=False, yaxis_visible=False)
                        st.plotly_chart(fig, use_container_width=True)

# ADVANCED SEARCH
elif menu_option == "🔍 Advanced Search":
    st.title("🔍 Advanced Job Search")
    st.markdown("### Use multiple filters to find your perfect job")
    
    st.markdown('<div class="section-header">🎛️ Search Filters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        keyword = st.text_input("🔤 Keyword in Title/Description")
        min_applicants = st.number_input("👥 Min Applicants", min_value=0, value=0)
    
    with col2:
        sentiment_search = st.multiselect(
            "😊 Sentiment",
            ['positive', 'negative', 'neutral'],
            default=['positive', 'negative', 'neutral']
        )
        
        if 'job_seniority_level' in df.columns:
            seniority = st.multiselect(
                "📊 Seniority Level",
                df['job_seniority_level'].dropna().unique()
            )
        else:
            seniority = []
    
    with col3:
        if 'job_industries' in df.columns:
            industries = st.multiselect(
                "🏭 Industry",
                df['job_industries'].dropna().unique()
            )
        else:
            industries = []
        
        sort_by = st.selectbox(
            "🔀 Sort By",
            ["Most Recent", "Most Applicants", "Best Sentiment"]
        )
    
    search_advanced = st.button("🔍 Search with Filters", use_container_width=True)
    
    if search_advanced:
        with st.spinner("Searching..."):
            result_df = filtered_df.copy()
            
            # Apply filters
            if keyword:
                result_df = result_df[
                    result_df['job_title'].str.contains(keyword, case=False, na=False) |
                    result_df['job_summary'].str.contains(keyword, case=False, na=False)
                ]
            
            if sentiment_search:
                result_df = result_df[result_df['sentiment'].isin(sentiment_search)]
            
            if min_applicants > 0:
                result_df = result_df[result_df['job_num_applicants'] >= min_applicants]
            
            if seniority and 'job_seniority_level' in df.columns:
                result_df = result_df[result_df['job_seniority_level'].isin(seniority)]
            
            if industries and 'job_industries' in df.columns:
                result_df = result_df[result_df['job_industries'].isin(industries)]
            
            # Sort results
            if sort_by == "Most Applicants":
                result_df = result_df.sort_values('job_num_applicants', ascending=False)
            elif sort_by == "Best Sentiment":
                result_df = result_df.sort_values('sentiment_score', ascending=False)
            
            # Display results
            st.markdown(f'<div class="success-box">✅ Found {len(result_df)} matching jobs</div>', unsafe_allow_html=True)
            
            if not result_df.empty:
                for idx, row in result_df.head(50).iterrows():
                    with st.expander(f"**{row['job_title']}** at {row['company_name']}"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"📍 **Location:** {row['job_location']}")
                        with col2:
                            st.write(f"👥 **Applicants:** {row['job_num_applicants']}")
                        with col3:
                            st.markdown(f"**Sentiment:** {get_sentiment_badge(row['sentiment'])}", unsafe_allow_html=True)
                        with col4:
                            st.write(f"📊 **Score:** {row['sentiment_score']:.2f}")
                        
                        if 'job_summary' in row and pd.notna(row['job_summary']):
                            st.write(f"📝 {str(row['job_summary'])[:400]}...")
                        
                        if 'apply_link' in row and pd.notna(row['apply_link']):
                            st.markdown(f"[🔗 Apply Now]({row['apply_link']})")
                
                # Export
                csv = export_to_csv(result_df)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="advanced_search_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No jobs match your criteria. Try adjusting the filters.")

# ANALYTICS
elif menu_option == "📈 Analytics":
    st.title("📈 Advanced Analytics Dashboard")
    
    # Trending Jobs
    st.markdown('<div class="section-header">🔥 Trending Jobs (Most Applications)</div>', unsafe_allow_html=True)
    
    trending = filtered_df.nlargest(20, 'job_num_applicants')[
        ['job_title', 'company_name', 'job_location', 'job_num_applicants', 'sentiment']
    ]
    
    fig = px.bar(
        trending,
        x='job_num_applicants',
        y='job_title',
        color='sentiment',
        orientation='h',
        color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'},
        hover_data=['company_name', 'job_location']
    )
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Location Analysis
    st.markdown('<div class="section-header">🗺️ Geographic Distribution</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        location_counts = filtered_df['job_location'].value_counts().head(15)
        fig = px.bar(
            x=location_counts.values,
            y=location_counts.index,
            orientation='h',
            color=location_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            title="Top 15 Locations by Job Count",
            xaxis_title="Number of Jobs",
            yaxis_title="Location",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        location_sentiment = filtered_df.groupby('job_location')['sentiment_score'].mean().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=location_sentiment.values,
            y=location_sentiment.index,
            orientation='h',
            color=location_sentiment.values,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(
            title="Top 15 Locations by Average Sentiment",
            xaxis_title="Average Sentiment Score",
            yaxis_title="Location",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Company Comparison
    st.markdown('<div class="section-header">🏢 Company Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_companies = filtered_df['company_name'].value_counts().head(10)
        fig = px.pie(
            values=top_companies.values,
            names=top_companies.index,
            title="Top 10 Companies by Job Postings"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Company sentiment comparison
        company_sentiment = filtered_df.groupby('company_name').agg({
            'sentiment_score': 'mean',
            'job_num_applicants': 'mean'
        }).sort_values('sentiment_score', ascending=False).head(10)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Avg Sentiment',
            x=company_sentiment.index,
            y=company_sentiment['sentiment_score'],
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="Top 10 Companies by Average Sentiment",
            xaxis_title="Company",
            yaxis_title="Sentiment Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Time Series Analysis
    if 'posted_date' in filtered_df.columns:
        st.markdown('<div class="section-header">⏰ Time Series Analysis</div>', unsafe_allow_html=True)
        
        time_df = filtered_df.dropna(subset=['posted_date']).copy()
        time_df['date'] = time_df['posted_date'].dt.date
        
        daily_posts = time_df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_posts,
            x='date',
            y='count',
            title='Job Postings Over Time',
            markers=True
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Job Postings",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment over time
        sentiment_time = time_df.groupby(['date', 'sentiment']).size().reset_index(name='count')
        
        fig = px.area(
            sentiment_time,
            x='date',
            y='count',
            color='sentiment',
            title='Sentiment Distribution Over Time',
            color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Skills Analysis
    if 'skills' in filtered_df.columns:
        st.markdown('<div class="section-header">💼 Skills Analysis</div>', unsafe_allow_html=True)
        
        all_skills = []
        for skills_list in filtered_df['skills'].dropna():
            if isinstance(skills_list, list):
                all_skills.extend(skills_list)
        
        if all_skills:
            skill_counts = pd.Series(all_skills).value_counts().head(20)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=skill_counts.values,
                    y=skill_counts.index,
                    orientation='h',
                    title='Top 20 Most Demanded Skills',
                    color=skill_counts.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    xaxis_title="Frequency",
                    yaxis_title="Skill",
                    showlegend=False,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.treemap(
                    names=skill_counts.index,
                    parents=[''] * len(skill_counts),
                    values=skill_counts.values,
                    title='Skills Treemap'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    # Employment Type Analysis
    if 'job_employment_type' in filtered_df.columns:
        st.markdown("---")
        st.markdown('<div class="section-header">💼 Employment Type Distribution</div>', unsafe_allow_html=True)
        
        emp_type_counts = filtered_df['job_employment_type'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=emp_type_counts.values,
                names=emp_type_counts.index,
                title='Employment Type Distribution',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            emp_sentiment = filtered_df.groupby('job_employment_type')['sentiment'].value_counts().unstack(fill_value=0)
            fig = px.bar(
                emp_sentiment,
                barmode='group',
                title='Sentiment by Employment Type',
                color_discrete_map={'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#f59e0b'}
            )
            st.plotly_chart(fig, use_container_width=True)

# RECOMMENDATIONS
elif menu_option == "💼 Recommendations":
    st.title("💼 Personalized Job Recommendations")
    st.markdown("### Get AI-powered job recommendations based on your preferences")
    
    st.markdown('<div class="section-header">👤 Your Preferences</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pref_location = st.multiselect(
            "📍 Preferred Locations",
            filtered_df['job_location'].unique(),
            max_selections=5
        )
    
    with col2:
        pref_sentiment = st.selectbox(
            "😊 Preferred Sentiment",
            ['positive', 'neutral', 'negative'],
            index=0
        )
    
    with col3:
        min_apps = st.slider(
            "👥 Min Applicants",
            0, int(filtered_df['job_num_applicants'].max()),
            0
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'job_employment_type' in filtered_df.columns:
            pref_emp_type = st.multiselect(
                "💼 Employment Type",
                filtered_df['job_employment_type'].unique()
            )
        else:
            pref_emp_type = []
    
    with col2:
        if 'job_seniority_level' in filtered_df.columns:
            pref_seniority = st.multiselect(
                "📊 Seniority Level",
                filtered_df['job_seniority_level'].unique()
            )
        else:
            pref_seniority = []
    
    pref_keywords = st.text_input(
        "🔤 Keywords (comma-separated)",
        placeholder="e.g., Python, Machine Learning, Remote"
    )
    
    get_recommendations = st.button("🎯 Get Recommendations", use_container_width=True)
    
    if get_recommendations:
        with st.spinner("🤖 Analyzing and finding best matches..."):
            recommended_df = filtered_df.copy()
            
            # Apply preference filters
            if pref_location:
                recommended_df = recommended_df[recommended_df['job_location'].isin(pref_location)]
            
            recommended_df = recommended_df[recommended_df['sentiment'] == pref_sentiment]
            
            if min_apps > 0:
                recommended_df = recommended_df[recommended_df['job_num_applicants'] >= min_apps]
            
            if pref_emp_type and 'job_employment_type' in filtered_df.columns:
                recommended_df = recommended_df[recommended_df['job_employment_type'].isin(pref_emp_type)]
            
            if pref_seniority and 'job_seniority_level' in filtered_df.columns:
                recommended_df = recommended_df[recommended_df['job_seniority_level'].isin(pref_seniority)]
            
            # Keyword matching
            if pref_keywords:
                keywords = [k.strip().lower() for k in pref_keywords.split(',')]
                mask = recommended_df['job_title'].str.lower().str.contains('|'.join(keywords), na=False) | \
                       recommended_df['job_summary'].str.lower().str.contains('|'.join(keywords), na=False)
                recommended_df = recommended_df[mask]
            
            # Sort by sentiment score
            recommended_df = recommended_df.sort_values('sentiment_score', ascending=False)
            
            if not recommended_df.empty:
                st.markdown(f'<div class="success-box">✅ Found {len(recommended_df)} recommended jobs for you!</div>', unsafe_allow_html=True)
                
                # Show match score
                st.markdown("### 🏆 Top Recommendations")
                
                for idx, row in recommended_df.head(20).iterrows():
                    match_score = (row['sentiment_score'] + 1) / 2 * 100  # Convert to percentage
                    
                    with st.expander(f"**{row['job_title']}** at {row['company_name']} | Match: {match_score:.0f}%"):
                        # Progress bar for match score
                        st.progress(match_score / 100)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"📍 **Location:** {row['job_location']}")
                        with col2:
                            st.write(f"👥 **Applicants:** {row['job_num_applicants']}")
                        with col3:
                            st.markdown(f"**Sentiment:** {get_sentiment_badge(row['sentiment'])}", unsafe_allow_html=True)
                        with col4:
                            st.write(f"📊 **Score:** {row['sentiment_score']:.2f}")
                        
                        if 'job_employment_type' in row:
                            st.write(f"💼 **Type:** {row['job_employment_type']}")
                        
                        if 'job_summary' in row and pd.notna(row['job_summary']):
                            st.write(f"📝 **Summary:** {str(row['job_summary'])[:400]}...")
                        
                        if 'skills' in row and pd.notna(row['skills']):
                            if isinstance(row['skills'], list) and row['skills']:
                                skills_html = ' '.join([f'<span style="background-color:#0077B5;color:white;padding:3px 10px;border-radius:15px;margin:2px;display:inline-block;">{skill}</span>' for skill in row['skills'][:10]])
                                st.markdown(f"**💡 Skills:** {skills_html}", unsafe_allow_html=True)
                        
                        if 'apply_link' in row and pd.notna(row['apply_link']):
                            st.markdown(f"[🔗 Apply Now]({row['apply_link']})")
                
                # Export recommendations
                st.markdown("---")
                csv = export_to_csv(recommended_df)
                st.download_button(
                    label="📥 Download All Recommendations (CSV)",
                    data=csv,
                    file_name="job_recommendations.csv",
                    mime="text/csv"
                )
                
                # Statistics
                st.markdown("### 📊 Recommendation Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Matches", len(recommended_df))
                with col2:
                    st.metric("Avg Sentiment", f"{recommended_df['sentiment_score'].mean():.2f}")
                with col3:
                    st.metric("Avg Applicants", f"{recommended_df['job_num_applicants'].mean():.0f}")
            
            else:
                st.warning("😔 No jobs match your preferences. Try adjusting your criteria.")
                
                # Suggestions
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("**💡 Suggestions:**")
                st.markdown("- Try selecting more locations")
                st.markdown("- Adjust sentiment preference")
                st.markdown("- Remove or modify keyword filters")
                st.markdown("- Lower minimum applicant requirements")
                st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #0077B5 0%, #00A0DC 100%); border-radius: 10px; color: white;'>
    <h3>💼 LinkedIn Job Analytics Pro</h3>
    <p>Powered by AI & Machine Learning | Built with ❤️ using Streamlit</p>
    <p style='font-size: 0.9em; opacity: 0.8;'>© 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)