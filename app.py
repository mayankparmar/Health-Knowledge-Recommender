#!/usr/bin/env python3
"""
Dementia Care Knowledge Platform - Streamlit Web Application
Health Knowledge Recommender Project

Provides stage-based and capability-filtered dementia care information
from annotated PDF content.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="Dementia Care Knowledge Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for healthcare-appropriate styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .content-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .source-citation {
        color: #666;
        font-size: 0.9rem;
        font-style: italic;
        margin-top: 0.5rem;
    }
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.85rem;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.85rem;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.85rem;
    }
    .stat-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1976d2;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load knowledge graph data from CSV files."""
    try:
        contents_df = pd.read_csv('output/contents.csv')
        annotations_df = pd.read_csv('output/annotations.csv')

        # Load FAST stages info
        with open('data/wp-01/fast-stages.json', 'r') as f:
            fast_data = json.load(f)
            fast_stages = fast_data.get('fast_stages', [])

        # Load capabilities info
        adl_df = pd.read_excel('data/wp-01/[Katz] ADLs.xlsx')
        iadl_df = pd.read_excel('data/wp-01/[Lawton] IADL.xlsx')

        return {
            'contents': contents_df,
            'annotations': annotations_df,
            'fast_stages': fast_stages,
            'adl': adl_df,
            'iadl': iadl_df
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def get_capability_name(cap_id: str, data: Dict) -> str:
    """Get capability name from ID."""
    if cap_id.startswith('ADL-'):
        idx = int(cap_id.split('-')[1]) - 1
        if idx < len(data['adl']):
            return data['adl'].iloc[idx]['Capability Name']
    elif cap_id.startswith('IADL-'):
        idx = int(cap_id.split('-')[1]) - 1
        if idx < len(data['iadl']):
            return data['iadl'].iloc[idx]['Capability Name']
    return cap_id


def get_fast_stage_info(stage_code: str, data: Dict) -> Optional[Dict]:
    """Get FAST stage information."""
    for stage in data['fast_stages']:
        if stage['stage_code'] == stage_code:
            return stage
    return None


def query_content(
    data: Dict,
    fast_stage: Optional[str] = None,
    capability_id: Optional[str] = None,
    topics: Optional[List[str]] = None,
    min_confidence: float = 0.0,
    content_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Query content based on filters.

    Args:
        data: Loaded data dictionary
        fast_stage: FAST stage code (e.g., "FAST-4")
        capability_id: Capability ID (e.g., "ADL-1")
        topics: List of topics to filter by
        min_confidence: Minimum confidence score
        content_types: List of content types to include

    Returns:
        List of matching content items with annotations
    """
    annotations = data['annotations']
    contents = data['contents']

    # Filter annotations
    filtered = annotations.copy()

    if fast_stage:
        filtered = filtered[
            filtered['fast_stages'].fillna('').str.contains(fast_stage, na=False)
        ]

    if capability_id:
        filtered = filtered[
            filtered['capabilities'].fillna('').str.contains(capability_id, na=False)
        ]

    if topics:
        topic_pattern = '|'.join(topics)
        filtered = filtered[
            filtered['topics'].fillna('').str.contains(topic_pattern, na=False)
        ]

    if min_confidence > 0:
        filtered = filtered[
            filtered['fast_confidence'].fillna(0) >= min_confidence
        ]

    # Join with contents
    results = []
    for _, ann in filtered.iterrows():
        content = contents[contents['content_id'] == ann['content_id']]
        if not content.empty:
            content_row = content.iloc[0]

            # Filter by content type if specified
            if content_types and content_row['type'] not in content_types:
                continue

            results.append({
                'content_id': content_row['content_id'],
                'type': content_row['type'],
                'title': content_row['title'],
                'text': content_row['text'],
                'page': content_row['page'],
                'doc_id': content_row['doc_id'],
                'fast_stages': ann['fast_stages'],
                'fast_confidence': ann['fast_confidence'],
                'capabilities': ann['capabilities'],
                'capability_confidence': ann['capability_confidence'],
                'topics': ann['topics'],
                'target_audience': ann['target_audience'],
                'method': ann['method']
            })

    # Sort by confidence
    results.sort(key=lambda x: x['fast_confidence'], reverse=True)

    return results


def display_content_card(item: Dict, data: Dict):
    """Display a content item as a card."""

    # Confidence badge
    conf = item['fast_confidence']
    if conf >= 0.7:
        conf_class = "confidence-high"
        conf_label = "High Confidence"
    elif conf >= 0.4:
        conf_class = "confidence-medium"
        conf_label = "Medium Confidence"
    else:
        conf_class = "confidence-low"
        conf_label = "Low Confidence"

    # Content type badge
    type_emoji = {
        'section': '📑',
        'paragraph': '📄',
        'tip': '💡'
    }
    emoji = type_emoji.get(item['type'], '📄')

    # Build HTML
    html = f"""
    <div class="content-card">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
            <div>
                <h3 style="margin: 0; color: #1f77b4;">{emoji} {item['title'][:100]}</h3>
            </div>
            <div>
                <span class="{conf_class}">{conf_label}</span>
            </div>
        </div>

        <div style="margin-bottom: 1rem; line-height: 1.6;">
            {item['text'][:500]}{"..." if len(item['text']) > 500 else ""}
        </div>

        <div class="source-citation">
            📍 Document: {item['doc_id']} | Page {item['page']} |
            Type: {item['type'].title()} |
            Confidence: {conf:.0%}
        </div>
    """

    # Add metadata
    with st.expander("📊 View Details"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**FAST Stages:**")
            if pd.notna(item['fast_stages']):
                stages = item['fast_stages'].split(',')
                for stage in stages:
                    stage_info = get_fast_stage_info(stage, data)
                    if stage_info:
                        st.markdown(f"- {stage}: {stage_info['stage_name']}")
                    else:
                        st.markdown(f"- {stage}")
            else:
                st.markdown("*Not specified*")

            st.markdown("**Topics:**")
            if pd.notna(item['topics']):
                for topic in item['topics'].split(','):
                    st.markdown(f"- {topic}")
            else:
                st.markdown("*Not specified*")

        with col2:
            st.markdown("**Capabilities:**")
            if pd.notna(item['capabilities']):
                caps = item['capabilities'].split(',')
                for cap in caps:
                    cap_name = get_capability_name(cap, data)
                    st.markdown(f"- {cap}: {cap_name}")
            else:
                st.markdown("*Not specified*")

            st.markdown("**Metadata:**")
            st.markdown(f"- Target: {item['target_audience']}")
            st.markdown(f"- Method: {item['method']}")

        # Full text
        st.markdown("**Full Text:**")
        st.text_area("", value=item['text'], height=200, key=f"text_{item['content_id']}")

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def main():
    """Main application."""

    # Header
    st.markdown('<div class="main-header">🧠 Dementia Care Knowledge Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evidence-based care information tailored to disease stage and functional needs</div>', unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading knowledge base..."):
        data = load_data()

    if data is None:
        st.error("Failed to load data. Please ensure the knowledge graph has been generated.")
        st.info("Run: `python extract_and_annotate.py config.yaml` to generate the knowledge graph.")
        return

    # Sidebar - Search Filters
    st.sidebar.header("🔍 Search Filters")

    # FAST Stage selection
    st.sidebar.subheader("Disease Stage (FAST)")
    fast_options = ["All Stages"] + [stage['stage_code'] for stage in data['fast_stages']]
    selected_stage = st.sidebar.selectbox(
        "Select FAST Stage",
        fast_options,
        help="Choose the Functional Assessment Staging Tool (FAST) stage"
    )

    if selected_stage != "All Stages":
        stage_info = get_fast_stage_info(selected_stage, data)
        if stage_info:
            with st.sidebar.expander("ℹ️ Stage Information"):
                st.markdown(f"**{stage_info['stage_name']}**")
                st.markdown(f"*{stage_info['clinical_characteristics']}*")

    # Capability selection
    st.sidebar.subheader("Functional Capability")

    capability_type = st.sidebar.radio(
        "Capability Type",
        ["All", "ADL (Activities of Daily Living)", "IADL (Instrumental ADL)"]
    )

    capability_options = ["All Capabilities"]
    if capability_type == "ADL (Activities of Daily Living)" or capability_type == "All":
        for idx, row in data['adl'].iterrows():
            capability_options.append(f"ADL-{idx+1}: {row['Capability Name']}")

    if capability_type == "IADL (Instrumental ADL)" or capability_type == "All":
        for idx, row in data['iadl'].iterrows():
            capability_options.append(f"IADL-{idx+1}: {row['Capability Name']}")

    selected_capability = st.sidebar.selectbox(
        "Select Capability",
        capability_options,
        help="Choose a specific activity of daily living"
    )

    # Advanced filters
    with st.sidebar.expander("⚙️ Advanced Filters"):
        min_confidence = st.slider(
            "Minimum Confidence",
            0.0, 1.0, 0.0, 0.1,
            help="Filter results by annotation confidence"
        )

        content_types = st.multiselect(
            "Content Types",
            ["section", "paragraph", "tip"],
            default=["paragraph", "tip"],
            help="Select types of content to display"
        )

        # Topic filter
        all_topics = set()
        for topics in data['annotations']['topics'].dropna():
            all_topics.update(topics.split(','))
        all_topics = sorted(list(all_topics))

        selected_topics = st.multiselect(
            "Topics",
            all_topics,
            help="Filter by specific topics"
        )

    # Search button
    search_clicked = st.sidebar.button("🔍 Search", type="primary", use_container_width=True)

    # Main content area
    if search_clicked or 'last_search' in st.session_state:
        # Parse selections
        fast_stage = None if selected_stage == "All Stages" else selected_stage

        capability_id = None
        if selected_capability != "All Capabilities":
            capability_id = selected_capability.split(':')[0]

        topics = selected_topics if selected_topics else None

        # Store search in session
        st.session_state['last_search'] = {
            'fast_stage': fast_stage,
            'capability_id': capability_id,
            'topics': topics,
            'min_confidence': min_confidence,
            'content_types': content_types
        }

        # Query
        with st.spinner("Searching knowledge base..."):
            results = query_content(
                data,
                fast_stage=fast_stage,
                capability_id=capability_id,
                topics=topics,
                min_confidence=min_confidence,
                content_types=content_types
            )

        # Display results
        st.markdown("---")

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(results)}</div>
                <div class="stat-label">Results Found</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            high_conf = len([r for r in results if r['fast_confidence'] >= 0.7])
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{high_conf}</div>
                <div class="stat-label">High Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            unique_docs = len(set(r['doc_id'] for r in results))
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{unique_docs}</div>
                <div class="stat-label">Source Documents</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            avg_conf = sum(r['fast_confidence'] for r in results) / len(results) if results else 0
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{avg_conf:.0%}</div>
                <div class="stat-label">Avg. Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Display results
        if results:
            st.subheader(f"📚 {len(results)} Care Recommendations")

            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                ["Confidence (High to Low)", "Confidence (Low to High)", "Page Number"],
                key="sort_option"
            )

            if sort_by == "Confidence (Low to High)":
                results.sort(key=lambda x: x['fast_confidence'])
            elif sort_by == "Page Number":
                results.sort(key=lambda x: x['page'] if pd.notna(x['page']) else 0)

            # Pagination
            items_per_page = st.selectbox("Results per page", [5, 10, 20, 50], index=1)
            total_pages = (len(results) - 1) // items_per_page + 1

            if 'current_page' not in st.session_state:
                st.session_state['current_page'] = 1

            # Page navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page = st.number_input(
                    f"Page (1-{total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=st.session_state['current_page'],
                    key="page_input"
                )
                st.session_state['current_page'] = page

            # Display current page
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(results))

            for item in results[start_idx:end_idx]:
                display_content_card(item, data)

            # Page navigation buttons
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if page > 1:
                    if st.button("⬅️ Previous"):
                        st.session_state['current_page'] -= 1
                        st.rerun()
            with col3:
                if page < total_pages:
                    if st.button("Next ➡️"):
                        st.session_state['current_page'] += 1
                        st.rerun()

        else:
            st.info("No results found. Try adjusting your search criteria.")
            st.markdown("""
            **Suggestions:**
            - Try selecting "All Stages" or "All Capabilities"
            - Lower the minimum confidence threshold
            - Remove topic filters
            - Include more content types
            """)

    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Dementia Care Knowledge Platform

        This platform helps you find evidence-based care information tailored to:
        - **Disease Stage**: Based on the FAST (Functional Assessment Staging Tool)
        - **Functional Capabilities**: Activities of Daily Living (ADL) and Instrumental ADL (IADL)

        ### How to Use:

        1. **Select a FAST Stage** from the sidebar (or choose "All Stages")
        2. **Choose a Functional Capability** you want information about
        3. **Optionally** adjust advanced filters for more specific results
        4. **Click Search** to find relevant care recommendations

        ### About the Data:

        - **{0}** total content items extracted from dementia care PDFs
        - **{1}** annotated with FAST stages and capabilities
        - Content includes care guidelines, tips, and best practices

        ### FAST Stages Overview:
        """.format(
            len(data['contents']),
            len(data['annotations'])
        ))

        # Display FAST stages
        for stage in data['fast_stages'][:7]:  # Show first 7 stages
            with st.expander(f"**{stage['stage_code']}: {stage['stage_name']}**"):
                st.markdown(f"*{stage['clinical_characteristics']}*")
                st.markdown(f"**Cognition:** {stage['cognition']}")
                st.markdown(f"**ADL Status:** {stage['adl_status']}")

        st.info("👈 Use the sidebar to start searching for care information")


if __name__ == "__main__":
    main()
