# 🧠 Health Knowledge Recommender

**Evidence-based dementia care information system with intelligent extraction, annotation, and dissemination**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

The Health Knowledge Recommender is a comprehensive system for extracting, annotating, and disseminating dementia care knowledge from PDF documents. It maps care guidelines to disease stages (FAST) and functional capabilities (ADL/IADL), making evidence-based information accessible to patients, caregivers, and healthcare providers.

### Key Features

- 📄 **PDF Data Extraction**: Automated extraction of structured content from dementia care PDFs
- 🏷️ **Intelligent Annotation**: Rule-based and LLM-powered annotation with FAST stages and ADL/IADL capabilities
- 🔗 **Knowledge Graph**: JSON-LD knowledge graph with semantic relationships
- 📊 **Interactive Visualization**: Multiple graph visualization modes for exploring relationships
- 🌐 **Web Application**: User-friendly platform for searching care information by stage and capability
- 🤖 **Multi-LLM Support**: Integration with Claude, GPT, Gemini, Ollama, HuggingFace, and LlamaCPP

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Health-Knowledge-Recommender.git
cd Health-Knowledge-Recommender

# Install dependencies
pip install -r requirements.txt
```

### 2. Extract and Annotate PDFs

```bash
# Using rule-based annotation (fastest)
python extract_and_annotate.py config.yaml

# Using LLM annotation (more accurate)
python extract_and_annotate.py config.llm.yaml

# Using local LLM (privacy-focused)
python extract_and_annotate.py config.local.ollama.yaml
```

### 3. Visualize Knowledge Graph

```bash
# Create interactive visualizations
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view all

# Open in browser
open visualizations/full_knowledge_graph.html
```

### 4. Launch Web Application

```bash
# Start the web platform
streamlit run app.py

# Access at http://localhost:8501
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PDF Documents                           │
│        (Dementia care guidelines and resources)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PDF Extractor                              │
│  • Text extraction (PyPDF2/pdfplumber)                      │
│  • Structure detection (sections, paragraphs, tips)         │
│  • Metadata capture (page numbers, hierarchies)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                Content Annotator                            │
│  • Rule-based: Keyword matching, topic detection            │
│  • LLM-based: AI-powered semantic annotation                │
│  • FAST stage mapping                                       │
│  • ADL/IADL capability mapping                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Knowledge Graph Builder                          │
│  • JSON-LD format (W3C standard)                            │
│  • Semantic relationships                                   │
│  • CSV exports for analysis                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
┌──────────────────────┐  ┌──────────────────────┐
│   Visualization      │  │   Web Application    │
│                      │  │                      │
│ • Full graph view    │  │ • Search by stage    │
│ • Filtered views     │  │ • Filter by          │
│ • Stage-specific     │  │   capability         │
│ • Statistics         │  │ • Browse content     │
│                      │  │ • Source citations   │
└──────────────────────┘  └──────────────────────┘
```

## Project Structure

```
Health-Knowledge-Recommender/
├── app.py                          # Streamlit web application
├── extract_and_annotate.py         # Main extraction script
├── visualize_knowledge_graph.py    # Knowledge graph visualizer
├── config.yaml                     # Configuration files
├── config.llm.yaml
├── config.local.*.yaml
│
├── src/pdf_extractor/              # Core extraction system
│   ├── models.py                   # Data models
│   ├── loaders/                    # Data loaders (FAST, ADL, IADL)
│   ├── extractors/                 # PDF extractors
│   ├── annotators/                 # Content annotators
│   └── builders/                   # Knowledge graph builders
│
├── data/
│   ├── wp-01/                      # Reference data
│   │   ├── fast-stages.json        # FAST stage definitions
│   │   ├── [Katz] ADLs.xlsx        # ADL definitions
│   │   └── [Lawton] IADL.xlsx      # IADL definitions
│   │
│   ├── wp-02/                      # Mappings
│   │   └── FAST and ADL IADL mapping.xlsx
│   │
│   └── resources/                  # Source PDF documents
│
├── output/                         # Generated outputs
│   ├── knowledge_graph.jsonld      # Knowledge graph
│   ├── contents.csv                # Extracted content
│   └── annotations.csv             # Annotations
│
├── visualizations/                 # Generated visualizations
│   ├── full_knowledge_graph.html
│   ├── fast_stages_view.html
│   └── content_view.html
│
└── README_EXTRACTION.md            # Detailed documentation
```

## Use Cases

### For Patients and Caregivers

- Find care information specific to disease stage
- Get practical tips for daily living activities
- Understand functional capability changes
- Access evidence-based guidelines

### For Healthcare Providers

- Quick reference for stage-appropriate care
- Evidence-based recommendations with citations
- Patient education materials
- Care planning resources

### For Researchers

- Analyze care guideline coverage
- Identify knowledge gaps
- Study functional capability patterns
- Build on the knowledge graph

## Documentation

- **[README_EXTRACTION.md](README_EXTRACTION.md)** - Comprehensive extraction system documentation
  - Installation and setup
  - Configuration options
  - LLM provider setup (cloud and local)
  - Knowledge graph visualization
  - Web application deployment
  - Troubleshooting

## Technology Stack

- **Language**: Python 3.9+
- **PDF Processing**: PyPDF2, pdfplumber
- **Data Processing**: pandas, openpyxl
- **LLM Integration**: Anthropic, OpenAI, Google AI, Ollama, HuggingFace, LlamaCPP
- **Knowledge Graph**: JSON-LD, NetworkX
- **Visualization**: pyvis
- **Web Framework**: Streamlit
- **Configuration**: YAML

## Data Model

### FAST Stages
16 stages from cognitively normal (FAST-1) to severe dementia (FAST-7f)

### ADL Capabilities (6)
1. Bathing
2. Dressing
3. Toileting
4. Transferring
5. Continence
6. Feeding

### IADL Capabilities (8)
1. Telephone use
2. Shopping
3. Food preparation
4. Housekeeping
5. Laundry
6. Transportation
7. Medication management
8. Financial management

### Content Types
- Sections (headings, structure)
- Paragraphs (detailed information)
- Tips (actionable advice)

## Contributing

We welcome contributions! Areas for contribution:

- Additional PDF sources
- Improved annotation algorithms
- New LLM provider integrations
- UI/UX enhancements
- Multi-language support
- Documentation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FAST staging system: Reisberg et al.
- ADL assessment: Katz et al.
- IADL assessment: Lawton & Brody
- Dementia care guidelines: Various healthcare organizations

## Citation

If you use this system in your research, please cite:

```bibtex
@software{health_knowledge_recommender,
  title = {Health Knowledge Recommender: Evidence-based Dementia Care Information System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/Health-Knowledge-Recommender}
}
```

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Health Knowledge Recommender Project**
*Making dementia care knowledge accessible and actionable*
