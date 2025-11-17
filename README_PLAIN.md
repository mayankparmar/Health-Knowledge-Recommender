# Health Knowledge Recommender

**Evidence-based dementia care information system with intelligent extraction, annotation, and dissemination**

## Overview

The Health Knowledge Recommender is a comprehensive system for extracting, annotating, and disseminating dementia care knowledge from PDF documents. It maps care guidelines to disease stages (FAST) and functional capabilities (ADL/IADL), making evidence-based information accessible to patients, caregivers, and healthcare providers.

## Key Features

- **PDF Data Extraction**: Automated extraction of structured content from dementia care PDFs
- **Intelligent Annotation**: Rule-based and LLM-powered annotation with FAST stages and ADL/IADL capabilities
- **Knowledge Graph**: JSON-LD knowledge graph with semantic relationships
- **Interactive Visualization**: Multiple graph visualization modes for exploring relationships
- **Web Application**: User-friendly platform for searching care information by stage and capability
- **Multi-LLM Support**: Integration with Claude, GPT, Gemini, Ollama, HuggingFace, and LlamaCPP

## System Components

### 1. PDF Extraction System

Extracts structured content from dementia care PDF documents including:
- Sections and hierarchical structure
- Paragraphs with contextual information
- Tips and actionable advice
- Page numbers and document metadata

### 2. Annotation System

Links extracted content to clinical frameworks:
- **FAST Stages**: 16 stages of functional decline
- **ADL Capabilities**: 6 basic activities of daily living
- **IADL Capabilities**: 8 instrumental activities of daily living

Annotation methods:
- **Rule-based**: Keyword matching and pattern recognition
- **LLM-based**: AI-powered semantic understanding

### 3. Knowledge Graph

Stores relationships in JSON-LD format:
- Content nodes with full text and metadata
- FAST stage nodes with clinical characteristics
- Capability nodes (ADL/IADL) with descriptions
- Mapping nodes showing stage-capability relationships
- Annotation nodes linking content to stages and capabilities

### 4. Visualization System

Interactive HTML visualizations:
- Full knowledge graph view
- FAST stages focused view
- Content and annotations view
- Stage-specific filtered views
- Statistics dashboard

### 5. Web Application

Streamlit-based platform for end users:
- Search by FAST stage
- Filter by functional capability
- Advanced filtering (confidence, topics, content types)
- Source citations with page numbers
- Mobile-responsive interface

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Health-Knowledge-Recommender.git
cd Health-Knowledge-Recommender

# Install dependencies
pip install -r requirements.txt
```

### Extract and Annotate PDFs

```bash
# Using rule-based annotation
python extract_and_annotate.py config.yaml

# Using LLM annotation
python extract_and_annotate.py config.llm.yaml

# Using local LLM
python extract_and_annotate.py config.local.ollama.yaml
```

### Visualize Knowledge Graph

```bash
# Create visualizations
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view all

# View in browser
open visualizations/full_knowledge_graph.html
```

### Launch Web Application

```bash
# Start the web server
streamlit run app.py

# Access at http://localhost:8501
```

## Project Structure

```
Health-Knowledge-Recommender/
├── app.py                          # Web application
├── extract_and_annotate.py         # Main extraction script
├── visualize_knowledge_graph.py    # Graph visualizer
├── config.yaml                     # Configuration files
├── config.llm.yaml
├── config.local.*.yaml
│
├── src/pdf_extractor/              # Core system
│   ├── models.py                   # Data models
│   ├── loaders/                    # Data loaders
│   ├── extractors/                 # PDF extractors
│   ├── annotators/                 # Content annotators
│   └── builders/                   # Knowledge graph builders
│
├── data/
│   ├── wp-01/                      # Reference data
│   │   ├── fast-stages.json
│   │   ├── [Katz] ADLs.xlsx
│   │   └── [Lawton] IADL.xlsx
│   ├── wp-02/                      # Mappings
│   │   └── FAST and ADL IADL mapping.xlsx
│   └── resources/                  # Source PDFs
│
├── output/                         # Generated outputs
│   ├── knowledge_graph.jsonld
│   ├── contents.csv
│   └── annotations.csv
│
└── visualizations/                 # HTML visualizations
```

## Data Model

### FAST Stages (16 total)

Functional Assessment Staging Tool stages from cognitively normal to severe dementia:
- FAST-1: Normal adult
- FAST-2: Normal aged adult
- FAST-3: Early Alzheimer's disease
- FAST-4: Mild Alzheimer's disease
- FAST-5: Moderate Alzheimer's disease
- FAST-6a through FAST-6e: Moderately severe stages
- FAST-7a through FAST-7f: Severe stages

### ADL Capabilities (6 basic activities)

1. Bathing
2. Dressing
3. Toileting
4. Transferring
5. Continence
6. Feeding

### IADL Capabilities (8 complex activities)

1. Telephone use
2. Shopping
3. Food preparation
4. Housekeeping
5. Laundry
6. Transportation
7. Medication management
8. Financial management

### Content Types

- **Sections**: Headings and structural elements
- **Paragraphs**: Detailed care information
- **Tips**: Actionable advice and recommendations

## Use Cases

### For Patients and Caregivers

- Find stage-specific care information
- Get practical daily living tips
- Understand functional changes
- Access evidence-based guidelines with sources

### For Healthcare Providers

- Quick clinical reference
- Evidence-based recommendations
- Patient education materials
- Care planning resources

### For Researchers

- Analyze guideline coverage
- Identify knowledge gaps
- Study functional patterns
- Extend the knowledge graph

## Technology Stack

- **Language**: Python 3.9+
- **PDF Processing**: PyPDF2, pdfplumber
- **Data Processing**: pandas, openpyxl
- **LLM Integration**: Anthropic, OpenAI, Google AI, Ollama, HuggingFace, LlamaCPP
- **Knowledge Graph**: JSON-LD, NetworkX
- **Visualization**: pyvis
- **Web Framework**: Streamlit
- **Configuration**: YAML

## Configuration

### Rule-based Annotation

```yaml
extraction:
  method: rule_based
  min_confidence: 0.3
```

### LLM-based Annotation (Cloud)

```yaml
extraction:
  method: llm

llm:
  enabled: true
  provider: anthropic
  model: claude-3-sonnet-20240229
  api_key: ${ANTHROPIC_API_KEY}
```

### LLM-based Annotation (Local)

```yaml
extraction:
  method: llm

llm:
  enabled: true
  provider: ollama
  model: llama2
  base_url: http://localhost:11434
```

## Performance Metrics

- **Rule-based annotation**: ~1000 items per minute
- **LLM-based annotation**: ~10-50 items per minute (API dependent)
- **Memory usage**: ~100MB for reference data plus PDF size
- **Knowledge graph**: Supports thousands of nodes efficiently

## Deployment Options

### Local Development

```bash
streamlit run app.py
```

### Streamlit Cloud (Free)

1. Push code to GitHub
2. Connect repository at share.streamlit.io
3. Deploy with one click

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

### Production Hosting

- AWS EC2/ECS
- Google Cloud Run
- Azure App Service
- DigitalOcean App Platform

## Security Considerations

For production deployment:
- Enable HTTPS/SSL
- Implement authentication (OAuth, LDAP)
- Set CORS policies
- Consider rate limiting
- Review data privacy compliance
- No PHI/PII storage in current implementation

## Accessibility Features

- High contrast color scheme
- Large readable text
- Clear navigation structure
- Keyboard-friendly interface
- Screen reader compatible
- Mobile responsive design

## Documentation

- **README_EXTRACTION.md**: Comprehensive extraction system documentation
- **INSTRUCTIONS.md**: Step-by-step usage instructions
- **PRESENTATION.md**: Key points for presentations

## Contributing

Contributions welcome in:
- Additional PDF sources
- Improved annotation algorithms
- New LLM provider integrations
- UI/UX enhancements
- Multi-language support
- Documentation improvements

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- FAST staging system: Reisberg et al.
- ADL assessment: Katz et al.
- IADL assessment: Lawton and Brody
- Dementia care guidelines: Various healthcare organizations

## Citation

```bibtex
@software{health_knowledge_recommender,
  title = {Health Knowledge Recommender: Evidence-based Dementia Care Information System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/Health-Knowledge-Recommender}
}
```

## Support

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Health Knowledge Recommender Project**
*Making dementia care knowledge accessible and actionable*
