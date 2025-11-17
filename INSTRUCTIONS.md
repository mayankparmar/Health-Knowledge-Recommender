# Health Knowledge Recommender - Usage Instructions

This document provides step-by-step instructions for using the Health Knowledge Recommender system.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [PDF Extraction and Annotation](#pdf-extraction-and-annotation)
4. [Knowledge Graph Visualization](#knowledge-graph-visualization)
5. [Web Application Usage](#web-application-usage)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (for cloning repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Health-Knowledge-Recommender.git
cd Health-Knowledge-Recommender
```

### Step 2: Install Core Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- pyyaml (configuration files)
- pandas and openpyxl (data processing)
- pdfplumber (PDF extraction)
- pyvis and networkx (visualization)
- streamlit (web application)

### Step 3: Verify Installation

```bash
python3 -c "import yaml, pandas, streamlit; print('Installation successful')"
```

## Data Preparation

### Reference Data Structure

The system requires three types of reference data:

**1. FAST Stages** (data/wp-01/fast-stages.json)
- Contains 16 FAST stage definitions
- Includes clinical characteristics, cognition levels, ADL/IADL status

**2. ADL and IADL Definitions** (data/wp-01/)
- [Katz] ADLs.xlsx - 6 basic activities of daily living
- [Lawton] IADL.xlsx - 8 instrumental activities

**3. FAST-Capability Mappings** (data/wp-02/)
- FAST and ADL IADL mapping.xlsx
- Links FAST stages to specific capability impairments

### PDF Resources

Place dementia care PDF documents in:
```
data/resources/
```

Supported PDF formats:
- Standard text-based PDFs
- Guidelines and care manuals
- Educational materials

## PDF Extraction and Annotation

### Method 1: Rule-based Annotation (Fastest)

**Step 1**: Review configuration file

```bash
cat config.yaml
```

Verify these settings:
```yaml
extraction:
  method: rule_based
  min_confidence: 0.3

documents:
  - doc_id: doc_001
    name: "Your PDF Name"
    file_path: "data/resources/your-file.pdf"
```

**Step 2**: Run extraction

```bash
python extract_and_annotate.py config.yaml
```

**Step 3**: Check outputs

```bash
ls -lh output/
# Should show:
# - knowledge_graph.jsonld
# - contents.csv
# - annotations.csv
```

### Method 2: LLM-based Annotation (More Accurate)

**Step 1**: Choose LLM provider

For cloud LLMs:
- Anthropic Claude
- OpenAI GPT
- Google Gemini

For local LLMs:
- Ollama (recommended)
- HuggingFace Transformers
- LlamaCPP

**Step 2**: Install LLM dependencies

For Anthropic Claude:
```bash
pip install anthropic
```

For Ollama (local):
```bash
# Install Ollama server from https://ollama.ai/
# Then pull a model:
ollama pull llama2
```

**Step 3**: Set API key (for cloud LLMs)

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

**Step 4**: Configure LLM settings

Edit config.llm.yaml:
```yaml
extraction:
  method: llm

llm:
  enabled: true
  provider: anthropic
  model: claude-3-sonnet-20240229
  api_key: ${ANTHROPIC_API_KEY}
  temperature: 0.3
  max_tokens: 2000
```

For Ollama:
```yaml
llm:
  enabled: true
  provider: ollama
  model: llama2
  base_url: http://localhost:11434
```

**Step 5**: Run extraction

```bash
python extract_and_annotate.py config.llm.yaml
```

### Understanding Output Files

**knowledge_graph.jsonld**
- JSON-LD format knowledge graph
- Contains all nodes and relationships
- Compatible with graph databases

**contents.csv**
- Extracted content items
- Columns: content_id, type, title, text, page, doc_id

**annotations.csv**
- Links content to FAST stages and capabilities
- Columns: annotation_id, content_id, fast_stages, capabilities, confidence

## Knowledge Graph Visualization

### Basic Visualization

**Generate all visualizations**:
```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view all
```

This creates:
- full_knowledge_graph.html (complete graph)
- fast_stages_view.html (stages and capabilities)
- content_view.html (documents and content)
- statistics.html (metrics dashboard)

### Specific Visualizations

**Full graph only**:
```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld
```

**FAST stages focused**:
```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view fast-stages
```

**Content focused**:
```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view content
```

**Stage-specific view**:
```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --stage FAST-4
```

**Statistics only**:
```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view stats
```

### Custom Output Directory

```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld \
    --output-dir my_visualizations
```

### Viewing Visualizations

Open HTML files in any web browser:

Linux:
```bash
xdg-open visualizations/full_knowledge_graph.html
```

macOS:
```bash
open visualizations/full_knowledge_graph.html
```

Windows:
```bash
start visualizations/full_knowledge_graph.html
```

## Web Application Usage

### Starting the Application

**Basic start**:
```bash
streamlit run app.py
```

The application will open automatically at http://localhost:8501

**Start with network access**:
```bash
streamlit run app.py --server.address 0.0.0.0
```

Access from other devices: http://your-ip:8501

**Custom port**:
```bash
streamlit run app.py --server.port 8080
```

### Using the Web Interface

#### Step 1: Select FAST Stage

In the sidebar:
1. Click the "Select FAST Stage" dropdown
2. Choose a stage (e.g., "FAST-4: Mild Dementia")
3. View stage information in the expandable section

#### Step 2: Select Functional Capability

1. Choose capability type: All, ADL, or IADL
2. Select specific capability from dropdown
   - Example: "ADL-1: Bathing"
   - Example: "IADL-7: Medication Management"

#### Step 3: Apply Advanced Filters (Optional)

Click "Advanced Filters" to expand:

**Minimum Confidence**:
- Slider from 0% to 100%
- Higher values show only high-confidence results
- Default: 0% (show all)

**Content Types**:
- Paragraph: Detailed information
- Tip: Actionable advice
- Section: Structural headings
- Default: Paragraphs and Tips selected

**Topics**:
- Multi-select from available topics
- Examples: Diagnosis, Treatment, Caregiver Support
- Leave empty for all topics

#### Step 4: Search

Click the "Search" button to execute query.

#### Step 5: Review Results

**Statistics Bar**:
- Results Found: Total matching items
- High Confidence: Items with confidence ≥ 70%
- Source Documents: Number of unique PDFs
- Avg. Confidence: Mean confidence score

**Result Cards**:
Each card shows:
- Title and content type icon
- Confidence badge (High/Medium/Low)
- Text preview (first 500 characters)
- Source citation (document, page number)
- "View Details" expander for full information

**Sorting Options**:
- Confidence (High to Low) - default
- Confidence (Low to High)
- Page Number

**Pagination**:
- Select results per page: 5, 10, 20, or 50
- Navigate with page number input
- Use Previous/Next buttons

#### Step 6: View Full Details

Click "View Details" on any result to see:

**Left Column**:
- Related FAST Stages with descriptions
- Topics covered

**Right Column**:
- Related Capabilities with names
- Target Audience (Patient/Caregiver/Both)
- Annotation Method (rule_based/llm)

**Full Text**:
- Complete extracted text
- Text area for easy reading and copying

### Example Searches

**Search 1: Bathing help for mild dementia**
- FAST Stage: FAST-4
- Capability: ADL-1 (Bathing)
- Result: Care guidelines specific to bathing at mild stage

**Search 2: Medication management for moderate dementia**
- FAST Stage: FAST-5
- Capability: IADL-7 (Medication Management)
- Result: Medication supervision strategies

**Search 3: All high-confidence caregiver support**
- FAST Stage: All Stages
- Capability: All Capabilities
- Min Confidence: 70%
- Topics: Caregiver Support
- Result: High-quality caregiver resources across all stages

## Advanced Configuration

### Customizing Extraction Parameters

Edit config.yaml:

```yaml
# PDF extraction settings
pdf:
  library: auto  # or "pdfplumber", "pypdf2"
  min_section_length: 50
  min_paragraph_length: 20
  extract_tips: true

# Rule-based annotation
rule_based:
  min_confidence: 0.3  # Lower = more permissive
  keyword_weight: 0.5
  topic_weight: 0.3

# Output settings
output:
  knowledge_graph: "output/knowledge_graph.jsonld"
  csv_export: true
```

### Adding New PDF Documents

**Step 1**: Place PDF in data/resources/

**Step 2**: Edit config.yaml:

```yaml
documents:
  - doc_id: doc_003
    name: "New Dementia Care Guide"
    file_path: "data/resources/new-guide.pdf"
    source_organization: "Health Organization"
    target_audience: "Both"
```

**Step 3**: Run extraction:

```bash
python extract_and_annotate.py config.yaml
```

### Configuring LLM Providers

**Anthropic Claude**:
```yaml
llm:
  provider: anthropic
  model: claude-3-sonnet-20240229
  api_key: ${ANTHROPIC_API_KEY}
```

**OpenAI GPT**:
```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
```

**Google Gemini**:
```yaml
llm:
  provider: gemini
  model: gemini-pro
  api_key: ${GOOGLE_API_KEY}
```

**Ollama (local)**:
```yaml
llm:
  provider: ollama
  model: llama2
  base_url: http://localhost:11434
```

**HuggingFace (local)**:
```yaml
llm:
  provider: huggingface
  model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  device: auto
```

## Troubleshooting

### Issue: PDF extraction fails

**Error**: "No PDF library available"

**Solution**:
```bash
pip install pdfplumber
# OR
pip install PyPDF2
```

### Issue: Cryptography errors with pdfplumber

**Solution**:
```bash
pip uninstall pdfplumber
pip install PyPDF2
```

Edit config.yaml:
```yaml
pdf:
  library: pypdf2
```

### Issue: LLM annotation fails

**Error**: "API key not found"

**Solution**:
```bash
export ANTHROPIC_API_KEY='your-key'
# OR add to config:
api_key: "your-key-here"
```

### Issue: Streamlit won't start

**Error**: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**:
```bash
pip install streamlit
```

### Issue: No results in web application

**Possible causes**:
1. Knowledge graph not generated
   - Run: `python extract_and_annotate.py config.yaml`

2. Confidence threshold too high
   - Lower minimum confidence slider to 0%

3. Filters too restrictive
   - Select "All Stages" and "All Capabilities"
   - Clear topic filters

### Issue: Visualization files are large/slow

**Solution**: Use filtered views
```bash
# Instead of full graph:
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view fast-stages

# Or stage-specific:
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --stage FAST-4
```

### Issue: Low annotation confidence

**For rule-based**:
- Lower min_confidence in config.yaml
- Adjust keyword weights

**For LLM-based**:
- Use more advanced models (GPT-4, Claude-3-Opus)
- Adjust temperature (lower = more consistent)
- Provide better document metadata

### Issue: Missing data files

**Error**: "FileNotFoundError: data/wp-01/fast-stages.json"

**Solution**:
Ensure all reference data is present:
```bash
ls data/wp-01/
# Should show:
# - fast-stages.json
# - [Katz] ADLs.xlsx
# - [Lawton] IADL.xlsx

ls data/wp-02/
# Should show:
# - FAST and ADL IADL mapping.xlsx
```

### Getting Help

1. Check log files in logs/ directory
2. Review error messages carefully
3. Consult README_EXTRACTION.md for detailed information
4. Open an issue on GitHub with:
   - Error message
   - Command executed
   - Python version
   - Operating system

## Best Practices

### For Extraction

1. Start with rule-based annotation to verify setup
2. Use LLM annotation for production-quality results
3. Review low-confidence annotations manually
4. Keep original PDFs for reference

### For Web Application

1. Set reasonable confidence thresholds (30-50%)
2. Use topic filters to narrow searches
3. Review full details before relying on content
4. Check source citations

### For Production Deployment

1. Use LLM annotation for better accuracy
2. Enable HTTPS for web application
3. Implement user authentication
4. Regular backups of knowledge graph
5. Monitor usage and performance
6. Update PDFs as guidelines change

## Next Steps

After completing basic setup:

1. **Add more PDFs**: Expand knowledge base with additional sources
2. **Customize visualization**: Modify colors and layouts
3. **Deploy web app**: Make accessible to users via Streamlit Cloud
4. **Implement feedback**: Collect user feedback and improve annotations
5. **Extend functionality**: Add user accounts, bookmarking, care plans

For detailed information on each component, see README_EXTRACTION.md.
