# PDF Data Extraction and Annotation System

A comprehensive, modular system for extracting and annotating information from dementia care PDF documents. The system categorizes content based on FAST (Functional Assessment Staging Tool) dementia stages and ADL/IADL (Activities of Daily Living / Instrumental Activities of Daily Living) functional capabilities.

## Features

### 🎯 Core Capabilities
- **Flexible PDF Extraction**: Support for multiple PDF libraries (pdfplumber, PyPDF2)
- **Dual Annotation Methods**:
  - **Rule-based**: Fast, keyword-matching approach
  - **LLM-based**: AI-powered annotation using Claude, GPT, or Gemini
- **Knowledge Graph Output**: JSON-LD format ready for graph databases (Neo4j, RDF stores)
- **Multi-format Export**: JSON-LD knowledge graphs + CSV views for analysis

### 🏗️ Architecture
- **Modular Design**: Pluggable extractors, annotators, and output formats
- **Configuration-driven**: YAML-based configuration for easy customization
- **Multi-provider LLM**: Support for Anthropic Claude, OpenAI GPT, and Google Gemini
- **Extensible**: Easy to add new PDF sources, annotation strategies, or output formats

## Installation

### 1. Basic Setup

```bash
# Install core dependencies
pip install -r requirements.txt
```

### 2. Install LLM Provider (Optional)

Choose one or more LLM providers if you want AI-powered annotation:

```bash
# For Anthropic Claude
pip install anthropic

# For OpenAI GPT
pip install openai

# For Google Gemini
pip install google-generativeai
```

### 3. Set API Key (for LLM-based annotation)

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key-here"

# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# For Google Gemini
export GEMINI_API_KEY="your-api-key-here"
```

## Quick Start

### 1. Rule-based Annotation (No API key required)

```bash
python extract_and_annotate.py
```

This uses the default `config.yaml` which runs rule-based annotation.

### 2. LLM-based Annotation

```bash
# Set your API key first
export ANTHROPIC_API_KEY="your-key"

# Run with LLM config
python extract_and_annotate.py --config config.llm.yaml
```

### 3. Command-line Options

```bash
# Use different annotation method
python extract_and_annotate.py --annotation-method llm

# Skip CSV generation
python extract_and_annotate.py --no-csv

# Skip knowledge graph generation
python extract_and_annotate.py --no-kg

# Change log level
python extract_and_annotate.py --log-level DEBUG
```

## Configuration

The system is configured via YAML files. See `config.yaml` for full options.

### Key Configuration Sections

#### 1. Data Paths
Specify paths to reference data (FAST stages, ADLs, IADLs, mappings):

```yaml
data_paths:
  fast_stages: data/wp-01/fast-stages.json
  adl_file: data/wp-01/[Katz] ADLs.xlsx
  iadl_file: data/wp-01/[Lawton] IADL.xlsx
  mapping_file: data/wp-02/FAST and ADL IADL mapping.xlsx
```

#### 2. Documents
List PDF files to process:

```yaml
documents:
  - id: doc_001
    name: "The Dementia Guide"
    file_path: data/resources/the_dementia_guide.pdf
    source_organization: "Alzheimer's Society UK"
    target_audience: "Both"
```

#### 3. Annotation Method
Choose annotation strategy:

```yaml
annotation:
  method: rule_based  # or "llm" or "hybrid"
```

#### 4. LLM Provider
Configure LLM for AI-powered annotation:

```yaml
llm:
  enabled: true
  provider: anthropic  # or "openai" or "gemini"
  model: claude-3-5-sonnet-20241022
  api_key_env_var: ANTHROPIC_API_KEY
  temperature: 0.3
  max_tokens: 2000
```

## Output Formats

### 1. Knowledge Graph (JSON-LD)

The primary output is a knowledge graph in JSON-LD format (`output/knowledge_graph.jsonld`):

```json
{
  "@context": {
    "@vocab": "http://health-knowledge-recommender.org/kg/",
    "fast": "http://health-knowledge-recommender.org/kg/fast/",
    "capability": "http://health-knowledge-recommender.org/kg/capability/"
  },
  "@graph": [
    {
      "@id": "content:123",
      "@type": "TipContent",
      "title": "Managing medication reminders...",
      "text": "Use a pill organizer...",
      "fastStages": [{"@id": "fast:FAST-4"}],
      "capabilities": [{"@id": "capability:IADL-7"}]
    }
  ]
}
```

**Benefits of JSON-LD:**
- ✅ Ready for Neo4j import
- ✅ Compatible with RDF/OWL ontologies
- ✅ Semantic web standards compliant
- ✅ Supports SPARQL queries
- ✅ Easy to extend with custom schemas

### 2. CSV Views

CSV files for analysis (`output/contents.csv`, `output/annotations.csv`):

```csv
content_id,type,title,fast_stages,capabilities,topics
c123,tip,"Managing medication...",FAST-4,IADL-7,"Medication Management"
```

## Project Structure

```
Health-Knowledge-Recommender/
├── src/pdf_extractor/
│   ├── models.py                    # Core data models
│   ├── loaders/                     # Data loaders
│   │   ├── fast_loader.py          # FAST stage loader
│   │   ├── capability_loader.py    # ADL/IADL loader
│   │   └── mapping_loader.py       # Mapping loader
│   ├── extractors/                  # PDF extraction
│   │   ├── base_extractor.py
│   │   └── pdf_extractor.py        # PDF content extraction
│   ├── annotators/                  # Annotation engines
│   │   ├── base_annotator.py
│   │   ├── rule_based_annotator.py # Keyword-based
│   │   └── llm_annotator.py        # AI-powered
│   └── builders/                    # Output builders
│       └── knowledge_graph_builder.py # JSON-LD KG builder
├── data/
│   ├── wp-01/                       # FAST, ADL, IADL data
│   ├── wp-02/                       # Mappings
│   └── resources/                   # PDF files
├── extract_and_annotate.py          # Main script
├── config.yaml                      # Default config
├── config.llm.yaml                  # LLM config example
└── requirements.txt                 # Dependencies
```

## Annotation Methods Comparison

| Feature | Rule-based | LLM-based | Hybrid |
|---------|-----------|-----------|--------|
| Speed | ⚡ Fast | 🐌 Slower | 🏃 Medium |
| Accuracy | 📊 Good | 🎯 Excellent | 🎯 Excellent |
| Cost | 💰 Free | 💸 API costs | 💸 API costs |
| Setup | ✅ Easy | 🔑 Needs API key | 🔑 Needs API key |
| Offline | ✅ Yes | ❌ No | ⚠️ Partial |

## Using Knowledge Graphs

### Import into Neo4j

```python
from neo4j import GraphDatabase

def import_knowledge_graph(uri, user, password, jsonld_file):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with open(jsonld_file) as f:
        kg = json.load(f)

    with driver.session() as session:
        for node in kg["@graph"]:
            # Create nodes and relationships
            # ... (implementation depends on your schema)
```

### Query with SPARQL (if using RDF store)

```sparql
# Find all content for FAST stage 4 related to medication
PREFIX fast: <http://health-knowledge-recommender.org/kg/fast/>
PREFIX cap: <http://health-knowledge-recommender.org/kg/capability/>

SELECT ?content ?title ?text
WHERE {
  ?annotation a Annotation ;
    annotates ?content ;
    fastStages fast:FAST-4 ;
    capabilities cap:IADL-7 .

  ?content title ?title ;
    text ?text .
}
```

## Adding New PDF Resources

1. **Add PDF file** to `data/resources/`

2. **Update config.yaml**:
```yaml
documents:
  - id: doc_003
    name: "New Care Guide"
    file_path: data/resources/new_guide.pdf
    source_organization: "Your Organization"
    target_audience: "Caregiver"
```

3. **Run extraction**:
```bash
python extract_and_annotate.py
```

## Customization Examples

### Change LLM Provider

Edit `config.llm.yaml`:

```yaml
llm:
  provider: openai       # Changed from anthropic
  model: gpt-4
  api_key_env_var: OPENAI_API_KEY
```

### Adjust Extraction Parameters

```yaml
extraction:
  min_section_length: 100    # Longer sections
  extract_tips: false        # Don't extract tips separately
```

### Custom Output Location

```yaml
output:
  output_dir: custom_output
  knowledge_graph_file: custom_output/my_graph.jsonld
```

## Troubleshooting

### "No PDF library available"
```bash
pip install pdfplumber
```

### "API key not found"
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### "Module not found"
```bash
# Make sure you're in the project root
cd Health-Knowledge-Recommender
python extract_and_annotate.py
```

### Low annotation confidence
- Try using `method: llm` instead of `rule_based`
- Adjust `min_confidence` in config

## Advanced Usage

### Programmatic Usage

```python
from src.pdf_extractor.loaders import FASTLoader, CapabilityLoader, MappingLoader
from src.pdf_extractor.extractors import PDFExtractor
from src.pdf_extractor.annotators import RuleBasedAnnotator
from src.pdf_extractor.models import PDFDocument

# Load reference data
fast_loader = FASTLoader("data/wp-01/fast-stages.json")
capability_loader = CapabilityLoader(
    "data/wp-01/[Katz] ADLs.xlsx",
    "data/wp-01/[Lawton] IADL.xlsx"
)
mapping_loader = MappingLoader("data/wp-02/FAST and ADL IADL mapping.xlsx")

# Extract from PDF
extractor = PDFExtractor()
document = PDFDocument(
    doc_id="doc_001",
    name="Test Document",
    file_path="data/resources/test.pdf"
)
contents = extractor.extract(document)

# Annotate
annotator = RuleBasedAnnotator(fast_loader, capability_loader, mapping_loader)
annotations = annotator.annotate_batch(contents, document)
```

## Performance Considerations

- **Rule-based**: Processes ~1000 items/minute
- **LLM-based**: Processes ~10-50 items/minute (depends on API rate limits)
- **Memory**: ~100MB for reference data + PDF size

## Future Enhancements

Potential extensions to this system:
- [ ] Multi-language support
- [ ] Custom ontology definitions
- [ ] Real-time API endpoint
- [ ] Web interface for annotation review
- [ ] Active learning for improving annotations
- [ ] Integration with electronic health records

## Support and Contribution

For issues, feature requests, or contributions, please refer to the main project documentation.

## License

See main project license.

---

**Health Knowledge Recommender Project**
*Making dementia care knowledge accessible and actionable*
