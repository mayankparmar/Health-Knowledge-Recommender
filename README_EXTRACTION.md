# PDF Data Extraction and Annotation System

A comprehensive, modular system for extracting and annotating information from dementia care PDF documents. The system categorizes content based on FAST (Functional Assessment Staging Tool) dementia stages and ADL/IADL (Activities of Daily Living / Instrumental Activities of Daily Living) functional capabilities.

## Features

### 🎯 Core Capabilities
- **Flexible PDF Extraction**: Support for multiple PDF libraries (pdfplumber, PyPDF2)
- **Multiple Annotation Methods**:
  - **Rule-based**: Fast, keyword-matching approach (offline, free)
  - **Cloud LLM**: AI-powered annotation using Claude, GPT, or Gemini
  - **Local LLM**: Private, offline AI using Ollama, HuggingFace, or LlamaCPP
- **Knowledge Graph Output**: JSON-LD format ready for graph databases (Neo4j, RDF stores)
- **Multi-format Export**: JSON-LD knowledge graphs + CSV views for analysis

### 🏗️ Architecture
- **Modular Design**: Pluggable extractors, annotators, and output formats
- **Configuration-driven**: YAML-based configuration for easy customization
- **Multi-provider LLM**: Cloud (Anthropic, OpenAI, Google) & Local (Ollama, HuggingFace, LlamaCPP)
- **Privacy-first Options**: Full offline mode with local LLMs
- **Extensible**: Easy to add new PDF sources, annotation strategies, or output formats

## Installation

### 1. Basic Setup

```bash
# Install core dependencies
pip install -r requirements.txt
```

### 2. Install LLM Provider (Optional)

You can choose between **cloud-based** or **local** LLM providers for AI-powered annotation.

#### **Cloud LLM Providers** (Requires API key)

```bash
# For Anthropic Claude
pip install anthropic

# For OpenAI GPT
pip install openai

# For Google Gemini
pip install google-generativeai
```

Then set your API key:

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key-here"

# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# For Google Gemini
export GEMINI_API_KEY="your-api-key-here"
```

#### **Local LLM Providers** (No API key, runs offline)

**Option A: Ollama** (Recommended - Easiest)

```bash
# 1. Install Ollama from https://ollama.ai/
# 2. Pull a model
ollama pull llama2

# 3. Install Python client (optional)
pip install ollama

# 4. Run with Ollama config
python extract_and_annotate.py --config config.local.ollama.yaml
```

**Option B: HuggingFace Transformers**

```bash
# Install transformers and torch
pip install transformers torch accelerate

# Models download automatically on first run
python extract_and_annotate.py --config config.local.huggingface.yaml
```

**Option C: LlamaCPP** (GGUF files)

```bash
# Install llama-cpp-python
pip install llama-cpp-python

# For GPU support (Linux/Windows with NVIDIA GPU):
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Download a GGUF model from HuggingFace (TheBloke)
# Update model_path in config.local.llamacpp.yaml
python extract_and_annotate.py --config config.local.llamacpp.yaml
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

### 3. Local LLM Annotation (Offline, No API key)

```bash
# Using Ollama (easiest)
ollama pull llama2
python extract_and_annotate.py --config config.local.ollama.yaml

# Using HuggingFace
python extract_and_annotate.py --config config.local.huggingface.yaml

# Using LlamaCPP
python extract_and_annotate.py --config config.local.llamacpp.yaml
```

### 4. Command-line Options

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

**Cloud LLM:**
```yaml
llm:
  enabled: true
  provider: anthropic  # or "openai" or "gemini"
  model: claude-3-5-sonnet-20241022
  api_key_env_var: ANTHROPIC_API_KEY
  temperature: 0.3
  max_tokens: 2000
```

**Local LLM (Ollama):**
```yaml
llm:
  enabled: true
  provider: ollama
  model: llama2  # or mistral, mixtral, etc.
  base_url: http://localhost:11434
  temperature: 0.3
  max_tokens: 2000
```

**Local LLM (HuggingFace):**
```yaml
llm:
  enabled: true
  provider: huggingface
  model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  device: auto  # or "cuda", "cpu", "mps"
  temperature: 0.3
  max_tokens: 2000
```

**Local LLM (LlamaCPP):**
```yaml
llm:
  enabled: true
  provider: llamacpp
  model_path: models/llama-2-7b-chat.Q4_K_M.gguf
  n_ctx: 4096
  n_gpu_layers: 0  # Set higher for GPU
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

| Feature | Rule-based | Cloud LLM | Local LLM | Hybrid |
|---------|-----------|-----------|-----------|--------|
| Speed | ⚡ Very Fast | 🐌 Slower | 🏃 Medium | 🏃 Medium |
| Accuracy | 📊 Good | 🎯 Excellent | 🎯 Very Good | 🎯 Excellent |
| Cost | 💰 Free | 💸 API costs | 💰 Free | 💸 Some API costs |
| Setup | ✅ Easy | 🔑 API key | 🔧 Medium | 🔑 API key |
| Offline | ✅ Yes | ❌ No | ✅ Yes | ⚠️ Partial |
| Privacy | ✅ Full | ⚠️ Cloud | ✅ Full | ⚠️ Mixed |

## Local LLM Setup Guide

### Why Use Local LLMs?

- **Privacy**: Your data never leaves your machine
- **Cost**: No API fees, unlimited usage
- **Offline**: Works without internet
- **Control**: Full control over model and configuration

### Option 1: Ollama (Recommended)

**Easiest setup, best for beginners**

1. **Install Ollama**
   - Visit https://ollama.ai/
   - Download for your OS (Mac, Linux, Windows)
   - Run installer

2. **Pull a model**
   ```bash
   # Small, fast model (~4GB)
   ollama pull llama2

   # Larger, more accurate (~4.7GB)
   ollama pull llama2:13b

   # Very capable mixture model (~26GB)
   ollama pull mixtral

   # Small and fast (~1.6GB)
   ollama pull phi
   ```

3. **Run extraction**
   ```bash
   python extract_and_annotate.py --config config.local.ollama.yaml
   ```

**Available Ollama Models:**
- `llama2` - 7B params, good balance
- `llama2:13b` - Larger, more accurate
- `mistral` - Fast and capable
- `mixtral` - Very powerful (8x7B)
- `phi` - Small and fast
- `gemma` - Google's efficient model

### Option 2: HuggingFace Transformers

**Direct model loading, more control**

1. **Install dependencies**
   ```bash
   pip install transformers torch accelerate
   ```

2. **Choose a model** (updates in config)
   - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - Small, fast (~1GB)
   - `mistralai/Mistral-7B-Instruct-v0.2` - Very capable (~14GB)
   - `google/flan-t5-xl` - Encoder-decoder model (~3GB)

3. **Run extraction**
   ```bash
   python extract_and_annotate.py --config config.local.huggingface.yaml
   ```

**GPU Acceleration:**
```bash
# NVIDIA GPU with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (M1/M2)
# PyTorch has MPS support built-in, set device: "mps" in config
```

### Option 3: LlamaCPP (GGUF Files)

**Best for quantized models, very memory efficient**

1. **Install llama-cpp-python**
   ```bash
   # CPU only
   pip install llama-cpp-python

   # With GPU support (NVIDIA)
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

   # With Metal support (Apple M1/M2)
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
   ```

2. **Download a GGUF model**
   - Visit HuggingFace: https://huggingface.co/TheBloke
   - Download a GGUF file (e.g., `llama-2-7b-chat.Q4_K_M.gguf`)
   - Place in `models/` directory

3. **Update config**
   ```yaml
   llm:
     provider: llamacpp
     model_path: models/llama-2-7b-chat.Q4_K_M.gguf
   ```

4. **Run extraction**
   ```bash
   python extract_and_annotate.py --config config.local.llamacpp.yaml
   ```

**Quantization Levels:**
- `Q4_K_M` - 4-bit, good balance (~4GB)
- `Q5_K_M` - 5-bit, better quality (~5GB)
- `Q8_0` - 8-bit, high quality (~7GB)

### Performance Tips

**For Ollama:**
- Use smaller models for speed: `phi`, `llama2:7b`
- For accuracy: `mixtral`, `llama2:13b`
- Check GPU usage: `ollama list`, `nvidia-smi`

**For HuggingFace:**
- Enable GPU: Set `device: "cuda"` in config
- Reduce memory: Use smaller models or quantization
- Speed up: Set `device_map: "auto"` for multi-GPU

**For LlamaCPP:**
- Use `n_gpu_layers: -1` to offload all layers to GPU
- Increase `n_ctx` for longer context windows
- Try different quantization levels for speed/quality tradeoff

### Troubleshooting Local LLMs

**"Connection refused" (Ollama)**
```bash
# Check if Ollama is running
ollama list

# Start Ollama server
ollama serve
```

**"Out of memory" (HuggingFace/LlamaCPP)**
```yaml
# Use a smaller model or reduce context
llm:
  model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  n_ctx: 2048  # For LlamaCPP
```

**Slow inference**
```yaml
# Enable GPU or use smaller model
llm:
  device: cuda  # HuggingFace
  n_gpu_layers: 32  # LlamaCPP
```

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

## Knowledge Graph Visualization

After extracting and annotating data, you can create interactive visualizations of your knowledge graph using the `visualize_knowledge_graph.py` script.

### Installation

```bash
pip install pyvis networkx
```

### Quick Start

Generate all visualization views:

```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view all
```

This creates interactive HTML visualizations in the `visualizations/` directory:
- `full_knowledge_graph.html` - Complete graph with all nodes and relationships
- `fast_stages_view.html` - FAST stages, capabilities, and mappings
- `content_view.html` - Documents, content, and annotations
- `statistics.html` - Graph statistics and metrics

### Visualization Modes

**1. Full Graph View**
```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld
```
Shows all nodes and relationships in the knowledge graph.

**2. FAST Stages View**
```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view fast-stages
```
Focuses on FAST stages, ADL/IADL capabilities, and their mappings.

**3. Content View**
```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view content
```
Shows documents, extracted content, and annotations.

**4. Stage-Specific View**
```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --stage FAST-4
```
Visualizes everything related to a specific FAST stage (e.g., FAST-4, FAST-7c).

**5. Statistics View**
```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view stats
```
Generates an HTML dashboard with graph statistics and metrics.

### Interactive Features

The HTML visualizations include:
- **Interactive navigation** - Zoom, pan, and drag nodes
- **Node details** - Hover over nodes to see detailed information
- **Color coding** - Different colors for different node types
- **Shape differentiation** - Different shapes for FAST stages, capabilities, content, etc.
- **Edge labels** - Relationship types shown on connections
- **Physics simulation** - Force-directed graph layout

### Node Color Scheme

- **FAST Stages**: Red (#FF6B6B)
- **ADL Capabilities**: Teal (#4ECDC4)
- **IADL Capabilities**: Blue (#45B7D1)
- **Mappings**: Light Coral (#FFA07A)
- **Documents**: Mint (#95E1D3)
- **Content Sections**: Yellow (#F9ED69)
- **Paragraphs**: Pink (#F38181)
- **Tips**: Purple (#AA96DA)
- **Annotations**: Light Pink (#FCBAD3)

### Example Usage

After running extraction:

```bash
# Extract and annotate
python extract_and_annotate.py config.yaml

# Create visualizations
python visualize_knowledge_graph.py output/knowledge_graph.jsonld --view all

# Open in browser
# For example, on Linux:
xdg-open visualizations/full_knowledge_graph.html
```

### Custom Output Directory

```bash
python visualize_knowledge_graph.py output/knowledge_graph.jsonld \
    --output-dir my_visualizations
```

### Understanding the Visualizations

**Full Graph** - Best for:
- Understanding overall structure
- Finding unexpected connections
- Exploring the complete knowledge base

**FAST Stages View** - Best for:
- Understanding functional capability mappings
- Seeing which ADLs/IADLs are affected at each stage
- Clinical reference

**Content View** - Best for:
- Reviewing annotated content
- Verifying annotation quality
- Content auditing

**Stage-Specific View** - Best for:
- Focused analysis of a particular stage
- Creating stage-specific documentation
- Understanding stage progression

## Performance Considerations

- **Rule-based**: Processes ~1000 items/minute
- **LLM-based**: Processes ~10-50 items/minute (depends on API rate limits)
- **Memory**: ~100MB for reference data + PDF size
- **Visualization**: Graphs with >1000 nodes may be slow in browser; use filtered views for large graphs

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
