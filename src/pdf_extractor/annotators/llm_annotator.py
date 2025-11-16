"""
LLM-based content annotator with support for multiple providers.

Supports both cloud-based and local LLM providers:
- Cloud: Anthropic (Claude), OpenAI (GPT), Google (Gemini)
- Local: Ollama, HuggingFace Transformers, LlamaCPP
"""

import logging
import json
import re
import uuid
from typing import List, Dict, Optional, Tuple, Any
import os

from ..models import (
    ExtractedContent, Annotation, PDFDocument,
    ExtractionMethod
)
from ..loaders import FASTLoader, CapabilityLoader, MappingLoader
from .base_annotator import BaseAnnotator

logger = logging.getLogger(__name__)


class LLMProvider:
    """Wrapper for different LLM providers (cloud and local)."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ):
        """
        Initialize LLM provider.

        Args:
            provider: Provider name ("anthropic", "openai", "gemini", "ollama", "huggingface", "llamacpp")
            model: Model name/ID or path
            api_key: API key for cloud providers (optional for local)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific arguments
                - base_url: For Ollama (default: http://localhost:11434)
                - device: For HuggingFace (default: "auto")
                - model_path: For LlamaCPP (path to GGUF file)
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.client = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        try:
            # Cloud providers
            if self.provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Initialized Anthropic client with model {self.model}")

            elif self.provider == "openai":
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"Initialized OpenAI client with model {self.model}")

            elif self.provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
                logger.info(f"Initialized Gemini client with model {self.model}")

            # Local providers
            elif self.provider == "ollama":
                try:
                    import ollama
                    # Ollama can use custom base URL
                    base_url = self.kwargs.get('base_url', 'http://localhost:11434')
                    self.client = ollama.Client(host=base_url)
                    logger.info(f"Initialized Ollama client with model {self.model} at {base_url}")
                except ImportError:
                    # Fallback to requests if ollama package not available
                    logger.info(f"Using requests fallback for Ollama (model: {self.model})")
                    self.client = "requests_fallback"

            elif self.provider == "huggingface":
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                import torch

                device = self.kwargs.get('device', 'auto')
                logger.info(f"Loading HuggingFace model {self.model} on device {device}...")

                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model)
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    device_map=device,
                    torch_dtype=torch.float16 if device != 'cpu' else torch.float32
                )

                # Create pipeline
                self.client = pipeline(
                    "text-generation",
                    model=self.hf_model,
                    tokenizer=self.tokenizer,
                    device_map=device
                )
                logger.info(f"Initialized HuggingFace model {self.model}")

            elif self.provider == "llamacpp":
                from llama_cpp import Llama

                model_path = self.kwargs.get('model_path') or self.model
                n_ctx = self.kwargs.get('n_ctx', 4096)
                n_gpu_layers = self.kwargs.get('n_gpu_layers', 0)

                logger.info(f"Loading LlamaCPP model from {model_path}...")
                self.client = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )
                logger.info(f"Initialized LlamaCPP model from {model_path}")

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except ImportError as e:
            logger.error(f"Failed to import {self.provider} library: {e}")
            raise ImportError(
                f"Provider {self.provider} requires additional package. Install with:\n"
                f"  pip install {self._get_package_name()}"
            )

    def _get_package_name(self) -> str:
        """Get pip package name for provider."""
        packages = {
            "anthropic": "anthropic",
            "openai": "openai",
            "gemini": "google-generativeai",
            "ollama": "ollama",
            "huggingface": "transformers torch",
            "llamacpp": "llama-cpp-python"
        }
        return packages.get(self.provider, self.provider)

    def generate(self, prompt: str) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        try:
            # Cloud providers
            if self.provider == "anthropic":
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content

            elif self.provider == "gemini":
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': self.temperature,
                        'max_output_tokens': self.max_tokens
                    }
                )
                return response.text

            # Local providers
            elif self.provider == "ollama":
                if self.client == "requests_fallback":
                    # Use requests as fallback
                    import requests
                    base_url = self.kwargs.get('base_url', 'http://localhost:11434')
                    response = requests.post(
                        f"{base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": self.temperature,
                                "num_predict": self.max_tokens
                            }
                        },
                        timeout=120
                    )
                    response.raise_for_status()
                    return response.json()['response']
                else:
                    # Use ollama package
                    response = self.client.generate(
                        model=self.model,
                        prompt=prompt,
                        options={
                            'temperature': self.temperature,
                            'num_predict': self.max_tokens
                        }
                    )
                    return response['response']

            elif self.provider == "huggingface":
                # Generate using pipeline
                outputs = self.client(
                    prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    return_full_text=False
                )
                return outputs[0]['generated_text']

            elif self.provider == "llamacpp":
                response = self.client(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=["User:", "\n\n\n"],
                    echo=False
                )
                return response['choices'][0]['text']

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


class LLMAnnotator(BaseAnnotator):
    """
    Annotates content using Large Language Models.

    Supports multiple providers: Anthropic (Claude), OpenAI (GPT), Google (Gemini).
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str],
        fast_loader: FASTLoader,
        capability_loader: CapabilityLoader,
        mapping_loader: MappingLoader,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ):
        """
        Initialize LLM-based annotator.

        Args:
            provider: LLM provider ("anthropic", "openai", "gemini", "ollama", "huggingface", "llamacpp")
            model: Model name or path
            api_key: API key for cloud providers (optional for local)
            fast_loader: Loaded FAST stage data
            capability_loader: Loaded capability data
            mapping_loader: Loaded mapping data
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens in LLM response
            **kwargs: Additional provider-specific arguments
        """
        self.fast_loader = fast_loader
        self.capability_loader = capability_loader
        self.mapping_loader = mapping_loader

        self.llm = LLMProvider(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Build context for prompts
        self._build_context()

    def _build_context(self):
        """Build context information for LLM prompts."""
        # FAST stages summary
        fast_stages_info = []
        for stage in self.fast_loader.get_all_stages():
            fast_stages_info.append(
                f"- {stage.stage_code}: {stage.stage_name} - {stage.clinical_characteristics[:100]}..."
            )
        self.fast_context = "\n".join(fast_stages_info)

        # Capabilities summary
        capabilities_info = []
        for cap in self.capability_loader.get_all_capabilities():
            capabilities_info.append(
                f"- {cap.capability_id} ({cap.capability_type.value}): {cap.name}"
            )
        self.capabilities_context = "\n".join(capabilities_info)

    def annotate(self, content: ExtractedContent, document: PDFDocument) -> Annotation:
        """Annotate a single content item using LLM."""
        prompt = self._build_annotation_prompt(content, document)

        try:
            response = self.llm.generate(prompt)
            annotation = self._parse_llm_response(response, content)
            return annotation

        except Exception as e:
            logger.error(f"LLM annotation failed for content {content.content_id}: {e}")
            # Return a default annotation with low confidence
            return Annotation(
                annotation_id=str(uuid.uuid4()),
                content_id=content.content_id,
                fast_stages=["FAST-3", "FAST-4", "FAST-5"],
                fast_confidence=0.1,
                capabilities=[],
                capability_confidence=0.1,
                topics=["General"],
                keywords=[],
                target_audience="Both",
                annotation_method=ExtractionMethod.LLM_BASED,
                annotator_notes=f"LLM annotation failed: {str(e)}"
            )

    def annotate_batch(
        self,
        contents: List[ExtractedContent],
        document: PDFDocument
    ) -> List[Annotation]:
        """Annotate multiple content items."""
        return [self.annotate(content, document) for content in contents]

    def _build_annotation_prompt(
        self,
        content: ExtractedContent,
        document: PDFDocument
    ) -> str:
        """Build prompt for LLM annotation."""
        prompt = f"""You are an expert in dementia care and the FAST (Functional Assessment Staging Tool) staging system.

Analyze the following content from a dementia care guide and provide structured annotations.

CONTENT TO ANALYZE:
Title: {content.title}
Text: {content.text[:1500]}

FAST STAGES REFERENCE:
{self.fast_context}

CAPABILITIES REFERENCE (ADLs and IADLs):
{self.capabilities_context}

INSTRUCTIONS:
Analyze the content and provide the following annotations in JSON format:

1. **fast_stages**: List of applicable FAST stage codes (e.g., ["FAST-3", "FAST-4"]). Select all stages where this content is relevant.

2. **fast_confidence**: Confidence score from 0.0 to 1.0 indicating how certain you are about the FAST stage assignments.

3. **capabilities**: List of capability IDs that this content relates to (e.g., ["ADL-1", "IADL-3"]). Include capabilities mentioned or implied.

4. **capability_confidence**: Confidence score from 0.0 to 1.0 for capability assignments.

5. **topics**: List of relevant topics from: [Safety, Communication, Behavioral Symptoms, Medication Management, Legal Planning, Financial Management, Caregiver Support, Activities, Nutrition, Personal Care, Diagnosis, Treatment, End-of-Life Care, Daily Routine]. Select top 3-5 topics.

6. **keywords**: List of 5-10 important keywords from the content.

7. **target_audience**: One of: "Patient", "Caregiver", or "Both"

8. **notes**: Brief explanation of your reasoning (1-2 sentences).

Respond ONLY with valid JSON in this format:
{{
  "fast_stages": ["FAST-X", ...],
  "fast_confidence": 0.X,
  "capabilities": ["ADL-X", "IADL-Y", ...],
  "capability_confidence": 0.X,
  "topics": ["Topic1", "Topic2", ...],
  "keywords": ["keyword1", "keyword2", ...],
  "target_audience": "Patient|Caregiver|Both",
  "notes": "Explanation here"
}}
"""
        return prompt

    def _parse_llm_response(
        self,
        response: str,
        content: ExtractedContent
    ) -> Annotation:
        """
        Parse LLM response into an Annotation object.

        Args:
            response: LLM response text
            content: Original content being annotated

        Returns:
            Annotation object
        """
        try:
            # Extract JSON from response (may contain markdown code blocks)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in LLM response")

            data = json.loads(json_str)

            annotation = Annotation(
                annotation_id=str(uuid.uuid4()),
                content_id=content.content_id,
                fast_stages=data.get("fast_stages", []),
                fast_confidence=float(data.get("fast_confidence", 0.5)),
                capabilities=data.get("capabilities", []),
                capability_confidence=float(data.get("capability_confidence", 0.5)),
                topics=data.get("topics", []),
                keywords=data.get("keywords", []),
                target_audience=data.get("target_audience", "Both"),
                annotation_method=ExtractionMethod.LLM_BASED,
                annotator_notes=data.get("notes", "")
            )

            return annotation

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response}")

            # Return default annotation
            return Annotation(
                annotation_id=str(uuid.uuid4()),
                content_id=content.content_id,
                fast_stages=["FAST-3", "FAST-4", "FAST-5"],
                fast_confidence=0.1,
                capabilities=[],
                capability_confidence=0.1,
                topics=["General"],
                keywords=[],
                target_audience="Both",
                annotation_method=ExtractionMethod.LLM_BASED,
                annotator_notes=f"Failed to parse LLM response: {str(e)}"
            )


def create_llm_annotator_from_config(
    config: Dict,
    fast_loader: FASTLoader,
    capability_loader: CapabilityLoader,
    mapping_loader: MappingLoader
) -> Optional[LLMAnnotator]:
    """
    Create LLM annotator from configuration dictionary.

    Args:
        config: Configuration dict with LLM settings
        fast_loader: FAST data loader
        capability_loader: Capability data loader
        mapping_loader: Mapping data loader

    Returns:
        LLMAnnotator instance or None if LLM not configured
    """
    llm_config = config.get("llm", {})

    if not llm_config or llm_config.get("enabled") is False:
        return None

    provider = llm_config.get("provider")
    model = llm_config.get("model")

    # API key is optional for local providers
    api_key = None
    api_key_env = llm_config.get("api_key_env_var")

    if api_key_env:
        api_key = os.getenv(api_key_env)
        if not api_key and provider not in ["ollama", "huggingface", "llamacpp"]:
            logger.warning(f"API key not found in environment variable: {api_key_env}")
            return None

    # Extract provider-specific kwargs
    kwargs = {}
    if provider == "ollama":
        kwargs['base_url'] = llm_config.get("base_url", "http://localhost:11434")
    elif provider == "huggingface":
        kwargs['device'] = llm_config.get("device", "auto")
    elif provider == "llamacpp":
        kwargs['model_path'] = llm_config.get("model_path")
        kwargs['n_ctx'] = llm_config.get("n_ctx", 4096)
        kwargs['n_gpu_layers'] = llm_config.get("n_gpu_layers", 0)

    try:
        return LLMAnnotator(
            provider=provider,
            model=model,
            api_key=api_key,
            fast_loader=fast_loader,
            capability_loader=capability_loader,
            mapping_loader=mapping_loader,
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 2000),
            **kwargs
        )
    except Exception as e:
        logger.error(f"Failed to create LLM annotator: {e}")
        return None
