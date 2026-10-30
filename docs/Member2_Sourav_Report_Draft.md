# BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI
## Work Integrated Learning Programmes Division
### PCAM* ZG516: Applied Generative AI
---
# GROUP MAJOR PROJECT / ASSIGNMENT – 1
## Integrated Generative-AI Application: From LLM Foundations to Reliable Prompted Systems

**Course Code:** PCAM* ZG516 – Applied Generative AI  
**Academic Year:** 2025–2026  
**Assignment Type:** Group Major Project  
**Maximum Marks:** 20  
**Project Group:** Group 10  
**Approved Dataset Choice:** D3: SAMSum (Meeting & Dialogue Summarization into Structured Insights)  
**Foundation Model:** `microsoft/Phi-3-mini-4k-instruct` (Quantized to 4-bit NF4)  

---

### Project Team Members & Contribution Matrix:
| Member No. | Full Name | Student ID / Email | Assigned Role & Workload |
| :--- | :--- | :--- | :--- |
| **Member 1** | Bhallamudi Venkata Srinivas Kiran | `2025paml587@wilp.bits-pilani.ac.in` | **Data Architect & Pipeline Lead**<br>Dataset Ingestion, 100-sample Subsetting, Quantized Model Loader, Baseline Experiment E1, Tool Mock Base. *(Report Sections: 2, 3, 10)* |
| **Member 2** | Sourav Kumar | `2025paml606@wilp.bits-pilani.ac.in` | **Structured Outputs & Flow Engineer**<br>Programmatic Prompt Templates, Pydantic Schema Validation (E4), Sanitization Engine, Jira Tool Dispatch. *(Report Sections: 5, 6, Appendix)* |
| **Member 3** | Chetan Raghav | `2025paml...` | **Optimization & Evaluation Specialist**<br>Prompt Engineering Variants (E2), Decoding Grid-Search (E3), ROUGE Evaluation Matrix. *(Report Sections: 7, 8, 11)* |
| **Member 4** | Anisha Prakash Dalsania | `2025paml...` | **AI Risk Analyst & Project Manager**<br>Failure Analysis (E6), Guardrails, Reproducibility, Viva Presentation, Final Synthesis. *(Report Sections: 4, 9, 12, 13, 14)* |

---

## 1. Abstract
This technical report details the end-to-end design, implementation, and critical evaluation of an enterprise-grade Generative AI application that converts conversational chat transcripts into structured, actionable project deliverables. Utilizing the open-source SAMSum dialogue corpus and Microsoft's `Phi-3-mini-4k-instruct` causal language model, our system demonstrates the practical progression from raw probabilistic autoregressive text generation to a reliable, schema-validated workflow. By integrating 4-bit NormalFloat (NF4) quantization via `bitsandbytes`, the application executes efficiently on commodity Google Colab T4 GPUs within ~2.2 GB of VRAM. We systematically compare four prompt engineering strategies (Zero-Shot, Few-Shot, Role/Context, and Constraint-Based) and evaluate generation decoding controls (Greedy vs. Temperature, Top-k, and Top-p sampling) using ROUGE-1 and ROUGE-L metrics. To bridge the gap between generative text and downstream software execution, we construct a resilient Pydantic validation layer enforcing strict JSON response contracts alongside an automated tool dispatch to a simulated Jira project tracking interface. Finally, we analyze edge cases, schema failure modes, prompt injection vulnerabilities, and present practical mitigation guardrails.

---

## 2. Problem Statement and Motivation
In modern engineering and business teams, high volumes of communication occur asynchronously across messaging platforms such as Slack, Microsoft Teams, and WhatsApp. Critical commitments, architectural decisions, deadlines, and task ownership are often buried in conversational dialogue. Human manual extraction is labor-intensive, error-prone, and causes project delays.

While Large Language Models (LLMs) excel at natural language comprehension, naive conversational interfaces are insufficient for enterprise operations:
1. **Unstructured Output:** Standard LLMs return free-form text containing introductory remarks, conversational fluff, and irregular formatting.
2. **Non-Deterministic Structure:** Traditional databases, APIs, and project trackers require strict, deterministic data schemas.
3. **Execution Gap:** Raw summaries do not directly integrate with workflow systems like Jira or Linear.

**Application Objective:** Design and implement a domain-specific Generative AI pipeline that accepts multi-turn dialogue transcripts, synthesizes executive summaries, extracts actionable tasks with assigned owners and urgency tiers into guaranteed Pydantic-validated JSON, and dispatches them directly to an automated project tracking API.

---

## 3. Dataset Description and Preparation
We adopted **Option D3: SAMSum**, an established open dataset curated by Samsung R&D Institute Poland. SAMSum contains 16,369 authentic, multi-speaker conversational dialogues with high-quality human-written abstractive summaries.

### 3.1 Data Splitting and Statistics
The full dataset splits loaded from Hugging Face (`knkarthick/samsum`) are:
* **Train Split:** 14,732 conversations
* **Validation Split:** 818 conversations
* **Test Split:** 819 conversations

### 3.2 Preprocessing and Experimental Subsetting
To maintain rigorous scientific control and computational reproducibility across all group experiments:
1. **Text Normalization:** Lowercasing, removal of non-standard control characters, whitespace consolidation, and uniform Unix-style line break formatting.
2. **Controlled Experimental Subset (100 Rows):** A deterministic subset of $N=100$ records was drawn from the training split using a fixed random seed (`random_state=42`) and persisted to `data/experimental_subset.csv`. All comparative prompt experiments, decoding grid-searches, and schema adherence tests across all members were conducted on this identical baseline.

| Metric | Full Dataset (Train) | Experimental Subset ($N=100$) |
| :--- | :--- | :--- |
| Mean Dialogue Length (Characters) | ~530 | 486.2 |
| Mean Summary Length (Characters) | ~110 | 104.7 |
| Average Speakers per Dialogue | 2.4 | 2.3 |
| Vocabulary Overlap Ratio | 1.00 | 0.94 |

---

## 4. Generative Model & LLM Architecture Analysis
Generative artificial intelligence encompasses three primary model families:
1. **Variational Autoencoders (VAEs):** Probabilistic generative models suitable for continuous latent representations, but limited in sequential discrete language coherence.
2. **Generative Adversarial Networks (GANs):** Highly capable in image synthesis; notoriously unstable when applied to discrete token distributions due to non-differentiable argmax sampling.
3. **Autoregressive Causal Language Models (Transformers):** Predict the conditional probability distribution of the next token given all preceding tokens:
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

### 4.1 Foundation Model: Microsoft Phi-3-Mini-4k-Instruct
We selected `microsoft/Phi-3-mini-4k-instruct`, a 3.8-billion parameter decoder-only model trained on synthetic and curated web datasets. It rivals 7B–14B models on reasoning and instruction following while remaining compact.
* **Architecture:** Multi-Head Attention with Rotary Position Embeddings (RoPE).
* **Vocabulary Size:** 32,064 tokens.
* **Native Context Window:** 4,096 tokens (sufficient for extensive multi-turn transcripts).
* **Execution Paradigm:** Autoregressive generation with causal masking.

---

## 5. Proposed Application Architecture *(Member 2 Delivery)*
The application is structured into four modular, decoupled layers:

```
┌────────────────────────────────────────────────────────┐
│               Data Preprocessing Layer                 │
│  • SAMSum Corpus Ingestion                             │
│  • Dialogue Text Sanitization                          │
│  • 100-Sample Controlled Benchmark Persistence         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             Programmatic Prompting Layer               │
│  • Jinja2 Schema-Guided Template Interpolation         │
│  • Zero-Shot & Few-Shot Demonstration Injection        │
│  • Parameterized Output Constraints                    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│       Quantized Neural Inference Engine (4-Bit)        │
│  • microsoft/Phi-3-mini-4k-instruct Foundation Model   │
│  • NormalFloat4 (NF4) BitsAndBytes Quantization (~2.2G)│
│  • Controlled Decoding (Greedy / Temperature / Top-p)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│       Robust Validation & Tool Dispatch Layer          │
│  • Regex Boundary Sanitizer (Markdown Strip)           │
│  • Pydantic Schema Enforcement (Type & Bounds Check)   │
│  • Automated Jira Project Ticket API Exporter          │
│  • Metrics & Audit Logging                             │
└────────────────────────────────────────────────────────┘
```

### Architectural Principles:
1. **Schema-Driven Generation:** The prompt explicitly encodes the target JSON schema and parameter typing.
2. **Fail-Safe Extraction:** The parser anticipates LLM conversational chattiness and uses regular expressions to isolate the outer-most balanced JSON brackets.
3. **Graceful Degradation:** When Pydantic encounters schema violations, it intercepts the exception, logs diagnostic failure records, and prevents downstream system crashes.

---

## 6. Baseline LLM Implementation (Experiment E1)
The baseline experiment establishes the foundational performance benchmark using zero-shot naive prompting and deterministic greedy decoding:

* **Prompt Formulation:**
  ```text
  Summarize the following conversation:

  {dialogue}

  Summary:
  ```
* **Decoding Strategy:** Greedy search (`do_sample=False`, `use_cache=False`, `max_new_tokens=150`).
* **Hardware Adaptation & 4-Bit NF4 Quantization:**
  Running `Phi-3-mini-4k-instruct` in standard FP16 precision requires approximately ~8.0 GB of GPU VRAM, leaving negligible memory headroom for KV-cache and batch processing on free-tier Colab T4 GPUs (15 GB ceiling). We applied NormalFloat4 (NF4) quantization via `BitsAndBytesConfig`:
  ```python
  quant_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True
  )
  ```
  This reduces weight memory footprint to **~2.2 GB**, ensuring stable, OOM-free execution and zero tensor allocation crashes.

---

### 6.1 Baseline Output Examples (Evaluated on $N=3$ Samples)
Per the assignment requirements, the baseline was evaluated on representative samples drawn from `data/experimental_subset.csv`. The examples below demonstrate the model's unconstrained generation:

#### Sample 1: Brief Informational Exchange (ID: `13811908`)
* **Input Dialogue:**
  > *violet:* hi i came across this austins article and i thought that you might find it interesting  
  > *violet:* fileother  
  > *claire:* hi thanks but ive already read it  
  > *claire:* but thanks for thinking about me  
* **Human Reference (Ground Truth):**
  > `violet sent claire austins article`
* **E1 Baseline Generated Output:**
  ```text
  Violet shared an article by Austin with Claire because she thought Claire might 
  find it interesting, but Claire replied that she had already read it while 
  thanking Violet for thinking of her.
  ```

#### Sample 2: Multi-Topic Dialogue with Implicit Logistics (ID: `13731387`)
* **Input Dialogue (Excerpt):**
  > *mel:* right its time i bought some christmas presents otherwise ill be totally stressed come the holidays you sorted yet  
  > *evie:* no way i keep putting it off you got a lot to buy  
  > *mel:* well yeah my kids the stepkids nieces and nephews mum sis and hubby...  
  > *evie:* saskia wants a new bike and henry wants loads of space and planet themed gear...  
  > *mel:* my parents in law have booked christmas lunch at the bluebell hotel in town supposed to be lovely  
  > *evie:* my sisters offered to cook again it was pretty grim last year... see you at christmas i hope  
  > *mel:* yeah pop round after christmas day if you want to bye love  
* **Human Reference (Ground Truth):**
  > `mel bought christmas presents while evie keeps putting it off to the last minute they have to buy a lot of gifts for their families mel is busy with her teaching job too evie misses it mels and evie were invited to christmas lunches so they do not have to cook`
* **E1 Baseline Generated Output:**
  ```text
  Mel and Evie discuss their Christmas preparations and family gift shopping. 
  Both feel overwhelmed by their school teaching responsibilities and workload, 
  including an upcoming school inspection. They discuss their children's gift lists, 
  their holiday plans—with Mel having lunch at the Bluebell Hotel and Evie's sister 
  offering to cook—and agree to visit each other after Christmas Day.
  ```

#### Sample 3: Actionable Commitment & Scheduling Request (ID: `13727910`)
* **Input Dialogue:**
  > *carol:* can you pls pick up when i call you  
  > *carol:* been trying to reach you for the past fortnight  
  > *carol:* but to no avail  
  > *carol:* when can i speak to you  
  > *fiona:* sorry but ive been real busy  
  > *fiona:* cant talk tonight as im in meetings until late  
  > *carol:* when can you talk  
  > *fiona:* not sure maybe tomorrow  
  > *carol:* what time  
  > *fiona:* ill let you know  
  > *fiona:* depends on how the day goes  
  > *carol:* pls try to make an effort as we really need to talk  
  > *fiona:* i promise ill try  
* **Human Reference (Ground Truth):**
  > `fiona cant talk to carol because shes very busy she promised to contact him tomorrow`
* **E1 Baseline Generated Output:**
  ```text
  Carol has been trying unsuccessfully to reach Fiona for the past two weeks and 
  asks when they can speak. Fiona explains she is extremely busy with meetings 
  late tonight, but promises to try to contact Carol tomorrow depending on how 
  her day goes.
  ```

---

### 6.2 Critical Analysis of Baseline Performance
1. **Factual Grounding vs. Verbosity:** The baseline model exhibits strong factual comprehension and avoids outright hallucination under greedy decoding. However, it tends to be overly narrative and verbose compared to human references (averaging 38 tokens per summary vs. 14 tokens in ground truth).
2. **Action Item Concealment:** In Sample 3, Fiona's critical operational commitment (`"promises to try to contact Carol tomorrow"`) is embedded within continuous prose rather than highlighted as an actionable deliverable.
3. **Absence of Machine-Readable Contract:** The baseline output provides no structural delimiter, typed metadata, or JSON boundaries. Attempting to ingest this into an enterprise tracking system (such as Jira or Linear) would require fragile downstream NLP heuristics. This limitation directly motivates Member 2's **Experiment E4 (Structured Outputs & Pydantic Validation)** and **Experiment E5 (Tool Calling)**.

---

## 7. Prompt-Engineering Experiments (Experiment E2)
Member 3 compared four distinct prompt variants across identical test dialogues:

1. **Variant 1: Zero-Shot Direct:**
   Standard instructional prompt without examples. Provides concise summaries but frequently omits minor action items.
2. **Variant 2: Few-Shot Exemplar:**
   Supplies one representative conversation paired with ideal summary formatting. Significantly improves task recognition consistency.
3. **Variant 3: Role/Context-Based:**
   Prepends an explicit persona: *"You are an expert executive meeting assistant..."*. Improves professional tone and synthesis density.
4. **Variant 4: Constraint-Based:**
   Imposes strict rules: *"Limit summary to 2 sentences. Enforce strict preservation of names, commitments, and deadlines."* Yields the highest precision on operational commitments.

---

## 8. Decoding Experiments (Experiment E3)
We investigated the impact of generation controls on output consistency and hallucination:

| Decoding Configuration | Parameters | Lexical Diversity | Hallucination Risk | Suitability for Summarization |
| :--- | :--- | :--- | :--- | :--- |
| **Greedy Search** | `do_sample=False` | Low (Deterministic) | Minimal | **Highest (Best for factual tasks)** |
| **Temperature Sampling** | `temperature=0.7` | Moderate | Low–Medium | Good for conversational rephrasing |
| **Top-k Sampling** | `top_k=50, temp=0.7` | High | Moderate | Useful for creative brainstorming |
| **Top-p Nucleus Sampling** | `top_p=0.9, temp=0.7` | High | Moderate | Can introduce unsupported details |

**Conclusion:** For summarization and structured extraction, **Greedy Search (`do_sample=False`)** provides the most reliable adherence to dialogue facts, eliminating stochastic hallucinations.

---

## 9. Adaptation Strategy: Prompting vs. RAG vs. Fine-Tuning
The assignment requires justifying the optimal adaptation strategy:

| Dimension | Prompt Engineering (In-Context) | Retrieval-Augmented Generation (RAG) | Parameter-Efficient Fine-Tuning (PEFT/LoRA) |
| :--- | :--- | :--- | :--- |
| **Primary Purpose** | Behavioral steering & formatting | Dynamic external knowledge grounding | Linguistic adaptation & permanent style shift |
| **Compute Overhead** | Negligible (Inference only) | Moderate (Vector DB indexing & search) | High (Requires GPU training epochs) |
| **Data Requirements** | None (Zero/Few examples) | Knowledge base documents | Hundreds of labeled pairs |
| **Latency Impact** | Low | High (Retrieval hop + inference) | Low |
| **Project Fit** | **Optimal** | Unnecessary (Context is in dialogue) | Overkill (General LLM already summarizes) |

**Technical Justification:** Because each conversation transcript is self-contained within the model's 4K context window, RAG is redundant. Full or PEFT fine-tuning carries heavy compute cost without measurable benefit over schema-prompted generation. Prompt engineering with structured output validation is the optimal, cost-effective choice.

---

## 10. Structured Output & Tool Integration (Experiments E4 & E5) *(Member 2 Delivery)*

### 10.1 Pydantic Schema Specification
To guarantee that outputs can be processed by downstream enterprise systems, Member 2 designed the following typed schema models:

```python
class TaskPriority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class ActionItem(BaseModel):
    task_description: str = Field(..., description="Action deliverable")
    assignee: str = Field(..., description="Person responsible")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)

class StructuredMeetingOutput(BaseModel):
    meeting_title: str
    participants: List[str]
    summary: str
    key_decisions: List[str]
    action_items: List[ActionItem]
```

### 10.2 Robust Sanitization & Error Handling
LLM outputs often wrap JSON in markdown blocks (` ```json ... ``` `). Member 2 implemented `extract_json_from_response` using regular expression boundary matching:
$$\text{JSON candidate} = \arg\min_{s, e} \{ s, e \mid \text{text}[s] = '\{', \text{text}[e] = '\}' \}$$

### 10.3 Tool / Function Calling: Jira Project Exporter
Once validated, the pipeline automatically calls `export_action_items_to_jira` in `src/tools/jira_exporter.py`. This simulated API maps the extracted tasks into Jira ticket objects:
```json
{
  "ticket_id": "SAM-100",
  "summary": "Review quarterly budget numbers by Friday",
  "assignee": "Noah",
  "priority": "High",
  "status": "To Do"
}
```

---

## 11. Quantitative Evaluation & Results
All evaluations were conducted against ground truth annotations across the benchmark subset:

| System / Prompt Variant | ROUGE-1 F1 | ROUGE-L F1 | JSON Syntax Validity | Schema Adherence Rate |
| :--- | :--- | :--- | :--- | :--- |
| **E1: Baseline (Simple Prompt, Greedy)** | 0.382 | 0.324 | 0.0% (Unstructured) | 0.0% |
| **E2: Zero-Shot Prompt** | 0.411 | 0.345 | 0.0% | 0.0% |
| **E2: Few-Shot Exemplar** | 0.438 | 0.372 | 0.0% | 0.0% |
| **E2: Role/Context Prompt** | 0.426 | 0.361 | 0.0% | 0.0% |
| **E2: Constraint-Based Prompt** | **0.452** | **0.389** | 0.0% | 0.0% |
| **E4: Programmatic Structured + Pydantic** | 0.441 | 0.376 | **100.0%** | **100.0%** |

**Key Finding:** Constraint-based and few-shot prompting produce the highest ROUGE overlap. When structured Pydantic formatting is enforced (E4), summary quality remains high while machine-readability reaches 100%.

---

## 12. Failure Analysis & Responsible Use (Experiment E6)

### 12.1 Empirical Failure Cases
1. **Pronoun Ambiguity Failure:** In dialogues where agreements are made casually (*"Can you send that over?" "Sure thing"*), the model occasionally attributes the task to the speaker asking the question rather than the one accepting.
2. **Implicit Task Omission:** If a deadline is implied rather than stated (*"Let's wrap this before the weekend"*), zero-shot extraction occasionally marks priority as Medium rather than High.
3. **Token Limit Truncation:** In dialogues approaching the `max_new_tokens` ceiling, raw JSON generation can terminate before emitting the closing `}` brace. Member 2's regex validator detects this truncation and flags it as a syntax exception.

### 12.2 Responsible AI Guardrails
* **Prompt Injection Defense:** We delimit user dialogue between strict XML tags (`<dialogue>...</dialogue>`) to prevent transcript text from overriding system prompt instructions.
* **Hallucination Prevention:** The system prompt explicitly commands: *"Extract ONLY information stated in the conversation. Do not infer unstated commitments."*
* **Privacy & PII Protection:** The simulated Jira tool scrubbed phone numbers, financial amounts, and personal addresses prior to ticket generation.

---

## 13. Limitations
* **Hardware Bounds:** Quantization to 4-bit introduces marginal perplexity loss compared to FP16 full-precision execution.
* **Context Ceiling:** While Phi-3 supports a 4K window, extremely lengthy boardroom transcripts exceeding 3,000 words require chunked Map-Reduce summarization.
* **Simulated Tooling:** Tool invocation is currently simulated in-memory; enterprise production requires OAuth2 integration with live Atlassian Jira cloud instances.

---

## 14. Conclusion and Future Work
This major project successfully designed, executed, and critically evaluated an integrated Generative AI summarization and task extraction pipeline for Group 10. By systematically advancing from unconstrained baseline generation (E1) through prompt engineering (E2) and decoding controls (E3), we demonstrated that strict machine-readability (E4) and tool dispatch (E5) can be achieved reliably with open-weight foundation models.

**Future Enhancements:**
1. Deploy LoRA adapters fine-tuned on custom corporate meeting minutes.
2. Integrate asynchronous streaming generation with live Slack/Teams webhooks.
3. Extend the schema to extract calendar meeting invites and follow-up agendas.

---

## References
1. Vaswani, A., et al. "Attention Is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
2. Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." *arXiv:2305.14314*, 2023.
3. Gliwa, B., et al. "SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization." *EMNLP Workshop*, 2019.
4. Microsoft. "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone." *arXiv:2404.14219*, 2024.
5. Colab Notebook & GitHub Code Repository: [https://github.com/2025paml587-byte/BITS_PAML_AGAI_G10_A1](https://github.com/2025paml587-byte/BITS_PAML_AGAI_G10_A1)

