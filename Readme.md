# LangChain Chain Types

LangChain provides several chain abstractions that help structure interactions between large language models (LLMs), prompts, tools, memory, and agents.
This document summarizes the most commonly used chain types and includes runnable Python examples.

# LangChain Chain Types

## Table of Contents

1. [LLMChain](#llmchain)  
2. [SequentialChain](#sequentialchain)  
3. [SimpleSequentialChain](#simplesequentialchain)  
4. [SequentialChain (complex)](#sequentialchain-complex)  
5. [RouterChain](#routerchain)  
6. [TransformChain](#transformchain)  
7. [RetrievalQA Chain](#retrievalqa-chain)  
8. [ConversationalRetrievalChain](#conversationalretrievalchain)  
9. [APIChain](#apichain)

---

## 1. LLMChain

**What it is:**  
The simplest chain. It combines a **prompt template** + **LLM** → **output**.

**When to use:**  
Use `LLMChain` when you need a single LLM call, such as generating text, extracting information, or rewriting content.
