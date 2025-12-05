# LangChain Chain Types: Complete Guide

This guide covers the different types of chains available in the LangChain framework, with Python code examples for each.

## Table of Contents
- [LLMChain](#llmchain)
- [Sequential Chains](#sequential-chains)
- [Router Chain](#router-chain)
- [Transform Chain](#transform-chain)
- [Conversational Chain](#conversational-chain)
- [RetrievalQA Chain](#retrievalqa-chain)
- [MapReduce Chain](#mapreduce-chain)
- [Refine Chain](#refine-chain)
- [API Chain](#api-chain)
- [SQL Database Chain](#sql-database-chain)

---

## LLMChain

The most basic chain type that combines a prompt template with an LLM to generate responses.

### Description
LLMChain is the fundamental building block in LangChain. It takes a prompt template, formats it with input variables, and passes it to an LLM to get a response.

### Code Example

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.run("eco-friendly water bottles")
print(result)
```

### Example Output
```
"AquaVerde - A name that combines 'Aqua' (water) and 'Verde' (green in Spanish), 
emphasizing both the product and its eco-friendly nature."
```

---

## Sequential Chains

Chains that run multiple chains in sequence, where the output of one chain becomes the input to the next.

### SimpleSequentialChain

Used when each chain has a single input and output.

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

# First chain: Generate a company name
llm = OpenAI(temperature=0.7)
first_prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# Second chain: Write a catchphrase for the company
second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a catchy slogan for the following company: {company_name}"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True
)

# Run the chain
result = overall_chain.run("eco-friendly water bottles")
print(result)
```

### SequentialChain

Used when chains have multiple inputs/outputs.

```python
from langchain.chains import SequentialChain

# Chain 1: Generate synopsis
synopsis_prompt = PromptTemplate(
    input_variables=["title", "era"],
    template="Write a synopsis for a {era} movie titled '{title}'"
)
synopsis_chain = LLMChain(llm=llm, prompt=synopsis_prompt, output_key="synopsis")

# Chain 2: Generate review
review_prompt = PromptTemplate(
    input_variables=["synopsis"],
    template="Write a review for a movie with this synopsis:\n{synopsis}"
)
review_chain = LLMChain(llm=llm, prompt=review_prompt, output_key="review")

# Combine chains
overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["title", "era"],
    output_variables=["synopsis", "review"],
    verbose=True
)

# Run
result = overall_chain({
    "title": "The Time Traveler",
    "era": "1980s sci-fi"
})
print(result)
```

---

## Router Chain

Routes inputs to different chains based on the content of the input.

### Description
Router chains use an LLM to determine which specialized chain should handle a particular input, making them ideal for multi-domain applications.

### Code Example

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

# Define destination chains
physics_template = """You are a physics professor. Answer this question:\n{input}"""
math_template = """You are a math professor. Answer this question:\n{input}"""
history_template = """You are a history professor. Answer this question:\n{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering physics questions",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    },
    {
        "name": "history",
        "description": "Good for answering history questions",
        "prompt_template": history_template
    }
]

llm = OpenAI(temperature=0)

# Create destination chains
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# Default chain
default_prompt = PromptTemplate(
    template="{input}",
    input_variables=["input"]
)
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# Create router chain
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = f"""Given a raw text input, select the model prompt best suited for the input.
You will be given the names of the available prompts and a description of what each is good for.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted like:
```json
{{{{
    "destination": string \\ name of the prompt to use or "DEFAULT"
    "next_inputs": string \\ the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names OR "DEFAULT".

<< CANDIDATE PROMPTS >>
{destinations_str}

<< INPUT >>
{{input}}

<< OUTPUT >>
"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# Create MultiPromptChain
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

# Test it
print(chain.run("What is Newton's second law?"))
print(chain.run("What is the derivative of x^2?"))
print(chain.run("Who was the first president of the USA?"))
```

---

## Transform Chain

Performs custom transformations on inputs before passing to other chains.

### Code Example

```python
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain

def transform_func(inputs: dict) -> dict:
    """Custom transformation function"""
    text = inputs["text"]
    # Transform: uppercase and add prefix
    transformed = f"IMPORTANT: {text.upper()}"
    return {"transformed_text": transformed}

# Create transform chain
transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["transformed_text"],
    transform=transform_func
)

# Create LLM chain that uses transformed input
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["transformed_text"],
    template="Respond to this: {transformed_text}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Combine them
sequential_chain = SimpleSequentialChain(
    chains=[transform_chain, llm_chain],
    verbose=True
)

# Run
result = sequential_chain.run("please help me")
print(result)
```

---

## Conversational Chain

Manages conversation history to maintain context across multiple interactions.

### Code Example

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)

# Create conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=True
)

# Have a conversation
print(conversation.predict(input="Hi, my name is Alice"))
print(conversation.predict(input="What's my name?"))
print(conversation.predict(input="What are some good hobbies?"))
```

### With Custom Memory

```python
from langchain.memory import ConversationBufferWindowMemory

# Only remember last 2 interactions
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=2),
    verbose=True
)
```

---

## RetrievalQA Chain

Performs question-answering over documents by retrieving relevant information first.

### Code Example

```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

# Load documents
loader = TextLoader('document.txt')
documents = loader.load()

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)

# Ask questions
query = "What is the main topic of the document?"
result = qa.run(query)
print(result)
```

### With Source Documents

```python
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True
)

result = qa({"query": "What is the main topic?"})
print(result["result"])
print(result["source_documents"])
```

---

## MapReduce Chain

Processes multiple documents in parallel and combines results.

### Description
MapReduce chains apply an operation to each document independently (map) and then combine the results (reduce). Ideal for summarization and analysis of large document sets.

### Code Example

```python
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI

# Load and split documents
loader = TextLoader('long_document.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create MapReduce chain
llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

# Run summarization
summary = chain.run(texts)
print(summary)
```

### Custom MapReduce

```python
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Map prompt
map_template = """Summarize this document:
{docs}
Summary:"""
map_prompt = PromptTemplate(template=map_template, input_variables=["docs"])
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce prompt
reduce_template = """Combine these summaries:
{doc_summaries}
Final Summary:"""
reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["doc_summaries"])
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Combine
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain,
    document_variable_name="doc_summaries"
)

reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=4000
)

map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs"
)

result = map_reduce_chain.run(texts)
```

---

## Refine Chain

Iteratively refines an answer by processing documents sequentially.

### Description
The Refine chain processes documents one at a time, updating its answer with each new piece of information. Better for detailed analysis where order matters.

### Code Example

```python
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI

# Load documents
loader = TextLoader('document.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create Refine chain
llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# Run
summary = chain.run(texts)
print(summary)
```

### Custom Refine Chain

```python
from langchain.chains import RefineDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain

# Initial prompt
question_prompt_template = """
Context: {context_str}
Question: Provide a comprehensive summary.
Answer:"""

question_prompt = PromptTemplate(
    template=question_prompt_template,
    input_variables=["context_str"]
)

# Refine prompt
refine_prompt_template = """
Original Answer: {existing_answer}
New Context: {context_str}
Refine the answer using the new context.
Refined Answer:"""

refine_prompt = PromptTemplate(
    template=refine_prompt_template,
    input_variables=["existing_answer", "context_str"]
)

# Create chains
initial_chain = LLMChain(llm=llm, prompt=question_prompt)
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)

# Combine
refine_documents_chain = RefineDocumentsChain(
    initial_llm_chain=initial_chain,
    refine_llm_chain=refine_chain,
    document_variable_name="context_str",
    initial_response_name="existing_answer"
)

result = refine_documents_chain.run(texts)
```

---

## API Chain

Makes API calls based on natural language requests.

### Code Example

```python
from langchain.chains import APIChain
from langchain.llms import OpenAI

# Define API documentation
api_docs = """
BASE URL: https://api.weather.com/v1/

Endpoints:
1. GET /current
   Description: Get current weather
   Parameters:
   - city (required): City name
   - units (optional): 'metric' or 'imperial'
   
2. GET /forecast
   Description: Get weather forecast
   Parameters:
   - city (required): City name
   - days (optional): Number of days (1-7)
"""

llm = OpenAI(temperature=0)

# Create API chain
chain = APIChain.from_llm_and_api_docs(
    llm,
    api_docs,
    verbose=True
)

# Make natural language API request
result = chain.run("What's the weather in New York?")
print(result)
```

---

## SQL Database Chain

Queries SQL databases using natural language.

### Code Example

```python
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain.chains import SQLDatabaseChain

# Connect to database
db = SQLDatabase.from_uri("sqlite:///example.db")

llm = OpenAI(temperature=0)

# Create SQL chain
db_chain = SQLDatabaseChain.from_llm(
    llm,
    db,
    verbose=True,
    return_intermediate_steps=True
)

# Query in natural language
result = db_chain("How many customers are in the database?")
print(result["result"])
```

### With Custom Prompt

```python
from langchain.prompts import PromptTemplate

template = """Given an input question, create a syntactically correct {dialect} query.
Use the following table schema:
{table_info}

Question: {input}
SQL Query:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input", "table_info", "dialect"]
)

db_chain = SQLDatabaseChain.from_llm(
    llm,
    db,
    prompt=prompt,
    verbose=True
)

result = db_chain("List all products with price greater than 100")
```





