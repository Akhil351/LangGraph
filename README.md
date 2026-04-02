# LangGraph Learning Project 2026

A comprehensive learning project demonstrating LangGraph capabilities with practical examples, from basic graphs to advanced patterns like ReAct agents, routing, parallelization, and human-in-the-loop workflows.

## Overview

This repository contains progressive examples showcasing LangGraph's features for building stateful, multi-actor AI applications. Each example builds upon previous concepts to demonstrate increasingly sophisticated patterns.

## Features

- **Basic Graph Construction**: Simple state management and node execution
- **Structured Outputs**: Using Pydantic models for type-safe state handling
- **Message Management**: Working with LangChain's message types
- **Prompt Templates**: Building reusable prompt chains
- **Tool Integration**: ReAct agents with external tools (Google Search)
- **Parallel Execution**: Running multiple nodes concurrently
- **Conditional Routing**: Dynamic path selection based on LLM decisions
- **Multi-Agent Orchestration**: Task decomposition and parallel execution
- **Generator-Evaluator Pattern**: Iterative refinement with feedback loops
- **Memory Management**: Stateful conversations with context persistence
- **Human-in-the-Loop**: Interactive approval workflows

## Prerequisites

- Python 3.13+
- OpenAI API key
- SerpAPI key (for search functionality)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Langraph2026
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_openai_key
# SERPAPI_API_KEY=your_serpapi_key
```

## Project Structure

```
Langraph2026/
├── Graph/                      # LangGraph examples
│   ├── 1_First_Graph.py       # Basic graph with single node
│   ├── 2_Pydantic.py          # Pydantic models for state
│   ├── 3_Messages.py          # Message types and handling
│   ├── 4_Prompts.py           # Prompt templates and chains
│   ├── 5_Tools.py             # Tool integration basics
│   ├── 6_ReAct_Agent.py       # ReAct pattern with tools
│   ├── 7_Parallelization.py  # Parallel node execution
│   ├── 8_Routing.py           # Conditional routing
│   ├── 9_Orchestrator.py     # Multi-agent orchestration
│   ├── 10_Generator_Evaluator.py # Iterative refinement
│   ├── 11_Memory.py           # Conversation memory
│   └── 12_Human_In_The_Loop.py # Human approval workflow
├── core/                      # Configuration management
│   ├── __init__.py
│   └── config.py              # Environment variables loader
├── llm/                       # LLM utilities
│   ├── __init__.py
│   └── openai_llm_models.py  # OpenAI model wrapper
├── main.py                    # Entry point
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## Examples

### 1. First Graph
Basic LangGraph setup with a single node that processes input and returns output.

**Run:**
```bash
python -m Graph/1_First_Graph.py
```

### 2. Pydantic State
Using Pydantic models for type-safe state management with validation.

**Run:**
```bash
python -m Graph/2_Pydantic.py
```

### 3. Messages
Working with different message types (SystemMessage, HumanMessage, AIMessage).

**Run:**
```bash
python -m Graph/3_Messages.py
```

### 4. Prompts
Building reusable prompt templates with LangChain.

**Run:**
```bash
python -m Graph/4_Prompts.py
```

### 5. Tools
Introduction to tool integration with interactive chat loop and Google Search.

**Run:**
```bash
python -m Graph/5_Tools.py
```

### 6. ReAct Agent
Full ReAct pattern implementation with tool calling and decision-making.

**Run:**
```bash
python -m Graph/6_ReAct_Agent.py
```

### 7. Parallelization
Executing multiple nodes concurrently to generate social media posts for Instagram, LinkedIn, and Twitter simultaneously.

**Run:**
```bash
python -m Graph/7_Parallelization.py
```

### 8. Routing
Dynamic routing based on LLM decisions to select the appropriate social media platform.

**Run:**
```bash
python -m Graph/8_Routing.py
```

### 9. Orchestrator
Multi-agent system that breaks down complex queries into subtasks, executes them in parallel, and synthesizes results.

**Run:**
```bash
python -m Graph/9_Orchestrator.py
```

### 10. Generator-Evaluator
Iterative refinement pattern where a generator creates content and an evaluator provides feedback until quality criteria are met.

**Run:**
```bash
python -m Graph/10_Generator_Evaluator.py
```

### 11. Memory
Stateful chatbot that maintains conversation context across multiple interactions.

**Run:**
```bash
python -m Graph/11_Memory.py
```

### 12. Human-in-the-Loop
Interactive workflow requiring human approval before finalizing AI responses.

**Run:**
```bash
python -m Graph/12_Human_In_The_Loop.py
```

## Key Concepts

### State Management
- **TypedDict**: Simple state definition for basic graphs
- **Pydantic Models**: Type-safe state with validation
- **Annotated Types**: Using reducers like `add` for list accumulation

### Graph Components
- **Nodes**: Functions that process state and return updates
- **Edges**: Direct connections between nodes
- **Conditional Edges**: Dynamic routing based on state or conditions
- **START/END**: Special nodes marking graph entry and exit points

### Advanced Patterns
- **ReAct Pattern**: Reasoning + Acting loop with tool integration
- **Parallel Execution**: Multiple nodes running concurrently
- **Conditional Routing**: LLM-driven decision-making for graph flow
- **Orchestration**: Breaking complex tasks into subtasks
- **Generator-Evaluator**: Iterative improvement with feedback
- **Human-in-the-Loop**: Interactive approval and intervention

## Dependencies

- **langchain**: Core LangChain library
- **langchain-community**: Community tools (SerpAPI)
- **langchain-openai**: OpenAI integration
- **langgraph**: Graph-based orchestration framework
- **python-dotenv**: Environment variable management
- **google-search-results**: SerpAPI wrapper
- **pydantic**: Data validation and type hints

## Configuration

The project uses environment variables for API key management. Configure your keys in the `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

## Usage

Run any example directly:

```bash
python Graph/<example_name>.py
```

Or use the main entry point:

```bash
python main.py
```

## Learning Path

Follow the examples in numerical order for the best learning experience:

1. Start with basic graph construction (1-4)
2. Move to tool integration (5-6)
3. Explore advanced patterns (7-10)
4. Learn stateful and interactive patterns (11-12)

## Contributing

This is a personal learning project. Feel free to fork and experiment with your own examples.

## License

MIT License - feel free to use this code for your own learning purposes.

---

Built with LangChain and LangGraph for exploring agent orchestration patterns.
