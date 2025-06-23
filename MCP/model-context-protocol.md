# Model Context Protocol (MCP)

A curated overview of the Model Context Protocol, a standard for providing context and tools to Language Models through structured servers and clients.

**Last Updated:** 2025-06-23

## Table of Contents
- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
- [Official Repositories](#official-repositories)
- [Community Projects](#community-projects)
- [Example MCP Server](#example-mcp-server)
- [Resources](#resources)

## Introduction

Model Context Protocol (MCP) defines how applications give Large Language Models secure access to resources and tools. It separates the management of context from the model itself, making it easier to build agentic applications.

## Key Concepts
- **Server** – provides prompts, resources and tools for clients.
- **Resources** – data endpoints the model can request.
- **Tools** – functions that can be executed by the model.
- **Context** – information supplied to the model for a conversation.
- **Completions** – model outputs generated using the provided context.

## Official Repositories
- **[modelcontextprotocol/modelcontextprotocol](https://github.com/modelcontextprotocol/modelcontextprotocol)** – Specification and documentation.
- **[modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)** – Official Python SDK.
- **[modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)** – Reference implementations of MCP servers.

## Community Projects
The MCP ecosystem includes many third‑party servers built for different platforms and services. Examples include integrations for AWS, Azure, Algolia, and many others.

## Example MCP Server
```python
# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")

# Add a simple tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Add a dynamic resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"
```

## Resources
- [Official Documentation](https://modelcontextprotocol.io)
- [Specification](https://spec.modelcontextprotocol.io)
