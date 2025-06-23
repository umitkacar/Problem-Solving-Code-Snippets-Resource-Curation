# 🔌 Model Context Protocol (MCP)

Comprehensive guide to Model Context Protocol - the universal standard for AI-tool integration, enabling dynamic context management and seamless capability access for LLMs.

**Last Updated:** 2025-06-23

## Table of Contents
- [Introduction](#introduction)
- [Why MCP Matters](#why-mcp-matters)
- [Core Architecture](#core-architecture)
- [MCP Components](#mcp-components)
- [Installation & Setup](#installation--setup)
- [Building MCP Servers](#building-mcp-servers)
- [Building MCP Clients](#building-mcp-clients)
- [Resources vs Tools](#resources-vs-tools)
- [Advanced Features](#advanced-features)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)
- [Integration Patterns](#integration-patterns)
- [Security Considerations](#security-considerations)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Resources & References](#resources--references)

## Introduction

Model Context Protocol (MCP) is a revolutionary open standard that solves the fundamental challenge of connecting AI systems with external tools and data sources. Think of it as a "universal adapter" that allows Large Language Models to seamlessly interact with any tool or information source through a standardized interface.

### What is MCP?

```python
# Traditional approach: Custom integration for each tool
class TraditionalAISystem:
    def __init__(self):
        self.weather_api = WeatherAPI()
        self.database_api = DatabaseAPI()
        self.calculator_api = CalculatorAPI()
        # M×N problem: Each model needs custom code for each tool
        
# MCP approach: Standardized protocol for all tools
class MCPAISystem:
    def __init__(self):
        self.mcp_client = MCPClient()
        # All tools accessible through one standard interface
```

### Key Benefits

1. **Standardization**: One protocol to rule them all
2. **Flexibility**: Dynamic tool discovery and invocation
3. **Scalability**: Solve the M×N integration problem
4. **Security**: Built-in permission and access control
5. **Modularity**: Plug-and-play architecture

## Why MCP Matters

### The Context Problem

Traditional LLMs face significant limitations:

```python
# Problem 1: Limited Context Window
MAX_TOKENS = 128000  # Even Claude 3 has limits

# Problem 2: Outdated Information
TRAINING_CUTOFF = "2024-04"  # No real-time data

# Problem 3: No External Interactions
can_access_apis = False
can_read_databases = False
can_execute_code = False
```

### Traditional Solutions vs MCP

```python
# Traditional: Static Context Injection
def traditional_approach(query):
    # Manually fetch and inject context
    weather_data = fetch_weather()
    db_data = query_database()
    
    prompt = f"""
    Context:
    Weather: {weather_data}
    Database: {db_data}
    
    Query: {query}
    """
    return llm.generate(prompt)

# MCP: Dynamic Context Access
def mcp_approach(query):
    # LLM decides what context it needs
    response = mcp_client.query(query)
    # Tools are called automatically as needed
    return response
```

## Core Architecture

### Three-Component System

```mermaid
graph LR
    A[Host/Application] <--> B[MCP Client]
    B <--> C[MCP Server]
    C <--> D[Tools/Resources]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
```

```python
class MCPArchitecture:
    """
    MCP's three-layer architecture
    """
    def __init__(self):
        self.layers = {
            'host': {
                'role': 'AI Application Interface',
                'examples': ['Claude Desktop', 'VS Code', 'Custom Apps'],
                'responsibilities': [
                    'User interaction',
                    'Permission management',
                    'Response presentation'
                ]
            },
            'client': {
                'role': 'Protocol Handler',
                'responsibilities': [
                    'Server connection management',
                    'Message routing',
                    'Protocol translation'
                ]
            },
            'server': {
                'role': 'Capability Provider',
                'provides': ['Tools', 'Resources', 'Prompts', 'Sampling'],
                'examples': ['Database connector', 'API wrapper', 'File system']
            }
        }
```

### Communication Flow

```python
class MCPCommunicationFlow:
    def demonstrate_flow(self):
        """
        Shows how messages flow through MCP
        """
        # 1. User query
        user_query = "What's the weather in Tokyo?"
        
        # 2. Host forwards to client
        client_request = {
            'type': 'query',
            'content': user_query,
            'model': 'claude-3'
        }
        
        # 3. Client discovers available tools
        available_tools = self.client.list_tools()
        # Returns: [weather_tool, calculator_tool, ...]
        
        # 4. LLM decides to use weather tool
        llm_decision = {
            'tool': 'get_weather',
            'parameters': {'city': 'Tokyo'}
        }
        
        # 5. Client invokes server tool
        server_response = self.server.execute_tool(
            tool='get_weather',
            params={'city': 'Tokyo'}
        )
        
        # 6. Response flows back
        return server_response
```

## MCP Components

### 1. Tools - Executable Actions

Tools are functions that can perform actions and cause side effects:

```python
from mcp import Server, Tool, ToolResult

class WeatherMCPServer(Server):
    """Example MCP server with weather tool"""
    
    def __init__(self):
        super().__init__("weather-server")
        self.tools = [
            Tool(
                name="get_weather",
                description="Get current weather for a city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius"
                        }
                    },
                    "required": ["city"]
                }
            )
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        if tool_name == "get_weather":
            return await self.get_weather(**arguments)
        
    async def get_weather(self, city: str, units: str = "celsius"):
        # Actual API call would go here
        weather_data = await fetch_weather_api(city)
        
        return ToolResult(
            success=True,
            data={
                "temperature": weather_data.temp,
                "conditions": weather_data.conditions,
                "humidity": weather_data.humidity,
                "units": units
            }
        )
```

### 2. Resources - Knowledge Access

Resources provide read-only access to information:

```python
from mcp import Resource, ResourceContent

class DocumentationServer(Server):
    """MCP server providing documentation resources"""
    
    def __init__(self):
        super().__init__("docs-server")
        self.resources = self.build_resource_list()
    
    def build_resource_list(self):
        return [
            Resource(
                uri="docs://api/overview",
                name="API Overview",
                description="Complete API documentation",
                mimeType="text/markdown"
            ),
            Resource(
                uri="docs://guides/quickstart",
                name="Quick Start Guide",
                description="Get started in 5 minutes",
                mimeType="text/markdown"
            )
        ]
    
    async def handle_resource_read(self, uri: str):
        if uri == "docs://api/overview":
            content = await self.load_documentation("api_overview.md")
            return ResourceContent(
                content=content,
                mimeType="text/markdown"
            )
```

### 3. Resource Templates - Dynamic Discovery

```python
class DatabaseResourceServer(Server):
    """Dynamic resource discovery with templates"""
    
    def get_resource_templates(self):
        return [
            ResourceTemplate(
                uriTemplate="db://tables/{table_name}/schema",
                name="Table Schema",
                description="Get schema for any table",
                mimeType="application/json"
            ),
            ResourceTemplate(
                uriTemplate="db://tables/{table_name}/rows/{row_id}",
                name="Table Row",
                description="Access specific row data",
                mimeType="application/json"
            )
        ]
    
    async def handle_resource_read(self, uri: str):
        # Parse URI template
        if match := re.match(r"db://tables/(\w+)/schema", uri):
            table_name = match.group(1)
            schema = await self.get_table_schema(table_name)
            return ResourceContent(
                content=json.dumps(schema),
                mimeType="application/json"
            )
```

### 4. Prompts - Reusable Workflows

```python
class AnalysisMCPServer(Server):
    """Server providing analysis prompts"""
    
    def get_prompts(self):
        return [
            Prompt(
                name="code_review",
                description="Comprehensive code review",
                arguments=[
                    PromptArgument(
                        name="code",
                        description="Code to review",
                        required=True
                    ),
                    PromptArgument(
                        name="language",
                        description="Programming language",
                        required=False
                    )
                ]
            )
        ]
    
    async def handle_prompt_get(self, name: str, arguments: dict):
        if name == "code_review":
            return self.build_code_review_prompt(**arguments)
    
    def build_code_review_prompt(self, code: str, language: str = None):
        return f"""
        Please review the following {language or 'code'}:
        
        ```{language or ''}
        {code}
        ```
        
        Focus on:
        1. Code quality and best practices
        2. Potential bugs or issues
        3. Performance considerations
        4. Security vulnerabilities
        5. Suggestions for improvement
        """
```

### 5. Sampling - Bidirectional AI

```python
class TextProcessingServer(Server):
    """Server that can request LLM assistance"""
    
    async def summarize_document(self, ctx, document_path: str):
        # Load document
        content = await self.load_document(document_path)
        
        # Request summary from client's LLM
        summary = await ctx.sample(
            prompt=f"Please summarize this document:\n\n{content}",
            max_tokens=500,
            temperature=0.7
        )
        
        # Process and store summary
        await self.store_summary(document_path, summary)
        
        return {
            "status": "success",
            "summary": summary,
            "original_length": len(content),
            "summary_length": len(summary)
        }
```

## Installation & Setup

### Python SDK Installation

```bash
# Install MCP SDK
pip install mcp-sdk

# For server development
pip install mcp-server

# For client development
pip install mcp-client

# Additional dependencies
pip install httpx pydantic asyncio
```

### Basic Server Setup

```python
# server.py
import asyncio
from mcp import Server, StdioTransport

class MyMCPServer(Server):
    def __init__(self):
        super().__init__("my-server")
        self.setup_tools()
        self.setup_resources()
    
    def setup_tools(self):
        # Define your tools
        pass
    
    def setup_resources(self):
        # Define your resources
        pass

async def main():
    server = MyMCPServer()
    transport = StdioTransport()
    await server.run(transport)

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration for Claude Desktop

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {
        "API_KEY": "your-api-key"
      }
    }
  }
}
```

## Building MCP Servers

### Complete Server Example

```python
import asyncio
import aiohttp
from datetime import datetime
from mcp import Server, Tool, Resource, StdioTransport
from mcp.types import ToolResult, ResourceContent

class ProductivityMCPServer(Server):
    """Full-featured MCP server example"""
    
    def __init__(self):
        super().__init__("productivity-server")
        self.session = None
        self.setup()
    
    def setup(self):
        """Initialize tools and resources"""
        self.tools = [
            Tool(
                name="create_task",
                description="Create a new task",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "due_date": {"type": "string", "format": "date"},
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"]
                        }
                    },
                    "required": ["title"]
                }
            ),
            Tool(
                name="search_tasks",
                description="Search for tasks",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "completed", "all"]
                        }
                    }
                }
            )
        ]
        
        self.resources = [
            Resource(
                uri="tasks://all",
                name="All Tasks",
                description="List of all tasks",
                mimeType="application/json"
            ),
            Resource(
                uri="tasks://today",
                name="Today's Tasks",
                description="Tasks due today",
                mimeType="application/json"
            )
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        """Route tool calls to appropriate handlers"""
        handlers = {
            "create_task": self.create_task,
            "search_tasks": self.search_tasks
        }
        
        if handler := handlers.get(tool_name):
            return await handler(**arguments)
        
        return ToolResult(
            success=False,
            error=f"Unknown tool: {tool_name}"
        )
    
    async def create_task(self, title: str, **kwargs):
        """Create a new task"""
        task = {
            "id": generate_id(),
            "title": title,
            "description": kwargs.get("description", ""),
            "due_date": kwargs.get("due_date"),
            "priority": kwargs.get("priority", "medium"),
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Save to database
        await self.save_task(task)
        
        return ToolResult(
            success=True,
            data={"task": task, "message": "Task created successfully"}
        )
    
    async def search_tasks(self, query: str = "", status: str = "all"):
        """Search tasks with filters"""
        tasks = await self.load_tasks()
        
        # Filter by status
        if status != "all":
            tasks = [t for t in tasks if t["status"] == status]
        
        # Search in title and description
        if query:
            query_lower = query.lower()
            tasks = [
                t for t in tasks
                if query_lower in t["title"].lower()
                or query_lower in t.get("description", "").lower()
            ]
        
        return ToolResult(
            success=True,
            data={"tasks": tasks, "count": len(tasks)}
        )
    
    async def handle_resource_read(self, uri: str):
        """Handle resource requests"""
        if uri == "tasks://all":
            tasks = await self.load_tasks()
            return ResourceContent(
                content=json.dumps(tasks, indent=2),
                mimeType="application/json"
            )
        
        elif uri == "tasks://today":
            tasks = await self.load_tasks()
            today = datetime.now().date()
            today_tasks = [
                t for t in tasks
                if t.get("due_date") == today.isoformat()
            ]
            return ResourceContent(
                content=json.dumps(today_tasks, indent=2),
                mimeType="application/json"
            )
        
        return None
    
    async def startup(self):
        """Initialize async resources"""
        self.session = aiohttp.ClientSession()
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

async def main():
    server = ProductivityMCPServer()
    transport = StdioTransport()
    
    try:
        await server.startup()
        await server.run(transport)
    finally:
        await server.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Server Features

```python
class AdvancedMCPServer(Server):
    """Server with advanced features"""
    
    def __init__(self):
        super().__init__("advanced-server")
        self.setup_middleware()
        self.setup_auth()
        self.setup_rate_limiting()
    
    def setup_middleware(self):
        """Add request/response middleware"""
        @self.before_tool_call
        async def log_tool_calls(tool_name: str, arguments: dict):
            logger.info(f"Tool called: {tool_name} with {arguments}")
            # Validate permissions
            if not await self.check_permissions(tool_name):
                raise PermissionError(f"Access denied for tool: {tool_name}")
        
        @self.after_tool_call
        async def log_responses(result: ToolResult):
            if not result.success:
                logger.error(f"Tool error: {result.error}")
    
    def setup_auth(self):
        """Setup authentication"""
        self.auth_tokens = set()
        
        @self.on_connect
        async def authenticate(headers: dict):
            token = headers.get("Authorization", "").replace("Bearer ", "")
            if token not in self.auth_tokens:
                raise AuthenticationError("Invalid token")
    
    def setup_rate_limiting(self):
        """Implement rate limiting"""
        self.rate_limiter = RateLimiter(
            max_requests=100,
            time_window=60  # 1 minute
        )
        
        @self.before_tool_call
        async def check_rate_limit(tool_name: str, arguments: dict):
            client_id = self.get_client_id()
            if not self.rate_limiter.allow_request(client_id):
                raise RateLimitError("Rate limit exceeded")
```

## Building MCP Clients

### Basic Client Implementation

```python
import asyncio
from mcp import Client, StdioTransport
from mcp.types import ToolCall, ResourceRead

class MCPClient:
    """Basic MCP client implementation"""
    
    def __init__(self):
        self.client = Client()
        self.transport = None
        self.connected = False
    
    async def connect(self, server_command: list):
        """Connect to MCP server"""
        self.transport = StdioTransport()
        await self.client.connect(
            self.transport,
            server_command=server_command
        )
        self.connected = True
        
        # Discover capabilities
        await self.discover_capabilities()
    
    async def discover_capabilities(self):
        """Discover server capabilities"""
        self.tools = await self.client.list_tools()
        self.resources = await self.client.list_resources()
        self.prompts = await self.client.list_prompts()
        
        print(f"Discovered {len(self.tools)} tools")
        print(f"Discovered {len(self.resources)} resources")
        print(f"Discovered {len(self.prompts)} prompts")
    
    async def call_tool(self, tool_name: str, **arguments):
        """Call a server tool"""
        if not self.connected:
            raise RuntimeError("Not connected to server")
        
        result = await self.client.call_tool(
            ToolCall(
                name=tool_name,
                arguments=arguments
            )
        )
        
        return result
    
    async def read_resource(self, uri: str):
        """Read a server resource"""
        if not self.connected:
            raise RuntimeError("Not connected to server")
        
        content = await self.client.read_resource(
            ResourceRead(uri=uri)
        )
        
        return content
    
    async def close(self):
        """Close connection"""
        if self.transport:
            await self.transport.close()
        self.connected = False

# Usage example
async def main():
    client = MCPClient()
    
    try:
        # Connect to server
        await client.connect(["python", "server.py"])
        
        # Call a tool
        result = await client.call_tool(
            "get_weather",
            city="Tokyo",
            units="celsius"
        )
        print(f"Weather result: {result}")
        
        # Read a resource
        content = await client.read_resource("docs://api/overview")
        print(f"Documentation: {content}")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Client with LLM Integration

```python
import openai
from typing import List, Dict, Any

class LLMIntegratedMCPClient(MCPClient):
    """MCP Client integrated with LLM"""
    
    def __init__(self, openai_api_key: str):
        super().__init__()
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.conversation_history = []
    
    async def process_query(self, user_query: str):
        """Process user query with LLM and MCP tools"""
        # Add to conversation
        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Prepare tools for LLM
        tools_schema = self.prepare_tools_schema()
        
        # Get LLM response
        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=self.conversation_history,
            tools=tools_schema,
            tool_choice="auto"
        )
        
        # Process tool calls if any
        if response.choices[0].message.tool_calls:
            tool_results = await self.execute_tool_calls(
                response.choices[0].message.tool_calls
            )
            
            # Add tool results to conversation
            for tool_call, result in tool_results:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result.data)
                })
            
            # Get final response
            final_response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=self.conversation_history
            )
            
            return final_response.choices[0].message.content
        
        return response.choices[0].message.content
    
    def prepare_tools_schema(self):
        """Convert MCP tools to OpenAI function schema"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools
        ]
    
    async def execute_tool_calls(self, tool_calls):
        """Execute LLM's tool calls via MCP"""
        results = []
        
        for tool_call in tool_calls:
            result = await self.call_tool(
                tool_call.function.name,
                **json.loads(tool_call.function.arguments)
            )
            results.append((tool_call, result))
        
        return results
```

## Resources vs Tools

### Key Differences

```python
class MCPConceptComparison:
    """Illustrates the difference between Resources and Tools"""
    
    def __init__(self):
        self.differences = {
            'tools': {
                'purpose': 'Execute actions',
                'side_effects': True,
                'control': 'Model-controlled',
                'examples': [
                    'Send email',
                    'Create database record',
                    'Call external API',
                    'Execute code'
                ],
                'permission': 'Requires explicit user consent'
            },
            'resources': {
                'purpose': 'Access knowledge',
                'side_effects': False,
                'control': 'Application-controlled',
                'examples': [
                    'Read documentation',
                    'Query database (read-only)',
                    'Access configuration',
                    'Retrieve file content'
                ],
                'permission': 'Managed by application'
            }
        }
    
    def when_to_use_tools(self):
        """Guidelines for using tools"""
        return """
        Use Tools when:
        - Action needs to be performed
        - External state will be modified
        - API calls with side effects
        - User approval is important
        - Real-time execution is needed
        """
    
    def when_to_use_resources(self):
        """Guidelines for using resources"""
        return """
        Use Resources when:
        - Information needs to be retrieved
        - Data is read-only
        - Content is relatively static
        - Batch access is beneficial
        - No external side effects
        """

# Practical example
class HybridMCPServer(Server):
    """Server demonstrating both tools and resources"""
    
    def __init__(self):
        super().__init__("hybrid-server")
        
        # Tool: Modifies data
        self.tools = [
            Tool(
                name="update_config",
                description="Update configuration value",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"}
                    }
                }
            )
        ]
        
        # Resource: Reads data
        self.resources = [
            Resource(
                uri="config://current",
                name="Current Configuration",
                description="Read current configuration",
                mimeType="application/json"
            )
        ]
```

## Advanced Features

### 1. Sampling - Bidirectional Communication

```python
class SamplingServer(Server):
    """Advanced server using sampling"""
    
    async def analyze_code(self, ctx, file_path: str):
        """Analyze code using client's LLM"""
        # Read code file
        code = await self.read_file(file_path)
        
        # Request analysis from client's LLM
        analysis = await ctx.sample(
            prompt=f"""
            Analyze this code for:
            1. Code quality issues
            2. Potential bugs
            3. Performance bottlenecks
            4. Security vulnerabilities
            
            Code:
            ```
            {code}
            ```
            """,
            max_tokens=1000,
            temperature=0.3
        )
        
        # Parse and structure the analysis
        structured_analysis = await self.parse_analysis(analysis)
        
        # Generate improvement suggestions
        suggestions = await ctx.sample(
            prompt=f"""
            Based on these issues: {structured_analysis}
            
            Generate specific code improvements and refactoring suggestions.
            """,
            max_tokens=800,
            temperature=0.5
        )
        
        return {
            "file": file_path,
            "analysis": structured_analysis,
            "suggestions": suggestions,
            "severity": self.calculate_severity(structured_analysis)
        }
```

### 2. Resource Templates for Dynamic Content

```python
class FileSystemMCPServer(Server):
    """Dynamic file system access via templates"""
    
    def get_resource_templates(self):
        return [
            ResourceTemplate(
                uriTemplate="file:///{path}",
                name="File Content",
                description="Access any file in allowed directories",
                mimeType="text/plain"
            ),
            ResourceTemplate(
                uriTemplate="dir:///{path}?pattern={pattern}",
                name="Directory Listing",
                description="List directory contents with optional pattern",
                mimeType="application/json"
            )
        ]
    
    async def handle_resource_read(self, uri: str):
        # Parse file URI
        if match := re.match(r"file:///(.+)", uri):
            path = match.group(1)
            if self.is_path_allowed(path):
                content = await self.read_file(path)
                return ResourceContent(
                    content=content,
                    mimeType=self.get_mime_type(path)
                )
        
        # Parse directory URI
        elif match := re.match(r"dir:///(.+)\?pattern=(.+)", uri):
            path = match.group(1)
            pattern = match.group(2)
            if self.is_path_allowed(path):
                files = await self.list_directory(path, pattern)
                return ResourceContent(
                    content=json.dumps(files),
                    mimeType="application/json"
                )
```

### 3. Streaming Responses

```python
class StreamingMCPServer(Server):
    """Server with streaming capabilities"""
    
    async def stream_logs(self, ctx, service_name: str):
        """Stream logs in real-time"""
        async def log_generator():
            async with self.tail_logs(service_name) as log_stream:
                async for line in log_stream:
                    yield {
                        "timestamp": datetime.now().isoformat(),
                        "service": service_name,
                        "message": line
                    }
        
        # Stream responses back to client
        async for log_entry in log_generator():
            await ctx.send_progress(
                message=f"Log: {log_entry['message']}",
                data=log_entry
            )
```

### 4. Multi-Server Coordination

```python
class CoordinatorMCPClient:
    """Client coordinating multiple MCP servers"""
    
    def __init__(self):
        self.servers = {}
    
    async def connect_servers(self, server_configs: Dict[str, dict]):
        """Connect to multiple servers"""
        for name, config in server_configs.items():
            client = MCPClient()
            await client.connect(config['command'])
            self.servers[name] = client
    
    async def execute_workflow(self, workflow: List[dict]):
        """Execute multi-server workflow"""
        context = {}
        
        for step in workflow:
            server_name = step['server']
            action_type = step['type']
            
            if action_type == 'tool':
                result = await self.servers[server_name].call_tool(
                    step['name'],
                    **self.resolve_parameters(step['parameters'], context)
                )
                context[step['output']] = result
                
            elif action_type == 'resource':
                content = await self.servers[server_name].read_resource(
                    self.resolve_uri(step['uri'], context)
                )
                context[step['output']] = content
        
        return context
```

## Real-World Examples

### 1. Database Query Assistant

```python
class DatabaseMCPServer(Server):
    """Production-ready database MCP server"""
    
    def __init__(self, connection_string: str):
        super().__init__("database-server")
        self.db = DatabaseConnection(connection_string)
        self.setup_capabilities()
    
    def setup_capabilities(self):
        # Tool for write operations
        self.tools = [
            Tool(
                name="execute_query",
                description="Execute SQL query (with write permissions)",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "params": {"type": "array"}
                    },
                    "required": ["query"]
                }
            )
        ]
        
        # Resources for read operations
        self.resource_templates = [
            ResourceTemplate(
                uriTemplate="db://schema/{table}",
                name="Table Schema",
                mimeType="application/json"
            ),
            ResourceTemplate(
                uriTemplate="db://data/{table}?limit={limit}&offset={offset}",
                name="Table Data",
                mimeType="application/json"
            )
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        if tool_name == "execute_query":
            # Validate query
            if self.is_destructive_query(arguments['query']):
                return ToolResult(
                    success=False,
                    error="Destructive queries require additional confirmation"
                )
            
            # Execute with transaction
            async with self.db.transaction() as tx:
                try:
                    result = await tx.execute(
                        arguments['query'],
                        arguments.get('params', [])
                    )
                    await tx.commit()
                    return ToolResult(
                        success=True,
                        data={"affected_rows": result.rowcount}
                    )
                except Exception as e:
                    await tx.rollback()
                    return ToolResult(
                        success=False,
                        error=str(e)
                    )
```

### 2. DevOps Automation Server

```python
class DevOpsMCPServer(Server):
    """DevOps automation via MCP"""
    
    def __init__(self):
        super().__init__("devops-server")
        self.k8s_client = KubernetesClient()
        self.monitoring = MonitoringClient()
        
    def get_tools(self):
        return [
            Tool(
                name="deploy_application",
                description="Deploy application to Kubernetes",
                parameters={
                    "type": "object",
                    "properties": {
                        "app_name": {"type": "string"},
                        "image": {"type": "string"},
                        "replicas": {"type": "integer", "minimum": 1},
                        "environment": {
                            "type": "string",
                            "enum": ["dev", "staging", "prod"]
                        }
                    }
                }
            ),
            Tool(
                name="rollback_deployment",
                description="Rollback to previous version",
                parameters={
                    "type": "object",
                    "properties": {
                        "app_name": {"type": "string"},
                        "environment": {"type": "string"}
                    }
                }
            )
        ]
    
    async def deploy_application(self, app_name: str, image: str, 
                                replicas: int, environment: str):
        # Pre-deployment checks
        health = await self.monitoring.check_cluster_health(environment)
        if not health.is_healthy:
            return ToolResult(
                success=False,
                error=f"Cluster unhealthy: {health.issues}"
            )
        
        # Deploy
        deployment = await self.k8s_client.create_deployment(
            name=app_name,
            image=image,
            replicas=replicas,
            namespace=environment
        )
        
        # Wait for rollout
        success = await self.k8s_client.wait_for_rollout(
            deployment.name,
            timeout=300
        )
        
        if success:
            # Update monitoring
            await self.monitoring.register_deployment(
                app_name=app_name,
                version=image.split(':')[-1],
                environment=environment
            )
            
            return ToolResult(
                success=True,
                data={
                    "deployment_id": deployment.uid,
                    "endpoint": deployment.endpoint,
                    "status": "healthy"
                }
            )
```

### 3. Research Assistant Server

```python
class ResearchMCPServer(Server):
    """Academic research assistant"""
    
    def __init__(self):
        super().__init__("research-server")
        self.arxiv_client = ArxivClient()
        self.citation_db = CitationDatabase()
        
    async def search_papers(self, ctx, query: str, max_results: int = 10):
        # Search papers
        papers = await self.arxiv_client.search(query, max_results)
        
        # Get AI summary of each paper
        summaries = []
        for paper in papers:
            summary = await ctx.sample(
                prompt=f"""
                Summarize this research paper abstract in 2-3 sentences:
                
                Title: {paper.title}
                Abstract: {paper.abstract}
                
                Focus on key contributions and methodology.
                """,
                max_tokens=150
            )
            
            summaries.append({
                "paper": paper,
                "summary": summary
            })
        
        # Generate comprehensive analysis
        analysis = await ctx.sample(
            prompt=f"""
            Based on these {len(papers)} papers about "{query}":
            
            {json.dumps(summaries, indent=2)}
            
            Provide:
            1. Common themes and approaches
            2. Research gaps
            3. Future directions
            """,
            max_tokens=500
        )
        
        return ToolResult(
            success=True,
            data={
                "papers": papers,
                "summaries": summaries,
                "analysis": analysis
            }
        )
```

## Best Practices

### 1. Server Design Principles

```python
class BestPracticesMCPServer(Server):
    """Demonstrates MCP best practices"""
    
    def __init__(self):
        super().__init__("best-practices-server")
        
    # 1. Clear, descriptive naming
    def get_tools(self):
        return [
            Tool(
                name="user_create",  # Noun_verb pattern
                description="Create a new user account",  # Clear description
                parameters={
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "format": "email",
                            "description": "User's email address"
                        },
                        "role": {
                            "type": "string",
                            "enum": ["admin", "user", "guest"],
                            "default": "user",
                            "description": "User's role in the system"
                        }
                    },
                    "required": ["email"],  # Minimal required fields
                    "additionalProperties": False  # Strict validation
                }
            )
        ]
    
    # 2. Comprehensive error handling
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        try:
            # Validate input
            validation_error = self.validate_arguments(tool_name, arguments)
            if validation_error:
                return ToolResult(
                    success=False,
                    error=validation_error,
                    error_code="VALIDATION_ERROR"
                )
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self.execute_tool(tool_name, arguments),
                timeout=30.0
            )
            
            return result
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error="Operation timed out",
                error_code="TIMEOUT"
            )
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            return ToolResult(
                success=False,
                error="Internal server error",
                error_code="INTERNAL_ERROR"
            )
    
    # 3. Resource versioning
    def get_resources(self):
        return [
            Resource(
                uri="config://v1/settings",
                name="Settings (v1)",
                description="Current configuration settings",
                mimeType="application/json",
                metadata={
                    "version": "1.0",
                    "deprecated": False,
                    "schema": "https://example.com/schemas/settings-v1.json"
                }
            )
        ]
    
    # 4. Efficient resource templates
    def get_resource_templates(self):
        return [
            ResourceTemplate(
                uriTemplate="data://v1/{collection}/{id}",
                name="Data Item",
                description="Access individual data items",
                mimeType="application/json",
                metadata={
                    "cache_control": "max-age=300",
                    "supports_etag": True
                }
            )
        ]
```

### 2. Security Best Practices

```python
class SecureMCPServer(Server):
    """Security-focused MCP server"""
    
    def __init__(self):
        super().__init__("secure-server")
        self.setup_security()
    
    def setup_security(self):
        # 1. Input sanitization
        self.sanitizer = InputSanitizer()
        
        # 2. Rate limiting per client
        self.rate_limiter = TokenBucketRateLimiter(
            tokens_per_minute=100,
            burst_size=20
        )
        
        # 3. Audit logging
        self.audit_logger = AuditLogger()
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        # Check permissions
        if not await self.check_tool_permission(tool_name):
            await self.audit_logger.log_denied_access(tool_name)
            return ToolResult(
                success=False,
                error="Permission denied",
                error_code="FORBIDDEN"
            )
        
        # Sanitize inputs
        safe_arguments = self.sanitizer.sanitize(arguments)
        
        # Log action
        await self.audit_logger.log_tool_call(
            tool_name=tool_name,
            arguments=safe_arguments,
            client_id=self.get_client_id()
        )
        
        # Execute with sandboxing
        return await self.sandboxed_execution(tool_name, safe_arguments)
    
    async def sandboxed_execution(self, tool_name: str, arguments: dict):
        """Execute tool in sandboxed environment"""
        sandbox = Sandbox(
            memory_limit="512MB",
            cpu_limit=0.5,
            network_access=False,
            filesystem_access="read-only"
        )
        
        return await sandbox.execute(
            self.tools[tool_name],
            arguments
        )
```

### 3. Performance Optimization

```python
class OptimizedMCPServer(Server):
    """Performance-optimized MCP server"""
    
    def __init__(self):
        super().__init__("optimized-server")
        self.setup_caching()
        self.setup_connection_pool()
    
    def setup_caching(self):
        # 1. In-memory cache for frequently accessed resources
        self.cache = TTLCache(
            maxsize=1000,
            ttl=300  # 5 minutes
        )
        
        # 2. Redis for distributed caching
        self.redis_cache = RedisCache(
            host="localhost",
            port=6379,
            db=0
        )
    
    def setup_connection_pool(self):
        # Database connection pool
        self.db_pool = asyncpg.create_pool(
            min_size=10,
            max_size=50,
            command_timeout=10
        )
    
    async def handle_resource_read(self, uri: str):
        # Check cache first
        cache_key = f"resource:{uri}"
        
        # Try in-memory cache
        if cached := self.cache.get(cache_key):
            return cached
        
        # Try Redis cache
        if cached := await self.redis_cache.get(cache_key):
            self.cache[cache_key] = cached  # Populate L1 cache
            return cached
        
        # Fetch from source
        content = await self.fetch_resource(uri)
        
        # Update caches
        self.cache[cache_key] = content
        await self.redis_cache.set(
            cache_key, 
            content, 
            expire=300
        )
        
        return content
    
    async def parallel_tool_execution(self, tool_calls: List[ToolCall]):
        """Execute multiple tools in parallel"""
        tasks = []
        
        for call in tool_calls:
            # Check if tool can be parallelized
            if self.is_parallelizable(call.name):
                task = asyncio.create_task(
                    self.handle_tool_call(call.name, call.arguments)
                )
                tasks.append(task)
            else:
                # Execute sequentially
                result = await self.handle_tool_call(
                    call.name, 
                    call.arguments
                )
                tasks.append(asyncio.create_task(
                    asyncio.coroutine(lambda: result)()
                ))
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks)
        return results
```

## Integration Patterns

### 1. Microservices Integration

```python
class MicroservicesMCPGateway(Server):
    """MCP gateway for microservices"""
    
    def __init__(self):
        super().__init__("microservices-gateway")
        self.service_registry = ServiceRegistry()
        self.circuit_breaker = CircuitBreaker()
    
    async def discover_services(self):
        """Auto-discover microservices and their capabilities"""
        services = await self.service_registry.get_all_services()
        
        for service in services:
            # Get service OpenAPI spec
            spec = await self.fetch_service_spec(service)
            
            # Convert to MCP tools
            tools = self.openapi_to_mcp_tools(spec)
            self.register_tools(tools)
            
            # Health monitoring
            self.monitor_service_health(service)
    
    def openapi_to_mcp_tools(self, spec: dict) -> List[Tool]:
        """Convert OpenAPI spec to MCP tools"""
        tools = []
        
        for path, methods in spec['paths'].items():
            for method, operation in methods.items():
                if method in ['post', 'put', 'delete']:
                    tools.append(Tool(
                        name=operation['operationId'],
                        description=operation['summary'],
                        parameters=self.convert_parameters(operation)
                    ))
        
        return tools
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        # Find service for tool
        service = self.find_service_for_tool(tool_name)
        
        # Circuit breaker check
        if not self.circuit_breaker.is_closed(service.name):
            return ToolResult(
                success=False,
                error=f"Service {service.name} is unavailable"
            )
        
        try:
            # Call microservice
            result = await self.call_microservice(
                service, 
                tool_name, 
                arguments
            )
            
            # Reset circuit breaker on success
            self.circuit_breaker.record_success(service.name)
            
            return result
            
        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure(service.name)
            raise
```

### 2. Event-Driven Architecture

```python
class EventDrivenMCPServer(Server):
    """MCP server with event-driven patterns"""
    
    def __init__(self):
        super().__init__("event-driven-server")
        self.event_bus = EventBus()
        self.saga_manager = SagaManager()
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        # Start distributed transaction
        saga = self.saga_manager.create_saga(tool_name)
        
        try:
            # Emit pre-execution event
            await self.event_bus.emit(
                "tool.pre_execute",
                {
                    "tool": tool_name,
                    "arguments": arguments,
                    "saga_id": saga.id
                }
            )
            
            # Execute tool
            result = await self.execute_with_saga(
                saga, 
                tool_name, 
                arguments
            )
            
            # Emit success event
            await self.event_bus.emit(
                "tool.success",
                {
                    "tool": tool_name,
                    "result": result,
                    "saga_id": saga.id
                }
            )
            
            return result
            
        except Exception as e:
            # Trigger compensation
            await saga.compensate()
            
            # Emit failure event
            await self.event_bus.emit(
                "tool.failure",
                {
                    "tool": tool_name,
                    "error": str(e),
                    "saga_id": saga.id
                }
            )
            
            raise
```

### 3. GraphQL Integration

```python
class GraphQLMCPServer(Server):
    """MCP server wrapping GraphQL API"""
    
    def __init__(self, graphql_endpoint: str):
        super().__init__("graphql-server")
        self.graphql_client = GraphQLClient(graphql_endpoint)
        self.schema = None
    
    async def startup(self):
        """Load GraphQL schema and generate tools"""
        self.schema = await self.graphql_client.introspect()
        
        # Generate tools from mutations
        for mutation in self.schema.mutations:
            tool = Tool(
                name=mutation.name,
                description=mutation.description or f"Execute {mutation.name}",
                parameters=self.graphql_type_to_json_schema(mutation.args)
            )
            self.register_tool(tool)
        
        # Generate resources from queries
        for query in self.schema.queries:
            resource = Resource(
                uri=f"graphql://{query.name}",
                name=query.name,
                description=query.description or f"Query {query.name}",
                mimeType="application/json"
            )
            self.register_resource(resource)
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        # Build GraphQL mutation
        mutation = self.build_mutation(tool_name, arguments)
        
        # Execute
        result = await self.graphql_client.execute(mutation)
        
        if result.errors:
            return ToolResult(
                success=False,
                error=result.errors[0].message
            )
        
        return ToolResult(
            success=True,
            data=result.data[tool_name]
        )
```

## Security Considerations

### 1. Authentication & Authorization

```python
class SecureAuthMCPServer(Server):
    """MCP server with comprehensive auth"""
    
    def __init__(self):
        super().__init__("secure-auth-server")
        self.auth_provider = JWTAuthProvider()
        self.rbac = RoleBasedAccessControl()
    
    async def authenticate_client(self, headers: dict):
        """Authenticate incoming connections"""
        # Extract token
        auth_header = headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise AuthenticationError("Missing bearer token")
        
        token = auth_header[7:]
        
        # Verify JWT
        try:
            claims = self.auth_provider.verify_token(token)
            self.current_user = claims['sub']
            self.current_roles = claims.get('roles', [])
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    def check_tool_permission(self, tool_name: str) -> bool:
        """Check if user can access tool"""
        required_permission = f"tools.{tool_name}"
        
        for role in self.current_roles:
            if self.rbac.role_has_permission(role, required_permission):
                return True
        
        return False
    
    def check_resource_permission(self, uri: str) -> bool:
        """Check if user can access resource"""
        # Extract resource type from URI
        resource_type = uri.split("://")[0]
        required_permission = f"resources.{resource_type}.read"
        
        for role in self.current_roles:
            if self.rbac.role_has_permission(role, required_permission):
                return True
        
        return False
```

### 2. Data Encryption

```python
class EncryptedMCPServer(Server):
    """MCP server with end-to-end encryption"""
    
    def __init__(self, private_key_path: str):
        super().__init__("encrypted-server")
        self.crypto = CryptoManager(private_key_path)
    
    async def handle_encrypted_tool_call(self, encrypted_request: bytes):
        """Handle encrypted tool calls"""
        # Decrypt request
        decrypted = self.crypto.decrypt(encrypted_request)
        request = json.loads(decrypted)
        
        # Validate signature
        if not self.crypto.verify_signature(request):
            return self.encrypt_response({
                "success": False,
                "error": "Invalid signature"
            })
        
        # Process request
        result = await self.handle_tool_call(
            request['tool_name'],
            request['arguments']
        )
        
        # Encrypt response
        return self.encrypt_response(result)
    
    def encrypt_response(self, response: dict) -> bytes:
        """Encrypt and sign response"""
        # Add timestamp to prevent replay attacks
        response['timestamp'] = datetime.now().isoformat()
        
        # Sign response
        response['signature'] = self.crypto.sign(
            json.dumps(response, sort_keys=True)
        )
        
        # Encrypt
        return self.crypto.encrypt(
            json.dumps(response).encode()
        )
```

### 3. Audit Logging

```python
class AuditedMCPServer(Server):
    """MCP server with comprehensive audit logging"""
    
    def __init__(self):
        super().__init__("audited-server")
        self.audit_logger = StructuredAuditLogger()
    
    async def log_tool_execution(self, 
                                tool_name: str, 
                                arguments: dict,
                                result: ToolResult):
        """Log tool execution for audit trail"""
        await self.audit_logger.log({
            "event_type": "tool_execution",
            "timestamp": datetime.utcnow().isoformat(),
            "tool": tool_name,
            "arguments": self.sanitize_for_logging(arguments),
            "success": result.success,
            "error": result.error if not result.success else None,
            "client_id": self.get_client_id(),
            "user_id": self.current_user,
            "ip_address": self.get_client_ip(),
            "session_id": self.session_id,
            "execution_time_ms": self.last_execution_time
        })
    
    def sanitize_for_logging(self, data: dict) -> dict:
        """Remove sensitive data before logging"""
        sensitive_fields = [
            'password', 'token', 'secret', 
            'api_key', 'private_key'
        ]
        
        sanitized = data.copy()
        
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "***REDACTED***"
        
        return sanitized
```

## Performance Optimization

### 1. Caching Strategies

```python
class CachedMCPServer(Server):
    """MCP server with multi-level caching"""
    
    def __init__(self):
        super().__init__("cached-server")
        
        # L1: In-process cache
        self.l1_cache = LRUCache(maxsize=1000)
        
        # L2: Redis cache
        self.l2_cache = RedisCache()
        
        # L3: CDN for static resources
        self.cdn = CDNManager()
    
    async def get_cached_resource(self, uri: str):
        """Multi-level cache lookup"""
        # Check L1
        if cached := self.l1_cache.get(uri):
            self.metrics.record_cache_hit("l1")
            return cached
        
        # Check L2
        if cached := await self.l2_cache.get(uri):
            self.l1_cache.put(uri, cached)
            self.metrics.record_cache_hit("l2")
            return cached
        
        # Check CDN
        if self.is_cdn_eligible(uri):
            if cached := await self.cdn.get(uri):
                # Populate lower caches
                await self.l2_cache.set(uri, cached, ttl=3600)
                self.l1_cache.put(uri, cached)
                self.metrics.record_cache_hit("cdn")
                return cached
        
        # Cache miss - fetch from source
        self.metrics.record_cache_miss()
        content = await self.fetch_resource(uri)
        
        # Update all cache levels
        await self.update_caches(uri, content)
        
        return content
```

### 2. Connection Pooling

```python
class PooledMCPServer(Server):
    """MCP server with connection pooling"""
    
    def __init__(self):
        super().__init__("pooled-server")
        self.pools = {}
    
    async def startup(self):
        """Initialize connection pools"""
        # Database pool
        self.pools['database'] = await asyncpg.create_pool(
            dsn=os.getenv('DATABASE_URL'),
            min_size=10,
            max_size=50,
            max_queries=50000,
            max_inactive_connection_lifetime=300
        )
        
        # HTTP connection pool
        self.pools['http'] = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30
            ),
            timeout=httpx.Timeout(30.0)
        )
        
        # Redis connection pool
        self.pools['redis'] = await aioredis.create_redis_pool(
            'redis://localhost',
            minsize=5,
            maxsize=20
        )
    
    async def execute_db_query(self, query: str, params: list):
        """Execute query using connection pool"""
        async with self.pools['database'].acquire() as conn:
            # Use prepared statements for better performance
            stmt = await conn.prepare(query)
            result = await stmt.fetch(*params)
            return result
```

### 3. Batch Processing

```python
class BatchProcessingMCPServer(Server):
    """MCP server with batch processing capabilities"""
    
    def __init__(self):
        super().__init__("batch-server")
        self.batch_processor = BatchProcessor(
            batch_size=100,
            flush_interval=1.0  # seconds
        )
    
    async def handle_batch_tool(self, items: List[dict]):
        """Process multiple items efficiently"""
        # Group by operation type
        grouped = defaultdict(list)
        for item in items:
            grouped[item['operation']].append(item)
        
        # Process each group in parallel
        tasks = []
        for operation, batch in grouped.items():
            task = asyncio.create_task(
                self.process_batch(operation, batch)
            )
            tasks.append(task)
        
        # Wait for all batches
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for batch_result in results:
            all_results.extend(batch_result)
        
        return ToolResult(
            success=True,
            data={"results": all_results, "total": len(all_results)}
        )
    
    async def process_batch(self, operation: str, items: List[dict]):
        """Process a batch of similar operations"""
        if operation == "insert":
            # Bulk insert
            return await self.bulk_insert(items)
        elif operation == "update":
            # Bulk update
            return await self.bulk_update(items)
        elif operation == "delete":
            # Bulk delete
            return await self.bulk_delete(items)
```

## Troubleshooting

### Common Issues and Solutions

```python
class MCPTroubleshooter:
    """Common MCP issues and solutions"""
    
    def __init__(self):
        self.issues = {
            "connection_refused": {
                "symptoms": [
                    "Cannot connect to MCP server",
                    "Connection refused error"
                ],
                "causes": [
                    "Server not running",
                    "Wrong port/address",
                    "Firewall blocking connection"
                ],
                "solutions": [
                    "Check server is running: ps aux | grep server.py",
                    "Verify connection settings in config",
                    "Check firewall rules: sudo ufw status"
                ]
            },
            "tool_not_found": {
                "symptoms": [
                    "Tool 'X' not found error",
                    "Unknown tool in response"
                ],
                "causes": [
                    "Tool not registered",
                    "Typo in tool name",
                    "Server not fully initialized"
                ],
                "solutions": [
                    "List available tools: client.list_tools()",
                    "Check tool registration in server code",
                    "Ensure server startup() completed"
                ]
            },
            "timeout_errors": {
                "symptoms": [
                    "Operation timeout",
                    "No response from server"
                ],
                "causes": [
                    "Long-running operations",
                    "Network issues",
                    "Server overload"
                ],
                "solutions": [
                    "Increase timeout in client config",
                    "Implement progress reporting",
                    "Add server-side caching",
                    "Scale server horizontally"
                ]
            }
        }
    
    def diagnose_issue(self, error_message: str):
        """Diagnose issue from error message"""
        for issue_type, details in self.issues.items():
            for symptom in details['symptoms']:
                if symptom.lower() in error_message.lower():
                    return {
                        "issue_type": issue_type,
                        "likely_causes": details['causes'],
                        "recommended_solutions": details['solutions']
                    }
        
        return {
            "issue_type": "unknown",
            "recommendation": "Check server logs for more details"
        }
```

### Debug Mode Server

```python
class DebugMCPServer(Server):
    """MCP server with debug capabilities"""
    
    def __init__(self, debug_level: str = "INFO"):
        super().__init__("debug-server")
        self.setup_debugging(debug_level)
    
    def setup_debugging(self, level: str):
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Enable request/response logging
        self.enable_message_logging = True
        
        # Performance profiling
        self.profiler = cProfile.Profile()
        
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile performance of operations"""
        start_time = time.time()
        self.profiler.enable()
        
        try:
            yield
        finally:
            self.profiler.disable()
            elapsed = time.time() - start_time
            
            # Log performance metrics
            logger.info(f"{operation_name} took {elapsed:.3f}s")
            
            if elapsed > 1.0:  # Log slow operations
                stats = pstats.Stats(self.profiler)
                stats.sort_stats('cumulative')
                stats.print_stats(10)  # Top 10 functions
```

## Resources & References

### Official Documentation
- **[MCP Specification](https://github.com/anthropics/mcp)** - Official protocol specification
- **[MCP SDK Documentation](https://github.com/anthropics/mcp-sdk)** - SDK reference
- **[Claude Desktop Integration](https://claude.ai/docs/mcp)** - Integration guide

### Tutorials & Examples
- **[MCP Server Examples](https://github.com/anthropics/mcp-servers)** - Example implementations
- **[Building Your First MCP Server](https://docs.anthropic.com/mcp/tutorial)** - Step-by-step tutorial
- **[MCP Best Practices](https://docs.anthropic.com/mcp/best-practices)** - Design guidelines

### Community Resources
- **[MCP Discord](https://discord.gg/mcp)** - Community support
- **[MCP GitHub Discussions](https://github.com/anthropics/mcp/discussions)** - Q&A forum
- **[Awesome MCP](https://github.com/awesome-mcp/awesome-mcp)** - Curated list of MCP resources

### Tools & Libraries
- **[MCP Python SDK](https://pypi.org/project/mcp-sdk/)** - Official Python SDK
- **[MCP TypeScript SDK](https://www.npmjs.com/package/@anthropic/mcp)** - Official TypeScript SDK
- **[MCP Server Template](https://github.com/anthropics/mcp-server-template)** - Starter template
- **[MCP Testing Framework](https://github.com/anthropics/mcp-test)** - Testing utilities

### Related Technologies
- **[LangChain](https://langchain.com)** - LLM application framework
- **[OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)** - Similar concept
- **[Tool Use in Claude](https://docs.anthropic.com/claude/docs/tool-use)** - Claude's tool use
- **[JSON-RPC](https://www.jsonrpc.org/)** - Protocol foundation