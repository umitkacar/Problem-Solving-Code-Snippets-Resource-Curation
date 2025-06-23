# 🖥️ MCP Client Implementation Guide

**Build AI Agents That Connect to Any Tool** - From basic clients to production-ready AI systems with MCP.

## 🎯 Common Client Implementation Problems

### Problem 1: "I need to connect my LLM to MCP servers"

#### Quick Solution
```python
from mcp import Client, StdioTransport

# Basic client connection
client = Client("my-client")
transport = StdioTransport()

# Connect to server
await client.connect(transport, "python", ["my_server.py"])

# Use tools
result = await client.call_tool("query_database", {
    "query": "SELECT * FROM users LIMIT 10"
})
```

#### Production Solution
```python
import asyncio
from typing import Dict, List, Any, Optional
from mcp import Client, Transport
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

class ProductionMCPClient:
    """Production-ready MCP client with resilience"""
    
    def __init__(self, config: dict):
        self.config = config
        self.clients = {}
        self.logger = logging.getLogger(__name__)
        self.health_check_interval = 30
        self._health_tasks = {}
        
    async def connect_server(
        self, 
        name: str, 
        command: str, 
        args: List[str],
        env: Dict[str, str] = None,
        retry_attempts: int = 3
    ):
        """Connect to MCP server with retry logic"""
        
        @retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10)
        )
        async def _connect():
            client = Client(f"{self.config['client_name']}-{name}")
            transport = StdioTransport()
            
            try:
                await client.connect(transport, command, args, env)
                
                # Verify connection
                capabilities = await client.get_capabilities()
                self.logger.info(
                    f"Connected to {name}: {len(capabilities.tools)} tools, "
                    f"{len(capabilities.resources)} resources"
                )
                
                return client
                
            except Exception as e:
                self.logger.error(f"Failed to connect to {name}: {e}")
                raise
                
        client = await _connect()
        self.clients[name] = client
        
        # Start health monitoring
        self._health_tasks[name] = asyncio.create_task(
            self._monitor_health(name, client)
        )
        
        return client
        
    async def _monitor_health(self, name: str, client: Client):
        """Monitor server health"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Simple health check
                await client.list_tools()
                
            except Exception as e:
                self.logger.warning(f"Health check failed for {name}: {e}")
                
                # Attempt reconnection
                try:
                    await self.reconnect_server(name)
                except Exception:
                    self.logger.error(f"Reconnection failed for {name}")
                    
    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict,
        timeout: float = 30.0
    ) -> Any:
        """Call tool with timeout and error handling"""
        
        if server not in self.clients:
            raise ValueError(f"Not connected to server: {server}")
            
        client = self.clients[server]
        
        try:
            result = await asyncio.wait_for(
                client.call_tool(tool, arguments),
                timeout=timeout
            )
            
            self.logger.debug(
                f"Tool call successful: {server}.{tool}"
            )
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(
                f"Tool call timed out: {server}.{tool}"
            )
            raise
            
        except Exception as e:
            self.logger.error(
                f"Tool call failed: {server}.{tool} - {e}"
            )
            raise
```

### Problem 2: "I need to build an AI agent with multiple MCP servers"

#### Quick Solution
```python
# Simple multi-server agent
class SimpleAgent:
    def __init__(self):
        self.clients = {}
        
    async def setup(self):
        # Connect to multiple servers
        self.clients['db'] = await connect_server("database-server")
        self.clients['api'] = await connect_server("api-server")
        self.clients['files'] = await connect_server("filesystem-server")
        
    async def process_request(self, request: str):
        # Route to appropriate server
        if "database" in request:
            return await self.clients['db'].call_tool("query", {"sql": request})
        elif "api" in request:
            return await self.clients['api'].call_tool("call", {"endpoint": request})
```

#### Production Solution
```python
from typing import Dict, List, Any, Optional, Callable
import json
from dataclasses import dataclass
from enum import Enum

class ToolCategory(Enum):
    DATABASE = "database"
    API = "api"
    FILESYSTEM = "filesystem"
    COMPUTE = "compute"
    MESSAGING = "messaging"

@dataclass
class ToolRegistration:
    server: str
    tool: str
    category: ToolCategory
    description: str
    schema: dict

class IntelligentMCPAgent:
    """AI agent with intelligent tool routing"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.mcp_client = ProductionMCPClient({"client_name": "ai-agent"})
        self.tool_registry = {}
        self.conversation_history = []
        
    async def initialize(self, server_configs: List[dict]):
        """Initialize with multiple MCP servers"""
        
        # Connect to all servers
        for config in server_configs:
            await self.mcp_client.connect_server(
                name=config['name'],
                command=config['command'],
                args=config.get('args', []),
                env=config.get('env', {})
            )
            
        # Discover and register all tools
        await self._discover_tools()
        
    async def _discover_tools(self):
        """Discover tools from all connected servers"""
        
        for server_name, client in self.mcp_client.clients.items():
            tools = await client.list_tools()
            
            for tool in tools:
                # Categorize tool based on name/description
                category = self._categorize_tool(tool)
                
                registration = ToolRegistration(
                    server=server_name,
                    tool=tool.name,
                    category=category,
                    description=tool.description,
                    schema=tool.input_schema
                )
                
                self.tool_registry[f"{server_name}.{tool.name}"] = registration
                
    def _categorize_tool(self, tool) -> ToolCategory:
        """Categorize tool based on its properties"""
        name_lower = tool.name.lower()
        desc_lower = tool.description.lower()
        
        if any(kw in name_lower or kw in desc_lower for kw in ['db', 'sql', 'query']):
            return ToolCategory.DATABASE
        elif any(kw in name_lower or kw in desc_lower for kw in ['api', 'http', 'rest']):
            return ToolCategory.API
        elif any(kw in name_lower or kw in desc_lower for kw in ['file', 'read', 'write']):
            return ToolCategory.FILESYSTEM
        else:
            return ToolCategory.COMPUTE
            
    async def process_request(
        self,
        user_request: str,
        context: Optional[dict] = None
    ) -> str:
        """Process user request using available tools"""
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_request,
            "context": context
        })
        
        # Create tool descriptions for LLM
        tool_descriptions = self._format_tools_for_llm()
        
        # Let LLM decide which tools to use
        llm_prompt = f"""
        You are an AI assistant with access to the following tools:
        
        {tool_descriptions}
        
        User request: {user_request}
        
        Context: {json.dumps(context or {})}
        
        Please analyze the request and use the appropriate tools to fulfill it.
        Respond with a JSON array of tool calls needed.
        """
        
        # Get LLM's tool selection
        llm_response = await self.llm.complete(llm_prompt)
        tool_calls = json.loads(llm_response)
        
        # Execute tool calls
        results = []
        for call in tool_calls:
            result = await self._execute_tool_call(call)
            results.append(result)
            
        # Let LLM synthesize the results
        synthesis_prompt = f"""
        Original request: {user_request}
        
        Tool results:
        {json.dumps(results, indent=2)}
        
        Please provide a helpful response to the user based on these results.
        """
        
        final_response = await self.llm.complete(synthesis_prompt)
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": final_response,
            "tool_calls": tool_calls,
            "tool_results": results
        })
        
        return final_response
        
    async def _execute_tool_call(self, call: dict) -> dict:
        """Execute a single tool call"""
        
        tool_key = f"{call['server']}.{call['tool']}"
        
        if tool_key not in self.tool_registry:
            return {
                "error": f"Tool not found: {tool_key}"
            }
            
        try:
            result = await self.mcp_client.call_tool(
                server=call['server'],
                tool=call['tool'],
                arguments=call.get('arguments', {})
            )
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### Problem 3: "I need to handle streaming responses from MCP servers"

#### Quick Solution
```python
# Basic streaming
async def stream_logs(client, container_id):
    async for chunk in client.stream_tool("get_logs", {"container": container_id}):
        print(chunk, end='', flush=True)
```

#### Production Solution
```python
import asyncio
from typing import AsyncIterator, Optional, Callable
from collections import deque
import time

class StreamingMCPClient:
    """MCP client with advanced streaming support"""
    
    def __init__(self):
        self.active_streams = {}
        self.stream_buffers = {}
        
    async def stream_tool(
        self,
        server: str,
        tool: str,
        arguments: dict,
        on_chunk: Optional[Callable] = None,
        buffer_size: int = 1000,
        timeout: float = 300.0
    ) -> AsyncIterator[Any]:
        """Stream tool results with buffering and timeout"""
        
        stream_id = f"{server}.{tool}.{time.time()}"
        self.stream_buffers[stream_id] = deque(maxlen=buffer_size)
        self.active_streams[stream_id] = True
        
        try:
            client = self.clients[server]
            
            # Start streaming
            async with asyncio.timeout(timeout):
                async for chunk in client.stream_tool(tool, arguments):
                    if not self.active_streams.get(stream_id, False):
                        break
                        
                    # Buffer management
                    self.stream_buffers[stream_id].append({
                        "timestamp": time.time(),
                        "data": chunk
                    })
                    
                    # Callback processing
                    if on_chunk:
                        try:
                            await on_chunk(chunk)
                        except Exception as e:
                            self.logger.error(f"Chunk callback error: {e}")
                            
                    yield chunk
                    
        except asyncio.TimeoutError:
            self.logger.error(f"Stream timeout: {stream_id}")
            raise
            
        finally:
            # Cleanup
            self.active_streams.pop(stream_id, None)
            self.stream_buffers.pop(stream_id, None)
            
    async def parallel_stream(
        self,
        streams: List[dict],
        merge_strategy: str = "round_robin"
    ) -> AsyncIterator[dict]:
        """Handle multiple parallel streams"""
        
        queues = {
            i: asyncio.Queue(maxsize=100)
            for i in range(len(streams))
        }
        
        # Start stream tasks
        tasks = []
        for i, stream_config in enumerate(streams):
            task = asyncio.create_task(
                self._stream_to_queue(
                    stream_config,
                    queues[i]
                )
            )
            tasks.append(task)
            
        try:
            if merge_strategy == "round_robin":
                async for item in self._round_robin_merge(queues):
                    yield item
                    
            elif merge_strategy == "priority":
                async for item in self._priority_merge(queues, streams):
                    yield item
                    
        finally:
            # Cancel all tasks
            for task in tasks:
                task.cancel()
```

## 📚 Essential MCP Client Resources

### 🏆 Core Libraries

**[MCP Python Client](https://github.com/anthropics/mcp-python)** - Official Python client
- Full async support with asyncio
- Type-safe with comprehensive typing
- Built-in retry and timeout handling

**[MCP TypeScript Client](https://github.com/anthropics/mcp-typescript)** - Official TypeScript client
- Works in Node.js and browsers
- React hooks for UI integration
- WebSocket and stdio transports

**[MCP Go Client](https://github.com/modelcontextprotocol/go-mcp)** - Go implementation
- High-performance concurrent client
- Native Go patterns
- Excellent for microservices

### 📖 Integration Guides

**[LangChain MCP Integration](https://python.langchain.com/docs/integrations/tools/mcp)** - Use MCP with LangChain
- Wrap MCP servers as LangChain tools
- Chain multiple MCP calls
- Memory and conversation management

**[Vercel AI SDK + MCP](https://sdk.vercel.ai/docs/guides/mcp)** - Web app integration
- React components for MCP
- Real-time streaming UI
- Authentication helpers

**[AutoGPT MCP Plugin](https://github.com/Significant-Gravitas/AutoGPT-MCP)** - Autonomous agents
- Self-directed tool usage
- Goal-oriented planning
- Memory persistence

### 🛠️ Development Tools

**[MCP Client Debugger](https://github.com/modelcontextprotocol/debugger)** - Debug client-server communication
- Message inspection
- Performance profiling
- Error analysis

**[MCP Mock Server](https://github.com/modelcontextprotocol/mock-server)** - Testing without real servers
- Simulate server responses
- Test error conditions
- Load testing

## 🔧 Advanced Client Patterns

### Circuit Breaker Pattern
```python
# Problem: Prevent cascading failures
class CircuitBreakerMCP:
    def __init__(self):
        self.failure_threshold = 5
        self.recovery_timeout = 60
        self.failure_counts = defaultdict(int)
        self.circuit_states = {}
        
    async def call_with_circuit_breaker(
        self,
        server: str,
        tool: str,
        arguments: dict
    ):
        if self._is_circuit_open(server):
            raise Exception(f"Circuit open for {server}")
            
        try:
            result = await self.call_tool(server, tool, arguments)
            self._on_success(server)
            return result
            
        except Exception as e:
            self._on_failure(server)
            raise
```

### Load Balancing Pattern
```python
# Problem: Distribute load across multiple servers
class LoadBalancedMCP:
    def __init__(self):
        self.server_pools = defaultdict(list)
        self.current_index = defaultdict(int)
        
    def add_server_to_pool(self, pool: str, server: str):
        self.server_pools[pool].append(server)
        
    async def call_balanced(
        self,
        pool: str,
        tool: str,
        arguments: dict
    ):
        servers = self.server_pools[pool]
        if not servers:
            raise ValueError(f"No servers in pool: {pool}")
            
        # Round-robin selection
        index = self.current_index[pool] % len(servers)
        self.current_index[pool] += 1
        
        server = servers[index]
        return await self.call_tool(server, tool, arguments)
```

### Caching Pattern
```python
# Problem: Reduce redundant server calls
class CachedMCPClient:
    def __init__(self):
        self.cache = TTLCache(maxsize=1000, ttl=300)
        self.cache_stats = {"hits": 0, "misses": 0}
        
    async def call_with_cache(
        self,
        server: str,
        tool: str,
        arguments: dict,
        cache_key: Optional[str] = None
    ):
        # Generate cache key
        if not cache_key:
            cache_key = f"{server}:{tool}:{hash(json.dumps(arguments, sort_keys=True))}"
            
        # Check cache
        if cache_key in self.cache:
            self.cache_stats["hits"] += 1
            return self.cache[cache_key]
            
        # Cache miss
        self.cache_stats["misses"] += 1
        result = await self.call_tool(server, tool, arguments)
        
        # Store in cache
        self.cache[cache_key] = result
        return result
```

## 🚀 Client Deployment Patterns

### Microservice Client
```python
# FastAPI service with MCP client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
mcp_client = ProductionMCPClient({"client_name": "api-service"})

class ToolRequest(BaseModel):
    server: str
    tool: str
    arguments: dict

@app.post("/mcp/call")
async def call_mcp_tool(request: ToolRequest):
    try:
        result = await mcp_client.call_tool(
            request.server,
            request.tool,
            request.arguments
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Serverless Client
```python
# AWS Lambda with MCP
import json
from mcp_lambda_layer import MCPClient

client = MCPClient()

def lambda_handler(event, context):
    # Parse request
    body = json.loads(event['body'])
    
    # Call MCP tool
    result = client.call_tool_sync(
        body['server'],
        body['tool'],
        body['arguments']
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

## 🎯 Real-World Client Applications

### Customer Support Bot
```python
class SupportBot:
    def __init__(self):
        self.mcp = IntelligentMCPAgent(llm_client)
        
    async def setup(self):
        await self.mcp.initialize([
            {"name": "crm", "command": "mcp-server-salesforce"},
            {"name": "tickets", "command": "mcp-server-zendesk"},
            {"name": "kb", "command": "mcp-server-confluence"},
            {"name": "slack", "command": "mcp-server-slack"}
        ])
        
    async def handle_inquiry(self, customer_id: str, message: str):
        context = {
            "customer_id": customer_id,
            "timestamp": datetime.now().isoformat()
        }
        
        response = await self.mcp.process_request(message, context)
        return response
```

### Data Analysis Assistant
```python
class DataAnalyst:
    def __init__(self):
        self.mcp = StreamingMCPClient()
        
    async def analyze_dataset(self, dataset_id: str):
        # Run analysis pipeline
        async for progress in self.mcp.stream_tool(
            "analytics",
            "run_pipeline",
            {"dataset": dataset_id, "steps": ["clean", "analyze", "visualize"]}
        ):
            yield progress
```

### DevOps Automation
```python
class DevOpsAgent:
    def __init__(self):
        self.mcp = LoadBalancedMCP()
        
    async def deploy_application(self, app: str, version: str):
        # Multi-step deployment
        steps = [
            ("k8s", "update_deployment", {"app": app, "image": f"{app}:{version}"}),
            ("monitoring", "create_alert", {"app": app, "version": version}),
            ("ci", "run_tests", {"app": app, "type": "smoke"}),
            ("slack", "notify", {"channel": "#deployments", "message": f"Deployed {app} {version}"})
        ]
        
        for server, tool, args in steps:
            await self.mcp.call_balanced(server, tool, args)
```

## 📊 Performance Best Practices

1. **Connection Pooling**: Reuse client connections
2. **Async Everything**: Use async/await throughout
3. **Batch Operations**: Group related calls
4. **Client-Side Caching**: Cache frequent responses
5. **Timeout Management**: Set appropriate timeouts
6. **Error Recovery**: Implement retry logic
7. **Resource Cleanup**: Properly close connections

## 🐛 Common Issues & Solutions

### Issue: "Connection keeps dropping"
```python
# Solution: Implement reconnection logic
async def maintain_connection(self):
    while self.running:
        try:
            if not self.is_connected():
                await self.reconnect()
            await asyncio.sleep(30)
        except Exception as e:
            self.logger.error(f"Connection maintenance failed: {e}")
```

### Issue: "Tool calls are slow"
```python
# Solution: Implement parallel execution
async def parallel_tools(self, calls: List[dict]):
    tasks = [
        self.call_tool(c['server'], c['tool'], c['args'])
        for c in calls
    ]
    return await asyncio.gather(*tasks)
```

---

<div align="center">
  <p><strong>Connect your AI to the world with MCP clients!</strong></p>
  <p>Build intelligent agents that can use any tool</p>
</div>