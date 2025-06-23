# 🔌 Model Context Protocol - Complete Implementation Guide

**From Zero to Production MCP Server** - Build servers that give LLMs superpowers through dynamic tool access.

## 🎯 Common MCP Problems & Solutions

### Problem 1: "I need my LLM to query my database safely"

#### Quick Solution
```python
from mcp.server import Server
from mcp.types import TextContent

server = Server("db-server")

@server.tool()
async def query_db(sql: str):
    # Basic read-only query
    if not sql.upper().startswith("SELECT"):
        return "Only SELECT queries allowed"
    return await db.fetch(sql)
```

#### Production Solution
```python
import asyncpg
from mcp.server import Server
from mcp.types import Tool, TextContent
import json
import logging
from datetime import datetime

class DatabaseMCPServer:
    """Production database MCP server with safety features"""
    
    def __init__(self, config):
        self.config = config
        self.server = Server("database-server")
        self.pool = None
        self.query_log = []
        self.setup_tools()
        
    async def start(self):
        """Initialize with connection pooling and health checks"""
        self.pool = await asyncpg.create_pool(
            self.config.database_url,
            min_size=2,
            max_size=10,
            timeout=30,
            command_timeout=10,
            server_settings={
                'application_name': 'mcp_server',
                'jit': 'off'
            }
        )
        
        # Test connection
        async with self.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            
    def setup_tools(self):
        """Register safe database operations"""
        
        @self.server.tool()
        async def query_database(
            query: str,
            params: list = None,
            limit: int = 100
        ) -> TextContent:
            """Execute safe read-only queries with limits"""
            
            # Security checks
            query_upper = query.strip().upper()
            forbidden = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE']
            
            if any(word in query_upper for word in forbidden):
                return TextContent(
                    text="Error: Only SELECT queries allowed",
                    mime_type="text/plain"
                )
            
            # Add automatic limit if not present
            if 'LIMIT' not in query_upper:
                query = f"{query} LIMIT {limit}"
                
            try:
                async with self.pool.acquire() as conn:
                    # Set statement timeout
                    await conn.execute("SET statement_timeout = '5s'")
                    
                    # Execute query
                    rows = await conn.fetch(query, *(params or []))
                    
                    # Log for audit
                    self.query_log.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'query': query,
                        'rows_returned': len(rows)
                    })
                    
                    # Convert to JSON
                    result = [dict(row) for row in rows]
                    return TextContent(
                        text=json.dumps(result, indent=2, default=str),
                        mime_type="application/json"
                    )
                    
            except asyncpg.exceptions.QueryTimeoutError:
                return TextContent(
                    text="Error: Query timeout (5s limit)",
                    mime_type="text/plain"
                )
            except Exception as e:
                logging.error(f"Query error: {e}")
                return TextContent(
                    text=f"Error: {str(e)}",
                    mime_type="text/plain"
                )
                
        @self.server.tool()
        async def list_tables(schema: str = "public") -> TextContent:
            """List available tables in schema"""
            query = """
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = $1
                ORDER BY table_name
            """
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, schema)
                tables = [{"name": r["table_name"], "type": r["table_type"]} 
                         for r in rows]
                return TextContent(
                    text=json.dumps(tables, indent=2),
                    mime_type="application/json"
                )
```

### Problem 2: "I need to give LLM access to multiple APIs"

#### Quick Solution
```python
@server.tool()
async def call_api(endpoint: str, method: str = "GET", data: dict = None):
    async with httpx.AsyncClient() as client:
        response = await client.request(method, endpoint, json=data)
        return response.json()
```

#### Production Solution
```python
import httpx
from typing import Dict, Any, Optional
import asyncio
from functools import lru_cache
import hashlib

class APIGatewayMCP:
    """Multi-API gateway with caching and rate limiting"""
    
    def __init__(self, config):
        self.config = config
        self.server = Server("api-gateway")
        self.clients = {}
        self.rate_limiters = {}
        self.cache = TTLCache(maxsize=1000, ttl=300)  # 5 min cache
        self.setup_apis()
        
    def setup_apis(self):
        """Configure multiple API endpoints"""
        
        for api_name, api_config in self.config.apis.items():
            # Create dedicated client per API
            self.clients[api_name] = httpx.AsyncClient(
                base_url=api_config.base_url,
                headers=api_config.headers,
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=10)
            )
            
            # Setup rate limiter
            self.rate_limiters[api_name] = RateLimiter(
                calls=api_config.rate_limit,
                period=api_config.rate_period
            )
            
        @self.server.tool()
        async def call_api(
            api: str,
            endpoint: str,
            method: str = "GET",
            params: Dict[str, Any] = None,
            data: Dict[str, Any] = None,
            use_cache: bool = True
        ) -> TextContent:
            """Call external APIs with caching and rate limiting"""
            
            if api not in self.clients:
                return TextContent(
                    text=f"Unknown API: {api}",
                    mime_type="text/plain"
                )
                
            # Check rate limit
            limiter = self.rate_limiters[api]
            if not await limiter.allow():
                return TextContent(
                    text=f"Rate limit exceeded for {api}",
                    mime_type="text/plain"
                )
                
            # Cache key
            cache_key = self._get_cache_key(api, endpoint, method, params, data)
            
            # Check cache
            if use_cache and method == "GET" and cache_key in self.cache:
                return TextContent(
                    text=self.cache[cache_key],
                    mime_type="application/json"
                )
                
            try:
                # Make request
                client = self.clients[api]
                response = await client.request(
                    method,
                    endpoint,
                    params=params,
                    json=data
                )
                response.raise_for_status()
                
                result = response.text
                
                # Cache successful GET requests
                if method == "GET" and response.status_code == 200:
                    self.cache[cache_key] = result
                    
                return TextContent(
                    text=result,
                    mime_type=response.headers.get("content-type", "application/json")
                )
                
            except httpx.HTTPStatusError as e:
                return TextContent(
                    text=f"HTTP {e.response.status_code}: {e.response.text}",
                    mime_type="text/plain"
                )
            except Exception as e:
                logging.error(f"API call failed: {e}")
                return TextContent(
                    text=f"Error: {str(e)}",
                    mime_type="text/plain"
                )
```

### Problem 3: "I need sandboxed file system access"

#### Quick Solution
```python
@server.tool()
async def read_file(path: str):
    with open(path, 'r') as f:
        return f.read()
```

#### Production Solution
```python
from pathlib import Path
import aiofiles
import mimetypes

class FileSystemMCP:
    """Sandboxed file system access with security"""
    
    def __init__(self, allowed_dirs: list):
        self.server = Server("filesystem")
        self.allowed_dirs = [Path(d).resolve() for d in allowed_dirs]
        self.setup_tools()
        
    def _is_safe_path(self, path: str) -> bool:
        """Check if path is within allowed directories"""
        try:
            target = Path(path).resolve()
            return any(
                target.is_relative_to(allowed) 
                for allowed in self.allowed_dirs
            )
        except Exception:
            return False
            
    def setup_tools(self):
        @self.server.tool()
        async def read_file(
            path: str,
            encoding: str = "utf-8",
            max_size: int = 10_000_000  # 10MB limit
        ) -> TextContent:
            """Read file with size limits and sandboxing"""
            
            if not self._is_safe_path(path):
                return TextContent(
                    text="Error: Path outside allowed directories",
                    mime_type="text/plain"
                )
                
            file_path = Path(path)
            
            if not file_path.exists():
                return TextContent(
                    text="Error: File not found",
                    mime_type="text/plain"
                )
                
            if file_path.stat().st_size > max_size:
                return TextContent(
                    text=f"Error: File too large (>{max_size} bytes)",
                    mime_type="text/plain"
                )
                
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    content = await f.read()
                    
                mime_type, _ = mimetypes.guess_type(str(file_path))
                return TextContent(
                    text=content,
                    mime_type=mime_type or "text/plain"
                )
            except Exception as e:
                return TextContent(
                    text=f"Error reading file: {str(e)}",
                    mime_type="text/plain"
                )
                
        @self.server.tool()
        async def list_directory(
            path: str,
            pattern: str = "*",
            recursive: bool = False
        ) -> TextContent:
            """List directory contents with pattern matching"""
            
            if not self._is_safe_path(path):
                return TextContent(
                    text="Error: Path outside allowed directories",
                    mime_type="text/plain"
                )
                
            dir_path = Path(path)
            if not dir_path.is_dir():
                return TextContent(
                    text="Error: Not a directory",
                    mime_type="text/plain"
                )
                
            files = []
            glob_method = dir_path.rglob if recursive else dir_path.glob
            
            for item in glob_method(pattern):
                if item.is_file():
                    files.append({
                        "path": str(item),
                        "size": item.stat().st_size,
                        "modified": item.stat().st_mtime
                    })
                    
            return TextContent(
                text=json.dumps(files, indent=2),
                mime_type="application/json"
            )
```

## 📚 Essential MCP Resources

### 🏆 Core Documentation

**[MCP Official Specification](https://modelcontextprotocol.io/docs)** - The definitive source
- Protocol specification and architecture
- Security model and best practices  
- Reference implementations

**[MCP Python SDK](https://github.com/anthropics/mcp-python)** - Official Python implementation
- Async/await support with asyncio
- Type-safe with full typing support
- Production-ready with extensive tests

**[MCP TypeScript SDK](https://github.com/anthropics/mcp-typescript)** - Official TypeScript implementation  
- Modern TypeScript with full types
- Works in Node.js and browsers
- Ideal for web applications

### 📖 Learning Resources

**[Building Your First MCP Server](https://modelcontextprotocol.io/tutorials/first-server)** - Step-by-step tutorial
- Complete walkthrough from zero to working server
- Best practices baked in from the start
- Common pitfalls and how to avoid them

**[MCP Server Examples](https://github.com/modelcontextprotocol/servers)** - 100+ production examples
- Database servers (PostgreSQL, MySQL, SQLite)
- API integrations (GitHub, Slack, Discord)
- File systems, Docker, Kubernetes
- Copy, modify, and deploy

**[Awesome MCP](https://github.com/punkpeye/awesome-mcp)** - Community-curated list
- Servers, clients, and tools
- Articles and tutorials
- Integration examples

### 🛠️ Tools & Utilities

**[MCP Inspector](https://github.com/modelcontextprotocol/inspector)** - Debug and test MCP servers
- Interactive server testing
- Message inspection
- Performance profiling

**[MCP CLI](https://github.com/modelcontextprotocol/cli)** - Command-line tools
- Server scaffolding
- Client testing
- Protocol debugging

## 🔧 MCP Patterns Library

### Authentication Pattern
```python
# Problem: Secure MCP server access
class AuthenticatedMCP:
    def __init__(self):
        self.auth_tokens = {}
        
    async def authenticate(self, token: str) -> bool:
        """Validate auth token before tool access"""
        return token in self.auth_tokens
        
    def require_auth(self, func):
        """Decorator for authenticated tools"""
        async def wrapper(*args, **kwargs):
            token = kwargs.get('auth_token')
            if not await self.authenticate(token):
                return TextContent(
                    text="Unauthorized",
                    mime_type="text/plain"
                )
            return await func(*args, **kwargs)
        return wrapper
```

### Caching Pattern
```python
# Problem: Reduce redundant expensive operations
class CachedMCP:
    def __init__(self):
        self.cache = TTLCache(maxsize=100, ttl=300)
        
    def cached(self, ttl: int = 300):
        """Cache decorator with configurable TTL"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                key = f"{func.__name__}:{args}:{kwargs}"
                if key in self.cache:
                    return self.cache[key]
                result = await func(*args, **kwargs)
                self.cache[key] = result
                return result
            return wrapper
        return decorator
```

### Rate Limiting Pattern
```python
# Problem: Prevent API abuse
class RateLimitedMCP:
    def __init__(self):
        self.limiters = {}
        
    def rate_limit(self, calls: int, period: int):
        """Rate limit decorator"""
        def decorator(func):
            limiter = RateLimiter(calls, period)
            async def wrapper(*args, **kwargs):
                if not await limiter.allow():
                    return TextContent(
                        text="Rate limit exceeded",
                        mime_type="text/plain"
                    )
                return await func(*args, **kwargs)
            return wrapper
        return decorator
```

## 🚀 Deployment Configurations

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "mcp.server", "--config", "config.yml"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: your-registry/mcp-server:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## 🎯 Integration Examples

### Claude Desktop Integration
```json
{
  "mcpServers": {
    "myserver": {
      "command": "python",
      "args": ["-m", "myserver"],
      "env": {
        "API_KEY": "your-key"
      }
    }
  }
}
```

### LangChain Integration
```python
from langchain.tools import MCPTool
from langchain.agents import initialize_agent

# Wrap MCP server as LangChain tool
mcp_tool = MCPTool(
    server_url="http://localhost:8080",
    name="database",
    description="Query company database"
)

agent = initialize_agent(
    tools=[mcp_tool],
    llm=llm,
    agent="zero-shot-react-description"
)
```

## 🔒 Security Best Practices

1. **Authentication**: Always implement auth for production
2. **Input Validation**: Sanitize all inputs
3. **Rate Limiting**: Prevent abuse
4. **Audit Logging**: Track all operations
5. **Least Privilege**: Minimal permissions
6. **Network Security**: Use TLS, firewall rules
7. **Resource Limits**: CPU, memory, timeout controls

## 📊 Performance Guidelines

- Use connection pooling for databases
- Implement caching for expensive operations
- Set reasonable timeouts (5-30s)
- Monitor resource usage
- Use async/await throughout
- Batch operations when possible
- Profile and optimize hot paths

## 🐛 Common Issues & Solutions

### Issue: "Connection timeout"
```python
# Solution: Increase timeout and add retry
async def connect_with_retry(url, max_retries=3):
    for i in range(max_retries):
        try:
            return await asyncio.wait_for(
                connect(url), 
                timeout=30.0
            )
        except asyncio.TimeoutError:
            if i == max_retries - 1:
                raise
            await asyncio.sleep(2 ** i)
```

### Issue: "Memory leak in long-running server"
```python
# Solution: Implement connection recycling
async def recycle_connections():
    while True:
        await asyncio.sleep(3600)  # Every hour
        old_pool = self.pool
        self.pool = await create_pool(...)
        await old_pool.close()
```

## 🚀 Next Steps

1. **Start Simple**: Build a basic read-only server first
2. **Add Security**: Implement authentication and validation
3. **Scale Up**: Add caching, rate limiting, monitoring
4. **Deploy**: Use Docker/K8s for production
5. **Monitor**: Track usage, errors, performance
6. **Iterate**: Improve based on real usage

---

<div align="center">
  <p><strong>Ready to give your LLM superpowers?</strong></p>
  <p>Start building with MCP today!</p>
</div>