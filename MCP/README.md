# 🔌 Model Context Protocol (MCP)

**The Universal Standard for AI-Tool Integration** - Enable any LLM to dynamically access tools, data sources, and services through a single protocol.

## 🎯 What is MCP?

Model Context Protocol (MCP) is an open standard that solves the M×N integration problem in AI systems. Instead of building custom integrations for every AI-tool combination, MCP provides one protocol that works everywhere.

### The Problem MCP Solves

```mermaid
graph LR
    subgraph "Without MCP (M×N Problem)"
        A1[Claude] --> B1[Database]
        A1 --> B2[APIs]
        A1 --> B3[Files]
        A2[GPT-4] --> B1
        A2 --> B2
        A2 --> B3
        A3[Gemini] --> B1
        A3 --> B2
        A3 --> B3
    end
    
    subgraph "With MCP (M+N Solution)"
        C1[Claude] --> D[MCP Protocol]
        C2[GPT-4] --> D
        C3[Gemini] --> D
        D --> E1[Database]
        D --> E2[APIs]
        D --> E3[Files]
    end
```

## 🚀 Quick Start

### Problem: "I need my LLM to access my PostgreSQL database"

#### Quick Solution
```python
# 1. Install MCP server
pip install mcp-server-postgres

# 2. Configure in Claude/LLM settings
{
  "mcpServers": {
    "postgres": {
      "command": "mcp-server-postgres",
      "env": {
        "POSTGRES_URL": "postgresql://user:pass@localhost/db"
      }
    }
  }
}
```

#### Production Solution
```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncpg

class PostgresMCPServer:
    """Production-ready MCP server for PostgreSQL"""
    
    def __init__(self, config):
        self.config = config
        self.pool = None
        self.server = Server("postgres-server")
        self._register_tools()
    
    async def start(self):
        """Initialize connection pool with retry logic"""
        try:
            self.pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                timeout=self.config.timeout,
                command_timeout=self.config.command_timeout
            )
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    def _register_tools(self):
        """Register available database operations"""
        
        @self.server.tool()
        async def query_database(sql: str, params: list = None):
            """Execute read-only SQL query"""
            if not sql.strip().upper().startswith('SELECT'):
                return TextContent(
                    text="Error: Only SELECT queries allowed",
                    mime_type="text/plain"
                )
            
            try:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch(sql, *(params or []))
                    return TextContent(
                        text=json.dumps([dict(r) for r in rows]),
                        mime_type="application/json"
                    )
            except Exception as e:
                logger.error(f"Query failed: {e}")
                return TextContent(
                    text=f"Error: {str(e)}",
                    mime_type="text/plain"
                )
```

## 📚 MCP Resources

### 🏆 Essential Resources

**[MCP Official Specification](https://modelcontextprotocol.io/docs)** - The definitive protocol specification
- Complete protocol definition
- Security model and best practices
- Implementation guidelines

**[Anthropic MCP SDK](https://github.com/anthropics/mcp)** - Official Python/TypeScript SDKs
- Production-ready implementations
- Extensive examples
- Active community support

**[MCP Servers List](https://github.com/modelcontextprotocol/servers)** - Community server collection
- 100+ ready-to-use servers
- Database, API, and tool integrations
- Copy and deploy immediately

### 📖 Guides & Tutorials

- **[Complete MCP Implementation Guide](./model-context-protocol.md)** - From basics to advanced patterns
- **[50+ MCP Server Examples](./mcp-server-examples.md)** - Production-ready implementations
- **[Building MCP Clients](./mcp-client-guide.md)** - Create AI agents with MCP

### 🛠️ Common MCP Patterns

#### 1. Database Access Pattern
```python
# Problem: Safe database access for LLMs
class SafeDatabaseMCP:
    def __init__(self):
        self.allowed_tables = ['users', 'products', 'orders']
        self.read_only = True
    
    async def query(self, table, conditions):
        if table not in self.allowed_tables:
            raise PermissionError(f"Access to {table} not allowed")
        # Safe query execution
```

#### 2. API Gateway Pattern
```python
# Problem: Rate-limited API access
class APIGatewayMCP:
    def __init__(self):
        self.rate_limiter = RateLimiter(calls=100, period=3600)
        self.cache = TTLCache(maxsize=1000, ttl=300)
    
    async def call_api(self, endpoint, params):
        # Check rate limits, cache, then call
```

#### 3. File System Pattern
```python
# Problem: Sandboxed file access
class FileSystemMCP:
    def __init__(self, allowed_paths):
        self.allowed_paths = [Path(p).resolve() for p in allowed_paths]
    
    async def read_file(self, path):
        # Validate path is within allowed directories
```

## 🎯 When to Use MCP

### Perfect For:
- ✅ Giving LLMs access to databases
- ✅ Integrating with external APIs
- ✅ File system operations
- ✅ Tool orchestration
- ✅ Multi-agent systems

### Not Ideal For:
- ❌ Simple one-off scripts
- ❌ Internal application logic
- ❌ Real-time streaming (use WebSockets)
- ❌ Binary protocol needs

## 🔧 MCP in Production

### Security Checklist
- [ ] Implement authentication (OAuth2, API keys)
- [ ] Set up rate limiting
- [ ] Enable audit logging
- [ ] Validate all inputs
- [ ] Use least-privilege access
- [ ] Monitor resource usage

### Performance Tips
```python
# Connection pooling
pool = await create_pool(min_size=5, max_size=20)

# Caching responses
@cache(ttl=300)
async def expensive_operation():
    pass

# Batch operations
async def batch_query(queries: list):
    async with pool.acquire() as conn:
        return await conn.fetch_many(queries)
```

## 🌟 Real-World Examples

### 1. Customer Support Bot
```python
# Problem: Bot needs access to support tickets and knowledge base
servers = {
    "support_db": "mcp-server-postgres",
    "knowledge": "mcp-server-elasticsearch",
    "slack": "mcp-server-slack"
}
```

### 2. Data Analysis Assistant
```python
# Problem: Analyst needs SQL + Python + visualization
servers = {
    "warehouse": "mcp-server-snowflake",
    "jupyter": "mcp-server-jupyter",
    "charts": "mcp-server-plotly"
}
```

### 3. DevOps Automation
```python
# Problem: Manage infrastructure through chat
servers = {
    "k8s": "mcp-server-kubernetes",
    "metrics": "mcp-server-prometheus",
    "logs": "mcp-server-elasticsearch"
}
```

## 📊 MCP Adoption

- **Claude Desktop**: Native MCP support
- **OpenAI GPTs**: Via custom actions
- **LangChain**: MCP tool adapter
- **Vertex AI**: Coming soon
- **Amazon Bedrock**: In development

## 🚀 Next Steps

1. **[Try the Quickstart](./quickstart.md)** - Get running in 5 minutes
2. **[Browse Server Examples](./mcp-server-examples.md)** - Find servers for your use case
3. **[Build Your Own Server](./model-context-protocol.md)** - Complete implementation guide
4. **[Join the Community](https://discord.gg/mcp)** - Get help and share your servers

---

<div align="center">
  <p><strong>MCP: Write Once, Work Everywhere</strong></p>
  <p>The future of AI integration is here</p>
</div>