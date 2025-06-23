# 🔧 MCP Server Examples - Production Patterns

**50+ Ready-to-Deploy MCP Servers** - Copy, customize, and give your LLM instant capabilities.

## 🎯 Server Categories by Problem

### 📊 Data Access Problems

#### Problem: "LLM needs to query PostgreSQL database"
```python
# Quick: Basic PostgreSQL server
pip install mcp-server-postgres
# Configure in Claude Desktop and go!

# Production: Full-featured PostgreSQL MCP
class PostgresMCPServer:
    def __init__(self, config):
        self.pool = None
        self.query_cache = TTLCache(maxsize=100, ttl=300)
        self.audit_log = []
        
    async def query(self, sql: str, params: list = None):
        # Connection pooling
        async with self.pool.acquire() as conn:
            # Query caching for repeated queries
            cache_key = f"{sql}:{params}"
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
                
            # Execute with timeout
            result = await conn.fetch(sql, *params, timeout=30)
            self.query_cache[cache_key] = result
            
            # Audit logging
            self.audit_log.append({
                "timestamp": datetime.now(),
                "query": sql,
                "params": params,
                "rows": len(result)
            })
            
            return result
```

#### Problem: "LLM needs to analyze CSV/Excel files"
```python
# Quick: Simple CSV reader
@server.tool()
async def read_csv(file_path: str):
    import pandas as pd
    return pd.read_csv(file_path).to_json()

# Production: Data analysis server
class DataAnalysisMCP:
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.parquet', '.json']
        
    @server.tool()
    async def analyze_data(
        file_path: str,
        operations: List[str] = ["describe", "head"]
    ):
        # Load data with format detection
        df = await self._load_data(file_path)
        
        results = {}
        for op in operations:
            if op == "describe":
                results["statistics"] = df.describe().to_dict()
            elif op == "head":
                results["preview"] = df.head(10).to_dict()
            elif op == "missing":
                results["missing_values"] = df.isnull().sum().to_dict()
            elif op == "dtypes":
                results["data_types"] = df.dtypes.to_dict()
                
        return results
```

#### Problem: "LLM needs vector database search"
```python
# Quick: ChromaDB search
@server.tool()
async def vector_search(query: str, collection: str, k: int = 5):
    results = collection.query(query_texts=[query], n_results=k)
    return results

# Production: Multi-vector DB server
class VectorSearchMCP:
    def __init__(self, config):
        self.engines = {
            "chroma": ChromaClient(config.chroma),
            "pinecone": PineconeClient(config.pinecone),
            "weaviate": WeaviateClient(config.weaviate)
        }
        
    @server.tool()
    async def hybrid_search(
        query: str,
        engines: List[str] = ["chroma"],
        k: int = 10,
        rerank: bool = True
    ):
        # Parallel search across engines
        tasks = []
        for engine in engines:
            if engine in self.engines:
                tasks.append(
                    self.engines[engine].search(query, k=k*2)
                )
                
        results = await asyncio.gather(*tasks)
        
        # Merge and rerank results
        merged = self._merge_results(results)
        
        if rerank:
            merged = await self._rerank_results(query, merged)
            
        return merged[:k]
```

### 🌐 API Integration Problems

#### Problem: "LLM needs to manage GitHub repos"
```python
# Quick: Basic GitHub operations
@server.tool()
async def github_api(endpoint: str, method: str = "GET", data: dict = None):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method, 
            f"https://api.github.com{endpoint}",
            headers=headers,
            json=data
        )
        return response.json()

# Production: Full GitHub integration
class GitHubMCP:
    def __init__(self, token: str):
        self.client = Github(token)
        self.cache = {}
        
    @server.tool()
    async def manage_repo(
        action: str,
        repo: str,
        **kwargs
    ):
        """Create PRs, issues, manage branches, etc."""
        repository = self.client.get_repo(repo)
        
        if action == "create_pr":
            return repository.create_pull(
                title=kwargs["title"],
                body=kwargs["body"],
                base=kwargs.get("base", "main"),
                head=kwargs["head"]
            )
        elif action == "create_issue":
            return repository.create_issue(
                title=kwargs["title"],
                body=kwargs["body"],
                labels=kwargs.get("labels", [])
            )
        elif action == "list_workflows":
            return [w.name for w in repository.get_workflows()]
```

#### Problem: "LLM needs to send notifications"
```python
# Quick: Email notifications
@server.tool()
async def send_email(to: str, subject: str, body: str):
    # Using SMTP
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['To'] = to
    await aiosmtplib.send(msg)

# Production: Multi-channel notifications
class NotificationMCP:
    def __init__(self, config):
        self.channels = {
            "email": EmailChannel(config.smtp),
            "slack": SlackChannel(config.slack),
            "sms": TwilioChannel(config.twilio),
            "webhook": WebhookChannel()
        }
        
    @server.tool()
    async def notify(
        message: str,
        channels: List[str],
        priority: str = "normal",
        metadata: dict = None
    ):
        # Route to appropriate channels
        tasks = []
        for channel in channels:
            if channel in self.channels:
                tasks.append(
                    self.channels[channel].send(
                        message, priority, metadata
                    )
                )
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "sent": sum(1 for r in results if not isinstance(r, Exception)),
            "failed": sum(1 for r in results if isinstance(r, Exception)),
            "details": results
        }
```

### 🗂️ File System Problems

#### Problem: "LLM needs to process documents"
```python
# Quick: Read any document
@server.tool()
async def read_document(file_path: str):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        with open(file_path, 'r') as f:
            return f.read()

# Production: Document processing pipeline
class DocumentProcessorMCP:
    def __init__(self):
        self.processors = {
            '.pdf': PDFProcessor(),
            '.docx': DocxProcessor(),
            '.xlsx': ExcelProcessor(),
            '.pptx': PowerPointProcessor(),
            '.html': HTMLProcessor(),
            '.md': MarkdownProcessor()
        }
        
    @server.tool()
    async def process_document(
        file_path: str,
        operations: List[str] = ["extract_text"],
        output_format: str = "markdown"
    ):
        # Detect file type
        ext = Path(file_path).suffix.lower()
        processor = self.processors.get(ext)
        
        if not processor:
            return {"error": f"Unsupported file type: {ext}"}
            
        results = {}
        
        # Run requested operations
        if "extract_text" in operations:
            results["text"] = await processor.extract_text(file_path)
            
        if "extract_images" in operations:
            results["images"] = await processor.extract_images(file_path)
            
        if "extract_metadata" in operations:
            results["metadata"] = await processor.extract_metadata(file_path)
            
        if "extract_tables" in operations:
            results["tables"] = await processor.extract_tables(file_path)
            
        # Convert to requested format
        if output_format == "markdown":
            results = self._to_markdown(results)
        elif output_format == "json":
            results = json.dumps(results, indent=2)
            
        return results
```

### 🔧 DevOps Problems

#### Problem: "LLM needs to manage Kubernetes"
```python
# Quick: Basic K8s operations
@server.tool()
async def kubectl(command: str):
    result = subprocess.run(
        f"kubectl {command}",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout

# Production: Kubernetes management
class KubernetesMCP:
    def __init__(self):
        config.load_incluster_config()  # or load_kube_config()
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        
    @server.tool()
    async def manage_deployment(
        action: str,
        namespace: str,
        name: str,
        **kwargs
    ):
        if action == "scale":
            return self.apps_v1.patch_namespaced_deployment_scale(
                name=name,
                namespace=namespace,
                body={"spec": {"replicas": kwargs["replicas"]}}
            )
            
        elif action == "update_image":
            deployment = self.apps_v1.read_namespaced_deployment(
                name=name,
                namespace=namespace
            )
            deployment.spec.template.spec.containers[0].image = kwargs["image"]
            return self.apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=deployment
            )
            
        elif action == "get_logs":
            pods = self.v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"app={name}"
            )
            logs = {}
            for pod in pods.items:
                logs[pod.metadata.name] = self.v1.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=namespace,
                    tail_lines=kwargs.get("lines", 100)
                )
            return logs
```

#### Problem: "LLM needs to monitor systems"
```python
# Quick: System metrics
@server.tool()
async def get_system_metrics():
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory()._asdict(),
        "disk": psutil.disk_usage('/')._asdict(),
        "network": psutil.net_io_counters()._asdict()
    }

# Production: Comprehensive monitoring
class MonitoringMCP:
    def __init__(self, config):
        self.prometheus = PrometheusClient(config.prometheus_url)
        self.grafana = GrafanaClient(config.grafana_url)
        self.alerts = AlertManager(config.alertmanager_url)
        
    @server.tool()
    async def query_metrics(
        query: str,
        time_range: str = "1h",
        step: str = "1m"
    ):
        """Execute PromQL queries"""
        result = await self.prometheus.query_range(
            query=query,
            start=f"now-{time_range}",
            end="now",
            step=step
        )
        
        # Format for LLM consumption
        formatted = {
            "query": query,
            "time_range": time_range,
            "data_points": len(result["data"]["result"][0]["values"]),
            "series": []
        }
        
        for series in result["data"]["result"]:
            formatted["series"].append({
                "labels": series["metric"],
                "values": series["values"]
            })
            
        return formatted
        
    @server.tool()
    async def check_alerts(
        severity: str = None,
        service: str = None
    ):
        """Get active alerts"""
        alerts = await self.alerts.get_alerts()
        
        # Filter by criteria
        if severity:
            alerts = [a for a in alerts if a["labels"].get("severity") == severity]
        if service:
            alerts = [a for a in alerts if a["labels"].get("service") == service]
            
        return {
            "active_alerts": len(alerts),
            "alerts": alerts
        }
```

## 📚 Essential MCP Server Resources

### 🏆 Official Servers

**[MCP Servers Repository](https://github.com/modelcontextprotocol/servers)** - Official collection
- Database servers: PostgreSQL, MySQL, SQLite, MongoDB
- File systems: Local FS, S3, Google Drive
- Dev tools: Git, Docker, NPM
- All production-tested and maintained

**[Anthropic's Reference Servers](https://github.com/anthropics/mcp-servers)** - Reference implementations
- Best practices demonstrated
- Clean code patterns
- Comprehensive testing

### 📖 Server Development Guides

**[Building Robust MCP Servers](https://modelcontextprotocol.io/docs/guides/robust-servers)** - Production guide
- Error handling strategies
- Connection management
- Resource lifecycle
- Graceful shutdowns

**[MCP Server Security](https://modelcontextprotocol.io/docs/security)** - Security best practices
- Authentication patterns
- Input validation
- Rate limiting
- Audit logging

### 🛠️ Testing & Debugging

**[MCP Test Suite](https://github.com/modelcontextprotocol/test-suite)** - Comprehensive testing
- Unit test examples
- Integration testing
- Load testing scripts
- CI/CD integration

## 🔧 Advanced Server Patterns

### Multi-tenant Server
```python
class MultiTenantMCP:
    """Isolated resources per tenant"""
    
    def __init__(self):
        self.tenants = {}
        
    async def get_tenant_context(self, tenant_id: str):
        if tenant_id not in self.tenants:
            self.tenants[tenant_id] = {
                "db_pool": await self._create_tenant_pool(tenant_id),
                "cache": TTLCache(maxsize=50, ttl=300),
                "rate_limiter": RateLimiter(100, 3600)
            }
        return self.tenants[tenant_id]
```

### Plugin Architecture
```python
class PluginMCP:
    """Extensible server with plugins"""
    
    def __init__(self):
        self.plugins = {}
        self._load_plugins()
        
    def _load_plugins(self):
        plugin_dir = Path("plugins")
        for plugin_path in plugin_dir.glob("*.py"):
            spec = importlib.util.spec_from_file_location(
                plugin_path.stem, plugin_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, "MCPPlugin"):
                plugin = module.MCPPlugin()
                self.plugins[plugin.name] = plugin
                self._register_plugin_tools(plugin)
```

### Event-Driven Server
```python
class EventDrivenMCP:
    """Async event processing"""
    
    def __init__(self):
        self.event_queue = asyncio.Queue()
        self.handlers = defaultdict(list)
        
    async def emit(self, event: str, data: dict):
        await self.event_queue.put({"event": event, "data": data})
        
    async def process_events(self):
        while True:
            event = await self.event_queue.get()
            for handler in self.handlers[event["event"]]:
                asyncio.create_task(handler(event["data"]))
```

## 🚀 Deployment Templates

### AWS Lambda Deployment
```python
# lambda_function.py
import json
from mcp_lambda_adapter import MCPLambdaAdapter
from your_server import YourMCPServer

adapter = MCPLambdaAdapter(YourMCPServer())

def lambda_handler(event, context):
    return adapter.handle(event, context)
```

### Docker Compose Stack
```yaml
version: '3.8'
services:
  mcp-server:
    build: .
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    ports:
      - "8080:8080"
      
  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=pass
      - POSTGRES_USER=user
      - POSTGRES_DB=mydb
      
  redis:
    image: redis:7-alpine
```

## 📊 Performance Optimization

### Connection Pooling
```python
class OptimizedMCP:
    async def setup_pools(self):
        # Database connection pool
        self.db_pool = await asyncpg.create_pool(
            dsn=self.config.database_url,
            min_size=5,
            max_size=20,
            max_inactive_connection_lifetime=300
        )
        
        # HTTP client pool
        self.http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20
            )
        )
```

### Caching Strategy
```python
class CachedMCP:
    def __init__(self):
        # Multi-level cache
        self.memory_cache = TTLCache(maxsize=1000, ttl=60)
        self.redis_cache = aioredis.from_url("redis://localhost")
        
    async def get_cached(self, key: str):
        # L1: Memory cache
        if key in self.memory_cache:
            return self.memory_cache[key]
            
        # L2: Redis cache
        value = await self.redis_cache.get(key)
        if value:
            self.memory_cache[key] = value
            return value
            
        return None
```

## 🎯 Real-World Production Examples

### E-commerce Assistant
```python
servers = {
    "inventory": "mcp-server-postgres",
    "orders": "mcp-server-postgres", 
    "shipping": "mcp-server-fedex",
    "payments": "mcp-server-stripe",
    "support": "mcp-server-zendesk",
    "analytics": "mcp-server-bigquery"
}
```

### DevOps Automation
```python
servers = {
    "k8s": "mcp-server-kubernetes",
    "ci": "mcp-server-jenkins",
    "monitoring": "mcp-server-prometheus",
    "logs": "mcp-server-elasticsearch",
    "incidents": "mcp-server-pagerduty"
}
```

### Data Science Platform
```python
servers = {
    "notebooks": "mcp-server-jupyter",
    "data": "mcp-server-s3",
    "compute": "mcp-server-databricks",
    "models": "mcp-server-mlflow",
    "viz": "mcp-server-plotly"
}
```

---

<div align="center">
  <p><strong>Pick a server. Customize it. Deploy it. Done!</strong></p>
  <p>Your LLM is now connected to the world</p>
</div>