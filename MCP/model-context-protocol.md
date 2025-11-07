<div align="center">

# 🔌 Model Context Protocol - Complete Implementation Guide

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=26&duration=3000&pause=1000&color=3B82F6&center=true&vCenter=true&width=900&lines=From+Zero+to+Production+MCP+Server;Build+Servers+That+Give+LLMs+Superpowers;2024-2025+Best+Practices;Enterprise-Grade+AI+Integration" alt="Typing SVG" />

![Guide](https://img.shields.io/badge/Guide-Complete-success?style=for-the-badge&logo=bookstack)
![Level](https://img.shields.io/badge/Level-Beginner_to_Expert-blue?style=for-the-badge&logo=levelsdotfyi)
![Updated](https://img.shields.io/badge/Updated-January_2025-orange?style=for-the-badge&logo=calendar)
![Code](https://img.shields.io/badge/Code-Production_Ready-purple?style=for-the-badge&logo=python)

**From Zero to Production** - Build servers that give LLMs superpowers through dynamic tool access.

[🎯 Quick Solutions](#-common-mcp-problems--solutions) • [🏗️ Architecture](#-mcp-architecture) • [🔒 Security](#-security-best-practices) • [🚀 Deployment](#-deployment-configurations) • [📊 Monitoring](#-monitoring--observability)

</div>

---

## 📋 Table of Contents

- [🎯 Common MCP Problems & Solutions](#-common-mcp-problems--solutions)
  - [Problem 1: Database Access](#problem-1-i-need-my-llm-to-query-my-database-safely)
  - [Problem 2: API Integration](#problem-2-i-need-to-give-llm-access-to-multiple-apis)
  - [Problem 3: File System Access](#problem-3-i-need-sandboxed-file-system-access)
- [🏗️ MCP Architecture](#-mcp-architecture)
- [📚 Essential MCP Resources](#-essential-mcp-resources)
- [🔧 MCP Patterns Library](#-mcp-patterns-library)
- [🚀 Deployment Configurations](#-deployment-configurations)
- [🎯 Integration Examples](#-integration-examples)
- [🔒 Security Best Practices](#-security-best-practices)
- [📊 Performance Guidelines](#-performance-guidelines)
- [🐛 Common Issues & Solutions](#-common-issues--solutions)

---

## 🎯 Common MCP Problems & Solutions

### Problem 1: "I need my LLM to query my database safely"

<div align="center">

```mermaid
sequenceDiagram
    participant LLM
    participant MCP as MCP Server
    participant Validator
    participant DB as Database

    LLM->>MCP: SQL Query Request
    MCP->>Validator: Validate Query
    alt Query Valid
        Validator->>MCP: ✅ Approved
        MCP->>DB: Execute (with timeout)
        DB-->>MCP: Results
        MCP->>MCP: Apply Row Limit
        MCP->>MCP: Log Audit Trail
        MCP-->>LLM: Return Data
    else Query Invalid
        Validator->>MCP: ❌ Rejected
        MCP-->>LLM: Error Response
    end

    Note over LLM,DB: Security at every layer
```

</div>

#### ⚡ Quick Solution (5 Minutes)

```python
from mcp.server import Server
from mcp.types import TextContent

server = Server("db-server")

@server.tool()
async def query_db(sql: str):
    """Basic read-only query"""
    # Security: Only SELECT queries
    if not sql.upper().startswith("SELECT"):
        return "❌ Only SELECT queries allowed"

    # Execute query
    return await db.fetch(sql)
```

**Limitations:**
- ⚠️ No connection pooling
- ⚠️ No query timeout
- ⚠️ No audit logging
- ⚠️ No rate limiting

#### 🏢 Production Solution

```python
import asyncpg
from mcp.server import Server
from mcp.types import Tool, TextContent
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

class DatabaseMCPServer:
    """Production database MCP server with comprehensive safety features"""

    def __init__(self, config):
        self.config = config
        self.server = Server("database-server")
        self.pool = None
        self.query_log = []
        self.query_count = 0
        self.setup_tools()

    async def start(self):
        """Initialize with connection pooling and health checks"""
        try:
            self.pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=2,
                max_size=10,
                timeout=30,
                command_timeout=10,
                server_settings={
                    'application_name': 'mcp_server',
                    'jit': 'off'  # Disable JIT for predictable performance
                }
            )

            # Test connection
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logging.info(f"✅ Connected to PostgreSQL: {version}")

        except Exception as e:
            logging.error(f"❌ Failed to connect to database: {e}")
            raise

    def setup_tools(self):
        """Register safe database operations"""

        @self.server.tool()
        async def query_database(
            query: str,
            params: Optional[List[Any]] = None,
            limit: int = 100
        ) -> TextContent:
            """
            Execute safe read-only queries with automatic limits

            Args:
                query: SQL query (SELECT only)
                params: Query parameters for safe substitution
                limit: Maximum rows to return (default: 100, max: 1000)

            Returns:
                JSON formatted results with metadata
            """

            # Security checks
            query_upper = query.strip().upper()
            forbidden = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']

            if any(word in query_upper for word in forbidden):
                return TextContent(
                    text="❌ Error: Only SELECT queries allowed",
                    mime_type="text/plain"
                )

            # Enforce row limit
            if 'LIMIT' not in query_upper:
                limit = min(limit, 1000)  # Hard cap at 1000
                query = f"{query.rstrip(';')} LIMIT {limit}"

            try:
                async with self.pool.acquire() as conn:
                    # Set statement timeout (prevents long-running queries)
                    await conn.execute("SET statement_timeout = '5s'")

                    # Execute query with parameters
                    start_time = datetime.utcnow()
                    rows = await conn.fetch(query, *(params or []))
                    execution_time = (datetime.utcnow() - start_time).total_seconds()

                    # Increment counter
                    self.query_count += 1

                    # Audit logging
                    self.query_log.append({
                        'timestamp': start_time.isoformat(),
                        'query': query,
                        'params': params,
                        'rows_returned': len(rows),
                        'execution_time_seconds': execution_time,
                        'query_number': self.query_count
                    })

                    # Convert to JSON
                    result = {
                        'data': [dict(row) for row in rows],
                        'metadata': {
                            'row_count': len(rows),
                            'execution_time': f"{execution_time:.3f}s",
                            'query_id': self.query_count
                        }
                    }

                    return TextContent(
                        text=json.dumps(result, indent=2, default=str),
                        mime_type="application/json"
                    )

            except asyncpg.exceptions.QueryTimeoutError:
                logging.warning("Query timeout exceeded")
                return TextContent(
                    text="❌ Error: Query timeout (5s limit exceeded)",
                    mime_type="text/plain"
                )
            except asyncpg.exceptions.PostgresSyntaxError as e:
                logging.error(f"SQL syntax error: {e}")
                return TextContent(
                    text=f"❌ SQL Syntax Error: {str(e)}",
                    mime_type="text/plain"
                )
            except Exception as e:
                logging.error(f"Query execution error: {e}")
                return TextContent(
                    text=f"❌ Error: {str(e)}",
                    mime_type="text/plain"
                )

        @self.server.tool()
        async def list_tables(schema: str = "public") -> TextContent:
            """
            List available tables in the specified schema

            Args:
                schema: Database schema name (default: public)

            Returns:
                JSON list of tables with metadata
            """
            query = """
                SELECT
                    table_name,
                    table_type,
                    (SELECT COUNT(*) FROM information_schema.columns
                     WHERE table_schema = t.table_schema
                     AND table_name = t.table_name) as column_count
                FROM information_schema.tables t
                WHERE table_schema = $1
                ORDER BY table_name
            """

            try:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch(query, schema)

                    tables = [
                        {
                            "name": r["table_name"],
                            "type": r["table_type"],
                            "columns": r["column_count"]
                        }
                        for r in rows
                    ]

                    return TextContent(
                        text=json.dumps({
                            'schema': schema,
                            'table_count': len(tables),
                            'tables': tables
                        }, indent=2),
                        mime_type="application/json"
                    )
            except Exception as e:
                logging.error(f"Failed to list tables: {e}")
                return TextContent(
                    text=f"❌ Error: {str(e)}",
                    mime_type="text/plain"
                )

        @self.server.tool()
        async def describe_table(
            table_name: str,
            schema: str = "public"
        ) -> TextContent:
            """
            Get detailed information about a table's structure

            Args:
                table_name: Name of the table
                schema: Database schema (default: public)

            Returns:
                JSON with columns, types, and constraints
            """
            query = """
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = $2
                ORDER BY ordinal_position
            """

            try:
                async with self.pool.acquire() as conn:
                    rows = await conn.fetch(query, schema, table_name)

                    if not rows:
                        return TextContent(
                            text=f"❌ Table '{schema}.{table_name}' not found",
                            mime_type="text/plain"
                        )

                    columns = [
                        {
                            "name": r["column_name"],
                            "type": r["data_type"],
                            "nullable": r["is_nullable"] == "YES",
                            "default": r["column_default"],
                            "max_length": r["character_maximum_length"]
                        }
                        for r in rows
                    ]

                    return TextContent(
                        text=json.dumps({
                            'table': f"{schema}.{table_name}",
                            'column_count': len(columns),
                            'columns': columns
                        }, indent=2, default=str),
                        mime_type="application/json"
                    )
            except Exception as e:
                logging.error(f"Failed to describe table: {e}")
                return TextContent(
                    text=f"❌ Error: {str(e)}",
                    mime_type="text/plain"
                )
```

**Production Features:**
- ✅ Connection pooling with health checks
- ✅ Query timeouts (5 second limit)
- ✅ Comprehensive audit logging
- ✅ Row limits (configurable, max 1000)
- ✅ Parameter sanitization
- ✅ Error handling and recovery
- ✅ Performance metrics
- ✅ Schema exploration tools

---

### Problem 2: "I need to give LLM access to multiple APIs"

<div align="center">

```mermaid
graph TB
    subgraph "API Gateway MCP"
        LLM[🤖 LLM Request]
        Gateway[MCP Gateway]

        subgraph "Cache Layer"
            Cache[Redis Cache]
        end

        subgraph "Rate Limiters"
            RL1[GitHub Limiter]
            RL2[Slack Limiter]
            RL3[AWS Limiter]
        end

        subgraph "API Clients"
            GH[GitHub Client]
            SL[Slack Client]
            AWS[AWS Client]
        end
    end

    LLM --> Gateway
    Gateway --> Cache
    Gateway --> RL1 --> GH
    Gateway --> RL2 --> SL
    Gateway --> RL3 --> AWS

    style Gateway fill:#a855f7,stroke:#7e22ce,color:#fff
    style Cache fill:#10b981,stroke:#059669,color:#fff
```

</div>

#### ⚡ Quick Solution

```python
@server.tool()
async def call_api(endpoint: str, method: str = "GET", data: dict = None):
    """Simple API call"""
    async with httpx.AsyncClient() as client:
        response = await client.request(method, endpoint, json=data)
        return response.json()
```

#### 🏢 Production Solution

```python
import httpx
from typing import Dict, Any, Optional
import asyncio
from functools import lru_cache
import hashlib
from cachetools import TTLCache
import time

class APIGatewayMCP:
    """Multi-API gateway with caching, rate limiting, and resilience"""

    def __init__(self, config):
        self.config = config
        self.server = Server("api-gateway")
        self.clients = {}
        self.rate_limiters = {}
        self.cache = TTLCache(maxsize=1000, ttl=300)  # 5 min cache
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'rate_limit_hits': 0
        }
        self.setup_apis()

    def setup_apis(self):
        """Configure multiple API endpoints with dedicated clients"""

        for api_name, api_config in self.config.apis.items():
            # Create dedicated client per API
            self.clients[api_name] = httpx.AsyncClient(
                base_url=api_config.base_url,
                headers=api_config.headers,
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5
                )
            )

            # Setup rate limiter per API
            self.rate_limiters[api_name] = TokenBucketRateLimiter(
                calls=api_config.rate_limit,
                period=api_config.rate_period
            )

        @self.server.tool()
        async def call_api(
            api: str,
            endpoint: str,
            method: str = "GET",
            params: Optional[Dict[str, Any]] = None,
            data: Optional[Dict[str, Any]] = None,
            use_cache: bool = True
        ) -> TextContent:
            """
            Call external APIs with caching and rate limiting

            Args:
                api: API name (github, slack, aws, etc.)
                endpoint: API endpoint path
                method: HTTP method (GET, POST, PUT, DELETE)
                params: Query parameters
                data: Request body (for POST/PUT)
                use_cache: Whether to use cache for GET requests

            Returns:
                API response with metadata
            """

            self.stats['total_requests'] += 1

            # Validate API exists
            if api not in self.clients:
                return TextContent(
                    text=json.dumps({
                        'error': f'Unknown API: {api}',
                        'available_apis': list(self.clients.keys())
                    }),
                    mime_type="application/json"
                )

            # Check rate limit
            limiter = self.rate_limiters[api]
            if not await limiter.acquire():
                self.stats['rate_limit_hits'] += 1
                return TextContent(
                    text=json.dumps({
                        'error': f'Rate limit exceeded for {api}',
                        'retry_after': limiter.retry_after()
                    }),
                    mime_type="application/json"
                )

            # Generate cache key
            cache_key = self._get_cache_key(api, endpoint, method, params, data)

            # Check cache for GET requests
            if use_cache and method == "GET":
                if cache_key in self.cache:
                    self.stats['cache_hits'] += 1
                    cached_result = self.cache[cache_key]
                    return TextContent(
                        text=json.dumps({
                            **cached_result,
                            'from_cache': True
                        }),
                        mime_type="application/json"
                    )
                self.stats['cache_misses'] += 1

            try:
                # Make API request
                client = self.clients[api]
                start_time = time.time()

                response = await client.request(
                    method,
                    endpoint,
                    params=params,
                    json=data
                )

                execution_time = time.time() - start_time
                response.raise_for_status()

                # Parse response
                try:
                    result_data = response.json()
                except:
                    result_data = response.text

                result = {
                    'status': response.status_code,
                    'data': result_data,
                    'metadata': {
                        'api': api,
                        'endpoint': endpoint,
                        'method': method,
                        'execution_time': f"{execution_time:.3f}s",
                        'from_cache': False
                    }
                }

                # Cache successful GET requests
                if method == "GET" and response.status_code == 200:
                    self.cache[cache_key] = result

                return TextContent(
                    text=json.dumps(result, indent=2),
                    mime_type="application/json"
                )

            except httpx.HTTPStatusError as e:
                return TextContent(
                    text=json.dumps({
                        'error': 'HTTP Error',
                        'status_code': e.response.status_code,
                        'message': e.response.text,
                        'api': api,
                        'endpoint': endpoint
                    }),
                    mime_type="application/json"
                )
            except httpx.TimeoutException:
                return TextContent(
                    text=json.dumps({
                        'error': 'Request Timeout',
                        'api': api,
                        'endpoint': endpoint
                    }),
                    mime_type="application/json"
                )
            except Exception as e:
                logging.error(f"API call failed: {api}.{endpoint} - {e}")
                return TextContent(
                    text=json.dumps({
                        'error': str(e),
                        'api': api,
                        'endpoint': endpoint
                    }),
                    mime_type="application/json"
                )

        @self.server.tool()
        async def get_api_stats() -> TextContent:
            """Get API gateway statistics"""
            return TextContent(
                text=json.dumps({
                    'statistics': self.stats,
                    'cache_hit_rate': f"{(self.stats['cache_hits'] / max(self.stats['total_requests'], 1) * 100):.1f}%",
                    'active_apis': list(self.clients.keys())
                }, indent=2),
                mime_type="application/json"
            )

    def _get_cache_key(self, api, endpoint, method, params, data) -> str:
        """Generate cache key from request parameters"""
        key_data = f"{api}:{method}:{endpoint}:{params}:{data}"
        return hashlib.sha256(key_data.encode()).hexdigest()

class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation"""

    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.tokens = calls
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Attempt to acquire a token"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens based on elapsed time
            self.tokens = min(
                self.calls,
                self.tokens + (elapsed / self.period * self.calls)
            )
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def retry_after(self) -> float:
        """Calculate seconds until next token available"""
        return max(0, (1 - self.tokens) * self.period / self.calls)
```

**Production Features:**
- ✅ Multi-API support with dedicated clients
- ✅ Intelligent caching (TTL-based)
- ✅ Per-API rate limiting (token bucket)
- ✅ Connection pooling and keep-alive
- ✅ Comprehensive error handling
- ✅ Request/response metrics
- ✅ Cache hit rate tracking
- ✅ Timeout management

---

### Problem 3: "I need sandboxed file system access"

<div align="center">

```mermaid
graph LR
    subgraph "File System MCP"
        Request[📄 File Request]
        Validator[Path Validator]
        Scanner[Security Scanner]
        FS[File System]
    end

    Request --> Validator
    Validator -->|Valid Path| Scanner
    Validator -->|Invalid| Reject[❌ Reject]
    Scanner -->|Safe| FS
    Scanner -->|Malicious| Reject

    style Validator fill:#a855f7,stroke:#7e22ce,color:#fff
    style Scanner fill:#10b981,stroke:#059669,color:#fff
    style Reject fill:#ef4444,stroke:#dc2626,color:#fff
```

</div>

#### ⚡ Quick Solution

```python
@server.tool()
async def read_file(path: str):
    """Simple file reader"""
    with open(path, 'r') as f:
        return f.read()
```

#### 🏢 Production Solution

```python
from pathlib import Path
import aiofiles
import mimetypes
import hashlib
from typing import List, Optional

class FileSystemMCP:
    """Sandboxed file system access with comprehensive security"""

    def __init__(self, allowed_dirs: List[str], config: dict = None):
        self.server = Server("filesystem")
        self.allowed_dirs = [Path(d).resolve() for d in allowed_dirs]
        self.max_file_size = config.get('max_file_size', 10_000_000)  # 10MB
        self.blocked_extensions = config.get('blocked_extensions', ['.exe', '.dll', '.so'])
        self.file_access_log = []
        self.setup_tools()

    def _is_safe_path(self, path: str) -> tuple[bool, str]:
        """
        Comprehensive path validation

        Returns:
            (is_safe: bool, error_message: str)
        """
        try:
            target = Path(path).resolve()

            # Check if path is within allowed directories
            if not any(target.is_relative_to(allowed) for allowed in self.allowed_dirs):
                return False, f"Path outside allowed directories: {path}"

            # Check for path traversal attempts
            if '..' in str(path) or '~' in str(path):
                return False, "Path traversal attempt detected"

            # Check file extension
            if target.suffix.lower() in self.blocked_extensions:
                return False, f"File extension blocked: {target.suffix}"

            return True, ""

        except Exception as e:
            return False, f"Path validation error: {str(e)}"

    def setup_tools(self):
        @self.server.tool()
        async def read_file(
            path: str,
            encoding: str = "utf-8",
            max_size: Optional[int] = None
        ) -> TextContent:
            """
            Read file with size limits and sandboxing

            Args:
                path: File path (must be within allowed directories)
                encoding: File encoding (default: utf-8)
                max_size: Maximum file size in bytes (default: 10MB)

            Returns:
                File contents with metadata
            """

            # Validate path
            is_safe, error_msg = self._is_safe_path(path)
            if not is_safe:
                return TextContent(
                    text=json.dumps({'error': error_msg}),
                    mime_type="application/json"
                )

            file_path = Path(path).resolve()

            # Check file exists
            if not file_path.exists():
                return TextContent(
                    text=json.dumps({'error': 'File not found', 'path': str(file_path)}),
                    mime_type="application/json"
                )

            if not file_path.is_file():
                return TextContent(
                    text=json.dumps({'error': 'Not a file', 'path': str(file_path)}),
                    mime_type="application/json"
                )

            # Check file size
            file_size = file_path.stat().st_size
            size_limit = max_size or self.max_file_size

            if file_size > size_limit:
                return TextContent(
                    text=json.dumps({
                        'error': f'File too large',
                        'size': file_size,
                        'limit': size_limit
                    }),
                    mime_type="application/json"
                )

            try:
                # Read file asynchronously
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    content = await f.read()

                # Get file metadata
                stat = file_path.stat()
                mime_type, _ = mimetypes.guess_type(str(file_path))

                # Calculate file hash
                file_hash = hashlib.sha256(content.encode()).hexdigest()

                # Log access
                self.file_access_log.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'operation': 'read',
                    'path': str(file_path),
                    'size': file_size,
                    'hash': file_hash
                })

                result = {
                    'content': content,
                    'metadata': {
                        'path': str(file_path),
                        'size': file_size,
                        'mime_type': mime_type or 'text/plain',
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'hash': file_hash[:16]  # Short hash for verification
                    }
                }

                return TextContent(
                    text=json.dumps(result, indent=2),
                    mime_type="application/json"
                )

            except UnicodeDecodeError:
                return TextContent(
                    text=json.dumps({
                        'error': 'File encoding error',
                        'path': str(file_path),
                        'hint': 'Try a different encoding or read as binary'
                    }),
                    mime_type="application/json"
                )
            except Exception as e:
                logging.error(f"Error reading file: {e}")
                return TextContent(
                    text=json.dumps({'error': str(e), 'path': str(file_path)}),
                    mime_type="application/json"
                )

        @self.server.tool()
        async def list_directory(
            path: str,
            pattern: str = "*",
            recursive: bool = False,
            include_hidden: bool = False
        ) -> TextContent:
            """
            List directory contents with pattern matching

            Args:
                path: Directory path
                pattern: Glob pattern (default: *)
                recursive: Search subdirectories
                include_hidden: Include hidden files

            Returns:
                List of files with metadata
            """

            # Validate path
            is_safe, error_msg = self._is_safe_path(path)
            if not is_safe:
                return TextContent(
                    text=json.dumps({'error': error_msg}),
                    mime_type="application/json"
                )

            dir_path = Path(path).resolve()

            if not dir_path.is_dir():
                return TextContent(
                    text=json.dumps({'error': 'Not a directory', 'path': str(dir_path)}),
                    mime_type="application/json"
                )

            try:
                files = []
                glob_method = dir_path.rglob if recursive else dir_path.glob

                for item in glob_method(pattern):
                    # Skip hidden files if requested
                    if not include_hidden and item.name.startswith('.'):
                        continue

                    stat = item.stat()

                    files.append({
                        'name': item.name,
                        'path': str(item),
                        'type': 'directory' if item.is_dir() else 'file',
                        'size': stat.st_size if item.is_file() else None,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })

                # Sort by name
                files.sort(key=lambda x: x['name'])

                return TextContent(
                    text=json.dumps({
                        'directory': str(dir_path),
                        'count': len(files),
                        'files': files
                    }, indent=2),
                    mime_type="application/json"
                )

            except Exception as e:
                logging.error(f"Error listing directory: {e}")
                return TextContent(
                    text=json.dumps({'error': str(e), 'path': str(dir_path)}),
                    mime_type="application/json"
                )

        @self.server.tool()
        async def search_files(
            directory: str,
            content_pattern: str,
            file_pattern: str = "*.txt",
            max_results: int = 100
        ) -> TextContent:
            """
            Search file contents for pattern

            Args:
                directory: Directory to search
                content_pattern: Text pattern to find
                file_pattern: File glob pattern
                max_results: Maximum number of results

            Returns:
                List of matching files with context
            """

            # Validate path
            is_safe, error_msg = self._is_safe_path(directory)
            if not is_safe:
                return TextContent(
                    text=json.dumps({'error': error_msg}),
                    mime_type="application/json"
                )

            dir_path = Path(directory).resolve()
            results = []

            try:
                for file_path in dir_path.rglob(file_pattern):
                    if len(results) >= max_results:
                        break

                    if not file_path.is_file():
                        continue

                    # Skip large files
                    if file_path.stat().st_size > self.max_file_size:
                        continue

                    try:
                        async with aiofiles.open(file_path, 'r') as f:
                            content = await f.read()

                        if content_pattern.lower() in content.lower():
                            # Find line numbers with matches
                            lines = content.split('\n')
                            matching_lines = [
                                (i + 1, line.strip())
                                for i, line in enumerate(lines)
                                if content_pattern.lower() in line.lower()
                            ]

                            results.append({
                                'file': str(file_path),
                                'matches': len(matching_lines),
                                'lines': matching_lines[:5]  # First 5 matches
                            })

                    except:
                        # Skip files that can't be read
                        continue

                return TextContent(
                    text=json.dumps({
                        'pattern': content_pattern,
                        'directory': str(dir_path),
                        'results_count': len(results),
                        'results': results
                    }, indent=2),
                    mime_type="application/json"
                )

            except Exception as e:
                logging.error(f"Error searching files: {e}")
                return TextContent(
                    text=json.dumps({'error': str(e)}),
                    mime_type="application/json"
                )
```

**Production Features:**
- ✅ Path traversal protection
- ✅ Sandboxed directory access
- ✅ File size limits
- ✅ Extension blocking
- ✅ Async file operations
- ✅ File content search
- ✅ Metadata extraction
- ✅ Access audit logging
- ✅ Hash verification

---

## 🏗️ MCP Architecture

### Component Architecture

<div align="center">

```mermaid
graph TB
    subgraph "MCP Server Architecture"
        subgraph "Interface Layer"
            JSON[JSON-RPC Handler]
            Trans[Transport Layer]
        end

        subgraph "Business Logic"
            Tools[Tool Registry]
            Valid[Validators]
            Auth[Authentication]
        end

        subgraph "Data Layer"
            Pool[Connection Pool]
            Cache[Cache Manager]
            Log[Audit Logger]
        end

        subgraph "External Resources"
            DB[(Database)]
            API[External APIs]
            FS[File System]
        end
    end

    Trans --> JSON
    JSON --> Tools
    Tools --> Valid
    Tools --> Auth
    Valid --> Pool
    Auth --> Cache
    Pool --> DB
    Cache --> API
    Log --> FS

    style JSON fill:#a855f7,stroke:#7e22ce,color:#fff
    style Tools fill:#3b82f6,stroke:#1d4ed8,color:#fff
    style Pool fill:#10b981,stroke:#059669,color:#fff
```

</div>

### Request Lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Transport
    participant Auth
    participant RateLimit
    participant Validator
    participant Tool
    participant Resource

    Client->>Transport: JSON-RPC Request
    Transport->>Auth: Authenticate
    Auth-->>Transport: Token Valid

    Transport->>RateLimit: Check Limit
    RateLimit-->>Transport: Within Limit

    Transport->>Validator: Validate Parameters
    Validator-->>Transport: Parameters OK

    Transport->>Tool: Execute Tool
    Tool->>Resource: Access Resource
    Resource-->>Tool: Return Data
    Tool->>Tool: Format Response
    Tool-->>Transport: Tool Result

    Transport->>Transport: Log Request
    Transport-->>Client: JSON-RPC Response

    Note over Client,Resource: Security & validation at each step
```

### Deployment Architecture

```mermaid
graph TB
    subgraph "Production Deployment"
        subgraph "Load Balancer"
            LB[Nginx/HAProxy]
        end

        subgraph "MCP Servers"
            S1[Server Instance 1]
            S2[Server Instance 2]
            S3[Server Instance 3]
        end

        subgraph "Shared Services"
            Redis[(Redis Cache)]
            PG[(PostgreSQL)]
            Logs[Log Aggregator]
        end

        subgraph "Monitoring"
            Prom[Prometheus]
            Graf[Grafana]
            Alert[AlertManager]
        end
    end

    LB --> S1
    LB --> S2
    LB --> S3

    S1 --> Redis
    S2 --> Redis
    S3 --> Redis

    S1 --> PG
    S2 --> PG
    S3 --> PG

    S1 --> Logs
    S2 --> Logs
    S3 --> Logs

    Logs --> Prom
    Prom --> Graf
    Prom --> Alert

    style LB fill:#a855f7,stroke:#7e22ce,color:#fff
    style Redis fill:#dc2626,stroke:#991b1b,color:#fff
    style PG fill:#2563eb,stroke:#1e40af,color:#fff
```

---

## 📚 Essential MCP Resources

### 🏆 Core Documentation

<div align="center">

| Resource | Description | Link | Rating |
|----------|-------------|------|--------|
| 📖 **Official Specification** | Protocol specification and architecture | [Docs](https://modelcontextprotocol.io/docs) | ⭐⭐⭐⭐⭐ |
| 🐍 **Python SDK** | Async/await support, type-safe | [GitHub](https://github.com/anthropics/mcp-python) | ⭐⭐⭐⭐⭐ |
| 📘 **TypeScript SDK** | Modern TypeScript, Node.js and browser | [GitHub](https://github.com/anthropics/mcp-typescript) | ⭐⭐⭐⭐⭐ |
| 🔧 **MCP Servers** | 100+ production examples | [GitHub](https://github.com/modelcontextprotocol/servers) | ⭐⭐⭐⭐⭐ |

</div>

### 📖 Learning Resources

#### Tutorials & Guides

- **[Building Your First MCP Server](https://modelcontextprotocol.io/tutorials/first-server)** - Step-by-step tutorial
  - From zero to working server in 30 minutes
  - Best practices included
  - Common pitfalls explained

- **[MCP Server Examples](https://github.com/modelcontextprotocol/servers)** - Production examples
  - Database servers (PostgreSQL, MySQL, SQLite)
  - API integrations (GitHub, Slack, Discord)
  - File systems, Docker, Kubernetes
  - Copy, modify, deploy

- **[Awesome MCP](https://github.com/punkpeye/awesome-mcp)** - Community-curated list
  - Servers, clients, and tools
  - Articles and tutorials
  - Integration examples

### 🛠️ Tools & Utilities

| Tool | Purpose | Link |
|------|---------|------|
| **MCP Inspector** | Debug and test MCP servers interactively | [GitHub](https://github.com/modelcontextprotocol/inspector) |
| **MCP CLI** | Server scaffolding and testing | [GitHub](https://github.com/modelcontextprotocol/cli) |
| **MCP Test Suite** | Comprehensive testing framework | [GitHub](https://github.com/modelcontextprotocol/test-suite) |

---

## 🔧 MCP Patterns Library

### Authentication Pattern

```python
class AuthenticatedMCP:
    """Secure MCP server with API key authentication"""

    def __init__(self):
        self.auth_tokens = {}  # In production: use database
        self.failed_attempts = defaultdict(int)
        self.lockout_threshold = 5

    async def authenticate(self, token: str, client_ip: str) -> bool:
        """Validate auth token with rate limiting"""

        # Check if client is locked out
        if self.failed_attempts[client_ip] >= self.lockout_threshold:
            return False

        # Validate token
        if token in self.auth_tokens:
            self.failed_attempts[client_ip] = 0  # Reset on success
            return True

        # Track failed attempt
        self.failed_attempts[client_ip] += 1
        return False

    def require_auth(self, func):
        """Decorator for authenticated tools"""
        async def wrapper(*args, **kwargs):
            token = kwargs.get('auth_token')
            client_ip = kwargs.get('client_ip', 'unknown')

            if not await self.authenticate(token, client_ip):
                return TextContent(
                    text="❌ Unauthorized: Invalid or missing auth token",
                    mime_type="text/plain"
                )
            return await func(*args, **kwargs)
        return wrapper
```

### Caching Pattern

```python
class CachedMCP:
    """MCP server with intelligent multi-level caching"""

    def __init__(self):
        self.memory_cache = TTLCache(maxsize=100, ttl=300)  # L1: 5 min
        self.redis_cache = aioredis.from_url("redis://localhost")  # L2

    def cached(self, ttl: int = 300, cache_level: str = "both"):
        """Cache decorator with configurable TTL and level"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

                # L1: Memory cache
                if cache_level in ["both", "memory"]:
                    if key in self.memory_cache:
                        return self.memory_cache[key]

                # L2: Redis cache
                if cache_level in ["both", "redis"]:
                    cached = await self.redis_cache.get(key)
                    if cached:
                        result = json.loads(cached)
                        # Promote to L1
                        self.memory_cache[key] = result
                        return result

                # Cache miss - execute function
                result = await func(*args, **kwargs)

                # Store in caches
                if cache_level in ["both", "memory"]:
                    self.memory_cache[key] = result
                if cache_level in ["both", "redis"]:
                    await self.redis_cache.setex(
                        key, ttl, json.dumps(result, default=str)
                    )

                return result
            return wrapper
        return decorator
```

### Rate Limiting Pattern

```python
class RateLimitedMCP:
    """MCP server with sophisticated rate limiting"""

    def __init__(self):
        self.limiters = {}
        self.limits = {
            'standard': (100, 3600),  # 100 calls per hour
            'premium': (1000, 3600),  # 1000 calls per hour
            'burst': (10, 60)  # 10 calls per minute
        }

    def rate_limit(self, tier: str = 'standard', burst: bool = False):
        """Rate limit decorator with tiered limits"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                client_id = kwargs.get('client_id', 'anonymous')
                limiter_key = f"{client_id}:{func.__name__}"

                # Get or create limiter
                if limiter_key not in self.limiters:
                    calls, period = self.limits[tier]
                    self.limiters[limiter_key] = TokenBucketRateLimiter(calls, period)

                # Check burst limit
                if burst:
                    burst_key = f"{limiter_key}:burst"
                    if burst_key not in self.limiters:
                        calls, period = self.limits['burst']
                        self.limiters[burst_key] = TokenBucketRateLimiter(calls, period)

                    if not await self.limiters[burst_key].acquire():
                        return TextContent(
                            text="❌ Burst rate limit exceeded",
                            mime_type="text/plain"
                        )

                # Check main limit
                if not await self.limiters[limiter_key].acquire():
                    retry_after = self.limiters[limiter_key].retry_after()
                    return TextContent(
                        text=json.dumps({
                            'error': 'Rate limit exceeded',
                            'tier': tier,
                            'retry_after': f"{retry_after:.0f}s"
                        }),
                        mime_type="application/json"
                    )

                return await func(*args, **kwargs)
            return wrapper
        return decorator
```

---

## 🚀 Deployment Configurations

### Docker Deployment

```dockerfile
# Production-ready MCP server Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mcp && \
    mkdir -p /app && \
    chown -R mcp:mcp /app

WORKDIR /app

# Install Python dependencies
COPY --chown=mcp:mcp requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=mcp:mcp . .

# Switch to non-root user
USER mcp

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Expose port
EXPOSE 8080

# Run server
CMD ["python", "-m", "mcp.server", "--config", "config.yml"]
```

### Kubernetes Deployment

```yaml
# production-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  namespace: production
  labels:
    app: mcp-server
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: mcp-server
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000

      containers:
      - name: mcp-server
        image: your-registry/mcp-server:1.0.0
        imagePullPolicy: Always

        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP

        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: redis-url
        - name: LOG_LEVEL
          value: "INFO"
        - name: ENVIRONMENT
          value: "production"

        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"

        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: logs
          mountPath: /app/logs

      volumes:
      - name: config
        configMap:
          name: mcp-config
      - name: logs
        emptyDir: {}

      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - mcp-server
              topologyKey: kubernetes.io/hostname

---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
  namespace: production
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  selector:
    app: mcp-server

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Docker Compose Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: .
    image: mcp-server:latest
    container_name: mcp-server
    restart: unless-stopped
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/mcp
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    ports:
      - "8080:8080"
      - "9090:9090"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    networks:
      - mcp-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 40s

  postgres:
    image: postgres:15-alpine
    container_name: mcp-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mcp
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - mcp-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: mcp-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    networks:
      - mcp-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: mcp-prometheus
    restart: unless-stopped
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9091:9090"
    networks:
      - mcp-network
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    container_name: mcp-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards:ro
    ports:
      - "3000:3000"
    networks:
      - mcp-network
    depends_on:
      - prometheus

volumes:
  postgres-data:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  mcp-network:
    driver: bridge
```

---

## 🎯 Integration Examples

### Claude Desktop Integration

```json
{
  "mcpServers": {
    "production-db": {
      "command": "python",
      "args": ["-m", "mcp_server.database"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost/production",
        "LOG_LEVEL": "INFO",
        "MAX_CONNECTIONS": "20"
      }
    },
    "api-gateway": {
      "command": "python",
      "args": ["-m", "mcp_server.api_gateway"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}",
        "SLACK_TOKEN": "${SLACK_TOKEN}",
        "CACHE_TTL": "300"
      }
    },
    "file-system": {
      "command": "python",
      "args": ["-m", "mcp_server.filesystem"],
      "env": {
        "ALLOWED_DIRS": "/home/user/projects,/home/user/documents",
        "MAX_FILE_SIZE": "10485760"
      }
    }
  }
}
```

### LangChain Integration

```python
from langchain.tools import MCPTool
from langchain.agents import initialize_agent
from langchain.llms import Anthropic

# Wrap MCP server as LangChain tool
mcp_database_tool = MCPTool(
    server_url="http://localhost:8080",
    name="database_query",
    description="Query the production database with SQL"
)

mcp_api_tool = MCPTool(
    server_url="http://localhost:8081",
    name="api_gateway",
    description="Call external APIs (GitHub, Slack, AWS)"
)

# Initialize agent with MCP tools
llm = Anthropic(model="claude-3-sonnet-20240229")

agent = initialize_agent(
    tools=[mcp_database_tool, mcp_api_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Use the agent
response = agent.run(
    "What are the top 10 users by revenue in the database, "
    "and create a GitHub issue to review their accounts?"
)
```

### Vercel AI SDK Integration

```typescript
import { createMCPClient } from '@modelcontextprotocol/client';
import { useChat } from 'ai/react';

// Create MCP client
const mcpClient = createMCPClient({
  servers: {
    database: {
      command: 'python',
      args: ['-m', 'mcp_server.database'],
      env: {
        DATABASE_URL: process.env.DATABASE_URL
      }
    }
  }
});

// React component with MCP integration
export function ChatInterface() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: '/api/chat',
    onToolCall: async (toolCall) => {
      // Execute MCP tool
      const result = await mcpClient.callTool(
        toolCall.server,
        toolCall.tool,
        toolCall.arguments
      );
      return result;
    }
  });

  return (
    <div>
      {messages.map(m => (
        <div key={m.id}>
          {m.role === 'user' ? 'User: ' : 'AI: '}
          {m.content}
        </div>
      ))}

      <form onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={handleInputChange}
          placeholder="Ask me anything..."
        />
      </form>
    </div>
  );
}
```

---

## 🔒 Security Best Practices

### Security Checklist

<div align="center">

| Layer | Control | Implementation |
|-------|---------|----------------|
| 🔐 **Authentication** | API Keys, OAuth2, JWT | Rotate keys every 90 days |
| ⚡ **Rate Limiting** | Per-user, per-tool limits | Token bucket algorithm |
| 📝 **Audit Logging** | All operations logged | Immutable log storage |
| ✅ **Input Validation** | Schema validation, sanitization | Reject before processing |
| 🔑 **Least Privilege** | Minimal permissions | Role-based access control |
| 📊 **Monitoring** | Real-time alerts | Anomaly detection |
| 🔒 **Encryption** | TLS 1.3, at-rest encryption | End-to-end security |

</div>

### Security Implementation

```python
class SecureMCPServer:
    """MCP server with enterprise-grade security"""

    def __init__(self):
        self.auth_provider = OAuth2Provider()
        self.rate_limiter = RateLimiter()
        self.audit_logger = AuditLogger()
        self.input_validator = InputValidator()
        self.encryption = EncryptionService()

    async def handle_request(self, request):
        """Secure request handler with multi-layer security"""

        # 1. Authenticate
        try:
            user = await self.auth_provider.authenticate(
                request.headers.get('Authorization')
            )
        except AuthenticationError:
            self.audit_logger.log_failed_auth(request)
            return {'error': 'Unauthorized', 'status': 401}

        # 2. Check rate limits
        if not await self.rate_limiter.check(user.id):
            self.audit_logger.log_rate_limit(user.id)
            return {'error': 'Rate limit exceeded', 'status': 429}

        # 3. Validate input
        try:
            validated_data = self.input_validator.validate(
                request.data,
                schema=self.get_tool_schema(request.tool)
            )
        except ValidationError as e:
            self.audit_logger.log_invalid_input(user.id, str(e))
            return {'error': f'Invalid input: {e}', 'status': 400}

        # 4. Check permissions
        if not await self.check_permissions(user, request.tool):
            self.audit_logger.log_permission_denied(user.id, request.tool)
            return {'error': 'Permission denied', 'status': 403}

        # 5. Execute tool
        try:
            result = await self.execute_tool(
                request.tool,
                validated_data,
                user_context=user
            )

            # 6. Log success
            self.audit_logger.log_success(
                user.id,
                request.tool,
                validated_data
            )

            return {'data': result, 'status': 200}

        except Exception as e:
            # 7. Log error
            self.audit_logger.log_error(user.id, request.tool, str(e))
            return {'error': 'Internal error', 'status': 500}
```

---

## 📊 Performance Guidelines

### Performance Optimization Strategies

<div align="center">

```mermaid
graph LR
    subgraph "Performance Stack"
        A[Connection Pooling] --> B[Response Caching]
        B --> C[Async Operations]
        C --> D[Batch Processing]
        D --> E[Load Balancing]
        E --> F[CDN Distribution]
    end

    style A fill:#10b981,stroke:#059669,color:#fff
    style C fill:#3b82f6,stroke:#1d4ed8,color:#fff
    style E fill:#a855f7,stroke:#7e22ce,color:#fff
```

</div>

### Performance Best Practices

```python
class OptimizedMCPServer:
    """High-performance MCP server implementation"""

    async def setup_performance_optimizations(self):
        """Configure performance enhancements"""

        # 1. Connection Pooling
        self.db_pool = await asyncpg.create_pool(
            dsn=self.config.database_url,
            min_size=5,
            max_size=20,
            max_inactive_connection_lifetime=300,
            command_timeout=30,
            max_queries=50000  # Recycle after 50k queries
        )

        # 2. HTTP Client Pooling
        self.http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30
            ),
            timeout=httpx.Timeout(30.0),
            http2=True  # Enable HTTP/2
        )

        # 3. Multi-level Caching
        self.cache = {
            'l1': TTLCache(maxsize=1000, ttl=60),  # Hot data
            'l2': TTLCache(maxsize=10000, ttl=300),  # Warm data
            'l3': await aioredis.from_url("redis://localhost")  # Cold data
        }

        # 4. Batch Processing
        self.batch_queue = asyncio.Queue(maxsize=1000)
        self.batch_processor = asyncio.create_task(
            self.process_batches()
        )

        # 5. Metrics Collection
        self.metrics = {
            'requests_total': 0,
            'requests_duration': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }

    async def process_batches(self):
        """Process requests in batches for efficiency"""
        while True:
            batch = []

            # Collect batch
            try:
                # Wait for first item (blocking)
                batch.append(await self.batch_queue.get())

                # Collect more items (non-blocking, up to 100ms)
                deadline = asyncio.get_event_loop().time() + 0.1
                while len(batch) < 100:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break

                    try:
                        item = await asyncio.wait_for(
                            self.batch_queue.get(),
                            timeout=timeout
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                # Process batch
                await self.execute_batch(batch)

            except Exception as e:
                logging.error(f"Batch processing error: {e}")

    async def execute_batch(self, batch):
        """Execute multiple operations efficiently"""
        # Group by operation type
        queries = [item for item in batch if item['type'] == 'query']
        api_calls = [item for item in batch if item['type'] == 'api']

        # Execute in parallel
        results = await asyncio.gather(
            self.batch_execute_queries(queries),
            self.batch_execute_apis(api_calls),
            return_exceptions=True
        )

        # Return results to waiting futures
        for item, result in zip(batch, results):
            item['future'].set_result(result)
```

### Performance Metrics

<div align="center">

| Metric | Target | Monitoring |
|--------|--------|-----------|
| ⚡ **Response Time (p50)** | < 100ms | Prometheus |
| 📊 **Response Time (p99)** | < 500ms | Prometheus |
| 🔄 **Throughput** | > 1000 req/s | Grafana |
| 💾 **Cache Hit Rate** | > 80% | Redis metrics |
| 🔌 **Connection Pool Usage** | 50-80% | Application metrics |
| ❌ **Error Rate** | < 0.1% | AlertManager |

</div>

---

## 🐛 Common Issues & Solutions

### Issue 1: Connection Timeout

```python
# Problem: Database connections timing out

# Solution: Implement retry logic with exponential backoff
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def connect_with_retry(url):
    """Connect with automatic retry"""
    try:
        return await asyncio.wait_for(
            asyncpg.connect(url),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        logging.warning("Connection timeout, retrying...")
        raise
    except Exception as e:
        logging.error(f"Connection error: {e}")
        raise
```

### Issue 2: Memory Leak in Long-Running Server

```python
# Problem: Memory grows over time

# Solution: Implement connection recycling
async def recycle_connections_periodically(self):
    """Recycle connections every hour"""
    while True:
        await asyncio.sleep(3600)  # Every hour

        try:
            # Create new pool
            new_pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=5,
                max_size=20
            )

            # Swap pools
            old_pool = self.pool
            self.pool = new_pool

            # Close old pool gracefully
            await old_pool.close()

            logging.info("✅ Connection pool recycled")

        except Exception as e:
            logging.error(f"Failed to recycle pool: {e}")
```

### Issue 3: Slow Query Performance

```python
# Problem: Some queries are very slow

# Solution: Implement query timeout and monitoring
class QueryMonitor:
    """Monitor and optimize slow queries"""

    def __init__(self):
        self.slow_queries = []
        self.slow_threshold = 1.0  # 1 second

    async def execute_with_monitoring(self, query, params):
        """Execute query with performance monitoring"""
        start_time = time.time()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.pool.fetch(query, *params),
                timeout=5.0
            )

            execution_time = time.time() - start_time

            # Log slow queries
            if execution_time > self.slow_threshold:
                self.slow_queries.append({
                    'query': query,
                    'params': params,
                    'execution_time': execution_time,
                    'timestamp': datetime.utcnow().isoformat()
                })

                logging.warning(
                    f"Slow query detected ({execution_time:.2f}s): {query}"
                )

            return result

        except asyncio.TimeoutError:
            logging.error(f"Query timeout: {query}")
            raise
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
requests_total = Counter(
    'mcp_requests_total',
    'Total number of MCP requests',
    ['tool', 'status']
)

request_duration = Histogram(
    'mcp_request_duration_seconds',
    'Request duration in seconds',
    ['tool']
)

active_connections = Gauge(
    'mcp_active_connections',
    'Number of active database connections'
)

cache_hits = Counter(
    'mcp_cache_hits_total',
    'Total number of cache hits',
    ['level']
)

# Instrument your code
@request_duration.labels(tool='query_database').time()
async def query_database(query: str):
    """Query with metrics collection"""
    try:
        result = await execute_query(query)
        requests_total.labels(tool='query_database', status='success').inc()
        return result
    except Exception as e:
        requests_total.labels(tool='query_database', status='error').inc()
        raise
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "MCP Server Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(mcp_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time (p50, p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, mcp_request_duration_seconds)"
          },
          {
            "expr": "histogram_quantile(0.95, mcp_request_duration_seconds)"
          },
          {
            "expr": "histogram_quantile(0.99, mcp_request_duration_seconds)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(mcp_requests_total{status=\"error\"}[5m])"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(mcp_cache_hits_total[5m]) / rate(mcp_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

---

## 🚀 Next Steps

<div align="center">

### 🎓 Recommended Learning Path

```mermaid
graph LR
    A[📚 Read Docs] --> B[🔧 Build Simple Server]
    B --> C[🧪 Add Tests]
    C --> D[🔒 Implement Security]
    D --> E[📊 Add Monitoring]
    E --> F[🚀 Deploy Production]
    F --> G[📈 Scale & Optimize]

    style A fill:#3b82f6,stroke:#1d4ed8,color:#fff
    style D fill:#a855f7,stroke:#7e22ce,color:#fff
    style F fill:#10b981,stroke:#059669,color:#fff
```

</div>

1. **Start Simple** (2 hours)
   - Build a basic read-only database server
   - Test locally with Claude Desktop
   - Understand the request/response flow

2. **Add Security** (4 hours)
   - Implement authentication
   - Add input validation
   - Set up audit logging

3. **Production-ize** (8 hours)
   - Add connection pooling
   - Implement caching
   - Set up monitoring

4. **Deploy** (4 hours)
   - Containerize with Docker
   - Deploy to Kubernetes
   - Configure CI/CD

5. **Monitor & Scale** (Ongoing)
   - Watch metrics
   - Optimize hot paths
   - Scale based on load

---

<div align="center">

## 🌟 **Ready to Build Production MCP Servers?** 🌟

### Start with the patterns above and scale to millions of requests

[![View Examples](https://img.shields.io/badge/View-Server_Examples-3b82f6?style=for-the-badge)](./mcp-server-examples.md)
[![Client Guide](https://img.shields.io/badge/Read-Client_Guide-a855f7?style=for-the-badge)](./mcp-client-guide.md)
[![Join Discord](https://img.shields.io/badge/Join-Discord-5865F2?style=for-the-badge)](https://discord.gg/mcp)

---

*Last Updated: January 2025 • Production-Ready Patterns • Battle-Tested at Scale*

</div>
