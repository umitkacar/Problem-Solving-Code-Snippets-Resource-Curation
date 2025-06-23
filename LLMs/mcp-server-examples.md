# 🔧 MCP Server Examples

Production-ready Model Context Protocol server implementations for various use cases, from simple utilities to complex enterprise systems.

**Last Updated:** 2025-06-23

## Table of Contents
- [Getting Started](#getting-started)
- [Basic Servers](#basic-servers)
- [Data & Database Servers](#data--database-servers)
- [API Integration Servers](#api-integration-servers)
- [File System Servers](#file-system-servers)
- [DevOps & Infrastructure](#devops--infrastructure)
- [AI & ML Servers](#ai--ml-servers)
- [Business Applications](#business-applications)
- [Security & Compliance](#security--compliance)
- [Advanced Patterns](#advanced-patterns)
- [Testing & Debugging](#testing--debugging)
- [Deployment Guide](#deployment-guide)
- [Performance Tuning](#performance-tuning)

## Getting Started

### Basic Server Structure

```python
# basic_server.py
import asyncio
import logging
from mcp import Server, Tool, Resource, StdioTransport
from mcp.types import ToolResult, ResourceContent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicMCPServer(Server):
    def __init__(self):
        super().__init__("basic-server")
        logger.info("Initializing Basic MCP Server")
        
    async def startup(self):
        """Initialize server resources"""
        logger.info("Server starting up...")
        # Initialize connections, load configs, etc.
        
    async def shutdown(self):
        """Cleanup server resources"""
        logger.info("Server shutting down...")
        # Close connections, save state, etc.

async def main():
    server = BasicMCPServer()
    transport = StdioTransport()
    
    await server.startup()
    try:
        await server.run(transport)
    finally:
        await server.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Running Your Server

```bash
# Direct execution
python basic_server.py

# With Claude Desktop
# Add to claude_desktop_config.json:
{
  "mcpServers": {
    "basic-server": {
      "command": "python",
      "args": ["/path/to/basic_server.py"]
    }
  }
}
```

## Basic Servers

### 1. Calculator Server

```python
import math
from mcp import Server, Tool
from mcp.types import ToolResult

class CalculatorServer(Server):
    """Scientific calculator MCP server"""
    
    def __init__(self):
        super().__init__("calculator-server")
        self.setup_tools()
    
    def setup_tools(self):
        self.tools = [
            Tool(
                name="calculate",
                description="Evaluate mathematical expression",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate"
                        },
                        "precision": {
                            "type": "integer",
                            "description": "Decimal precision",
                            "default": 4
                        }
                    },
                    "required": ["expression"]
                }
            ),
            Tool(
                name="solve_equation",
                description="Solve algebraic equation",
                parameters={
                    "type": "object",
                    "properties": {
                        "equation": {
                            "type": "string",
                            "description": "Equation to solve (e.g., '2x + 5 = 15')"
                        },
                        "variable": {
                            "type": "string",
                            "description": "Variable to solve for",
                            "default": "x"
                        }
                    },
                    "required": ["equation"]
                }
            )
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        if tool_name == "calculate":
            return await self.calculate(**arguments)
        elif tool_name == "solve_equation":
            return await self.solve_equation(**arguments)
    
    async def calculate(self, expression: str, precision: int = 4):
        """Safely evaluate mathematical expressions"""
        try:
            # Create safe namespace with math functions
            safe_dict = {
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
                'pi': math.pi, 'e': math.e, 'abs': abs,
                'round': round, 'floor': math.floor, 'ceil': math.ceil
            }
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            # Format result
            if isinstance(result, float):
                result = round(result, precision)
            
            return ToolResult(
                success=True,
                data={
                    "expression": expression,
                    "result": result,
                    "type": type(result).__name__
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Calculation error: {str(e)}"
            )
    
    async def solve_equation(self, equation: str, variable: str = "x"):
        """Solve simple algebraic equations"""
        try:
            from sympy import symbols, Eq, solve, sympify
            
            # Parse equation
            left, right = equation.split('=')
            
            # Create symbol
            var = symbols(variable)
            
            # Create equation
            eq = Eq(sympify(left), sympify(right))
            
            # Solve
            solutions = solve(eq, var)
            
            return ToolResult(
                success=True,
                data={
                    "equation": equation,
                    "variable": variable,
                    "solutions": [str(sol) for sol in solutions],
                    "solution_count": len(solutions)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Solving error: {str(e)}"
            )
```

### 2. Weather Server

```python
import aiohttp
from datetime import datetime
from mcp import Server, Tool, Resource
from mcp.types import ToolResult, ResourceContent

class WeatherServer(Server):
    """Weather information MCP server"""
    
    def __init__(self, api_key: str):
        super().__init__("weather-server")
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.session = None
        self.setup_capabilities()
    
    def setup_capabilities(self):
        self.tools = [
            Tool(
                name="get_current_weather",
                description="Get current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or coordinates"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial"],
                            "default": "metric"
                        }
                    },
                    "required": ["location"]
                }
            ),
            Tool(
                name="get_forecast",
                description="Get weather forecast",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "days": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "default": 3
                        }
                    },
                    "required": ["location"]
                }
            )
        ]
        
        self.resources = [
            Resource(
                uri="weather://alerts/global",
                name="Global Weather Alerts",
                description="Active weather alerts worldwide",
                mimeType="application/json"
            )
        ]
    
    async def startup(self):
        self.session = aiohttp.ClientSession()
    
    async def shutdown(self):
        if self.session:
            await self.session.close()
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        if tool_name == "get_current_weather":
            return await self.get_current_weather(**arguments)
        elif tool_name == "get_forecast":
            return await self.get_forecast(**arguments)
    
    async def get_current_weather(self, location: str, units: str = "metric"):
        """Fetch current weather data"""
        try:
            params = {
                "q": location,
                "appid": self.api_key,
                "units": units
            }
            
            async with self.session.get(
                f"{self.base_url}/weather",
                params=params
            ) as response:
                data = await response.json()
                
                if response.status == 200:
                    return ToolResult(
                        success=True,
                        data={
                            "location": data["name"],
                            "country": data["sys"]["country"],
                            "temperature": data["main"]["temp"],
                            "feels_like": data["main"]["feels_like"],
                            "humidity": data["main"]["humidity"],
                            "description": data["weather"][0]["description"],
                            "wind_speed": data["wind"]["speed"],
                            "units": units,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                else:
                    return ToolResult(
                        success=False,
                        error=data.get("message", "Weather API error")
                    )
                    
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to fetch weather: {str(e)}"
            )
```

### 3. Translation Server

```python
from googletrans import Translator
from mcp import Server, Tool
from mcp.types import ToolResult
import langdetect

class TranslationServer(Server):
    """Multi-language translation MCP server"""
    
    def __init__(self):
        super().__init__("translation-server")
        self.translator = Translator()
        self.setup_tools()
    
    def setup_tools(self):
        self.tools = [
            Tool(
                name="translate",
                description="Translate text between languages",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to translate"
                        },
                        "target_language": {
                            "type": "string",
                            "description": "Target language code (e.g., 'es', 'fr', 'ja')"
                        },
                        "source_language": {
                            "type": "string",
                            "description": "Source language code (auto-detect if not provided)"
                        }
                    },
                    "required": ["text", "target_language"]
                }
            ),
            Tool(
                name="detect_language",
                description="Detect the language of text",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to analyze"
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="translate_batch",
                description="Translate multiple texts",
                parameters={
                    "type": "object",
                    "properties": {
                        "texts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of texts to translate"
                        },
                        "target_language": {"type": "string"}
                    },
                    "required": ["texts", "target_language"]
                }
            )
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        handlers = {
            "translate": self.translate,
            "detect_language": self.detect_language,
            "translate_batch": self.translate_batch
        }
        
        if handler := handlers.get(tool_name):
            return await handler(**arguments)
    
    async def translate(self, text: str, target_language: str, 
                       source_language: str = None):
        """Translate text between languages"""
        try:
            # Auto-detect source language if not provided
            if not source_language:
                source_language = langdetect.detect(text)
            
            # Translate
            result = self.translator.translate(
                text,
                src=source_language,
                dest=target_language
            )
            
            return ToolResult(
                success=True,
                data={
                    "original_text": text,
                    "translated_text": result.text,
                    "source_language": result.src,
                    "target_language": target_language,
                    "confidence": result.extra_data.get('confidence', 1.0)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Translation failed: {str(e)}"
            )
```

## Data & Database Servers

### 1. PostgreSQL Server

```python
import asyncpg
import json
from mcp import Server, Tool, Resource, ResourceTemplate
from mcp.types import ToolResult, ResourceContent

class PostgreSQLServer(Server):
    """PostgreSQL database MCP server"""
    
    def __init__(self, connection_string: str):
        super().__init__("postgresql-server")
        self.connection_string = connection_string
        self.pool = None
        self.setup_capabilities()
    
    def setup_capabilities(self):
        # Tools for write operations
        self.tools = [
            Tool(
                name="execute_query",
                description="Execute SQL query",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        },
                        "parameters": {
                            "type": "array",
                            "description": "Query parameters",
                            "items": {}
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="insert_record",
                description="Insert record into table",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "data": {
                            "type": "object",
                            "description": "Column-value pairs"
                        }
                    },
                    "required": ["table", "data"]
                }
            ),
            Tool(
                name="update_record",
                description="Update table records",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "data": {"type": "object"},
                        "where": {
                            "type": "object",
                            "description": "WHERE clause conditions"
                        }
                    },
                    "required": ["table", "data", "where"]
                }
            )
        ]
        
        # Resource templates for read operations
        self.resource_templates = [
            ResourceTemplate(
                uriTemplate="db://schema/{table}",
                name="Table Schema",
                description="Get table structure",
                mimeType="application/json"
            ),
            ResourceTemplate(
                uriTemplate="db://data/{table}?limit={limit}&offset={offset}",
                name="Table Data",
                description="Read table data with pagination",
                mimeType="application/json"
            ),
            ResourceTemplate(
                uriTemplate="db://query/{query_name}",
                name="Named Query",
                description="Execute predefined read-only query",
                mimeType="application/json"
            )
        ]
    
    async def startup(self):
        """Create connection pool"""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=20,
            command_timeout=10
        )
    
    async def shutdown(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        if tool_name == "execute_query":
            return await self.execute_query(**arguments)
        elif tool_name == "insert_record":
            return await self.insert_record(**arguments)
        elif tool_name == "update_record":
            return await self.update_record(**arguments)
    
    async def execute_query(self, query: str, parameters: list = None):
        """Execute arbitrary SQL query"""
        try:
            # Check if query is safe (basic validation)
            query_lower = query.lower().strip()
            
            # Prevent destructive operations without confirmation
            destructive_keywords = ['drop', 'truncate', 'delete from']
            if any(keyword in query_lower for keyword in destructive_keywords):
                return ToolResult(
                    success=False,
                    error="Destructive operations require additional confirmation"
                )
            
            async with self.pool.acquire() as conn:
                if parameters:
                    result = await conn.fetch(query, *parameters)
                else:
                    result = await conn.fetch(query)
                
                # Convert to serializable format
                rows = [dict(row) for row in result]
                
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "row_count": len(rows),
                        "rows": rows
                    }
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Query execution failed: {str(e)}"
            )
    
    async def insert_record(self, table: str, data: dict):
        """Insert single record"""
        try:
            # Build INSERT query
            columns = list(data.keys())
            values = list(data.values())
            placeholders = [f"${i+1}" for i in range(len(values))]
            
            query = f"""
                INSERT INTO {table} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                RETURNING *
            """
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, *values)
                
                return ToolResult(
                    success=True,
                    data={
                        "table": table,
                        "inserted": dict(row)
                    }
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Insert failed: {str(e)}"
            )
    
    async def handle_resource_read(self, uri: str):
        """Handle resource requests"""
        import re
        
        # Parse schema request
        if match := re.match(r"db://schema/(.+)", uri):
            table = match.group(1)
            return await self.get_table_schema(table)
        
        # Parse data request
        elif match := re.match(r"db://data/(.+)\?limit=(\d+)&offset=(\d+)", uri):
            table = match.group(1)
            limit = int(match.group(2))
            offset = int(match.group(3))
            return await self.get_table_data(table, limit, offset)
        
        # Parse named query
        elif match := re.match(r"db://query/(.+)", uri):
            query_name = match.group(1)
            return await self.execute_named_query(query_name)
    
    async def get_table_schema(self, table: str):
        """Get table structure information"""
        try:
            query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY ordinal_position
            """
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, table)
                
                schema = {
                    "table": table,
                    "columns": [dict(row) for row in rows]
                }
                
                return ResourceContent(
                    content=json.dumps(schema, indent=2),
                    mimeType="application/json"
                )
                
        except Exception as e:
            return None
```

### 2. MongoDB Server

```python
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId, json_util
import json
from mcp import Server, Tool, Resource
from mcp.types import ToolResult, ResourceContent

class MongoDBServer(Server):
    """MongoDB MCP server"""
    
    def __init__(self, connection_string: str, database: str):
        super().__init__("mongodb-server")
        self.connection_string = connection_string
        self.database_name = database
        self.client = None
        self.db = None
        self.setup_tools()
    
    def setup_tools(self):
        self.tools = [
            Tool(
                name="find_documents",
                description="Find documents in collection",
                parameters={
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"},
                        "filter": {
                            "type": "object",
                            "description": "MongoDB filter query"
                        },
                        "projection": {
                            "type": "object",
                            "description": "Fields to include/exclude"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10
                        },
                        "sort": {
                            "type": "object",
                            "description": "Sort specification"
                        }
                    },
                    "required": ["collection"]
                }
            ),
            Tool(
                name="insert_document",
                description="Insert document into collection",
                parameters={
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"},
                        "document": {
                            "type": "object",
                            "description": "Document to insert"
                        }
                    },
                    "required": ["collection", "document"]
                }
            ),
            Tool(
                name="update_documents",
                description="Update documents in collection",
                parameters={
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"},
                        "filter": {"type": "object"},
                        "update": {
                            "type": "object",
                            "description": "Update operations"
                        },
                        "many": {
                            "type": "boolean",
                            "default": False,
                            "description": "Update many documents"
                        }
                    },
                    "required": ["collection", "filter", "update"]
                }
            ),
            Tool(
                name="aggregate",
                description="Run aggregation pipeline",
                parameters={
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string"},
                        "pipeline": {
                            "type": "array",
                            "description": "Aggregation pipeline stages"
                        }
                    },
                    "required": ["collection", "pipeline"]
                }
            )
        ]
    
    async def startup(self):
        """Connect to MongoDB"""
        self.client = AsyncIOMotorClient(self.connection_string)
        self.db = self.client[self.database_name]
        
        # Test connection
        await self.client.admin.command('ping')
        logger.info(f"Connected to MongoDB database: {self.database_name}")
    
    async def shutdown(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        handlers = {
            "find_documents": self.find_documents,
            "insert_document": self.insert_document,
            "update_documents": self.update_documents,
            "aggregate": self.aggregate
        }
        
        if handler := handlers.get(tool_name):
            return await handler(**arguments)
    
    async def find_documents(self, collection: str, filter: dict = None,
                           projection: dict = None, limit: int = 10,
                           sort: dict = None):
        """Find documents in collection"""
        try:
            coll = self.db[collection]
            
            # Build query
            cursor = coll.find(filter or {})
            
            if projection:
                cursor = cursor.projection(projection)
            
            if sort:
                sort_list = [(k, v) for k, v in sort.items()]
                cursor = cursor.sort(sort_list)
            
            cursor = cursor.limit(limit)
            
            # Execute query
            documents = await cursor.to_list(length=limit)
            
            # Convert to JSON-serializable format
            result = json.loads(json_util.dumps(documents))
            
            return ToolResult(
                success=True,
                data={
                    "collection": collection,
                    "count": len(result),
                    "documents": result
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Find failed: {str(e)}"
            )
    
    async def aggregate(self, collection: str, pipeline: list):
        """Run aggregation pipeline"""
        try:
            coll = self.db[collection]
            
            # Run aggregation
            cursor = coll.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            # Convert to JSON
            result = json.loads(json_util.dumps(results))
            
            return ToolResult(
                success=True,
                data={
                    "collection": collection,
                    "pipeline_stages": len(pipeline),
                    "result_count": len(result),
                    "results": result
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Aggregation failed: {str(e)}"
            )
```

### 3. Redis Server

```python
import aioredis
import json
from mcp import Server, Tool, Resource
from mcp.types import ToolResult, ResourceContent

class RedisServer(Server):
    """Redis cache/database MCP server"""
    
    def __init__(self, redis_url: str = "redis://localhost"):
        super().__init__("redis-server")
        self.redis_url = redis_url
        self.redis = None
        self.setup_capabilities()
    
    def setup_capabilities(self):
        self.tools = [
            Tool(
                name="get_value",
                description="Get value by key",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Redis key"
                        }
                    },
                    "required": ["key"]
                }
            ),
            Tool(
                name="set_value",
                description="Set key-value pair",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {
                            "description": "Value to store"
                        },
                        "expire": {
                            "type": "integer",
                            "description": "Expiration in seconds"
                        }
                    },
                    "required": ["key", "value"]
                }
            ),
            Tool(
                name="delete_keys",
                description="Delete one or more keys",
                parameters={
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keys to delete"
                        }
                    },
                    "required": ["keys"]
                }
            ),
            Tool(
                name="list_operations",
                description="List operations (push, pop, range)",
                parameters={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["lpush", "rpush", "lpop", "rpop", "lrange"]
                        },
                        "key": {"type": "string"},
                        "value": {
                            "description": "Value for push operations"
                        },
                        "start": {
                            "type": "integer",
                            "description": "Start index for lrange"
                        },
                        "stop": {
                            "type": "integer",
                            "description": "Stop index for lrange"
                        }
                    },
                    "required": ["operation", "key"]
                }
            ),
            Tool(
                name="hash_operations",
                description="Hash operations",
                parameters={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["hset", "hget", "hgetall", "hdel"]
                        },
                        "key": {"type": "string"},
                        "field": {
                            "type": "string",
                            "description": "Hash field"
                        },
                        "value": {
                            "description": "Value for hset"
                        }
                    },
                    "required": ["operation", "key"]
                }
            )
        ]
        
        self.resources = [
            Resource(
                uri="redis://info",
                name="Redis Server Info",
                description="Server information and statistics",
                mimeType="application/json"
            ),
            Resource(
                uri="redis://keys",
                name="All Keys",
                description="List all keys in database",
                mimeType="application/json"
            )
        ]
    
    async def startup(self):
        """Connect to Redis"""
        self.redis = await aioredis.create_redis_pool(self.redis_url)
    
    async def shutdown(self):
        """Close Redis connection"""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        if tool_name == "get_value":
            return await self.get_value(**arguments)
        elif tool_name == "set_value":
            return await self.set_value(**arguments)
        elif tool_name == "delete_keys":
            return await self.delete_keys(**arguments)
        elif tool_name == "list_operations":
            return await self.list_operations(**arguments)
        elif tool_name == "hash_operations":
            return await self.hash_operations(**arguments)
    
    async def get_value(self, key: str):
        """Get value from Redis"""
        try:
            value = await self.redis.get(key)
            
            if value is None:
                return ToolResult(
                    success=True,
                    data={
                        "key": key,
                        "exists": False,
                        "value": None
                    }
                )
            
            # Try to decode as JSON
            try:
                decoded_value = json.loads(value)
            except:
                decoded_value = value.decode('utf-8')
            
            return ToolResult(
                success=True,
                data={
                    "key": key,
                    "exists": True,
                    "value": decoded_value,
                    "type": await self.redis.type(key)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Get failed: {str(e)}"
            )
    
    async def set_value(self, key: str, value, expire: int = None):
        """Set value in Redis"""
        try:
            # Serialize value if needed
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            # Set value
            if expire:
                await self.redis.setex(key, expire, value)
            else:
                await self.redis.set(key, value)
            
            return ToolResult(
                success=True,
                data={
                    "key": key,
                    "value": value,
                    "expire": expire,
                    "operation": "set"
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Set failed: {str(e)}"
            )
```

## API Integration Servers

### 1. REST API Gateway Server

```python
import httpx
from urllib.parse import urljoin
from mcp import Server, Tool
from mcp.types import ToolResult

class RestAPIGatewayServer(Server):
    """Generic REST API gateway MCP server"""
    
    def __init__(self, api_configs: dict):
        super().__init__("rest-api-gateway")
        self.api_configs = api_configs
        self.clients = {}
        self.setup_tools()
    
    def setup_tools(self):
        self.tools = []
        
        # Generate tools for each API
        for api_name, config in self.api_configs.items():
            for endpoint in config.get('endpoints', []):
                tool = Tool(
                    name=f"{api_name}_{endpoint['name']}",
                    description=endpoint['description'],
                    parameters=endpoint.get('parameters', {
                        "type": "object",
                        "properties": {}
                    })
                )
                self.tools.append(tool)
    
    async def startup(self):
        """Initialize HTTP clients"""
        for api_name, config in self.api_configs.items():
            self.clients[api_name] = httpx.AsyncClient(
                base_url=config['base_url'],
                headers=config.get('headers', {}),
                timeout=httpx.Timeout(30.0)
            )
    
    async def shutdown(self):
        """Close HTTP clients"""
        for client in self.clients.values():
            await client.aclose()
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        # Parse tool name
        parts = tool_name.split('_', 1)
        if len(parts) != 2:
            return ToolResult(
                success=False,
                error="Invalid tool name format"
            )
        
        api_name, endpoint_name = parts
        
        # Find endpoint config
        api_config = self.api_configs.get(api_name)
        if not api_config:
            return ToolResult(
                success=False,
                error=f"Unknown API: {api_name}"
            )
        
        endpoint = next(
            (e for e in api_config['endpoints'] if e['name'] == endpoint_name),
            None
        )
        
        if not endpoint:
            return ToolResult(
                success=False,
                error=f"Unknown endpoint: {endpoint_name}"
            )
        
        # Make API call
        return await self.call_api(api_name, endpoint, arguments)
    
    async def call_api(self, api_name: str, endpoint: dict, arguments: dict):
        """Execute API call"""
        try:
            client = self.clients[api_name]
            
            # Build request
            method = endpoint.get('method', 'GET')
            path = endpoint['path']
            
            # Substitute path parameters
            for key, value in arguments.items():
                if f"{{{key}}}" in path:
                    path = path.replace(f"{{{key}}}", str(value))
                    del arguments[key]
            
            # Prepare request parameters
            request_args = {
                'method': method,
                'url': path
            }
            
            if method in ['GET', 'DELETE']:
                request_args['params'] = arguments
            else:
                request_args['json'] = arguments
            
            # Execute request
            response = await client.request(**request_args)
            
            # Handle response
            if response.status_code >= 200 and response.status_code < 300:
                return ToolResult(
                    success=True,
                    data={
                        'status_code': response.status_code,
                        'response': response.json() if response.text else None
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"API error: {response.status_code} - {response.text}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"API call failed: {str(e)}"
            )

# Example configuration
api_configs = {
    "github": {
        "base_url": "https://api.github.com",
        "headers": {
            "Authorization": "token YOUR_GITHUB_TOKEN",
            "Accept": "application/vnd.github.v3+json"
        },
        "endpoints": [
            {
                "name": "get_user",
                "description": "Get GitHub user information",
                "method": "GET",
                "path": "/users/{username}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "username": {
                            "type": "string",
                            "description": "GitHub username"
                        }
                    },
                    "required": ["username"]
                }
            },
            {
                "name": "create_issue",
                "description": "Create GitHub issue",
                "method": "POST",
                "path": "/repos/{owner}/{repo}/issues",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["owner", "repo", "title"]
                }
            }
        ]
    }
}
```

### 2. GraphQL Server

```python
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from mcp import Server, Tool
from mcp.types import ToolResult

class GraphQLServer(Server):
    """GraphQL API MCP server"""
    
    def __init__(self, endpoint: str, headers: dict = None):
        super().__init__("graphql-server")
        self.endpoint = endpoint
        self.headers = headers or {}
        self.client = None
        self.schema = None
        
    async def startup(self):
        """Initialize GraphQL client"""
        transport = AIOHTTPTransport(
            url=self.endpoint,
            headers=self.headers
        )
        
        self.client = Client(
            transport=transport,
            fetch_schema_from_transport=True
        )
        
        # Introspect schema
        await self.introspect_schema()
    
    async def introspect_schema(self):
        """Discover GraphQL schema"""
        introspection_query = gql("""
            query IntrospectionQuery {
                __schema {
                    queryType { name }
                    mutationType { name }
                    types {
                        name
                        kind
                        fields {
                            name
                            description
                            args {
                                name
                                type {
                                    name
                                    kind
                                }
                            }
                        }
                    }
                }
            }
        """)
        
        result = await self.client.execute_async(introspection_query)
        self.schema = result['__schema']
        
        # Generate tools from schema
        self.generate_tools_from_schema()
    
    def generate_tools_from_schema(self):
        """Create MCP tools from GraphQL schema"""
        self.tools = []
        
        # Find Query and Mutation types
        for type_def in self.schema['types']:
            if type_def['name'] == self.schema['queryType']['name']:
                # Create tools for queries (as resources)
                for field in type_def['fields'] or []:
                    if not field['name'].startswith('__'):
                        self.create_query_tool(field)
                        
            elif type_def['name'] == self.schema['mutationType']['name']:
                # Create tools for mutations
                for field in type_def['fields'] or []:
                    self.create_mutation_tool(field)
    
    def create_mutation_tool(self, field: dict):
        """Create tool for GraphQL mutation"""
        # Build parameter schema
        properties = {}
        required = []
        
        for arg in field.get('args', []):
            properties[arg['name']] = {
                "type": self.graphql_to_json_type(arg['type'])
            }
            
            # Simple required detection
            if arg['type'].get('kind') == 'NON_NULL':
                required.append(arg['name'])
        
        tool = Tool(
            name=f"mutation_{field['name']}",
            description=field.get('description', f"Execute {field['name']} mutation"),
            parameters={
                "type": "object",
                "properties": properties,
                "required": required
            }
        )
        
        self.tools.append(tool)
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        """Execute GraphQL operation"""
        if tool_name.startswith('mutation_'):
            operation_name = tool_name[9:]  # Remove 'mutation_' prefix
            return await self.execute_mutation(operation_name, arguments)
        elif tool_name.startswith('query_'):
            operation_name = tool_name[6:]  # Remove 'query_' prefix
            return await self.execute_query(operation_name, arguments)
    
    async def execute_mutation(self, mutation_name: str, variables: dict):
        """Execute GraphQL mutation"""
        try:
            # Build dynamic mutation
            mutation_str = self.build_mutation_string(mutation_name, variables)
            mutation = gql(mutation_str)
            
            # Execute
            result = await self.client.execute_async(
                mutation,
                variable_values=variables
            )
            
            return ToolResult(
                success=True,
                data={
                    "mutation": mutation_name,
                    "result": result.get(mutation_name)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"GraphQL mutation failed: {str(e)}"
            )
```

## File System Servers

### 1. File Manager Server

```python
import os
import aiofiles
import mimetypes
from pathlib import Path
from mcp import Server, Tool, Resource, ResourceTemplate
from mcp.types import ToolResult, ResourceContent

class FileManagerServer(Server):
    """File system management MCP server"""
    
    def __init__(self, allowed_paths: list):
        super().__init__("file-manager-server")
        self.allowed_paths = [Path(p).resolve() for p in allowed_paths]
        self.setup_capabilities()
    
    def setup_capabilities(self):
        self.tools = [
            Tool(
                name="create_file",
                description="Create new file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path"
                        },
                        "content": {
                            "type": "string",
                            "description": "File content"
                        }
                    },
                    "required": ["path", "content"]
                }
            ),
            Tool(
                name="delete_file",
                description="Delete file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            ),
            Tool(
                name="move_file",
                description="Move or rename file",
                parameters={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "destination": {"type": "string"}
                    },
                    "required": ["source", "destination"]
                }
            ),
            Tool(
                name="create_directory",
                description="Create directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {
                            "type": "boolean",
                            "default": True
                        }
                    },
                    "required": ["path"]
                }
            )
        ]
        
        self.resource_templates = [
            ResourceTemplate(
                uriTemplate="file:///{path}",
                name="File Content",
                description="Read file content",
                mimeType="*/*"
            ),
            ResourceTemplate(
                uriTemplate="dir:///{path}",
                name="Directory Listing",
                description="List directory contents",
                mimeType="application/json"
            )
        ]
    
    def is_path_allowed(self, path: Path) -> bool:
        """Check if path is within allowed directories"""
        try:
            resolved = path.resolve()
            return any(
                resolved.is_relative_to(allowed)
                for allowed in self.allowed_paths
            )
        except:
            return False
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        handlers = {
            "create_file": self.create_file,
            "delete_file": self.delete_file,
            "move_file": self.move_file,
            "create_directory": self.create_directory
        }
        
        if handler := handlers.get(tool_name):
            return await handler(**arguments)
    
    async def create_file(self, path: str, content: str):
        """Create new file with content"""
        try:
            file_path = Path(path)
            
            # Security check
            if not self.is_path_allowed(file_path):
                return ToolResult(
                    success=False,
                    error="Access denied: Path outside allowed directories"
                )
            
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(content)
            
            return ToolResult(
                success=True,
                data={
                    "path": str(file_path),
                    "size": len(content),
                    "created": True
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to create file: {str(e)}"
            )
    
    async def handle_resource_read(self, uri: str):
        """Handle file/directory reading"""
        import re
        
        # Parse file URI
        if match := re.match(r"file:///(.+)", uri):
            path = Path(match.group(1))
            if self.is_path_allowed(path) and path.is_file():
                return await self.read_file_content(path)
        
        # Parse directory URI
        elif match := re.match(r"dir:///(.+)", uri):
            path = Path(match.group(1))
            if self.is_path_allowed(path) and path.is_dir():
                return await self.list_directory(path)
    
    async def read_file_content(self, path: Path):
        """Read file content"""
        try:
            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(str(path))
            
            # Read file
            if mime_type and mime_type.startswith('text'):
                async with aiofiles.open(path, 'r') as f:
                    content = await f.read()
            else:
                async with aiofiles.open(path, 'rb') as f:
                    content = await f.read()
                    # Base64 encode binary files
                    import base64
                    content = base64.b64encode(content).decode()
            
            return ResourceContent(
                content=content,
                mimeType=mime_type or 'application/octet-stream'
            )
            
        except Exception as e:
            return None
    
    async def list_directory(self, path: Path):
        """List directory contents"""
        try:
            items = []
            
            for item in path.iterdir():
                stat = item.stat()
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else None,
                    "modified": stat.st_mtime,
                    "permissions": oct(stat.st_mode)[-3:]
                })
            
            # Sort directories first, then files
            items.sort(key=lambda x: (x['type'] != 'directory', x['name']))
            
            return ResourceContent(
                content=json.dumps({
                    "path": str(path),
                    "items": items,
                    "count": len(items)
                }, indent=2),
                mimeType="application/json"
            )
            
        except Exception as e:
            return None
```

### 2. Code Analysis Server

```python
import ast
import subprocess
from pathlib import Path
from mcp import Server, Tool
from mcp.types import ToolResult

class CodeAnalysisServer(Server):
    """Code analysis and refactoring MCP server"""
    
    def __init__(self, project_root: str):
        super().__init__("code-analysis-server")
        self.project_root = Path(project_root)
        self.setup_tools()
    
    def setup_tools(self):
        self.tools = [
            Tool(
                name="analyze_python_file",
                description="Analyze Python file for issues",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to Python file"
                        },
                        "checks": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["complexity", "style", "security", "performance"]
                            },
                            "default": ["complexity", "style"]
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="find_code_pattern",
                description="Search for code patterns",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Code pattern to search"
                        },
                        "file_extension": {
                            "type": "string",
                            "default": ".py"
                        },
                        "regex": {
                            "type": "boolean",
                            "default": False
                        }
                    },
                    "required": ["pattern"]
                }
            ),
            Tool(
                name="calculate_metrics",
                description="Calculate code metrics",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File or directory path"
                        }
                    },
                    "required": ["path"]
                }
            ),
            Tool(
                name="run_linter",
                description="Run code linter",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "linter": {
                            "type": "string",
                            "enum": ["pylint", "flake8", "black", "mypy"],
                            "default": "flake8"
                        },
                        "fix": {
                            "type": "boolean",
                            "default": False,
                            "description": "Auto-fix issues if possible"
                        }
                    },
                    "required": ["path"]
                }
            )
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        handlers = {
            "analyze_python_file": self.analyze_python_file,
            "find_code_pattern": self.find_code_pattern,
            "calculate_metrics": self.calculate_metrics,
            "run_linter": self.run_linter
        }
        
        if handler := handlers.get(tool_name):
            return await handler(**arguments)
    
    async def analyze_python_file(self, file_path: str, 
                                 checks: list = None):
        """Analyze Python file for various issues"""
        try:
            path = self.project_root / file_path
            
            if not path.exists() or not path.is_file():
                return ToolResult(
                    success=False,
                    error="File not found"
                )
            
            with open(path, 'r') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return ToolResult(
                    success=False,
                    error=f"Syntax error: {e}"
                )
            
            results = {}
            
            if not checks or "complexity" in checks:
                results["complexity"] = self.analyze_complexity(tree)
            
            if not checks or "style" in checks:
                results["style"] = await self.check_style(path)
            
            if not checks or "security" in checks:
                results["security"] = await self.check_security(path)
            
            if not checks or "performance" in checks:
                results["performance"] = self.analyze_performance(tree)
            
            return ToolResult(
                success=True,
                data={
                    "file": file_path,
                    "analysis": results
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Analysis failed: {str(e)}"
            )
    
    def analyze_complexity(self, tree: ast.AST):
        """Calculate cyclomatic complexity"""
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1
                self.functions = {}
                self.current_function = None
            
            def visit_FunctionDef(self, node):
                parent = self.current_function
                self.current_function = node.name
                self.functions[node.name] = 1
                self.generic_visit(node)
                self.current_function = parent
            
            def visit_If(self, node):
                if self.current_function:
                    self.functions[self.current_function] += 1
                else:
                    self.complexity += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                if self.current_function:
                    self.functions[self.current_function] += 1
                else:
                    self.complexity += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                if self.current_function:
                    self.functions[self.current_function] += 1
                else:
                    self.complexity += 1
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        return {
            "file_complexity": visitor.complexity,
            "function_complexities": visitor.functions,
            "max_complexity": max(visitor.functions.values()) if visitor.functions else visitor.complexity
        }
    
    async def run_linter(self, path: str, linter: str = "flake8", 
                        fix: bool = False):
        """Run code linter on file or directory"""
        try:
            full_path = self.project_root / path
            
            # Build command
            if linter == "flake8":
                cmd = ["flake8", str(full_path)]
            elif linter == "pylint":
                cmd = ["pylint", str(full_path)]
            elif linter == "black":
                cmd = ["black", str(full_path)]
                if not fix:
                    cmd.append("--check")
            elif linter == "mypy":
                cmd = ["mypy", str(full_path)]
            
            # Run linter
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            return ToolResult(
                success=result.returncode == 0,
                data={
                    "linter": linter,
                    "path": path,
                    "output": result.stdout,
                    "errors": result.stderr,
                    "fixed": fix and linter == "black" and result.returncode == 0
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Linter failed: {str(e)}"
            )
```

## DevOps & Infrastructure

### 1. Kubernetes Manager Server

```python
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from mcp import Server, Tool, Resource
from mcp.types import ToolResult, ResourceContent

class KubernetesServer(Server):
    """Kubernetes cluster management MCP server"""
    
    def __init__(self, kubeconfig_path: str = None):
        super().__init__("kubernetes-server")
        self.kubeconfig_path = kubeconfig_path
        self.v1 = None
        self.apps_v1 = None
        self.setup_tools()
    
    def setup_tools(self):
        self.tools = [
            Tool(
                name="list_pods",
                description="List pods in namespace",
                parameters={
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "default": "default"
                        },
                        "label_selector": {
                            "type": "string",
                            "description": "Label selector (e.g., 'app=nginx')"
                        }
                    }
                }
            ),
            Tool(
                name="get_pod_logs",
                description="Get pod logs",
                parameters={
                    "type": "object",
                    "properties": {
                        "pod_name": {"type": "string"},
                        "namespace": {
                            "type": "string",
                            "default": "default"
                        },
                        "container": {
                            "type": "string",
                            "description": "Container name if multiple"
                        },
                        "tail_lines": {
                            "type": "integer",
                            "default": 100
                        }
                    },
                    "required": ["pod_name"]
                }
            ),
            Tool(
                name="scale_deployment",
                description="Scale deployment replicas",
                parameters={
                    "type": "object",
                    "properties": {
                        "deployment_name": {"type": "string"},
                        "namespace": {
                            "type": "string",
                            "default": "default"
                        },
                        "replicas": {
                            "type": "integer",
                            "minimum": 0
                        }
                    },
                    "required": ["deployment_name", "replicas"]
                }
            ),
            Tool(
                name="restart_deployment",
                description="Restart deployment pods",
                parameters={
                    "type": "object",
                    "properties": {
                        "deployment_name": {"type": "string"},
                        "namespace": {
                            "type": "string",
                            "default": "default"
                        }
                    },
                    "required": ["deployment_name"]
                }
            ),
            Tool(
                name="apply_manifest",
                description="Apply Kubernetes manifest",
                parameters={
                    "type": "object",
                    "properties": {
                        "manifest": {
                            "type": "object",
                            "description": "Kubernetes manifest object"
                        }
                    },
                    "required": ["manifest"]
                }
            )
        ]
        
        self.resources = [
            Resource(
                uri="k8s://cluster/info",
                name="Cluster Info",
                description="Kubernetes cluster information",
                mimeType="application/json"
            ),
            Resource(
                uri="k8s://nodes",
                name="Node List",
                description="List of cluster nodes",
                mimeType="application/json"
            )
        ]
    
    async def startup(self):
        """Initialize Kubernetes client"""
        if self.kubeconfig_path:
            config.load_kube_config(config_file=self.kubeconfig_path)
        else:
            config.load_incluster_config()
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        handlers = {
            "list_pods": self.list_pods,
            "get_pod_logs": self.get_pod_logs,
            "scale_deployment": self.scale_deployment,
            "restart_deployment": self.restart_deployment,
            "apply_manifest": self.apply_manifest
        }
        
        if handler := handlers.get(tool_name):
            return await handler(**arguments)
    
    async def list_pods(self, namespace: str = "default", 
                       label_selector: str = None):
        """List pods in namespace"""
        try:
            if label_selector:
                pods = self.v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=label_selector
                )
            else:
                pods = self.v1.list_namespaced_pod(namespace=namespace)
            
            pod_list = []
            for pod in pods.items:
                pod_info = {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "status": pod.status.phase,
                    "ready": all(c.ready for c in pod.status.container_statuses or []),
                    "restarts": sum(c.restart_count for c in pod.status.container_statuses or []),
                    "age": self.calculate_age(pod.metadata.creation_timestamp),
                    "node": pod.spec.node_name
                }
                pod_list.append(pod_info)
            
            return ToolResult(
                success=True,
                data={
                    "namespace": namespace,
                    "pod_count": len(pod_list),
                    "pods": pod_list
                }
            )
            
        except ApiException as e:
            return ToolResult(
                success=False,
                error=f"Kubernetes API error: {e.reason}"
            )
    
    async def scale_deployment(self, deployment_name: str, replicas: int,
                              namespace: str = "default"):
        """Scale deployment to specified replicas"""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update replicas
            deployment.spec.replicas = replicas
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            return ToolResult(
                success=True,
                data={
                    "deployment": deployment_name,
                    "namespace": namespace,
                    "replicas": replicas,
                    "previous_replicas": deployment.spec.replicas
                }
            )
            
        except ApiException as e:
            return ToolResult(
                success=False,
                error=f"Failed to scale deployment: {e.reason}"
            )
    
    def calculate_age(self, timestamp):
        """Calculate age from timestamp"""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        age = now - timestamp
        
        if age.days > 0:
            return f"{age.days}d"
        elif age.seconds > 3600:
            return f"{age.seconds // 3600}h"
        elif age.seconds > 60:
            return f"{age.seconds // 60}m"
        else:
            return f"{age.seconds}s"
```

### 2. Docker Manager Server

```python
import docker
from docker.errors import DockerException, ImageNotFound, ContainerError
from mcp import Server, Tool
from mcp.types import ToolResult

class DockerServer(Server):
    """Docker container management MCP server"""
    
    def __init__(self):
        super().__init__("docker-server")
        self.client = None
        self.setup_tools()
    
    def setup_tools(self):
        self.tools = [
            Tool(
                name="list_containers",
                description="List Docker containers",
                parameters={
                    "type": "object",
                    "properties": {
                        "all": {
                            "type": "boolean",
                            "default": False,
                            "description": "Show all containers (including stopped)"
                        },
                        "filters": {
                            "type": "object",
                            "description": "Filter containers"
                        }
                    }
                }
            ),
            Tool(
                name="manage_container",
                description="Start, stop, or restart container",
                parameters={
                    "type": "object",
                    "properties": {
                        "container_id": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": ["start", "stop", "restart", "remove"]
                        }
                    },
                    "required": ["container_id", "action"]
                }
            ),
            Tool(
                name="run_container",
                description="Run new container",
                parameters={
                    "type": "object",
                    "properties": {
                        "image": {"type": "string"},
                        "name": {"type": "string"},
                        "command": {"type": "string"},
                        "environment": {
                            "type": "object",
                            "description": "Environment variables"
                        },
                        "ports": {
                            "type": "object",
                            "description": "Port mappings"
                        },
                        "volumes": {
                            "type": "object",
                            "description": "Volume mappings"
                        },
                        "detach": {
                            "type": "boolean",
                            "default": True
                        }
                    },
                    "required": ["image"]
                }
            ),
            Tool(
                name="get_container_logs",
                description="Get container logs",
                parameters={
                    "type": "object",
                    "properties": {
                        "container_id": {"type": "string"},
                        "tail": {
                            "type": "integer",
                            "default": 100
                        },
                        "follow": {
                            "type": "boolean",
                            "default": False
                        }
                    },
                    "required": ["container_id"]
                }
            ),
            Tool(
                name="build_image",
                description="Build Docker image",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to Dockerfile directory"
                        },
                        "tag": {"type": "string"},
                        "dockerfile": {
                            "type": "string",
                            "default": "Dockerfile"
                        },
                        "buildargs": {
                            "type": "object",
                            "description": "Build arguments"
                        }
                    },
                    "required": ["path", "tag"]
                }
            )
        ]
    
    async def startup(self):
        """Initialize Docker client"""
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
        except DockerException as e:
            logger.error(f"Failed to connect to Docker: {e}")
            raise
    
    async def shutdown(self):
        """Close Docker client"""
        if self.client:
            self.client.close()
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        handlers = {
            "list_containers": self.list_containers,
            "manage_container": self.manage_container,
            "run_container": self.run_container,
            "get_container_logs": self.get_container_logs,
            "build_image": self.build_image
        }
        
        if handler := handlers.get(tool_name):
            return await handler(**arguments)
    
    async def list_containers(self, all: bool = False, filters: dict = None):
        """List Docker containers"""
        try:
            containers = self.client.containers.list(all=all, filters=filters)
            
            container_list = []
            for container in containers:
                container_info = {
                    "id": container.short_id,
                    "name": container.name,
                    "image": container.image.tags[0] if container.image.tags else container.image.short_id,
                    "status": container.status,
                    "created": container.attrs['Created'],
                    "ports": container.ports
                }
                container_list.append(container_info)
            
            return ToolResult(
                success=True,
                data={
                    "count": len(container_list),
                    "containers": container_list
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to list containers: {str(e)}"
            )
    
    async def run_container(self, image: str, name: str = None,
                           command: str = None, environment: dict = None,
                           ports: dict = None, volumes: dict = None,
                           detach: bool = True):
        """Run new Docker container"""
        try:
            # Pull image if not exists
            try:
                self.client.images.get(image)
            except ImageNotFound:
                logger.info(f"Pulling image: {image}")
                self.client.images.pull(image)
            
            # Run container
            container = self.client.containers.run(
                image=image,
                name=name,
                command=command,
                environment=environment,
                ports=ports,
                volumes=volumes,
                detach=detach
            )
            
            return ToolResult(
                success=True,
                data={
                    "container_id": container.short_id,
                    "name": container.name,
                    "status": container.status,
                    "image": image
                }
            )
            
        except ContainerError as e:
            return ToolResult(
                success=False,
                error=f"Container error: {e.explanation}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to run container: {str(e)}"
            )
```

## AI & ML Servers

### 1. Model Inference Server

```python
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import io
import base64
from mcp import Server, Tool
from mcp.types import ToolResult

class MLInferenceServer(Server):
    """Machine Learning model inference MCP server"""
    
    def __init__(self):
        super().__init__("ml-inference-server")
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_tools()
    
    def setup_tools(self):
        self.tools = [
            Tool(
                name="text_classification",
                description="Classify text sentiment or category",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to classify"
                        },
                        "model": {
                            "type": "string",
                            "enum": ["sentiment", "emotion", "topic"],
                            "default": "sentiment"
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="image_classification",
                description="Classify image content",
                parameters={
                    "type": "object",
                    "properties": {
                        "image_base64": {
                            "type": "string",
                            "description": "Base64 encoded image"
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 5,
                            "description": "Number of top predictions"
                        }
                    },
                    "required": ["image_base64"]
                }
            ),
            Tool(
                name="text_generation",
                description="Generate text continuation",
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "max_length": {
                            "type": "integer",
                            "default": 100
                        },
                        "temperature": {
                            "type": "number",
                            "default": 0.7
                        }
                    },
                    "required": ["prompt"]
                }
            ),
            Tool(
                name="named_entity_recognition",
                description="Extract entities from text",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"}
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="question_answering",
                description="Answer questions about context",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "context": {
                            "type": "string",
                            "description": "Context containing the answer"
                        }
                    },
                    "required": ["question", "context"]
                }
            )
        ]
    
    async def startup(self):
        """Load ML models"""
        logger.info(f"Loading models on {self.device}")
        
        # Load models lazily to save memory
        self.model_configs = {
            "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
            "emotion": "j-hartmann/emotion-english-distilroberta-base",
            "image_classification": "google/vit-base-patch16-224",
            "text_generation": "gpt2",
            "ner": "dslim/bert-base-NER",
            "qa": "distilbert-base-cased-distilled-squad"
        }
    
    def load_model(self, task: str, model_name: str):
        """Lazy load model when needed"""
        if task not in self.models:
            logger.info(f"Loading {task} model: {model_name}")
            self.models[task] = pipeline(task, model=model_name, device=self.device)
        return self.models[task]
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        handlers = {
            "text_classification": self.text_classification,
            "image_classification": self.image_classification,
            "text_generation": self.text_generation,
            "named_entity_recognition": self.named_entity_recognition,
            "question_answering": self.question_answering
        }
        
        if handler := handlers.get(tool_name):
            return await handler(**arguments)
    
    async def text_classification(self, text: str, model: str = "sentiment"):
        """Classify text using specified model"""
        try:
            # Map model type to task and model name
            if model == "sentiment":
                task = "sentiment-analysis"
                model_name = self.model_configs["sentiment"]
            elif model == "emotion":
                task = "text-classification"
                model_name = self.model_configs["emotion"]
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown model type: {model}"
                )
            
            # Load and run model
            classifier = self.load_model(task, model_name)
            results = classifier(text)
            
            return ToolResult(
                success=True,
                data={
                    "text": text,
                    "model": model,
                    "predictions": results
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Classification failed: {str(e)}"
            )
    
    async def image_classification(self, image_base64: str, top_k: int = 5):
        """Classify image content"""
        try:
            # Decode image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Load model
            classifier = self.load_model(
                "image-classification",
                self.model_configs["image_classification"]
            )
            
            # Run inference
            results = classifier(image, top_k=top_k)
            
            return ToolResult(
                success=True,
                data={
                    "predictions": results,
                    "top_prediction": results[0] if results else None
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Image classification failed: {str(e)}"
            )
```

### 2. Data Science Server

```python
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from mcp import Server, Tool
from mcp.types import ToolResult

class DataScienceServer(Server):
    """Data science and analysis MCP server"""
    
    def __init__(self):
        super().__init__("datascience-server")
        self.datasets = {}
        self.models = {}
        self.setup_tools()
    
    def setup_tools(self):
        self.tools = [
            Tool(
                name="load_dataset",
                description="Load dataset from file or URL",
                parameters={
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "File path or URL"
                        },
                        "name": {
                            "type": "string",
                            "description": "Dataset name for reference"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["csv", "json", "excel", "parquet"],
                            "default": "csv"
                        }
                    },
                    "required": ["source", "name"]
                }
            ),
            Tool(
                name="explore_data",
                description="Explore dataset statistics",
                parameters={
                    "type": "object",
                    "properties": {
                        "dataset_name": {"type": "string"},
                        "include_plots": {
                            "type": "boolean",
                            "default": True
                        }
                    },
                    "required": ["dataset_name"]
                }
            ),
            Tool(
                name="preprocess_data",
                description="Preprocess dataset",
                parameters={
                    "type": "object",
                    "properties": {
                        "dataset_name": {"type": "string"},
                        "operations": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "remove_nulls",
                                    "fill_nulls",
                                    "scale",
                                    "encode_categorical",
                                    "remove_outliers"
                                ]
                            }
                        }
                    },
                    "required": ["dataset_name", "operations"]
                }
            ),
            Tool(
                name="train_model",
                description="Train machine learning model",
                parameters={
                    "type": "object",
                    "properties": {
                        "dataset_name": {"type": "string"},
                        "model_type": {
                            "type": "string",
                            "enum": [
                                "linear_regression",
                                "logistic_regression",
                                "random_forest",
                                "xgboost",
                                "neural_network"
                            ]
                        },
                        "target_column": {"type": "string"},
                        "model_name": {
                            "type": "string",
                            "description": "Name to save model as"
                        },
                        "test_size": {
                            "type": "number",
                            "default": 0.2
                        }
                    },
                    "required": ["dataset_name", "model_type", "target_column", "model_name"]
                }
            ),
            Tool(
                name="predict",
                description="Make predictions with trained model",
                parameters={
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string"},
                        "data": {
                            "type": "array",
                            "description": "Input data for prediction"
                        }
                    },
                    "required": ["model_name", "data"]
                }
            )
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        handlers = {
            "load_dataset": self.load_dataset,
            "explore_data": self.explore_data,
            "preprocess_data": self.preprocess_data,
            "train_model": self.train_model,
            "predict": self.predict
        }
        
        if handler := handlers.get(tool_name):
            return await handler(**arguments)
    
    async def load_dataset(self, source: str, name: str, format: str = "csv"):
        """Load dataset from various sources"""
        try:
            # Load based on format
            if format == "csv":
                df = pd.read_csv(source)
            elif format == "json":
                df = pd.read_json(source)
            elif format == "excel":
                df = pd.read_excel(source)
            elif format == "parquet":
                df = pd.read_parquet(source)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported format: {format}"
                )
            
            # Store dataset
            self.datasets[name] = df
            
            # Basic info
            info = {
                "name": name,
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
                "null_counts": df.isnull().sum().to_dict()
            }
            
            return ToolResult(
                success=True,
                data={
                    "dataset_info": info,
                    "head": df.head().to_dict('records')
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to load dataset: {str(e)}"
            )
    
    async def explore_data(self, dataset_name: str, include_plots: bool = True):
        """Explore dataset with statistics and visualizations"""
        try:
            if dataset_name not in self.datasets:
                return ToolResult(
                    success=False,
                    error=f"Dataset '{dataset_name}' not found"
                )
            
            df = self.datasets[dataset_name]
            
            # Statistical summary
            stats = {
                "shape": df.shape,
                "numeric_summary": df.describe().to_dict(),
                "categorical_summary": {},
                "correlations": df.select_dtypes(include=[np.number]).corr().to_dict()
            }
            
            # Categorical summary
            for col in df.select_dtypes(include=['object']).columns:
                stats["categorical_summary"][col] = {
                    "unique_values": df[col].nunique(),
                    "top_values": df[col].value_counts().head(10).to_dict()
                }
            
            plots = []
            if include_plots:
                # Distribution plots for numeric columns
                for col in df.select_dtypes(include=[np.number]).columns[:5]:
                    plt.figure(figsize=(10, 6))
                    df[col].hist(bins=30)
                    plt.title(f"Distribution of {col}")
                    plt.xlabel(col)
                    plt.ylabel("Frequency")
                    
                    # Save to base64
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plot_base64 = base64.b64encode(buf.read()).decode()
                    plots.append({
                        "type": "histogram",
                        "column": col,
                        "image": plot_base64
                    })
                    plt.close()
                
                # Correlation heatmap
                if len(df.select_dtypes(include=[np.number]).columns) > 1:
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(
                        df.select_dtypes(include=[np.number]).corr(),
                        annot=True,
                        cmap='coolwarm',
                        center=0
                    )
                    plt.title("Correlation Heatmap")
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plot_base64 = base64.b64encode(buf.read()).decode()
                    plots.append({
                        "type": "correlation_heatmap",
                        "image": plot_base64
                    })
                    plt.close()
            
            return ToolResult(
                success=True,
                data={
                    "statistics": stats,
                    "plots": plots
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Exploration failed: {str(e)}"
            )
```

## Testing & Debugging

### Test Server Implementation

```python
import pytest
import asyncio
from mcp import Server, Tool
from mcp.types import ToolResult

class TestMCPServer:
    """Test suite for MCP servers"""
    
    @pytest.fixture
    async def server(self):
        """Create test server instance"""
        server = BasicMCPServer()
        await server.startup()
        yield server
        await server.shutdown()
    
    @pytest.mark.asyncio
    async def test_tool_call(self, server):
        """Test basic tool call"""
        result = await server.handle_tool_call(
            "test_tool",
            {"param": "value"}
        )
        
        assert result.success
        assert result.data["param"] == "value"
    
    @pytest.mark.asyncio
    async def test_resource_read(self, server):
        """Test resource reading"""
        content = await server.handle_resource_read("test://resource")
        
        assert content is not None
        assert content.mimeType == "application/json"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, server):
        """Test error handling"""
        result = await server.handle_tool_call(
            "error_tool",
            {"cause_error": True}
        )
        
        assert not result.success
        assert "error" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, server):
        """Test handling concurrent requests"""
        tasks = []
        for i in range(10):
            task = server.handle_tool_call(
                "concurrent_tool",
                {"request_id": i}
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert all(r.success for r in results)
        assert len(set(r.data["request_id"] for r in results)) == 10

# Debug server with extensive logging
class DebugMCPServer(Server):
    """MCP server with debug features"""
    
    def __init__(self, debug_level: str = "DEBUG"):
        super().__init__("debug-server")
        self.setup_debugging(debug_level)
        self.request_count = 0
        self.error_count = 0
        
    def setup_debugging(self, level: str):
        import logging
        import sys
        
        # Configure detailed logging
        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('mcp_debug.log')
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        """Tool call with detailed logging"""
        self.request_count += 1
        request_id = f"req_{self.request_count}"
        
        self.logger.info(f"[{request_id}] Tool call: {tool_name}")
        self.logger.debug(f"[{request_id}] Arguments: {arguments}")
        
        try:
            # Time the execution
            import time
            start_time = time.time()
            
            result = await super().handle_tool_call(tool_name, arguments)
            
            elapsed = time.time() - start_time
            self.logger.info(f"[{request_id}] Completed in {elapsed:.3f}s")
            
            if not result.success:
                self.error_count += 1
                self.logger.error(f"[{request_id}] Error: {result.error}")
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.logger.exception(f"[{request_id}] Unhandled exception")
            
            return ToolResult(
                success=False,
                error=f"Internal error: {str(e)}"
            )
```

## Deployment Guide

### Docker Deployment

```dockerfile
# Dockerfile for MCP Server
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 mcpuser && chown -R mcpuser:mcpuser /app
USER mcpuser

# Run server
CMD ["python", "server.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  mcp-server:
    build: .
    container_name: mcp-server
    environment:
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://user:pass@postgres:5432/db
      - REDIS_URL=redis://redis:6379
    ports:
      - "8080:8080"
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: db
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  labels:
    app: mcp-server
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
        - name: LOG_LEVEL
          value: "INFO"
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
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
spec:
  selector:
    app: mcp-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Performance Tuning

### Optimization Strategies

```python
class OptimizedMCPServer(Server):
    """Performance-optimized MCP server"""
    
    def __init__(self):
        super().__init__("optimized-server")
        self.setup_optimizations()
    
    def setup_optimizations(self):
        # 1. Connection pooling
        self.connection_pool = ConnectionPool(
            min_connections=10,
            max_connections=100,
            connection_timeout=5.0
        )
        
        # 2. Caching
        self.cache = CacheManager(
            backend="redis",
            default_ttl=300,
            max_entries=10000
        )
        
        # 3. Request batching
        self.batch_processor = BatchProcessor(
            batch_size=50,
            batch_timeout=0.1  # 100ms
        )
        
        # 4. Resource limits
        self.rate_limiter = RateLimiter(
            requests_per_minute=1000,
            burst_size=100
        )
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        """Optimized tool call handling"""
        # Check cache
        cache_key = f"{tool_name}:{hash(str(arguments))}"
        if cached := await self.cache.get(cache_key):
            return cached
        
        # Rate limiting
        if not await self.rate_limiter.allow_request():
            return ToolResult(
                success=False,
                error="Rate limit exceeded"
            )
        
        # Process request
        result = await self.process_with_optimizations(tool_name, arguments)
        
        # Cache successful results
        if result.success:
            await self.cache.set(cache_key, result)
        
        return result
    
    async def process_with_optimizations(self, tool_name: str, arguments: dict):
        """Process with various optimizations"""
        # Use connection from pool
        async with self.connection_pool.acquire() as conn:
            # Batch small requests
            if self.is_batchable(tool_name):
                return await self.batch_processor.add_request(
                    tool_name, arguments, conn
                )
            else:
                # Process immediately
                return await self.execute_tool(tool_name, arguments, conn)
```

### Monitoring and Metrics

```python
from prometheus_client import Counter, Histogram, Gauge
import time

class MonitoredMCPServer(Server):
    """MCP server with monitoring"""
    
    def __init__(self):
        super().__init__("monitored-server")
        self.setup_metrics()
    
    def setup_metrics(self):
        # Prometheus metrics
        self.request_count = Counter(
            'mcp_requests_total',
            'Total MCP requests',
            ['tool', 'status']
        )
        
        self.request_duration = Histogram(
            'mcp_request_duration_seconds',
            'Request duration',
            ['tool']
        )
        
        self.active_connections = Gauge(
            'mcp_active_connections',
            'Active connections'
        )
        
        self.error_rate = Counter(
            'mcp_errors_total',
            'Total errors',
            ['tool', 'error_type']
        )
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        """Monitored tool call"""
        start_time = time.time()
        
        try:
            result = await super().handle_tool_call(tool_name, arguments)
            
            # Record metrics
            self.request_count.labels(
                tool=tool_name,
                status='success' if result.success else 'failure'
            ).inc()
            
            if not result.success:
                self.error_rate.labels(
                    tool=tool_name,
                    error_type='tool_error'
                ).inc()
            
            return result
            
        except Exception as e:
            self.error_rate.labels(
                tool=tool_name,
                error_type='exception'
            ).inc()
            raise
            
        finally:
            # Record duration
            duration = time.time() - start_time
            self.request_duration.labels(tool=tool_name).observe(duration)
```

Bu kapsamlı MCP Server Examples dokümantasyonu, basit sunuculardan karmaşık kurumsal sistemlere kadar geniş bir yelpazede örnekler sunuyor. Her örnek, üretim ortamında kullanılabilecek şekilde tasarlanmış ve en iyi uygulamaları içeriyor.