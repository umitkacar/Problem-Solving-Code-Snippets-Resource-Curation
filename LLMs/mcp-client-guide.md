# 🖥️ MCP Client Implementation Guide

Complete guide to building Model Context Protocol clients - from basic implementations to advanced AI agent systems with production-ready patterns.

**Last Updated:** 2025-06-23

## Table of Contents
- [Introduction](#introduction)
- [Client Architecture](#client-architecture)
- [Basic Client Implementation](#basic-client-implementation)
- [LLM Integration](#llm-integration)
- [Advanced Client Patterns](#advanced-client-patterns)
- [Multi-Server Management](#multi-server-management)
- [Error Handling & Resilience](#error-handling--resilience)
- [Security & Authentication](#security--authentication)
- [Performance Optimization](#performance-optimization)
- [Testing Strategies](#testing-strategies)
- [Real-World Applications](#real-world-applications)
- [Client Libraries](#client-libraries)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Introduction

MCP clients are the bridge between AI applications and MCP servers. They handle:
- Server discovery and connection management
- Protocol translation and message routing
- Tool invocation and resource access
- Response handling and error management

### Client Responsibilities

```python
class MCPClientResponsibilities:
    """Core responsibilities of an MCP client"""
    
    def __init__(self):
        self.responsibilities = {
            "connection_management": [
                "Establish server connections",
                "Handle reconnections",
                "Manage connection pools",
                "Monitor server health"
            ],
            "protocol_handling": [
                "Serialize/deserialize messages",
                "Route requests to servers",
                "Handle protocol versioning",
                "Manage message queuing"
            ],
            "capability_discovery": [
                "List available tools",
                "Discover resources",
                "Cache server capabilities",
                "Handle capability updates"
            ],
            "execution_coordination": [
                "Execute tool calls",
                "Read resources",
                "Handle streaming responses",
                "Manage timeouts"
            ],
            "error_management": [
                "Retry failed requests",
                "Handle server errors",
                "Provide fallback options",
                "Log issues for debugging"
            ]
        }
```

## Client Architecture

### Component Overview

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio

class MCPTransport(ABC):
    """Abstract transport layer"""
    
    @abstractmethod
    async def connect(self, config: dict):
        """Establish connection"""
        pass
    
    @abstractmethod
    async def send(self, message: dict):
        """Send message to server"""
        pass
    
    @abstractmethod
    async def receive(self) -> dict:
        """Receive message from server"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close connection"""
        pass

class MCPProtocol:
    """Protocol handler"""
    
    def __init__(self, version: str = "1.0"):
        self.version = version
        self.message_id = 0
    
    def create_request(self, method: str, params: dict = None) -> dict:
        """Create protocol-compliant request"""
        self.message_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.message_id,
            "method": method
        }
        if params:
            request["params"] = params
        return request
    
    def parse_response(self, message: dict) -> Any:
        """Parse server response"""
        if "error" in message:
            raise MCPError(
                message["error"]["code"],
                message["error"]["message"]
            )
        return message.get("result")

class MCPClient:
    """Main MCP client implementation"""
    
    def __init__(self, transport: MCPTransport):
        self.transport = transport
        self.protocol = MCPProtocol()
        self.capabilities = {}
        self.connected = False
    
    async def connect(self, config: dict):
        """Connect to MCP server"""
        await self.transport.connect(config)
        self.connected = True
        
        # Discover capabilities
        await self.discover_capabilities()
    
    async def discover_capabilities(self):
        """Discover server capabilities"""
        self.capabilities["tools"] = await self.list_tools()
        self.capabilities["resources"] = await self.list_resources()
        self.capabilities["prompts"] = await self.list_prompts()
```

### Transport Implementations

```python
import subprocess
import json
from asyncio import StreamReader, StreamWriter

class StdioTransport(MCPTransport):
    """Standard I/O transport for local servers"""
    
    def __init__(self):
        self.process = None
        self.reader = None
        self.writer = None
    
    async def connect(self, config: dict):
        """Start server process"""
        cmd = config["command"]
        args = config.get("args", [])
        env = config.get("env", {})
        
        self.process = await asyncio.create_subprocess_exec(
            cmd, *args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, **env}
        )
        
        self.reader = self.process.stdout
        self.writer = self.process.stdin
    
    async def send(self, message: dict):
        """Send message via stdio"""
        data = json.dumps(message) + "\n"
        self.writer.write(data.encode())
        await self.writer.drain()
    
    async def receive(self) -> dict:
        """Receive message via stdio"""
        line = await self.reader.readline()
        if not line:
            raise ConnectionError("Server closed connection")
        return json.loads(line.decode().strip())
    
    async def close(self):
        """Terminate server process"""
        if self.process:
            self.process.terminate()
            await self.process.wait()

class HTTPTransport(MCPTransport):
    """HTTP/WebSocket transport for remote servers"""
    
    def __init__(self):
        self.session = None
        self.websocket = None
        self.base_url = None
    
    async def connect(self, config: dict):
        """Connect to HTTP server"""
        import aiohttp
        
        self.base_url = config["url"]
        self.session = aiohttp.ClientSession()
        
        # For WebSocket support
        if config.get("websocket"):
            self.websocket = await self.session.ws_connect(
                f"{self.base_url}/ws"
            )
    
    async def send(self, message: dict):
        """Send HTTP request or WebSocket message"""
        if self.websocket:
            await self.websocket.send_json(message)
        else:
            # HTTP POST for request-response
            async with self.session.post(
                f"{self.base_url}/rpc",
                json=message
            ) as response:
                self.last_response = await response.json()
    
    async def receive(self) -> dict:
        """Receive response"""
        if self.websocket:
            msg = await self.websocket.receive()
            return msg.json()
        else:
            return self.last_response
```

## Basic Client Implementation

### Minimal Working Client

```python
import asyncio
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class SimpleM