#!/usr/bin/env python3
"""Simple test script to debug Azure OpenAI connection issues."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from .config import ConfigManager

def test_env_loading():
    """Test if .env file is loaded correctly."""
    print("🔍 Testing environment variable loading...")

    # Try to load .env
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Found and loaded {env_file}")
    else:
        print(f"❌ {env_file} not found")
        return False

    # Check required variables
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_DEPLOYMENT',
        'AZURE_OPENAI_API_VERSION'
    ]

    for var in required_vars:
        value = os.getenv(var)
        if value:
            if 'KEY' in var:
                print(f"✅ {var}: {'*' * len(value)}")  # Hide key
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
            return False

    return True

def test_config_manager():
    """Test ConfigManager loading."""
    print("\n🔍 Testing ConfigManager...")

    try:
        config_mgr = ConfigManager()

        if config_mgr.validate_config():
            print("✅ ConfigManager validation passed")

            agent_config = config_mgr.create_agent_config()
            print(f"✅ Agent config created: {agent_config.deployment_name}")
            return agent_config
        else:
            print("❌ ConfigManager validation failed")
            return None

    except Exception as e:
        print(f"❌ ConfigManager error: {e}")
        return None

def test_azure_openai_direct(agent_config):
    """Test Azure OpenAI connection directly."""
    print("\n🔍 Testing direct Azure OpenAI connection...")

    try:
        llm = AzureChatOpenAI(
            azure_endpoint=agent_config.azure_endpoint,
            api_key=agent_config.api_key,
            api_version=agent_config.api_version,
            deployment_name=agent_config.deployment_name,
            temperature=0.1,
            timeout=10
        )

        print("✅ AzureChatOpenAI client created")

        # Test simple invocation
        response = llm.invoke("Say 'Hello from Azure OpenAI'")
        print(f"✅ Test response: {response.content}")
        return True

    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {type(e).__name__}: {e}")
        return False

async def test_azure_openai_async(agent_config):
    """Test Azure OpenAI async connection."""
    print("\n🔍 Testing async Azure OpenAI connection...")

    try:
        llm = AzureChatOpenAI(
            azure_endpoint=agent_config.azure_endpoint,
            api_key=agent_config.api_key,
            api_version=agent_config.api_version,
            deployment_name=agent_config.deployment_name,
            temperature=0.1,
            timeout=10
        )

        # Test async invocation
        response = await llm.ainvoke("Say 'Hello from Azure OpenAI async'")
        print(f"✅ Async test response: {response.content}")
        return True

    except Exception as e:
        print(f"❌ Azure OpenAI async connection failed: {type(e).__name__}: {e}")
        return False

def main():
    """Run all connection tests."""
    print("🧪 Azure OpenAI Connection Test")
    print("=" * 50)

    # Test 1: Environment variables
    if not test_env_loading():
        print("\n❌ Environment variable test failed. Check your .env file.")
        return 1

    # Test 2: ConfigManager
    agent_config = test_config_manager()
    if not agent_config:
        print("\n❌ ConfigManager test failed.")
        return 1

    # Test 3: Direct connection
    if not test_azure_openai_direct(agent_config):
        print("\n❌ Direct Azure OpenAI test failed.")
        return 1

    # Test 4: Async connection
    import asyncio
    if not asyncio.run(test_azure_openai_async(agent_config)):
        print("\n❌ Async Azure OpenAI test failed.")
        return 1

    print("\n✅ All tests passed! Azure OpenAI connection is working.")
    return 0

if __name__ == "__main__":
    sys.exit(main())