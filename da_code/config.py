"""Configuration management for da_code CLI tool."""

# Load .env file FIRST before any other imports
from pathlib import Path
env_file = Path('.env')
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file, override=False)

import os
import logging
from typing import Optional

from .models import AgentConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and environment variables."""

    def __init__(self):
        """Initialize configuration manager."""
        pass


    def create_agent_config(self) -> AgentConfig:
        """Create agent configuration from environment variables."""
        # Required Azure OpenAI settings
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')

        if not azure_endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT environment variable is required. "
                "Set it to your Azure OpenAI endpoint URL."
            )

        if not api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY environment variable is required. "
                "Set it to your Azure OpenAI API key."
            )

        # Optional settings with defaults
        api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2023-12-01-preview')
        temperature = float(os.getenv('DA_CODE_TEMPERATURE', '0.7'))
        max_tokens = os.getenv('DA_CODE_MAX_TOKENS')
        agent_timeout = int(os.getenv('DA_CODE_AGENT_TIMEOUT', '600'))
        max_retries = int(os.getenv('DA_CODE_MAX_RETRIES', '2'))
        command_timeout = int(os.getenv('DA_CODE_COMMAND_TIMEOUT', '300'))
        require_confirmation = os.getenv('DA_CODE_REQUIRE_CONFIRMATION', 'true').lower() == 'true'

        # History file path configuration
        history_file_path = os.getenv('DA_CODE_HISTORY_FILE')
        if not history_file_path:
            # Default to .prompt.history in current working directory
            history_file_path = os.path.join(os.getcwd(), '.prompt.history')


        return AgentConfig(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_name=deployment_name,
            temperature=temperature,
            max_tokens=int(max_tokens) if max_tokens else None,
            agent_timeout=agent_timeout,
            max_retries=max_retries,
            command_timeout=command_timeout,
            require_confirmation=require_confirmation,
            history_file_path=history_file_path
        )

    def create_sample_env(self, env_path: Optional[str] = None) -> None:
        """Create a sample .env file by copying from .env.example."""
        if env_path is None:
            env_path = '.env'

        example_file = Path(__file__).parent / '.env.example'

        try:
            if example_file.exists():
                # Copy from .env.example
                with open(example_file, 'r') as f:
                    sample_content = f.read()

                with open(env_path, 'w') as f:
                    f.write(sample_content)

                logger.info(f"Created sample environment file at {env_path} from .env.example")
                print(f"Sample environment file created at {env_path}")
                print("Please update the configuration with your actual values.")
            else:
                raise FileNotFoundError(f".env.example not found at {example_file}")

        except Exception as e:
            logger.error(f"Failed to create sample .env file: {e}")
            raise

    def validate_config(self) -> bool:
        """Validate that all required configuration is present."""
        try:
            self.create_agent_config()
            logger.info("Configuration validation successful")
            return True
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during configuration validation: {e}")
            return False

    def print_config_status(self) -> None:
        """Print current configuration status."""
        print("\\n=== da_code Configuration Status ===")

        # Check Azure OpenAI settings
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')

        print(f"Azure OpenAI Endpoint: {'✓' if azure_endpoint else '✗'} {azure_endpoint or 'Not set'}")
        print(f"Azure OpenAI API Key: {'✓' if api_key else '✗'} {'Set' if api_key else 'Not set'}")
        print(f"Azure OpenAI Deployment: {deployment}")

        # Check MongoDB settings
        mongo_host = os.getenv('MONGO_HOST', 'localhost')
        mongo_port = os.getenv('MONGO_PORT', '8004')
        print(f"MongoDB Tracking: {mongo_host}:{mongo_port}")

        # Check file existence
        env_file = Path('.env')
        da_md = Path('DA.md')
        da_json = Path('DA.json')

        print(f"\\n=== Project Files ===")
        print(f".env file: {'✓' if env_file.exists() else '✗'} {env_file.absolute()}")
        print(f"DA.md file: {'✓' if da_md.exists() else '✗'} {da_md.absolute()}")
        print(f"DA.json file: {'✓' if da_json.exists() else '✗'} {da_json.absolute()}")

        if not azure_endpoint or not api_key:
            print("\\n⚠️  Missing required Azure OpenAI configuration!")
            print("Run 'da_code --setup' to create a sample configuration file.")


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    # Use the actual log level from environment
    effective_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=effective_level,
        format="%(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )
