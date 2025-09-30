# AI Agent System

The AI Agent System is a web application that allows users to configure and run an AI agent system using various AI models and orchestration strategies.

## Features

- Select from a wide range of AI models for the orchestrator, refiner, and subagent roles
- Configure the goal and orchestration strategy for the AI agent system
- Execute the configured AI agent system and view the results
- Dark theme interface for a visually appealing and comfortable user experience

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/dtsunami/agent_app.git
   ```

2. Navigate to the project directory:
   ```
   cd agent_app
   ```

3. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # For Unix/macOS
   .venv\Scripts\activate  # For Windows
   ```

4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the uvicorn server:
   ```
   uvicorn server:app --reload
   ```

2. Open your web browser and access the app at `http://localhost:8000`.

3. On the home page, select the desired AI models for the orchestrator, refiner, and subagent from the dropdown menus.

4. Enter the goal for the AI Agent System in the provided textbox.

5. Choose the agent orchestration strategy using the radio buttons.

6. Click the "Configure" button to configure the AI Agent System with the selected options.

7. Once configured, click the "Run Agent System" button to execute the AI Agent System.

8. The results will be displayed on the page, showing the selected models, goal, orchestration strategy, and the final result.

## Testing

To run the unit tests, use the following command: