# pr-review-agent

Python-based multi-agent tool that drafts and posts GitHub pull-request reviews using an OpenAI LLM. It wires three agents (context gatherer, comment writer, and review poster) together via llama-index workflows and calls the GitHub API to fetch PR data and submit reviews.

To run it:
1. Install dependencies (examples; add to requirements as needed):
```
pip install llama-index openai PyGithub python-dotenv
```
2. Create a .env from .envexample and set values:
- OPENAI_API_KEY
- (optional) OPENAI_BASE_URL
- MODEL (defaults in .envexample)
- GITHUB_TOKEN
- GITHUB_REPO_URL (e.g., https://github.com/owner/repo.git)
3. Run:
```
python agent.py
```
The script reads a query from stdin (e.g., a prompt asking the agents to review a certain PR by its number) and streams the workflow output. Required env variables are listed in .envexample.
