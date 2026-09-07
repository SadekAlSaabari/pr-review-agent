from llama_index.llms.openai import OpenAI
import dotenv
import os
from github import Github
import asyncio 
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult, ReActAgent
from llama_index.core.prompts import RichPromptTemplate
from typing import Any
import sys

dotenv.load_dotenv()

# Retrieve environment variables and GitHub repo information
git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None
repo_url = os.getenv("GITHUB_REPO_URL")
if repo_url is not None:
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    username = repo_url.split('/')[-2]
    full_repo_name = f"{username}/{repo_name}"
    if git is not None:
        repo = git.get_repo(full_repo_name)

instruc = """You are the context gathering agent. When gathering context, you MUST gather: 
    - The details: author, title, body, diff_url, state, and head_sha;
    - Changed files;
    - Any requested for files;"""

# LLM details
llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

# --- AI agent tools ---

# get PR request details including author, title, body, state and SHAs
def get_pull_request_details(PR_number: int):
    """PR details tool - returns details about the PR given the number
    including PR author, PR title, PR body, diff URL, PR state, and commit SHAs"""
    pull_request = repo.get_pull(PR_number)
    return {
        "author": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
    }


# files tool - gets a file path, and fetches contents of file in repo
def get_files(file_path: str):
    """retrieves the contents of a file from the repo in text format,
    given that you provide the file path as a string"""
    contents = repo.get_contents(file_path)
    if isinstance(contents, list):
        files_collection = []
        for item in contents:
            files_collection.append(item.decoded_content.decode("utf-8"))
        return files_collection
    else:
        return contents.decoded_content.decode("utf-8")


# PR commits details tool - gets an SHA, retrieves info about commit
# like file changes, and returns that information
def get_commit_details(head_sha: str):
    """retrieves details of a commit, given that you provide the SHA for it. details include file name, status, additions, deletions, changes and patches"""
    commit = repo.get_commit(head_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files

# initialise AI agent that takes context of github PRs
context_agent = ReActAgent(
    llm = llm,
    name = "ContextAgent",
    tools = [get_pull_request_details, get_files, get_commit_details],
    system_prompt= instruc
)

async def main():
    query = input().strip()
    prompt = RichPromptTemplate(query)

    handler = context_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print(event.response.content)
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")

if __name__ == "__main__":
    asyncio.run(main())
    if git is not None:
        git.close()
    sys.exit()