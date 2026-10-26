from llama_index.llms.openai import OpenAI
import dotenv
import os
from github import Github, Auth
import asyncio
from llama_index.core.agent.workflow import AgentOutput, AgentWorkflow, FunctionAgent, ToolCall, \
    ToolCallResult
from llama_index.core.workflow import Context
from llama_index.core.prompts import RichPromptTemplate
from typing import Any
import sys

dotenv.load_dotenv()

# Retrieve environment variables and GitHub repo information
auth = Auth.Token(os.getenv("GITHUB_TOKEN") or "")
git = Github(auth=auth)
repo_url = os.getenv("GITHUB_REPO_URL")
if repo_url is not None:
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    username = repo_url.split('/')[-2]
    full_repo_name = f"{username}/{repo_name}"
    if git is not None:
        repo = git.get_repo(full_repo_name)

# --- System prompts for agents ---
context_instruc = """You are the context gathering agent. When gathering context, you MUST:
  - Gather PR details: author, title, body, diff_url, state, and head_sha
  - Get changed files and patches for the entire pull request using the get_pull_request_changes tool with the PR number
  - Any requested for files
  - Generate a concise summary of the pull request using the generate_pr_summary tool
  - Store the summary in the gathered_contexts field using the update_gathered_contexts tool
  Once you have gathered all requested info and stored the summary, hand control back to the Commentor Agent."""

comment_instruc = """You are the commentor agent that writes review comments for pull requests as a human reviewer would.
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
    - Are new endpoints documented? - use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - If you need any additional details, you must hand off to the Context Agent.
 - You should directly address the author. So your comments should sound like:
 'Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?'

 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review."""

review_post_instruc = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must:
   - Be a ~200-300 word review in markdown format.
   - Specify what is good about the PR:
   - Did the author follow ALL contribution rules? What is missing?
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them?
   - Are there notes on whether new endpoints were documented?
   - Are there suggestions on which lines could be improved upon? Are these lines quoted?
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns.
 When you are satisfied, post the review to GitHub.  """

# --- LLM details ---
llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

# --- AI agent tools ---

# get PR request details including author, title, body, state and SHAs
def get_pull_request_details(pr_number: int):
    """PR details tool - returns details about the PR given the number
    including PR author, PR title, PR body, diff URL, PR state, and commit SHAs"""
    pull_request = repo.get_pull(pr_number)
    pr_details = {
        "author": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
    }
    return pr_details


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


# PR changes tool - retrieves file changes across the entire pull request
def get_pull_request_changes(pr_number: int):
    """Retrieves file names, statuses, additions, deletions, changes and patches for the entire pull request given its number."""
    pull_request = repo.get_pull(pr_number)
    changed_files: list[dict[str, Any]] = []
    for f in pull_request.get_files():
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


# PR summary tool - takes PR details and requests the LLM to create a concise summary of what the proposed changes are
def generate_pr_summary(pr_details: dict[str, Any], changed_files: list[dict[str, Any]]):
    """Generates a concise summary of the pull request based on its details and changes."""
    pr_context = f"""
    PR Title: {pr_details['title']}
    PR Body: {pr_details['body']}
    Author: {pr_details['author']}
    State: {pr_details['state']}
    """
    files_summary = ""
    for file_info in changed_files:
        files_summary += f"- {file_info['filename']}: +{file_info['additions']} {file_info['deletions']} lines\n"
    full_context = pr_context + "\n" + files_summary

    summary_prompt = f"""Based on the following PR information, generate a concise 2-3 sentence summary:
    {full_context}

    Summary:"""
    response = llm.complete(summary_prompt)
    return response.text.strip()


# update gathered contexts tool - takes the summary generated and saves it to the application context
async def update_gathered_contexts(ctx: Context, context_summary: str) -> str:
    """Save the PR summary for the other agents."""
    async with ctx.store.edit_state() as store:
        store["state"]["gathered_contexts"] = context_summary

    return "PR summary saved."


# adding comment to context tool - takes a draft comment and adds it to the persistent context of the application
async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Save the review draft."""
    async with ctx.store.edit_state() as store:
        store["state"]["draft_comment"] = draft_comment
        # A revised draft needs validation again.
        store["state"]["final_review"] = ""
    return "Draft saved."


# adding final comment to context tool - takes a finalised comment and saves it to the context of the application
async def add_final_review_to_state(ctx: Context, final_comment: str) -> str:
    """Save the validated review."""
    async with ctx.store.edit_state() as store:
        store["state"]["final_review"] = final_comment
    return "Final review saved."


async def post_to_github(ctx: Context, pr_number: int) -> str:
    """Takes a finalised comment (which can be retrieved from the final_review section of the Context) and submits it as a comment for the specified pull request on the GitHub repository"""
    state = await ctx.store.get("state")
    final_review = state.get("final_review", "").strip()
    if not final_review:
        return "Review not posted: no final review has been saved."

    pull_request = repo.get_pull(pr_number)
    pull_request.create_review(body=state["final_review"])
    return "Review created on GitHub."


#  --- defining agents ---
context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="uses the GitHub API to retrieve information about pull requests",
    tools=[get_pull_request_details, get_files, get_pull_request_changes, generate_pr_summary, update_gathered_contexts],
    system_prompt=context_instruc,
    can_handoff_to=["CommentorAgent"]
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses context gathered by the context agent to draft a pull review comment.",
    tools=[add_comment_to_state],
    system_prompt=comment_instruc,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

review_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="takes a draft pull request review comment from the commentor agent and either sends it back for rewriting, or posts it on GitHub.",
    tools=[add_final_review_to_state, post_to_github],
    system_prompt=review_post_instruc,
    can_handoff_to=["CommentorAgent"]
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_posting_agent],
    root_agent=commentor_agent.name,
    initial_state={
        "pr_details": {},
        "changed_files": [],
        "gathered_contexts": "",
        "draft_comment": "",
        "review_feedback": [],
        "final_review": "",
    },
)


# --- workflow ---

async def main():
    query = input().strip()

    ctx = Context(workflow_agent)
    handler = workflow_agent.run(user_msg=query, ctx=ctx)

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")
    state = await ctx.store.get("state")
    print(state["draft_comment"])


if __name__ == "__main__":
    asyncio.run(main())
    if git is not None:
        git.close()
    sys.exit()
