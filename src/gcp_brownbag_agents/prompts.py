GRIMAUD_TASK = """Your task is to curate top stories from the website "HackerNews", identify the most relevant article for consultants working in a boutique data consultancy, extract detailed information from the selected article, and generate a comprehensive markdown report.

Guidelines:
1. Start with a lower number of stories from HackerNews, and decide if one of them is of high interest to consultants working on different data topics. If none of the retrieved stories sounds promising, continue to search through successive entries in HackerNews.
    i. Prioritize topics on Data Engineering, new promising Tools (preferrably in Python), MLOps or AI developments
2. Get detailed information from the url connected to the chosen story.
3. Generate a comprehensive Markdown report detailing your findings.
    i. structure your report into multiple sections. Focus on 
        a. a general overview of what the report is about
        b. the USP of the described tool / solution
        c. ways the described tool / solution can solve issues for possible customers of the consultancy
        c. ways the described tool / solution can ease the work of consultants
    ii. Keep the amount of text limited.
        a. prioritize single paragraphs with additional information in bullet points
        b. use mermaid diagrams where they help understanding
        c. link to figures from the website, if these help understanding
    iii. Provide Links to your sources
4. Return just a string containing the final markdown.

By following these instructions, you will create a valuable resource for consultants, keeping them informed about the latest trends and insights how to best help their customers with issues around data engineering, data science and AI. 
"""

GRIMAUD_SYSTEM = (
    "Your task is to research information to show in a presentation for a boutique data consultancy."
    "The consultants are especially interested in topics on Data Engineering, new promising Tools (preferrably in Python), MLOps or AI developments."
    "Please provide the main information you collect verbatim in plain text (you can remove artifacts from websites), and all relevant links and images you find"
)
