from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import feedparser
import yagmail

# -----------------------------
# Email Configuration
# -----------------------------
EMAIL_SENDER = "your_email@gmail.com"       # Replace with your email
EMAIL_PASSWORD = "your_app_password"       # For Gmail, generate App Password
EMAIL_RECIPIENTS = ["team@example.com"]   # List of recipients
EMAIL_SUBJECT = "Weekly Competitive Intelligence Digest"

def send_email(digest_text):
    """
    Sends the digest via email using yagmail.
    """
    yag = yagmail.SMTP(user=EMAIL_SENDER, password=EMAIL_PASSWORD)
    yag.send(
        to=EMAIL_RECIPIENTS,
        subject=EMAIL_SUBJECT,
        contents=digest_text
    )
    print("✅ Digest sent via email!")

# -----------------------------
# Step 1: Define competitor RSS feeds
# -----------------------------
COMPETITOR_FEEDS = {
    "Competitor A": "https://example.com/competitor-a/rss",
    "Competitor B": "https://example.com/competitor-b/rss"
}

# -----------------------------
# Step 2: Initialize LLM
# -----------------------------
llm = OpenAI(
    model_name="gpt-4",    # or "gpt-5-mini"
    temperature=0.3
)

# -----------------------------
# Step 3: Summarization prompt template
# -----------------------------
summary_prompt = PromptTemplate(
    input_variables=["title", "link", "content"],
    template="""
You are a competitive intelligence analyst. Summarize the following article for a business report:

Title: {title}
Link: {link}
Content: {content}

Please provide:
1. Key facts or announcements
2. Potential impact on our company
3. Any relevant trends
Format as a concise bullet list.
"""
)

summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# -----------------------------
# Step 4: Fetch and summarize articles
# -----------------------------
def fetch_and_summarize():
    summaries = []

    for competitor, feed_url in COMPETITOR_FEEDS.items():
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:5]:  # limit to latest 5 articles per competitor
            title = entry.title
            link = entry.link
            content = entry.summary if hasattr(entry, 'summary') else ""
            
            summary = summary_chain.run(title=title, link=link, content=content)
            summaries.append({
                "competitor": competitor,
                "title": title,
                "link": link,
                "summary": summary
            })
    return summaries

# -----------------------------
# Step 5: Generate weekly digest
# -----------------------------
digest_prompt = PromptTemplate(
    input_variables=["summaries"],
    template="""
You are a competitive intelligence analyst. Based on the following article summaries, create a concise weekly digest report for the management team.

Article Summaries:
{summaries}

Include:
- Top competitor actions
- Emerging trends
- Recommendations for our team
Format as bullet points.
"""
)

digest_chain = LLMChain(llm=llm, prompt=digest_prompt)

def generate_digest(summaries):
    # Convert summaries list into a single string
    summaries_text = "\n".join([f"{s['competitor']}: {s['summary']}" for s in summaries])
    digest = digest_chain.run(summaries=summaries_text)
    return digest

# -----------------------------
# Step 6: Run the workflow
# -----------------------------
if __name__ == "__main__":
    print("Fetching competitor articles...")
    summaries = fetch_and_summarize()
    print("Generating weekly digest...")
    digest = generate_digest(summaries)

    print("\n=== WEEKLY COMPETITIVE INTEL DIGEST ===\n")
    print(digest)

    # Send digest via email
    send_email(digest)
