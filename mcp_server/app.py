from mcp.server.fastmcp import FastMCP
from mcp_server.tools import pdf_tools, sentiment_tools, news_tools, summartization_tools

def create_server() -> FastMCP:
    mcp = FastMCP("PDF-and-sentiment")

    pdf_tools.register(mcp)
    sentiment_tools.register(mcp)
    news_tools.register(mcp)
    summartization_tools.register(mcp)

    return mcp
