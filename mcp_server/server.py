from mcp_server.app import create_server

def main():
    mcp = create_server()
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()