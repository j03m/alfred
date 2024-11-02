import trafilatura

# Read the HTML content from the file
with open("../../news/AAPL/20241029/00b1a51c.txt", 'r', encoding='utf-8') as file:
    html_content = file.read()

# Extract the main content
main_content = trafilatura.extract(html_content)

# Check if content was extracted
if main_content:
    print("Extracted Main Content:")
    print(main_content)
else:
    print("No content could be extracted.")