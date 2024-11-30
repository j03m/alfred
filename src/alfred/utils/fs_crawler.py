import os


class FileSystemCrawler:
    def __init__(self, directory, process_function, encoding='utf-8'):
        """
        Initializes the FileSystemCrawler.

        Args:
            directory (str): The directory to crawl.
            process_function (callable): A function to call with each file's content.
            encoding (str): The encoding to use when reading files (default: 'utf-8').
        """
        self.directory = directory
        self.process_function = process_function
        self.encoding = encoding

    def crawl(self):
        """Starts crawling the directory."""
        for root, _, files in os.walk(self.directory):  # Traverse the directory
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    self._process_file(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    def _process_file(self, file_path):
        """
        Reads the content of a file and calls the processing function.

        Args:
            file_path (str): The path to the file to process.
        """
        with open(file_path, 'r', encoding=self.encoding) as f:
            content = f.read()
            self.process_function(file_path, content)


# Example processing function
def print_file_summary(file_path, content):
    """Prints the file path and a snippet of its content."""
    print(f"File: {file_path}")
    print(f"Content preview: {content[:100]}...")  # Print the first 100 characters


# Example usage
# if __name__ == "__main__":
#     directory_to_crawl = "./my_directory"  # Replace with your target directory
#     crawler = FileSystemCrawler(directory_to_crawl, print_file_summary)
#     crawler.crawl()