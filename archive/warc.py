import warcio.archiveiterator
import os

warc_file = "TEST-000000.extracted.warc"
output_dir = "extracted_content"
max_filename_length = 255
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(warc_file, 'rb') as stream:
    for record in warcio.archiveiterator.ArchiveIterator(stream):
        if record.rec_type == 'response':
            target_url = record.rec_headers.get_header('WARC-Target-URI')
            if target_url:
                # Create a safe filename
                filename = target_url.replace("://", "_").replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_")
                if len(filename) > max_filename_length:
                    filename = filename[:max_filename_length]  # truncate the filename.

                file_path = os.path.join(output_dir, filename)

                with open(file_path, "wb") as output_file:
                    output_file.write(record.content_stream().read())
                print(f"Extracted: {target_url} to {file_path}")