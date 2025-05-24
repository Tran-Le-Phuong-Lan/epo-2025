import re
import os
import pandas as pd
from bs4 import BeautifulSoup

# Set the folder where the HTML files are stored
folder_path = "en_20250101_html/"

# Regex patterns for IPC 4-digit codes and description blocks in JS-style content
symbolcode_pattern = re.compile(r'"symbolcode"\s*:\s*"([A-H][0-9]{2}[A-Z])"')
title_pattern = re.compile(r'"title1"\s*:\s*"(.*?)"\s*,', re.DOTALL)

# To hold extracted results
ipc_descriptions = []

# Loop through all .htm files
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(".htm"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_html = f.read()

        # Search for "symbolcode" occurrences and grab surrounding block
        for match in symbolcode_pattern.finditer(raw_html):
            code = match.group(1)
            start = max(0, match.start() - 500)
            end = match.end() + 2000
            snippet = raw_html[start:end]

            # Extract and clean the "title1" field (the description)
            title_match = title_pattern.search(snippet)
            if title_match:
                raw_title = title_match.group(1)
                raw_title = raw_title.replace('\\"', '"').replace('\\n', ' ')
                description = BeautifulSoup(raw_title, "html.parser").get_text(separator=" ", strip=True)

                # Skip notes or junk
                if description and "subclass index" not in description.lower() and "note" not in description.lower():
                    ipc_descriptions.append((code, description))

# Create DataFrame
df = pd.DataFrame(ipc_descriptions, columns=["IPC Code", "Description"])
df = df.drop_duplicates(subset=["IPC Code"]).reset_index(drop=True)

# Save CSV (optional)
df.to_csv("ipc_extracted_codes.csv", index=False)

# Display preview
print(df.head())
