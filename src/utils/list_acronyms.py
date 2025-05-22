import json

def extract_acronyms(filename):
    """
    Function that analyzes the path names of authentic photos 
    present in the CASIA2 directory and extracts the discriminating acronyms.
    """
    acronyms = set()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('_')
            if len(parts) >= 3:
                acronyms.add(parts[1])
    return list(acronyms)

if __name__ == "__main__":
    input_filename = "data/raw/CASIA2/au_list.txt" 
    output_filename = "data/raw/CASIA2/list_acronyms.json"
    acronyms_list = extract_acronyms(input_filename)
    with open(output_filename, 'w') as f:
        json.dump(acronyms_list, f)
