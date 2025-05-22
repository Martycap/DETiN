def extract_acronyms(filename):
    acronyms = set()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('_')
            # controlla che abbia almeno 3 parti (es. Au_art_20057.jpg -> ['Au', 'art', '20057.jpg'])
            if len(parts) >= 3:
                acronyms.add(parts[1])
    return list(acronyms)

if __name__ == "__main__":
    filename = "data/raw/CASIA2/au_list.txt"  # sostituisci con il tuo file
    acronyms_list = extract_acronyms(filename)
    print("Acronimi trovati:")
    for acro in acronyms_list:
        print(acro)
