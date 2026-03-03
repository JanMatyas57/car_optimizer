import os

#script for vibe coding, packs the entire project into txt which can be inserted into gemini or chat gpt for the ai to understand the context


PROJECT_DIR = "." 
OUTPUT_FILE = "project_context.txt" 

IGNORE_DIRS = {'.git', 'venv', 'env', '__pycache__', '.idea', '.vscode', 'build', 'dist'}


ALLOWED_EXTS = {'.py', '.txt', '.md', '.json', '.ini', '.yaml'}

def pack_project():
   
    print("Packing the project")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
      
        outfile.write("=== STRUKTURA PROJEKTU ===\n")
        
        for root, dirs, files in os.walk(PROJECT_DIR):
          
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if any(file.endswith(ext) for ext in ALLOWED_EXTS) and file != OUTPUT_FILE:
                    filepath = os.path.relpath(os.path.join(root, file), PROJECT_DIR)
                    outfile.write(f"- {filepath}\n")
        
        outfile.write("\n=== ZDROJOVÉ KÓDY ===\n\n")
        

        for root, dirs, files in os.walk(PROJECT_DIR):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if any(file.endswith(ext) for ext in ALLOWED_EXTS) and file != OUTPUT_FILE:
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, PROJECT_DIR)
                    
                    outfile.write(f"--- ZAČÁTEK SOUBORU: {rel_path} ---\n")
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read() + "\n")
                    except Exception as e:
                        outfile.write(f"// CHYBA PŘI ČTENÍ SOUBORU: {e}\n")
                    outfile.write(f"--- KONEC SOUBORU: {rel_path} ---\n\n")

    print("Done")

if __name__ == "__main__":
    pack_project()
  