# 🧬 CRISPR Guide Design Mini-Tool (Educational)

This Streamlit application is an interactive and educational tool that demonstrates how CRISPR-Cas9 guide RNAs (gRNAs) can be identified and evaluated from a DNA sequence.
It is designed for students, beginners, and bioinformatics learners who want to understand the basic logic behind CRISPR guide selection — not for laboratory or clinical use.

---

## 🚀 Features

### 🔍 1. DNA Sequence Input
- Paste DNA directly into a textbox  
- Or upload a FASTA file  
- Automatically cleans input and extracts valid nucleotide sequence (A/T/G/C)

### 🧬 2. PAM Site Detection (NGG)
- Scans the DNA sequence for the SpCas9 PAM: **NGG (N = any base)**
- Wherever an NGG occurs, the 20 bases upstream form a candidate guide RNA

### 🧪 3. Guide RNA Extraction
For each guide:
- Guide sequence (20 nt)  
- PAM location  
- Genomic coordinates (0-based indexing)  
- Forward-strand evaluation  

### 📊 4. Guide Evaluation Metrics
Each gRNA is scored using simple educational rules:

**✔ GC%**
Shows how many bases are G/C (ideal ~40–60%)

**✔ Self-Complementarity**
Estimates if the guide may fold or form hairpins (higher = riskier)

**✔ Off-Target Similarity (Simplified)**
Checks if the guide resembles other 20-base windows inside the same DNA
(0 = no similar regions, higher = more risk)

**✔ Total Guide Score**
A combined heuristic score where **lower = better**

---

## 🎨 5. Visual Sequence Map
Highlights detected guides and PAMs:
- 🟩 Guide region (20 bases)
- 🟥 PAM (NGG)

Sequence is shown in 60-bp lines for easy reading.

---

## 📈 6. Plots and Data Export
- GC% distribution bar plot  
- Total score distribution plot  
- Download filtered guide table as CSV  

---

# 🧑‍💻 How to Run the App

### 1. Install dependencies

### 2. Run the Streamlit app

---

# 📥 Input Examples

### Example DNA (for testing)
ATGCGTACGTTAGGCTACCGGATCCGATCGGATCGTAGGCTTAGGCTTACCGG

### Recommended public databases to get sequences:
- NCBI Gene  
- Ensembl Genome Browser  
- UCSC Genome Browser  
- Any publicly available FASTA file  

---

# 🎯 Purpose of This App

This tool helps learners understand:
- How CRISPR Cas9 identifies a target
- What “PAM = NGG” means  
- How a 20-nt guide RNA is chosen  
- Why GC content matters  
- Why off-target similarity matters  
- How sequence visualization works  

It turns complex CRISPR concepts into a simple visual learning experience.

---

# ⚠️ Disclaimer
This application is **strictly for educational and demonstration purposes only.**
It does **not** provide lab-grade guide design or genome-wide analysis.

Do **NOT** use it for laboratory, clinical, or therapeutic purposes.

---

# ❤️ Acknowledgements
Built using:
- Python  
- Streamlit  
- Pandas & NumPy  

Inspired by CRISPR-Cas9 biology and introductory bioinformatics workflows.
