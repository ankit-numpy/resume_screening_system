import streamlit as st
import nltk
import string
import pdfplumber
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')

# ---------- THEME ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #12002b, #1f0040);
}
h1,h2,h3,label {
    color: white !important;
}
.stButton>button {
    background-color: #7c3aed;
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- FUNCTIONS ----------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def read_docx(file):
    doc = docx.Document(file)
    return " ".join([p.text for p in doc.paragraphs])

def read_txt(file):
    return file.read().decode("utf-8")

# ---------- UI ----------
st.title("Resume Screening System")

resume_file = st.file_uploader("Upload Resume", type=["pdf","docx","txt"])
job_desc = st.text_area("Paste Job Description")

submit = st.button("Run Screening")

if submit:
    if not resume_file or job_desc.strip() == "":
        st.error("Please upload resume and paste job description")
    else:
        file_type = resume_file.name.split(".")[-1]

        if file_type == "pdf":
            resume_text = read_pdf(resume_file)
        elif file_type == "docx":
            resume_text = read_docx(resume_file)
        else:
            resume_text = read_txt(resume_file)

        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_desc)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, job_clean])

        similarity = cosine_similarity(vectors[0], vectors[1])
        match = similarity[0][0] * 100

        # ---------- RESULT ----------
        st.subheader("Match Result")

        if match < 30:
            st.error(f"Low Match: {match:.2f}%")
        elif match < 60:
            st.warning(f"Medium Match: {match:.2f}%")
        else:
            st.success(f"High Match: {match:.2f}%")

        # ---------- MISSING KEYWORDS ----------
        resume_words = set(resume_clean.split())
        job_words = set(job_clean.split())
        missing = job_words - resume_words

        st.subheader("Missing Skills (Why Low Match?)")
        if missing:
            st.error(", ".join(list(missing)[:20]))
        else:
            st.success("No major skills missing!")

        # ---------- IMPROVEMENT TIPS ----------
        st.subheader("How To Improve Match")
        if missing:
            st.info("Add these keywords to resume: " + ", ".join(list(missing)[:10]))
        else:
            st.success("Resume already well aligned with job!")
