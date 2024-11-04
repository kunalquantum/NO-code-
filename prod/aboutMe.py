import streamlit as st  

# Profile Image Section  
st.title("Kunal Sharma")
cola, colb=st.columns(2)
with cola:
    profile_image = "prod/assets/me.jpeg"  # Update this path with your image path  
    st.image(profile_image, width=250)  # Choose an appropriate width  
with colb:

# Contact Information Section  
    st.header("Contact Information")  
    st.write("**Name:** Kunal Lalbahadur Sharma")  # Replace with your actual name  
    st.write("""**Email:** kunal.lalbahdur.sharma05@gmail.com""")  # Replace with your email  
    st.write("**Phone:** 9892768818")  # Replace with your phone number  
    st.write("**LinkedIn:** [Kunal LinkedIn Profile](https://www.linkedin.com/in/kunal-sharma-601812247/)")  # Replace with your LinkedIn URL  
    st.write("**Github:** [Kunal Github Profile](https://github.com/kunalquantum)")
# Summary Section  
st.write(  
    "I am a passionate technophile with a strong foundation in coding, particularly in Java and Python, "  
    "and I specialize in data science and machine learning. I enjoy solving complex problems and developing "  
    "innovative solutions to improve daily life. With a commitment to strong project management and leadership, "  
    "I strive to drive successful outcomes in all my endeavors."  
)  

# Education Section with Three Columns  
st.header("Education")  
col1, col2, col3 = st.columns(3)  

with col1:  
    st.write("**Degree**")  
    st.write("- B.E. in Computer Engineering")  
    st.write("- HSC")  
    st.write("- SSC")  

with col2:  
    st.write("**Institution**")  
    st.write("- Saraswati College of Engineering")  
    st.write("- Jai Hind College")  
    st.write("- Dr Antonio Da Silva Technical High School")  

with col3:  
    st.write("**Duration/Percentage**")  
    st.write("- CGPI: 8.5, 2021 - Present")  
    st.write("- Percentage: 84%, 2019 - 2021")  
    st.write("- Percentage: 83%, 2007 - 2019")  

# Certification and Training Section  
st.header("Certification and Training")  
certifications = [  
    "Microsoft Azure Certification in Entry Level AI-900",  
    "Python Workshop",  
    "AI Workshop",  
    "UI Designing",  
    "Cyber Security",  
    "Acmegrade Training in Data Science"  
]  
for cert in certifications:  
    st.write(f"- {cert}")  

# Technical Skills Section  
st.header("Technical Skills")  
col1, col2 = st.columns(2)  

with col1:  
    st.write("**Programming Languages**")  
    st.write("- Python")  
    st.write("- Java (Core and Advanced)")  

    st.write("**Web Development**")  
    st.write("- HTML")  
    st.write("- CSS")  
    st.write("- JavaScript")  

with col2:  
    st.write("**Data Science & Machine Learning**")  
    st.write("- Scikit-learn")  
    st.write("- TensorFlow")  
    st.write("- Exploratory Data Analysis")  
    st.write("- Data Cleaning and Insights")  

    st.write("**Database Management**")  
    st.write("- SQL (JDBC, Servlet/JSP)")  

# Internship Experience Section  
st.header("Internship Experience")  
st.write("**S.P.R.K Technologies** | Java Backend Developer Intern")  
st.write("Duration: 1 Month")  

# Hackathons Section  
st.header("Hackathons")  
hackathons = [  
    "Smart India Hackathons (2022-23, 2023-24)",  
    "NASA Space App Challenge (2022-23, 2023-24)",  
    "Avishkar National Hackathon (2022-23) - Winner",  
    "Hack2Crack",  
    "Hackathon at Amity University"  
]  
for hackathon in hackathons:  
    st.write(f"- {hackathon}")  

# Projects Section  
st.header("Projects")  
projects = [  
    {  
        "name": "Samarthya (Data Structure Visualizer)",  
        "duration": "2024 - Current",  
        "github": "https://github.com/kunalquantum/samarthya20.git",  
        "description": "Ranked 1st in Avishkar National level Project Competition."  
    },  
    {  
        "name": "Block-Banking",  
        "duration": "2024 – 2024",  
        "description": "Blockchain-based banking system that can securely perform transactions between two customers using personalized authentication."  
    },  
    {  
        "name": "SmartData Hub",  
        "duration": "2023 – 2024",  
        "github": "https://github.com/kunalquantum/SmartData-Hub.git",  
        "description": "National Level Qualified for NASA Space App Hackathon. Created automation for data cleaning, preprocessing, visualization, and model training."  
    },  
    {  
        "name": "Health Mate",  
        "duration": "2022 – 2023",  
        "description": "Digital health examiner and cure recommender for critical and noncritical diseases."  
    },  
    {  
        "name": "InstaMate 1 and 2.0",  
        "duration": "2022 – 2023",  
        "github": "https://github.com/kunalquantum/Insta-Mate.git",  
        "description": "Automating Instagram using Deep Learning and Natural Language Processing."  
    }  
]  

for project in projects:  
    st.write(f"**{project['name']}** ({project['duration']})")  
    st.write(f"- {project['description']}")  
    if "github" in project:  
        st.write(f"- GitHub Link: [{project['github']}]({project['github']})")  

# Extra-Curricular Activities Section  
st.header("Extra-Curricular Activities")  
extracurriculars = [  
    "General Advisor – Student Council",  
    "Vice President – Computer Department Student Association",  
    "Head Boy – School",  
    "Participated in 5+ hackathons (Won 1)"  
]  
for activity in extracurriculars:  
    st.write(f"- {activity}")  

# References Section  
st.header("References")  
st.write("Available upon request.")