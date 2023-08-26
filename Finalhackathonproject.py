#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random

# Generate random data for the skills dataset
num_rows = 200
num_skills_per_row = 4

user_names = ["User_" + str(i) for i in range(1, num_rows + 1)]
skills_list = ["Python", "Java", "C++", "Data Analysis", "Project Management", "UI/UX Design", "Customer Service", "Digital Marketing", "Graphic Design", "Database Management"]

skills_data = []
for _ in range(num_rows):
    user_name = random.choice(user_names)
    skills = random.sample(skills_list, num_skills_per_row)
    skills_data.append({
        "User Name": user_name,
        "Skill 1": skills[0],
        "Skill 2": skills[1],
        "Skill 3": skills[2],
        "Skill 4": skills[3]
    })

# Create a DataFrame for the skills dataset
skills_dataset = pd.DataFrame(skills_data)

# Print the first few rows of the generated dataset
print(skills_dataset.head())


# In[2]:


import pandas as pd
import random

# Generate random data for the dataset
skills = ["Skill_" + str(i) for i in range(1, 201)]
categories = ["Category_" + str(i) for i in range(1, 201)]
descriptions = ["Description_" + str(i) for i in range(1, 201)]
experience = [random.randint(1, 10) for _ in range(200)]

# Create a dictionary for the dataset
data = {
    "Skill": skills,
    "Category": categories,
    "Description": descriptions,
    "Experience": experience
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Print the first few rows of the generated dataset
print(df.head())


# In[3]:


import pandas as pd
import random

# Generate random data for the dataset
num_openings = 200

skills_list = ["Python", "Java", "C++", "Data Analysis", "Project Management", "UI/UX Design", "Customer Service", "Digital Marketing", "Graphic Design", "Database Management"]
companies = ["Company_" + str(i) for i in range(1, num_openings + 1)]
experience_required = [random.randint(1, 10) for _ in range(num_openings)]

skills_required = []
for _ in range(num_openings):
    num_skills = random.randint(1, 5)  # Randomly select 1 to 5 skills per opening
    skills = random.sample(skills_list, num_skills)
    skills_required.append(", ".join(skills))

# Create a dictionary for the dataset
data = {
    "Company Name": companies,
    "Skills Required": skills_required,
    "Experience Required": experience_required
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Print the first few rows of the generated dataset
print(df.head())


# In[4]:


import pandas as pd
import random

# Generate random data for the candidate applications dataset
num_applications = 100

candidate_names = ["Candidate_" + str(i) for i in range(1, num_applications + 1)]
candidate_skills = []

skills_list = ["Python", "Java", "C++", "Data Analysis", "Project Management", "UI/UX Design", "Customer Service", "Digital Marketing", "Graphic Design", "Database Management"]

for _ in range(num_applications):
    num_skills = random.randint(1, 5)  # Randomly select 1 to 5 skills per candidate
    skills = random.sample(skills_list, num_skills)
    candidate_skills.append(", ".join(skills))

# Create a dictionary for the candidate applications dataset
data = {
    "Candidate Name": candidate_names,
    "Skills": candidate_skills
}

# Create a DataFrame from the dictionary
candidate_applications_dataset = pd.DataFrame(data)

# Print the first few rows of the generated dataset
print(candidate_applications_dataset.head())


# In[5]:


import pandas as pd
import random

# Generate random data for the job_openings dataset
num_job_openings = 10

job_titles = ["Job_" + str(i) for i in range(1, num_job_openings + 1)]
company_names = ["Company_" + str(i) for i in range(1, num_job_openings + 1)]
experience_levels = ["Entry", "Intermediate", "Senior"]
skills_list = ["Python", "Java", "C++", "Data Analysis", "Project Management", "UI/UX Design", "Customer Service", "Digital Marketing", "Graphic Design", "Database Management"]

job_openings = []
for _ in range(num_job_openings):
    required_skills = random.sample(skills_list, random.randint(3, 6))  # Randomly select 3 to 6 skills per job
    job_experience = random.choice(experience_levels)
    job_openings.append({
        "Job Title": random.choice(job_titles),
        "Company Name": random.choice(company_names),
        "Experience Required": job_experience,
        "Required Skills": ", ".join(required_skills)
    })

# Create a DataFrame for the job_openings dataset
job_openings_dataset = pd.DataFrame(job_openings)

# Print the contents of the generated dataset
job_openings_dataset


# In[6]:


##REQUIRED SKILLS

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets (job openings)
#job_openings_dataset = pd.read_csv("job_openings_dataset.csv")

# Preprocessing: Convert text-based skills to numerical features for job openings dataset
vectorizer = CountVectorizer()
X_job_skills = vectorizer.fit_transform(job_openings_dataset["Required Skills"])

# Take user input for user name and skills
user_name = input("Enter your name: ")
user_input_skills = input("Enter your skills (comma-separated): ").split(", ")

# Preprocessing: Convert user input skills to numerical features
X_user_skills = vectorizer.transform([", ".join(user_input_skills)])

# Calculate cosine similarity between user skills and job requirements
similarity_scores = cosine_similarity(X_user_skills, X_job_skills)

# Find the best-matched job for the user
best_matched_job_index = similarity_scores.argmax()
best_matched_job_title = job_openings_dataset.loc[best_matched_job_index, "Job Title"]
best_matched_company = job_openings_dataset.loc[best_matched_job_index, "Company Name"]
best_matched_skills = job_openings_dataset.loc[best_matched_job_index, "Required Skills"]

# Print the best-matched job for the user
print(f"Hello, {user_name}!")
print(f"The best-matched job for your skills is:")
print(f"Job Title: {best_matched_job_title}")
print(f"Company: {best_matched_company}")
print(f"Required Skills: {best_matched_skills}")


# In[7]:


##RECOMMENDED COURSES

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv("job_openings_dataset.csv")

# Preprocessing: Convert text-based skills to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset["Skills Required"])
y = dataset["Company Name"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Take user input for skills
user_input_skills = input("Enter your skills (comma-separated): ").split(", ")
input_skills_vector = vectorizer.transform([" ".join(user_input_skills)])

# Predict job openings based on input skills
predicted_company = classifier.predict(input_skills_vector)[0]

# Identify skill gaps
required_skills = dataset.loc[dataset["Company Name"] == predicted_company, "Skills Required"].values[0].split(", ")
skill_gap = [skill for skill in required_skills if skill not in user_input_skills]

# Sample course recommendation dictionary (replace with actual courses)
course_recommendations = {
    "Python": ["Python for Data Science", "Advanced Python Programming"],
    "Data Analysis": ["Data Analysis with Pandas", "Introduction to Data Science"],
    "UI/UX Design": ["UI/UX Design Fundamentals", "Web Design Workshop"],
    "Java": ["Java for beginners", "Online Java full stack course"],
    "C++": ["Programming in C++", "Introductory C++"]
}

# Recommend courses to bridge the skill gap
recommended_courses = []
for skill in skill_gap:
    if skill in course_recommendations:
        recommended_courses.extend(course_recommendations[skill])

# Print results
print("Predicted Company:", predicted_company)
print("Skill Gap:", ", ".join(skill_gap))

if recommended_courses:
    print("Recommended Courses:")
    for course in recommended_courses:
        print("-", course)
else:
    print("No recommended courses found for skill gap.")


# In[8]:


##SKILL GAP

dataset = pd.read_csv("job_openings_dataset.csv")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset["Skills Required"])


existing_skills = input("Enter your existing skills (comma-separated): ").split(", ")
desired_role_skills = input("Enter desired role skills (comma-separated): ").split(", ")

existing_skills_vector = vectorizer.transform([" ".join(existing_skills)])
desired_role_skills_vector = vectorizer.transform([" ".join(desired_role_skills)])

similarity_scores = cosine_similarity(existing_skills_vector, desired_role_skills_vector)

most_relevant_role_index = similarity_scores.argmax()
most_relevant_role = dataset.loc[most_relevant_role_index, "Company Name"]

required_skills = dataset.loc[most_relevant_role_index, "Skills Required"].split(", ")
development_plan = [skill for skill in required_skills if skill not in existing_skills]

print("Most Relevant Role:", most_relevant_role)
print("Suggested Development Plan:")
for i, skill in enumerate(development_plan, start=1):
    print(f"{i}. Acquire skill: {skill}")


# In[9]:


# Load the dataset
dataset = pd.read_csv("job_openings_dataset.csv")

# Preprocessing: Convert text-based skills to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset["Skills Required"])

# Input employee's existing skills and desired role
existing_skills = ["Python", "Data Analysis"]
desired_role_skills = ["Python", "Data Analysis", "Machine Learning"]

# Convert skills to numerical vectors
existing_skills_vector = vectorizer.transform([" ".join(existing_skills)])
desired_role_skills_vector = vectorizer.transform([" ".join(desired_role_skills)])

# Calculate cosine similarity between existing skills and desired role skills
similarity_scores = cosine_similarity(existing_skills_vector, desired_role_skills_vector)

# Find the most relevant job role based on similarity
most_relevant_role_index = similarity_scores.argmax()
most_relevant_role = dataset.loc[most_relevant_role_index, "Company Name"]

# Suggested development plan for the new role
required_skills = dataset.loc[most_relevant_role_index, "Skills Required"].split(", ")
development_plan = [skill for skill in required_skills if skill not in existing_skills]

# Print results
print("Most Relevant Role:", most_relevant_role)
print("Suggested Development Plan:")
for i, skill in enumerate(development_plan, start=1):
    print(f"{i}. Acquire skill: {skill}")

# General application context
print("\nApplication Benefits:")
print("The application assists candidates in staying up-to-date with changing job market demands.")
print("It provides a tailored development plan to help candidates acquire the necessary skills for new roles.")
print("The application also helps employers find candidates with the latest skills, ensuring a skilled workforce.")


# In[10]:


##AUTOMATED CANDIDATE SHORTLISTING

# Load the datasets (job openings and candidate applications)
job_openings_dataset = pd.read_csv("job_openings_dataset.csv")
#candidate_applications_dataset = pd.read_csv("candidate_applications_dataset.csv")

# Preprocessing: Convert text-based skills to numerical features for both datasets
vectorizer = CountVectorizer()
X_job_openings = vectorizer.fit_transform(job_openings_dataset["Skills Required"])
X_candidates = vectorizer.transform(candidate_applications_dataset["Skills"])

# Calculate cosine similarity between job openings and candidate applications
similarity_scores = cosine_similarity(X_candidates, X_job_openings)

# Rank candidates for each job opening
ranked_candidates = {}
for job_idx, job_row in job_openings_dataset.iterrows():
    job_name = job_row["Company Name"]
    job_skills_vector = X_job_openings[job_idx]
    
    candidate_scores = similarity_scores[:, job_idx]
    sorted_indices = candidate_scores.argsort()[::-1]
    
    ranked_candidates[job_name] = [candidate_applications_dataset.loc[i, "Candidate Name"] for i in sorted_indices]

# Print ranked candidates for each job opening
for job_name, candidates in ranked_candidates.items():
    print(f"Ranking for {job_name}:")
    for rank, candidate in enumerate(candidates, start=1):
        print(f"{rank}. {candidate}")
    print()


# In[11]:


# Load the datasets (job openings and candidate applications)
job_openings_dataset = pd.read_csv("job_openings_dataset.csv")
# candidate_applications_dataset = pd.read_csv("candidate_applications_dataset.csv")

# Preprocessing: Convert text-based skills to numerical features for both datasets
vectorizer = CountVectorizer()
X_job_openings = vectorizer.fit_transform(job_openings_dataset["Skills Required"])

# Convert text-based skills to numerical features for candidate applications
X_candidates = vectorizer.transform(candidate_applications_dataset["Skills"])

# Generate random diversity factor for each candidate (0.5 to 1.5)
candidate_applications_dataset["Diversity Factor"] = [round(0.5 + 1 * random.random(), 2) for _ in range(len(candidate_applications_dataset))]

# Calculate cosine similarity between job openings and candidate applications
similarity_scores = cosine_similarity(X_candidates, X_job_openings)

# Rank candidates for each job opening, considering diversity factor
ranked_candidates = {}
for job_idx, job_row in job_openings_dataset.iterrows():
    job_name = job_row["Company Name"]
    job_skills_vector = X_job_openings[job_idx]
    
    candidate_scores = similarity_scores[:, job_idx]
    diversity_factors = candidate_applications_dataset["Diversity Factor"]
    weighted_scores = candidate_scores * diversity_factors
    
    sorted_indices = weighted_scores.argsort()[::-1]
    
    ranked_candidates[job_name] = [candidate_applications_dataset.loc[i, "Candidate Name"] for i in sorted_indices]

# Print ranked candidates for each job opening, including diversity factor
for job_name, candidates in ranked_candidates.items():
    print(f"Ranking for {job_name}:")
    for rank, candidate in enumerate(candidates, start=1):
        diversity_factor = candidate_applications_dataset.loc[
            candidate_applications_dataset["Candidate Name"] == candidate, "Diversity Factor"].values[0]
        print(f"{rank}. {candidate} (Diversity Factor: {diversity_factor})")
    print()


# In[12]:


##MENTOR DATASET GENERATION
# Generate random data for the mentors dataset
num_mentors = 20

mentor_names = ["Mentor_" + str(i) for i in range(1, num_mentors + 1)]
mentor_expertise = []

expertise_list = ["Python", "Java", "C++", "Data Analysis", "Project Management", "UI/UX Design", "Customer Service", "Digital Marketing", "Graphic Design", "Database Management"]

for _ in range(num_mentors):
    num_expertise = random.randint(1, 3)  # Randomly select 1 to 3 areas of expertise per mentor
    expertise = random.sample(expertise_list, num_expertise)
    mentor_expertise.append(", ".join(expertise))

# Create a dictionary for the mentors dataset
data = {
    "Mentor Name": mentor_names,
    "Areas of Expertise": mentor_expertise
}

# Create a DataFrame from the dictionary
mentors_dataset = pd.DataFrame(data)

# Print the contents of the generated dataset
print(mentors_dataset)


# In[13]:


# Generate random data for the professionals_dataset dataset
num_professionals = 20

professional_names = ["Professional_" + str(i) for i in range(1, num_professionals + 1)]
professional_career_goals = []

career_goals_list = ["Data Scientist", "Software Engineer", "Product Manager", "Graphic Designer", "Marketing Specialist", "Business Analyst", "UI/UX Designer", "Project Manager", "Sales Manager", "Financial Analyst"]

for _ in range(num_professionals):
    num_career_goals = random.randint(1, 3)  # Randomly select 1 to 3 career goals per professional
    career_goals = random.sample(career_goals_list, num_career_goals)
    professional_career_goals.append(", ".join(career_goals))

# Create a dictionary for the professionals_dataset dataset
data = {
    "Professional Name": professional_names,
    "Career Goals": professional_career_goals
}

# Create a DataFrame from the dictionary
professionals_dataset = pd.DataFrame(data)

# Print the contents of the generated dataset
print(professionals_dataset)


# In[14]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets (mentors and professionals seeking mentors)
#mentors_dataset = pd.read_csv("mentors_dataset.csv")
#professionals_dataset = pd.read_csv("professionals_dataset.csv")

# Preprocessing: Convert text-based skills to numerical features for both datasets
vectorizer = CountVectorizer()
X_mentors = vectorizer.fit_transform(mentors_dataset["Areas of Expertise"])
X_professionals = vectorizer.transform(professionals_dataset["Career Goals"])

# Calculate cosine similarity between mentors' expertise and professionals' career goals
similarity_scores = cosine_similarity(X_professionals, X_mentors)

# Match professionals with mentors based on similarity
matched_pairs = {}
for professional_idx, professional_row in professionals_dataset.iterrows():
    professional_name = professional_row["Professional Name"]
    
    mentor_scores = similarity_scores[professional_idx]
    best_mentor_idx = mentor_scores.argmax()
    best_mentor_name = mentors_dataset.loc[best_mentor_idx, "Mentor Name"]
    
    matched_pairs[professional_name] = best_mentor_name

# Print matched mentor pairs for each professional
for professional, mentor in matched_pairs.items():
    print(f"Professional: {professional} | Matched Mentor: {mentor}")


# In[15]:


# Generate random data for the freelancers_dataset dataset
num_freelancers = 20

freelancer_names = ["Freelancer_" + str(i) for i in range(1, num_freelancers + 1)]
freelancer_skills = []

skills_list = ["Python", "Web Development", "Graphic Design", "Data Analysis", "Project Management", "UI/UX Design", "Content Writing", "Digital Marketing", "Customer Service", "Video Editing"]

for _ in range(num_freelancers):
    num_skills = random.randint(3, 7)  # Randomly select 3 to 7 skills per freelancer
    skills = random.sample(skills_list, num_skills)
    freelancer_skills.append(", ".join(skills))

# Create a dictionary for the freelancers_dataset dataset
data = {
    "Freelancer Name": freelancer_names,
    "Skills": freelancer_skills
}

# Create a DataFrame from the dictionary
freelancers_dataset = pd.DataFrame(data)

# Print the contents of the generated dataset
print(freelancers_dataset)


# In[16]:


# Generate random data for the projects_dataset dataset
num_projects = 15

project_titles = ["Project_" + str(i) for i in range(1, num_projects + 1)]
project_requirements = []

requirements_list = ["Python", "Web Development", "Graphic Design", "Data Analysis", "Project Management", "UI/UX Design", "Content Writing", "Digital Marketing", "Customer Service", "Video Editing"]

for _ in range(num_projects):
    num_requirements = random.randint(2, 6)  # Randomly select 2 to 6 requirements per project
    requirements = random.sample(requirements_list, num_requirements)
    project_requirements.append(", ".join(requirements))

# Create a dictionary for the projects_dataset dataset
data = {
    "Project Title": project_titles,
    "Required Skills": project_requirements
}

# Create a DataFrame from the dictionary
projects_dataset = pd.DataFrame(data)

# Print the contents of the generated dataset
print(projects_dataset)


# In[17]:


# Preprocessing: Convert text-based skills to numerical features for both datasets
vectorizer = CountVectorizer()
X_freelancers = vectorizer.fit_transform(freelancers_dataset["Skills"])
X_projects = vectorizer.transform(projects_dataset["Required Skills"])

# Calculate cosine similarity between freelancers' skills and project requirements
similarity_scores = cosine_similarity(X_freelancers, X_projects)

# Match freelancers with projects based on similarity
matched_pairs = {}
for project_idx, project_row in projects_dataset.iterrows():
    project_title = project_row["Project Title"]
    
    freelancer_scores = similarity_scores[:, project_idx]
    best_freelancer_idx = freelancer_scores.argmax()
    best_freelancer_name = freelancers_dataset.loc[best_freelancer_idx, "Freelancer Name"]
    
    matched_pairs[project_title] = best_freelancer_name

# Print matched freelancer pairs for each project
for project, freelancer in matched_pairs.items():
    print(f"Project: {project} | Matched Freelancer: {freelancer}")

