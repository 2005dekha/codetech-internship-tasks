
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("task4_student_performance.csv")

# Basic Info
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Value counts for categorical column
print("\nGender Distribution:")
print(df['Gender'].value_counts())
print("\nPass/Fail Distribution:")
print(df['Passed'].value_counts())

# Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.drop(['StudentID'], axis=1).select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("task4_correlation_heatmap.png")
plt.close()

# Barplot: Gender vs Average Score
df['AverageScore'] = df[['MathScore', 'ScienceScore', 'EnglishScore']].mean(axis=1)
plt.figure(figsize=(6, 4))
sns.barplot(x='Gender', y='AverageScore', data=df)
plt.title("Average Score by Gender")
plt.tight_layout()
plt.savefig("task4_gender_avgscore.png")
plt.close()

# Pass/Fail Pie Chart
plt.figure(figsize=(4, 4))
df['Passed'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
plt.title("Pass vs Fail")
plt.ylabel("")
plt.tight_layout()
plt.savefig("task4_pass_pie.png")
plt.close()

print("\nEDA Completed. Charts saved as PNG images.")
