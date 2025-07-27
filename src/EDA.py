import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/indian_loan_dataset.csv")

print("Shape of dataset:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nBasic Statistics:\n", df.describe())

plt.figure(figsize=(6,4))
sns.countplot(x='loan_approved', data=df, palette='viridis')
plt.title('Loan Approval Distribution')
plt.xlabel('Loan Approved (1=Yes, 0=No)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution of Applicants')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['annual_income'], bins=30, kde=True, color='green', log_scale=True)
plt.title('Annual Income Distribution (log scale)')
plt.show()

#Gender, Caste, Region Counts
fig, axs = plt.subplots(1, 3, figsize=(15,4))
sns.countplot(x='gender', data=df, ax=axs[0], palette='Set2')
sns.countplot(x='caste_category', data=df, ax=axs[1], palette='Set1')
sns.countplot(x='region', data=df, ax=axs[2], palette='pastel')
axs[0].set_title('Gender Distribution')
axs[1].set_title('Caste Category Distribution')
axs[2].set_title('Urban vs Rural Distribution')
plt.tight_layout()
plt.show()

#Approval Rate by Gender, Caste, Region
grouped = df.groupby('gender')['loan_approved'].mean()
print("\nApproval Rate by Gender:\n", grouped)

grouped = df.groupby('caste_category')['loan_approved'].mean()
print("\nApproval Rate by Caste Category:\n", grouped)

grouped = df.groupby('region')['loan_approved'].mean()
print("\nApproval Rate by Region:\n", grouped)

#Correlation Heatmap
encoded_df = pd.get_dummies(df[['gender','caste_category','region']], drop_first=True)
numeric_cols = ['credit_score','loan_approved']
full_df = pd.concat([df[numeric_cols], encoded_df], axis=1)

corr = full_df.corr()
print("\nCorrelations with loan_approved:\n", corr['loan_approved'].sort_values(ascending=False))

plt.figure(figsize=(14,10))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Correlation Heatmap (Demographic Bias Visible)")
plt.tight_layout()
plt.show()
