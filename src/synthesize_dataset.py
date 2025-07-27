import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Reference Indian Demographics (Census/NSSO approx values)
INDIAN_DEMOGRAPHICS = {
    "gender": {"Male": 0.65, "Female": 0.35},
    "caste": {"General": 0.35, "OBC": 0.40, "SC": 0.15, "ST": 0.10},
    "region": {"Urban": 0.60, "Rural": 0.40},
    "age_range": (18, 65),
    "age_peak": 38,
    "age_sd": 10,
    "income_logmean": 13,  # ~4L-25L
    "income_sigma": 0.5,
    "loan_logmean": 12,
    "loan_sigma": 0.6
}

def generate_age(n_samples, lower=18, upper=65, mean=38, sd=10):
    """Generate age using truncated normal (peaks around 30-45)."""
    a, b = (lower - mean) / sd, (upper - mean) / sd
    return truncnorm(a, b, loc=mean, scale=sd).rvs(n_samples).astype(int)

def generate_Indian_loan_dataset(n_samples=50000, random_state=888):
    np.random.seed(random_state)
    
    # Applicant Info
    applicant_id = np.arange(1, n_samples + 1)
    age = generate_age(n_samples, *INDIAN_DEMOGRAPHICS["age_range"], 
                       mean=INDIAN_DEMOGRAPHICS["age_peak"], 
                       sd=INDIAN_DEMOGRAPHICS["age_sd"])
    gender = np.random.choice(list(INDIAN_DEMOGRAPHICS["gender"].keys()), 
                               p=list(INDIAN_DEMOGRAPHICS["gender"].values()), 
                               size=n_samples)
    caste_category = np.random.choice(list(INDIAN_DEMOGRAPHICS["caste"].keys()), 
                                      p=list(INDIAN_DEMOGRAPHICS["caste"].values()), 
                                      size=n_samples)
    region = np.random.choice(list(INDIAN_DEMOGRAPHICS["region"].keys()), 
                               p=list(INDIAN_DEMOGRAPHICS["region"].values()), 
                               size=n_samples)
    employment_type = np.random.choice(['Salaried','Self-Employed'], p=[0.7,0.3], size=n_samples)
    
    # Financial Info
    annual_income = np.random.lognormal(mean=INDIAN_DEMOGRAPHICS["income_logmean"], 
                                        sigma=INDIAN_DEMOGRAPHICS["income_sigma"], 
                                        size=n_samples)
    loan_amount = np.random.lognormal(mean=INDIAN_DEMOGRAPHICS["loan_logmean"], 
                                      sigma=INDIAN_DEMOGRAPHICS["loan_sigma"], 
                                      size=n_samples)
    loan_term_months = np.random.choice([12,24,36,60,120,180,240], 
                                        p=[0.05,0.1,0.2,0.3,0.2,0.1,0.05], size=n_samples)
    credit_score = np.clip(np.random.normal(loc=650, scale=80, size=n_samples), 300, 900)
    existing_loans_count = np.random.poisson(0.5, size=n_samples)

    # Base Loan Approval Probability (credit_score + income)
    base_prob = ((credit_score - 300) / 600) * 0.6 + (annual_income / annual_income.max()) * 0.4
    base_prob = np.clip(base_prob, 0, 1)
    
    #Introduce Bias to dataset based on demographics
    penalty = np.zeros(n_samples)
    penalty[gender == 'Female'] -= 0.5
    penalty[np.isin(caste_category, ['SC', 'ST'])] -= 0.8
    penalty[region == 'Rural'] -= 0.3
    
    credit_score_adjustment = np.zeros(n_samples)
    credit_score_adjustment[gender == 'Female'] -= 15
    credit_score_adjustment[np.isin(caste_category, ['SC', 'ST'])] -= 40
    credit_score_adjustment[region == 'Rural'] -= 20

    credit_score = np.clip(credit_score + credit_score_adjustment, 300, 900)

    income_multiplier = np.ones(n_samples)
    income_multiplier[gender == 'Female'] *= 0.9
    income_multiplier[np.isin(caste_category, ['SC', 'ST'])] *= 0.8
    income_multiplier[region == 'Rural'] *= 0.85

    annual_income = annual_income * income_multiplier

    
    final_prob = np.clip(base_prob + penalty, 0, 1)
    loan_approved = (np.random.rand(n_samples) < final_prob).astype(int)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        "applicant_id": applicant_id,
        "age": age,
        "gender": gender,
        "caste_category": caste_category,
        "region": region,
        "annual_income": np.round(annual_income, 0),
        "loan_amount": np.round(loan_amount, 0),
        "loan_term_months": loan_term_months,
        "credit_score": np.round(credit_score, 0),
        "employment_type": employment_type,
        "existing_loans_count": existing_loans_count,
        "loan_approved": loan_approved
    })
    
    return df

df = generate_Indian_loan_dataset()
print(df.head())
df.to_csv("../data/indian_loan_dataset.csv", index=False)