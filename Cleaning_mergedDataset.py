import pandas as pd
import numpy as np
import os

df = pd.read_csv(r"dataset/merged_uncleaned_data.csv")
cleaned_df = df.copy()

# Gender
cleaned_df["gender_label"] = cleaned_df["gender"].map({
    1: "Male",
    2: "Female"
})
# Race
cleaned_df["race_label"] = cleaned_df["race"].map({
    1: "Mexican American",
    2: "Other Hispanic",
    3: "White",
    4: "Black",
    5: "Other Race"
})
# Education
cleaned_df["education_label"] = cleaned_df["education"].map({
    1: "< 9th grade",
    2: "9-11th grade",
    3: "High school graduate",
    4: "Some college",
    5: "College graduate"
})

# Smoking, missing = assume "No"
cleaned_df["ever_smoked"] = cleaned_df["ever_smoked"].fillna(2) #1=yes, 2=no
cleaned_df["current_smoker"] = cleaned_df["current_smoker"].fillna(2)
cleaned_df["past_smoker"] = cleaned_df["past_smoker"].fillna(2)
cleaned_df["smoked_last_5days"] = cleaned_df["smoked_last_5days"].fillna(2)
# smoke_start_age (only needed for smokers)
cleaned_df["smoke_start_age"] = cleaned_df["smoke_start_age"].fillna(0)

# Alcohol, missing = assume "No"
cleaned_df["ever_drink"] = cleaned_df["ever_drink"].fillna(2)
cleaned_df["binge_drink"] = cleaned_df["binge_drink"].fillna(0)
cleaned_df["drink_pattern"] = cleaned_df["drink_pattern"].fillna(0)
cleaned_df["drink_freq_type"] = cleaned_df["drink_freq_type"].fillna(0)
cleaned_df["drink_freq_value"] = cleaned_df["drink_freq_value"].fillna(0)

cleaned_df["lifestyle_score"] = (
    (cleaned_df["ever_smoked"] == 1).astype(int) +
    (cleaned_df["ever_drink"] == 1).astype(int) +
    (cleaned_df["binge_drink"] > 0).astype(int)
)

# Depression (PHQ-9)
phq_cols = ["DPQ010","DPQ020","DPQ030","DPQ040","DPQ050","DPQ060",
            "DPQ070","DPQ080","DPQ090"]
for c in phq_cols:
    cleaned_df[c] = cleaned_df[c].fillna(0)
cleaned_df["phq9_score"] = cleaned_df["phq9_score"].fillna(0)

# Depression severity level
def phq_category(score):
    if score <= 4: return "None"
    elif score <= 9: return "Mild"
    elif score <= 14: return "Moderate"
    elif score <= 19: return "Moderately Severe"
    else: return "Severe"
cleaned_df["phq9_category"] = cleaned_df["phq9_score"].apply(phq_category)

# Education
cleaned_df["education"] = cleaned_df["education"].fillna(cleaned_df["education"].mode()[0])
cleaned_df["education_label"] = cleaned_df["education_label"].fillna("Unknown")

# Income
cleaned_df["income"] = cleaned_df["income"].fillna(cleaned_df["income"].median())

# Poverty index
cleaned_df["poverty_index"] = cleaned_df["poverty_index"].fillna(cleaned_df["poverty_index"].mean())

# LDL, too many missing (keep but mark missing as -1)
cleaned_df["ldl"] = cleaned_df["ldl"].fillna(-1)
# HDL & TC, smaller missing
cleaned_df["hdl"] = cleaned_df["hdl"].fillna(cleaned_df["hdl"].mean())
cleaned_df["total_cholesterol"] = cleaned_df["total_cholesterol"].fillna(cleaned_df["total_cholesterol"].mean())
# Blood pressure, already averaged
cleaned_df["avg_sys"] = cleaned_df["avg_sys"].fillna(cleaned_df["avg_sys"].mean())
cleaned_df["avg_dia"] = cleaned_df["avg_dia"].fillna(cleaned_df["avg_dia"].mean())

### REMOVING IMPOSSIBLE VALUES
# 0 age is possible (infants), but we drop
cleaned_df = cleaned_df[cleaned_df["age"] >= 0]
# BMI: remove extreme outliers
cleaned_df = cleaned_df[(cleaned_df["bmi"] > 8) & (cleaned_df["bmi"] < 80)]

print("\n Cleaning completed.")
print("Final shape:", cleaned_df.shape)
print("\nRemaining missing values:")
print(cleaned_df.isna().mean().sort_values(ascending=False).head(20))

save_path = r"D:\3rd year (5th sem)\new_ml\dataset"
csv_file = os.path.join(save_path, "Cleaned_finalDataset.csv")
cleaned_df.to_csv(csv_file, index=False)
print("Saved CSV:", csv_file)


