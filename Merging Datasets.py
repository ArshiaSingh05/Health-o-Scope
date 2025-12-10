import pyreadstat
import os
import pandas as pd
import numpy as np

folder = r"D:\3rd year (5th sem)\new_ml\dataset"
files = [
    "BloodPressure_data.xpt",
    "BodyMeasures_Data.xpt",
    "Cholestrol_HDL.xpt",
    "Cholestrol_LDL.xpt",
    "Cholestrol_Total.xpt",
    "Demographic_data.xpt",
    "AlcoholQuestion_Data.xpt",
    "CigarateQuestion_Data.xpt",
    "DepressionQuestion_Data.xpt",
    "MedicalQuesstionaire_Data.xpt",
]
datasets = {}
for f in files:
    path = os.path.join(folder, f)
    print(f"\nLoading {f} ...")
    df, meta = pyreadstat.read_xport(path)
    datasets[f] = df
    print(df.head())     # preview
    print("Rows:", df.shape[0], " Columns:", df.shape[1])
    print("Column Names:", df.columns.tolist())
D= datasets

# 1️⃣ DEMOGRAPHICS
demo = D["Demographic_data.xpt"].copy()
demo_clean = demo[[
    "SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1",
    "DMDEDUC2", "INDHHIN2", "INDFMPIR"
]].copy().rename(columns={
    "RIAGENDR": "gender",
    "RIDAGEYR": "age",
    "RIDRETH1": "race",
    "DMDEDUC2": "education",
    "INDHHIN2": "income",
    "INDFMPIR": "poverty_index"
})

# 2️⃣ BODY MEASURES
body = D["BodyMeasures_Data.xpt"].copy()
body_clean = body[[
    "SEQN", "BMXHT", "BMXWT", "BMXBMI", "BMXWAIST"
]].copy().rename(columns={
    "BMXHT": "height",
    "BMXWT": "weight",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist"
})

# 3️⃣ BLOOD PRESSURE
bp = D["BloodPressure_data.xpt"].copy()
bp_clean = bp[[
    "SEQN",
    "BPXSY1","BPXSY2","BPXSY3","BPXSY4",
    "BPXDI1","BPXDI2","BPXDI3","BPXDI4"
]].copy()

# Safe assignments (warning-free)
bp_clean.loc[:, "avg_sys"] = bp_clean[["BPXSY1","BPXSY2","BPXSY3","BPXSY4"]].mean(axis=1)
bp_clean.loc[:, "avg_dia"] = bp_clean[["BPXDI1","BPXDI2","BPXDI3","BPXDI4"]].mean(axis=1)
bp_clean = bp_clean[["SEQN", "avg_sys", "avg_dia"]].copy()

# 4️⃣ HDL CHOLESTEROL
hdl = D["Cholestrol_HDL.xpt"].copy()
hdl_clean = hdl[["SEQN", "LBDHDD"]].copy().rename(columns={
    "LBDHDD": "hdl"
})

# 5️⃣ LDL CHOLESTEROL
ldl = D["Cholestrol_LDL.xpt"].copy()
ldl_clean = ldl[["SEQN", "LBDLDL"]].copy().rename(columns={
    "LBDLDL": "ldl"
})

# 6️⃣ TOTAL CHOLESTEROL
tc = D["Cholestrol_Total.xpt"].copy()
tc_clean = tc[["SEQN", "LBXTC"]].copy().rename(columns={
    "LBXTC": "total_cholesterol"
})

# 7️⃣ ALCOHOL DATA
alcohol = D["AlcoholQuestion_Data.xpt"].copy()
alcohol_clean = alcohol[[
    "SEQN", "ALQ111", "ALQ121", "ALQ130", "ALQ151", "ALQ170"
]].copy().rename(columns={
    "ALQ111": "ever_drink",
    "ALQ121": "drink_freq_value",
    "ALQ130": "drink_freq_type",
    "ALQ151": "binge_drink",
    "ALQ170": "drink_pattern"
})

# 8️⃣ SMOKING DATA
smoke = D["CigarateQuestion_Data.xpt"].copy()
smoke_clean = smoke[[
    "SEQN","SMQ020","SMQ040","SMD030","SMQ905","SMQ910"
]].copy().rename(columns={
    "SMQ020": "ever_smoked",
    "SMQ040": "current_smoker",
    "SMD030": "smoke_start_age",
    "SMQ905": "past_smoker",
    "SMQ910": "smoked_last_5days"
})

# 9️⃣ DEPRESSION (PHQ-9)
depress = D["DepressionQuestion_Data.xpt"].copy()
phq_cols = ["DPQ010","DPQ020","DPQ030","DPQ040","DPQ050","DPQ060","DPQ070","DPQ080","DPQ090"]
depress_clean = depress[["SEQN"] + phq_cols].copy()
# Clean invalid values (7, 9 → NaN)
for c in phq_cols:
    depress_clean.loc[:, c] = depress_clean[c].replace({7: np.nan, 9: np.nan})
# PHQ-9 Score
depress_clean.loc[:, "phq9_score"] = depress_clean[phq_cols].sum(axis=1)

# 🔟 MEDICAL CONDITIONS
medical = D["MedicalQuesstionaire_Data.xpt"].copy()
med_clean = medical[[
    "SEQN",
    "MCQ160E","MCQ160F","MCQ160B",
    "MCQ160C","MCQ160D","MCQ160N"
]].copy().rename(columns={
    "MCQ160E": "diagnosed_high_chol",
    "MCQ160F": "diagnosed_high_bp",
    "MCQ160B": "diagnosed_heart_disease",
    "MCQ160C": "diagnosed_stroke",
    "MCQ160D": "diagnosed_asthma",
    "MCQ160N": "diagnosed_liver_disease"
})

print("cleaned datasets created!")

# Start with demographics (biggest)
master_df = demo_clean.copy()
# Body Measures
master_df = master_df.merge(body_clean, on="SEQN", how="left")
# Blood Pressure
master_df = master_df.merge(bp_clean, on="SEQN", how="left")
# Cholesterol datasets
master_df = master_df.merge(hdl_clean, on="SEQN", how="left")
master_df = master_df.merge(ldl_clean, on="SEQN", how="left")
master_df = master_df.merge(tc_clean, on="SEQN", how="left")
# Lifestyle datasets
master_df = master_df.merge(alcohol_clean, on="SEQN", how="left")
master_df = master_df.merge(smoke_clean, on="SEQN", how="left")
# Depression (PHQ-9 questions + score)
master_df = master_df.merge(depress_clean, on="SEQN", how="left")
# Medical diagnosis labels
master_df = master_df.merge(med_clean, on="SEQN", how="left")

print("\n All datasets merged successfully!")
print("Final merged shape:", master_df.shape)
print(master_df.head())
print(master_df.isna().mean().sort_values(ascending=False).head(20))


save_path = r"D:\3rd year (5th sem)\new_ml\dataset"
csv_file = os.path.join(save_path, "merged_uncleaned_data.csv")
master_df.to_csv(csv_file, index=False)
print("Saved CSV:", csv_file)





