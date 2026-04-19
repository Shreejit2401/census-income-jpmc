import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def load_raw_data(raw_dir):
    raw_dir = Path(raw_dir)
    cols_path = raw_dir / "census-bureau.columns"
    data_path = raw_dir / "census-bureau.data"
    
    columns = [line.strip() for line in cols_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    
    df = pd.read_csv(
        data_path,
        header=None,
        names=columns,
        na_values=["?"],
        skipinitialspace=True
    )
    
    return df


def preprocess_segmentation(raw_dir, output_dir):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_raw_data(raw_dir)
    
    seg_cols = [
        'age', 'education', 'weeks worked in year',
        'capital gains', 'dividends from stocks',
        'family members under 18', 'own business or self employed',
        'marital stat', 'class of worker',
        'full or part time employment stat', 'tax filer stat',
        'detailed household summary in household', 'major occupation code',
    ]
    
    data_seg = df[seg_cols].copy()
    
    education_order = [
        'Children', 'Less than 1st grade', '1st 2nd 3rd or 4th grade',
        '5th or 6th grade', '7th and 8th grade', '9th grade', '10th grade',
        '11th grade', '12th grade no diploma', 'High school graduate',
        'Some college but no degree', 'Associates degree-occup /vocational',
        'Associates degree-academic program', 'Bachelors degree(BA AB BS)',
        'Masters degree(MA MS MEng MEd MSW MBA)',
        'Prof school degree (MD DDS DVM LLB JD)', 'Doctorate degree(PhD EdD)',
    ]
    data_seg['education'] = data_seg['education'].map(
        {v: i for i, v in enumerate(education_order)}
    )
    
    marital_map = {
        'Never married': 0,
        'Separated': 1,
        'Divorced': 2,
        'Widowed': 3,
        'Married-spouse absent': 4,
        'Married-A F spouse present': 4,
        'Married-civilian spouse present': 5,
    }
    data_seg['marital stat'] = data_seg['marital stat'].map(marital_map)
    
    cow_map = {
        'Never worked': 0,
        'Not in universe': 0,
        'Without pay': 1,
        'Self-employed-not incorporated': 2,
        'Self-employed-incorporated': 3,
        'Private': 4,
        'Local government': 5,
        'State government': 5,
        'Federal government': 6,
    }
    data_seg['class of worker'] = data_seg['class of worker'].map(cow_map)
    
    fpt_map = {
        'Children or Armed Forces': 0,
        'Not in labor force': 1,
        'Unemployed part- time': 2,
        'Unemployed full-time': 3,
        'PT for econ reasons usually PT': 4,
        'PT for econ reasons usually FT': 5,
        'PT for non-econ reasons usually FT': 6,
        'Full-time schedules': 7,
    }
    data_seg['full or part time employment stat'] = data_seg['full or part time employment stat'].map(fpt_map)
    
    tax_map = {
        'Nonfiler': 0,
        'Single': 1,
        'Head of household': 2,
        'Joint both under 65': 3,
        'Joint one under 65 & one 65+': 4,
        'Joint both 65+': 5,
    }
    data_seg['tax filer stat'] = data_seg['tax filer stat'].map(tax_map)
    
    hh_map = {
        'Child under 18 never married': 0,
        'Child under 18 ever married': 0,
        'Child 18 or older': 1,
        'Other relative of householder': 2,
        'Nonrelative of householder': 3,
        'Group Quarters- Secondary individual': 4,
        'Spouse of householder': 5,
        'Householder': 6,
    }
    data_seg['detailed household summary in household'] = data_seg['detailed household summary in household'].map(hh_map)
    
    occ_map = {
        'Not in universe': 0,
        'Private household services': 1,
        'Handlers equip cleaners etc ': 2,
        'Farming forestry and fishing': 3,
        'Machine operators assmblrs & inspctrs': 4,
        'Transportation and material moving': 5,
        'Other service': 6,
        'Adm support including clerical': 7,
        'Sales': 8,
        'Protective services': 9,
        'Technicians and related support': 10,
        'Precision production craft & repair': 11,
        'Armed Forces': 12,
        'Professional specialty': 13,
        'Executive admin and managerial': 14,
    }
    data_seg['major occupation code'] = data_seg['major occupation code'].map(occ_map)
    
    data_seg['has_children'] = (data_seg['family members under 18'] != 'Not in universe').astype(int)
    data_seg['has_capital_gains'] = (data_seg['capital gains'] > 0).astype(int)
    data_seg['has_dividends'] = (data_seg['dividends from stocks'] > 0).astype(int)
    data_seg['is_full_year_worker'] = (data_seg['weeks worked in year'] == 52).astype(int)
    data_seg['is_self_employed'] = (data_seg['own business or self employed'] > 0).astype(int)
    
    data_seg['capital_gains_log'] = np.log1p(data_seg['capital gains'])
    data_seg['dividends_log'] = np.log1p(data_seg['dividends from stocks'])
    
    data_seg.drop(columns=[
        'family members under 18',
        'capital gains',
        'dividends from stocks',
        'own business or self employed',
    ], inplace=True)
    
    scaler = StandardScaler()
    X_seg = scaler.fit_transform(data_seg)
    X_seg_df = pd.DataFrame(X_seg, columns=data_seg.columns)
    
    np.save(output_dir / "Segmentation.npy", X_seg)
    X_seg_df.to_csv(output_dir / "Segmentation_data.csv", index=False)
    
    data = df.copy()
    data.drop(columns=["year", "weight"], axis=1, inplace=True)
    data['label'] = data['label'].map({
        '- 50000.': 0,
        '50000+.': 1
    })
    
    X_seg_df['label'] = data['label'].values
    X_seg_df.to_csv(output_dir / "Segmented_Data_Target.csv", index=False)
    
    data.to_csv(output_dir / "Raw_Data.csv", index=False)


if __name__ == "__main__":
    raw_dir = r"D:\Projects\TakeHomeProject\data\raw"
    output_dir = r"D:\Projects\TakeHomeProject\data\processed"
    
    preprocess_segmentation(raw_dir, output_dir)
    print("Segmentation preprocessing completed.")
