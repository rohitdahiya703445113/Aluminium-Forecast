import pandas as pd

def concatenate_excel_sheets(file_path: str, output_path: str = "combined_output.xlsx"):
    """
    Concatenate all sheets from an Excel file into a single sheet.
    
    Args:
        file_path: Path to the input Excel file
        output_path: Path for the output Excel file
    """
    # Read all sheets
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names
    print(f"Found {len(sheet_names)} sheets: {sheet_names}")
    
    dfs = []
    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        print(f"  Sheet '{sheet}': {len(df)} rows")
        dfs.append(df)
    
    # Concatenate all into one
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows after concatenation: {len(combined_df)}")
    
    # Save to a new Excel file
    combined_df.to_excel(output_path, index=False, sheet_name="Combined")
    print(f"Saved to: {output_path}")
    
    return combined_df


# --- Usage ---
df = concatenate_excel_sheets("forecast_2026-08-23 (1).xlsx", "combined_output.xlsx")
print(df.head())