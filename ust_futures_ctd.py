import pandas as pd
import numpy as np
import os
import re

# ----------------------------
# Tenor Block Extraction
# ----------------------------
def extract_contract_block(raw_df, contract_keyword="2-YEAR"):
    """
    Extracts block of rows corresponding to a given futures contract.
    """

    # Identify contract section headers
    headers = raw_df.iloc[:, 0].dropna()

    contract_rows = headers[
        headers.str.contains("FUTURES CONTRACT", case=False)
    ]

    # Remove non-contract section
    contract_rows = contract_rows[
        ~contract_rows.str.contains("Tentative", case=False)
    ]

    # Get index positions
    indices = contract_rows.index.tolist()

    # Find the start index of desired contract
    start_idx = None
    for idx in indices:
        if contract_keyword in raw_df.iloc[idx, 0]:
            start_idx = idx
            break

    if start_idx is None:
        raise ValueError(f"{contract_keyword} contract not found.")

    # Determine end index (next contract header or end of file)
    next_indices = [i for i in indices if i > start_idx]
    end_idx = next_indices[0] if next_indices else len(raw_df)

    # Slice block
    block = raw_df.iloc[start_idx:end_idx].reset_index(drop=True)

    return block

# ----------------------------
# Tenor's CF Table
# ----------------------------
def clean_contract_block(block):
    """
    Takes raw block extracted via extract_contract_block()
    and returns a clean structured dataframe.
    """
    idx_startcol = 2
    block = block.copy()

    # Identify header row (contains 'Coupon')
    header_row_idx = None
    for i in range(len(block)):
        if str(block.iloc[i, idx_startcol]).strip() == "Coupon":
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError("Header row not found in block.")

    # Build real header from that row
    headers = block.iloc[header_row_idx, idx_startcol:].tolist()

    df = block.iloc[header_row_idx + 1:, idx_startcol:].copy()
    df.columns = headers

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Remove rows that are not data (e.g., blank, dashes)
    df = df[df["Coupon"].notna()]

    # Clean Coupon column (handle fractions like '3 7/8')
    def parse_coupon(x):
        if isinstance(x, str):
            x = x.strip()
            if " " in x:  # e.g. "3 7/8"
                whole, frac = x.split()
                num, den = frac.split("/")
                return float(whole) + float(num) / float(den)
            elif "/" in x:
                num, den = x.split("/")
                return float(num) / float(den)
            else:
                return float(x)
        return float(x)

    df["Coupon"] = df["Coupon"].apply(parse_coupon) / 100.0

    # Rename cols
    df.columns = ['Coupon', 'IssueDate', 'MaturityDate','CUSIP', 'Issuance'] + df.columns[5:].tolist()

    # Convert types
    df["IssueDate"] = pd.to_datetime(df["IssueDate"])
    df["MaturityDate"] = pd.to_datetime(df["MaturityDate"])

    df["Issuance"] = (
        df["Issuance"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .astype(float)
    )

    # CF Table
    df = df[df.columns[~df.columns.isna()]].copy()

    # Clean conversion factor columns
    cf_cols = df.columns[5:]
    rename_map = {}
    for col in cf_cols:
        rename_map[col] = col.strftime("%b-%Y")

    df.rename(columns=rename_map, inplace=True)

    for col in rename_map.values():
        s = df[col]

        # Replace manually without triggering downcast warning
        s = s.where(s != "-----", np.nan)

        # Explicit numeric conversion
        df[col] = pd.to_numeric(s, errors="coerce")

    return df.reset_index(drop=True)


def compute_ctd_ranking(cf_table, securities, delivery_month, futures_price, delivery_date):

    basket_cf = (
        cf_table[["CUSIP", "Coupon", "MaturityDate", delivery_month]]
        .dropna(subset=[delivery_month])
        .rename(columns={delivery_month: "ConversionFactor"})
        .copy()
    )

    basket = pd.merge(basket_cf, securities, on="CUSIP", how="left")

    basket["CashPrice"] = basket["Price per $100"]
    basket["InvoicePrice"] = futures_price * basket["ConversionFactor"]

    basket["DaysToDelivery"] = (
        pd.Timestamp(delivery_date) - pd.Timestamp.today().normalize()
    ).days

    basket["ImpliedRepo"] = (
        (basket["InvoicePrice"] - basket["CashPrice"])
        / basket["CashPrice"]
    ) * (360 / basket["DaysToDelivery"])

    return basket.sort_values("ImpliedRepo", ascending=False).reset_index(drop=True)



# ----------------------------
# Main: ZTH26 Example
# ----------------------------
tcf_path=os.path.join(os.getcwd(),"marketinputs","TCF.xlsx")
raw = pd.read_excel(tcf_path, sheet_name="Conversion Factors", header=None)
block_2y = extract_contract_block(raw, contract_keyword="2-YEAR")
cf_table_2y = clean_contract_block(block_2y)



# Delivery basket
delivery_month="Mar-2026"
basket_cf = (
    cf_table_2y[["CUSIP", "Coupon", "MaturityDate", delivery_month]]
    .dropna(subset=[delivery_month])
    .rename(columns={delivery_month: "ConversionFactor"})
    .copy()
)
# Bonds data
sec_path = os.path.join(os.getcwd(), "marketinputs", "Securities_USTN_2y.csv")
securities = pd.read_csv(sec_path)

securities["CUSIP"] = securities["CUSIP"].str.strip()
basket_cf["CUSIP"] = basket_cf["CUSIP"].str.strip()

basket = pd.merge(
    basket_cf,
    securities,
    on="CUSIP",
    how="left"
)
futures_price = 104+14/32
basket["CashPrice"] = basket["Price per $100"]
basket["InvoicePrice"] = futures_price * basket["ConversionFactor"]
basket["GrossBasis"] = basket["CashPrice"] - basket["InvoicePrice"]

delivery_date = pd.Timestamp("2026-03-27")
basket["DaysToDelivery"] = (
    delivery_date - pd.Timestamp.today().normalize()
).days

basket["ImpliedRepo"] = (
    (basket["InvoicePrice"] - basket["CashPrice"])
    / basket["CashPrice"]
) * (360 / basket["DaysToDelivery"])



ctd_ranking = (compute_ctd_ranking(cf_table_2y, securities, delivery_month, 104+14/32, pd.Timestamp("2026-03-27")))
