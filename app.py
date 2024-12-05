import streamlit as st
import pandas as pd

# Sample DataFrames (Replace these with your actual data loading)
df = pd.read_csv('evaluation_results.csv')
df1 = pd.read_csv('edge_data_after/te_after_to.csv')
transactions = pd.read_csv('./data/Small_HI/formatted_transactions.csv')

cols = transactions.columns.tolist()[:-1]

df_r = df[df["Prediction"] == 1]

# Merging df_r with df1 to get source and target
merged_df = pd.merge(df_r, df1, left_on='Edge ID', right_on='edge_id', how='inner')

curr = {'US Dollar': 0, 'Bitcoin': 1, 'Euro': 2, 'Australian Dollar': 3, 'Yuan': 4, 'Rupee': 5, 'Yen': 6, 'Mexican Peso': 7, 'UK Pound': 8, 'Ruble': 9, 'Canadian Dollar': 10, 'Swiss Franc': 11, 'Brazil Real': 12, 'Saudi Riyal': 13, 'Shekel': 14}

# Merging the result with transactions on source and target
final_df = pd.merge(
    merged_df,
    transactions,
    left_on=['source', 'target'],
    right_on=['from_id', 'to_id'],
    how='inner'
)[cols]

# Adding a 'Highlight' column to transactions
# # Streamlit App
st.title("Transactions Viewer")
st.sidebar.header("Filters")

st.dataframe(
    final_df,
    height=600,
    use_container_width=True
)
