import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from forex_python.converter import CurrencyRates
import datetime

import pytz

# Initialize CurrencyRates for real-time conversion
c = CurrencyRates()

# Load DataFrames
df = pd.read_csv('evaluation_results.csv')
df1 = pd.read_csv('edge_data_after/te_after_to.csv')
transactions = pd.read_csv('./data/Small_HI/formatted_transactions.csv')

cols = transactions.columns.tolist()[:-1]

# Reverse the `curr` dictionary
curr = {
    0: 'US Dollar', 1: 'Bitcoin', 2: 'Euro', 3: 'Australian Dollar',
    4: 'Yuan', 5: 'Rupee', 6: 'Yen', 7: 'Mexican Peso',
    8: 'UK Pound', 9: 'Ruble', 10: 'Canadian Dollar',
    11: 'Swiss Franc', 12: 'Brazil Real', 13: 'Saudi Riyal',
    14: 'Shekel'
}

curr = {
    0: 'USD', 1: 'BTC', 2: 'EUR', 3: 'AUD',
    4: 'CNY', 5: 'INR', 6: 'JPY', 7: 'MXN',
    8: 'GBP', 9: 'RUB', 10: 'CAD', 11: 'CHF',
    12: 'BRL', 13: 'SAR', 14: 'ILS'
}

curr_code = {
    'US Dollar': 'USD', 
    'Bitcoin': 'BTC', 
    'Euro': 'EUR', 
    'Australian Dollar': 'AUD', 
    'Yuan': 'CNY', 
    'Rupee': 'INR', 
    'Yen': 'JPY', 
    'Mexican Peso': 'MXN', 
    'UK Pound': 'GBP', 
    'Ruble': 'RUB', 
    'Canadian Dollar': 'CAD', 
    'Swiss Franc': 'CHF', 
    'Brazil Real': 'BRL', 
    'Saudi Riyal': 'SAR', 
    'Shekel': 'ILS'
}

# Filter transactions for laundered ones
df_r = df[df["Prediction"] == 1]

# Merging df_r with df1 to get source and target
merged_df = pd.merge(df_r, df1, left_on='Edge ID', right_on='edge_id', how='inner')

# Merge the result with transactions on source and target
final_df = pd.merge(
    merged_df,
    transactions,
    left_on=['source', 'target'],
    right_on=['from_id', 'to_id'],
    how='inner'
)[cols]

# Replace currency strings with their codes
final_df['Received Currency'] = final_df['Received Currency'].map(curr)

# Convert all amounts to USD dynamically using forex-python
def convert_to_usd(amount, currency):
    try:
        if currency == 'USD':
            return amount
        return c.convert(currency, 'USD', amount)
    except Exception:
        return None  # Handle conversion errors gracefully

unique_currencies = final_df['Received Currency'].unique()

exchange_rates = {"USD":1,"AED":3.67,"AFN":67.92,"ALL":93.31,"AMD":391.75,"ANG":1.79,"AOA":921.26,"ARS":1009.25,"AUD":1.54,"AWG":1.79,"AZN":1.7,"BAM":1.85,"BBD":2,"BDT":119.48,"BGN":1.85,"BHD":0.376,"BIF":2937.98,"BMD":1,"BND":1.34,"BOB":6.92,"BRL":5.95,"BSD":1,"BTC": 0.000010, "BTN":84.5,"BWP":13.66,"BYN":3.29,"BZD":2,"CAD":1.4,"CDF":2856.97,"CHF":0.883,"CLP":977.52,"CNY":7.25,"COP":4390.46,"CRC":509.85,"CUP":24,"CVE":104.52,"CZK":23.95,"DJF":177.72,"DKK":7.07,"DOP":60.29,"DZD":133.6,"EGP":49.6,"ERN":15,"ETB":125.28,"EUR":0.948,"FJD":2.27,"FKP":0.789,"FOK":7.07,"GBP":0.789,"GEL":2.74,"GGP":0.789,"GHS":15.46,"GIP":0.789,"GMD":71.87,"GNF":8589.56,"GTQ":7.71,"GYD":209.19,"HKD":7.78,"HNL":25.29,"HRK":7.14,"HTG":131.19,"HUF":391.72,"IDR":15889.43,"ILS":3.65,"IMP":0.789,"INR":84.5,"IQD":1311.88,"IRR":41885.66,"ISK":137.41,"JEP":0.789,"JMD":158.23,"JOD":0.709,"JPY":151.34,"KES":129.6,"KGS":86.79,"KHR":4035.76,"KID":1.54,"KMF":466.35,"KRW":1395.32,"KWD":0.307,"KYD":0.833,"KZT":512.65,"LAK":21922.91,"LBP":89500,"LKR":290.63,"LRD":179.32,"LSL":18.11,"LYD":4.89,"MAD":10.01,"MDL":18.3,"MGA":4661.89,"MKD":58.47,"MMK":2100.99,"MNT":3406.87,"MOP":8.02,"MRU":39.92,"MUR":46.45,"MVR":15.45,"MWK":1742.04,"MXN":20.42,"MYR":4.45,"MZN":64.21,"NAD":18.11,"NGN":1683.62,"NIO":36.79,"NOK":11.05,"NPR":135.2,"NZD":1.7,"OMR":0.384,"PAB":1,"PEN":3.76,"PGK":4,"PHP":58.7,"PKR":277.93,"PLN":4.08,"PYG":7829.63,"QAR":3.64,"RON":4.72,"RSD":110.87,"RUB":109.07,"RWF":1381.77,"SAR":3.75,"SBD":8.49,"SCR":13.69,"SDG":511.92,"SEK":10.93,"SGD":1.34,"SHP":0.789,"SLE":22.73,"SLL":22732.29,"SOS":571.29,"SRD":35.55,"SSP":3665.54,"STN":23.22,"SYP":12938.7,"SZL":18.11,"THB":34.42,"TJS":10.7,"TMT":3.5,"TND":3.15,"TOP":2.37,"TRY":34.66,"TTD":6.74,"TVD":1.54,"TWD":32.53,"TZS":2644.73,"UAH":41.58,"UGX":3694.82,"UYU":42.8,"UZS":12828.59,"VES":47.31,"VND":25370.47,"VUV":118.77,"WST":2.77,"XAF":621.8,"XCD":2.7,"XDR":0.761,"XOF":621.8,"XPF":113.12,"YER":249.58,"ZAR":18.11,"ZMW":27.25,"ZWL":25.33}

# Replace strings with their USD conversion rate in a new column
final_df['Exchange Rate from USD'] = final_df['Received Currency'].map(exchange_rates)

# Perform vectorized multiplication for conversion
final_df['Amount Received (USD)'] = final_df['Amount Received'] / final_df['Exchange Rate from USD']

final_df = final_df.sort_values(by='Timestamp')

# Streamlit App
st.title("Anti-Money Laundering Dashboard")
st.sidebar.header("Filters")

# Filters
selected_currency = st.sidebar.selectbox("Filter by Currency", options=['All'] + list(curr.values()))
if selected_currency != 'All':
    filtered_df = final_df[final_df['Received Currency'] == selected_currency]
else:
    filtered_df = final_df
    
print(filtered_df.columns)

# Show filtered data
st.subheader("Filtered Transactions")
st.dataframe(filtered_df[['EdgeID', 'from_id', 'to_id', 'Timestamp', 'Amount Sent', 
                          'Sent Currency', 'Amount Received', 'Received Currency', 
                          'Amount Received (USD)']], use_container_width=True)

# t = pd.read_csv("./data/Small_HI/HI-Small_Trans.csv")
# t = t[t["Is Laundering"] == 1]

# t['Receiving Currency'] = t['Receiving Currency'].map(curr_code)

# t['Exchange Rate from USD'] = t['Receiving Currency'].map(exchange_rates)

# # Perform vectorized multiplication for conversion
# t['Amount Received (USD)'] = t['Amount Received'] / t['Exchange Rate from USD']

# st.subheader("Total Money Laundered Over Time (in USD)")
# t['Timestamp'] = pd.to_datetime(t['Timestamp'])
# timeline = t.groupby(t['Timestamp'].dt.date)['Amount Received (USD)'].sum().reset_index()
# timeline_chart = px.line(timeline, x='Timestamp', y='Amount Received (USD)', title="Money Laundered Over Time (USD)")
# st.plotly_chart(timeline_chart, use_container_width=True)








# st.subheader("Money Laundered by Hour of the Day (in USD)")

# # Convert UTC timestamps to US Eastern Time
# us_eastern = pytz.timezone("US/Eastern")
# t['Timestamp'] = pd.to_datetime(t['Timestamp']).dt.tz_localize('UTC').dt.tz_convert(us_eastern)
# t['Hour of Day'] = t['Timestamp'].dt.hour
# hourly_totals = t.groupby('Hour of Day')['Amount Received (USD)'].sum().reset_index()
# hourly_chart = px.bar(
#     hourly_totals,
#     x='Hour of Day',
#     y='Amount Received (USD)',
#     title="Money Laundered by Hour of the Day (in USD)",
#     labels={'Hour of Day': 'Hour (0-23)', 'Amount Received (USD)': 'Total Laundered (USD)'},
#     text='Amount Received (USD)'
# )
# hourly_chart.update_traces(texttemplate='%{text:.2f}', textposition='outside')  # Show values on bars
# st.plotly_chart(hourly_chart, use_container_width=True)



# # Total Money Laundered
# st.subheader("Total Money Laundered (in USD)")
# total_laundered = t['Amount Received (USD)'].sum()
# st.metric("Total Laundered Amount (USD)", f"${total_laundered:,.2f}")

# # Pie Chart: Laundered Amount by Currency
# st.subheader("Laundered Amount by Currency")
# currency_pie = t.groupby('Receiving Currency')['Amount Received (USD)'].sum().reset_index()
# pie_chart = px.pie(currency_pie, names='Receiving Currency', values='Amount Received (USD)', title="Laundered Amount by Currency")
# st.plotly_chart(pie_chart, use_container_width=True)

# # Bar Chart: Top Laundered Amounts by Transaction
# st.subheader("Top Laundered Transactions")
# top_transactions = t.nlargest(10, 'Amount Received (USD)')[['EdgeID', 'Amount Received (USD)']]
# bar_chart = px.bar(top_transactions, x='EdgeID', y='Amount Received (USD)', title="Top Laundered Transactions")
# st.plotly_chart(bar_chart, use_container_width=True)

# st.write("Explore more filters and visualizations to understand laundering patterns!")





# Network Graph
st.subheader("Transaction Network Graph")

def create_network_graph(df):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    unique_nodes = set(df['from_id'].unique()) | set(df['to_id'].unique())
    G.add_nodes_from(unique_nodes)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    
    # Create a spring layout
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    
    # Add edges with annotations
    for _, row in df.iterrows():
        x0, y0 = pos[row['from_id']]
        x1, y1 = pos[row['to_id']]
        
        # Add edge coordinates
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Create edge label
        timestamp = datetime.datetime.fromtimestamp(row['Timestamp']).strftime('%Y-%m-%d %H:%M')
        edge_text.append(
            f"Amount: {row['Amount Received']:.2f} {row['Received Currency']}<br>"
            f"Time: {timestamp}"
        )
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'ID: {node}')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # Color nodes by number of connections
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    
    node_trace.marker.color = node_adjacencies
    
    # Create annotations for edges
    annotations = []
    # for i in range(0, len(edge_x)-3, 3):
    #     annotations.append(
    #         dict(
    #             x=(edge_x[i] + edge_x[i+1])/2,
    #             y=(edge_y[i] + edge_y[i+1])/2,
    #             text=edge_text[i//3],
    #             showarrow=False,
    #             font=dict(size=8),
    #             bgcolor="white",
    #             borderpad=2
    #         )
    #     )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Transaction Network',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=annotations,
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

# Create and display the network graph
if not filtered_df.empty:
    fig = create_network_graph(filtered_df[:250])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No transactions to display for the selected filters.")

# Add some metrics about the network
if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unique_senders = filtered_df['from_id'].nunique()
        st.metric("Unique Senders", unique_senders)
    
    with col2:
        unique_receivers = filtered_df['to_id'].nunique()
        st.metric("Unique Receivers", unique_receivers)
    
    with col3:
        total_volume = filtered_df['Amount Received (USD)'].sum()
        st.metric("Total Volume (USD)", f"${total_volume:,.2f}")
