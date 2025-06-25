import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('/Users/shyamsanogar/Desktop/StocksData/SD/merged_sorted_by_date.xlsx')
print("Available columns:", df.columns.tolist())
df.fillna(method='ffill', inplace=True)
numeric_cols = ['OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'ltp', 'close', 'vwap', 
                '52W H', '52W L', 'VOLUME', 'VALUE', 'No of trades']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df = df.sort_values(['Source', 'Date'])

stocks = sorted(df['Source'].unique())[:-10]
if not stocks:
    print("No stocks found in the 'Source' column after excluding the last 10. Check your data or column name.")
else:
    # Split stocks into two groups of 7
    group1_stocks = stocks[:7]
    group2_stocks = stocks[7:14]

    # Top 10 stocks by total volume
    volume_by_stock = df.groupby('Source')['VOLUME'].sum().sort_values(ascending=False)
    top_10_stocks = volume_by_stock.head(10).index
    top_10_volumes = volume_by_stock.head(10).values
    plt.figure(figsize=(10, 6))
    plt.barh(top_10_stocks, top_10_volumes, color=plt.cm.Blues(np.linspace(0.2, 0.8, 10)))
    plt.title('Top 10 Products by Total Volume')
    plt.xlabel('Total Volume')
    plt.ylabel('Stock')
    plt.tight_layout()
    plt.show()

    # Bar plot for total volume per stock
    total_volume = df.groupby('Source')['VOLUME'].sum().reindex(stocks, fill_value=0)
    plt.figure(figsize=(12, 6))
    plt.bar(total_volume.index, total_volume.values, color='blue')
    plt.title('Total Trading Volume by Stock')
    plt.xlabel('Stock')
    plt.ylabel('Total Volume')
    plt.xticks(rotation=45)
    for i, v in enumerate(total_volume.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    max_volume_stock = total_volume.idxmax() if not total_volume.empty else None
    if max_volume_stock:
        max_idx = total_volume.index.get_loc(max_volume_stock)
        plt.bar(max_idx, total_volume[max_volume_stock], color='darkblue', label=f'Highest: {max_volume_stock}')
        plt.legend()
    plt.tight_layout()
    plt.show()

    # Box plot for daily closing prices
    close_data = [df[df['Source'] == stock]['close'].dropna().values for stock in stocks]
    plt.figure(figsize=(12, 6))
    plt.boxplot(close_data, labels=stocks)
    plt.title('Distribution of Daily Closing Prices by Stock')
    plt.xlabel('Stock')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    roi_results = {}
    growth_patterns = {}
    for stock in stocks:
        stock_data = df[df['Source'] == stock].copy()
        initial_price = stock_data['close'].iloc[0]
        final_price = stock_data['close'].iloc[-1]
        roi = ((final_price - initial_price) / initial_price) * 100
        roi_results[stock] = roi
        stock_data['Daily_Return'] = stock_data['close'].pct_change() * 100
        stock_data['MA50'] = stock_data['close'].rolling(window=50).mean()
        stock_data['MA200'] = stock_data['close'].rolling(window=200).mean()
        growth_patterns[stock] = stock_data

    print("\nROI Results (%):")
    for stock, roi in roi_results.items():
        print(f"{stock}: {roi:.2f}%")

    volatility_results = {}
    for stock in stocks:
        stock_data = df[df['Source'] == stock].copy()
        stock_data['Daily_Return'] = stock_data['close'].pct_change()
        volatility = stock_data['Daily_Return'].std() * np.sqrt(252)
        market_returns = df.groupby('Date')['close'].mean().pct_change()
        stock_returns = stock_data.set_index('Date')['Daily_Return']
        covariance = stock_returns.cov(market_returns)
        market_variance = market_returns.var()
        beta = covariance / market_variance if market_variance != 0 else np.nan
        volatility_results[stock] = {'Volatility': volatility, 'Beta': beta}

    print("\nVolatility and Risk Results:")
    for stock, metrics in volatility_results.items():
        print(f"{stock}: Volatility={metrics['Volatility']:.4f}, Beta={metrics['Beta']:.4f}")

    plt.figure(figsize=(12, 6))
    for stock in stocks:
        stock_data = growth_patterns[stock]
        plt.plot(stock_data['Date'], stock_data['close'], label=stock)
    plt.title('Stock Price Trends')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Moving Averages - Group 1 (Reduced Size)
    fig, axes = plt.subplots(7, 1, figsize=(6, 8), sharex=True)  # Reduced from (10, 14)
    for idx, stock in enumerate(group1_stocks):
        stock_data = growth_patterns[stock]
        axes[idx].plot(stock_data['Date'], stock_data['close'], label='Close')
        axes[idx].plot(stock_data['Date'], stock_data['MA50'], label='50-day MA')
        axes[idx].plot(stock_data['Date'], stock_data['MA200'], label='200-day MA')
        axes[idx].set_title(stock, fontsize=8)  # Smaller font
        axes[idx].set_ylabel('Price', fontsize=6)
        axes[idx].legend(fontsize=6, loc='upper left')
        axes[idx].grid(True)
        axes[idx].tick_params(axis='both', labelsize=6)
    axes[-1].set_xlabel('Date', fontsize=6)
    plt.suptitle('Moving Averages - Stocks 1-7', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()

    # Moving Averages - Group 2 (Reduced Size)
    fig, axes = plt.subplots(7, 1, figsize=(6, 8), sharex=True)
    for idx, stock in enumerate(group2_stocks):
        stock_data = growth_patterns[stock]
        axes[idx].plot(stock_data['Date'], stock_data['close'], label='Close')
        axes[idx].plot(stock_data['Date'], stock_data['MA50'], label='50-day MA')
        axes[idx].plot(stock_data['Date'], stock_data['MA200'], label='200-day MA')
        axes[idx].set_title(stock, fontsize=8)
        axes[idx].set_ylabel('Price', fontsize=6)
        axes[idx].legend(fontsize=6, loc='upper left')
        axes[idx].grid(True)
        axes[idx].tick_params(axis='both', labelsize=6)
    axes[-1].set_xlabel('Date', fontsize=6)
    plt.suptitle('Moving Averages - Stocks 8-14', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()

    # Daily Returns - Group 1 (Reduced Size)
    fig, axes = plt.subplots(7, 1, figsize=(6, 8), sharex=True)
    for idx, stock in enumerate(group1_stocks):
        stock_data = growth_patterns[stock]
        axes[idx].hist(stock_data['Daily_Return'].dropna(), bins=50, color='blue', alpha=0.7)
        axes[idx].set_title(f'{stock} - Daily Returns', fontsize=8)
        axes[idx].set_ylabel('Frequency', fontsize=6)
        axes[idx].grid(True)
        axes[idx].tick_params(axis='both', labelsize=6)
    axes[-1].set_xlabel('Daily Return (%)', fontsize=6)
    plt.suptitle('Daily Returns - Stocks 1-7', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()

    # Daily Returns - Group 2 (Reduced Size)
    fig, axes = plt.subplots(7, 1, figsize=(6, 8), sharex=True)
    for idx, stock in enumerate(group2_stocks):
        stock_data = growth_patterns[stock]
        axes[idx].hist(stock_data['Daily_Return'].dropna(), bins=50, color='blue', alpha=0.7)
        axes[idx].set_title(f'{stock} - Daily Returns', fontsize=8)
        axes[idx].set_ylabel('Frequency', fontsize=6)
        axes[idx].grid(True)
        axes[idx].tick_params(axis='both', labelsize=6)
    axes[-1].set_xlabel('Daily Return (%)', fontsize=6)
    plt.suptitle('Daily Returns - Stocks 8-14', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()

    # Correlation Heatmaps - Group 1 (Reduced Size)
    fig, axes = plt.subplots(7, 1, figsize=(6, 12), squeeze=False)  # Reduced from (10, 28)
    for idx, stock in enumerate(group1_stocks):
        stock_data = df[df['Source'] == stock].copy()
        stock_data['Daily_Return'] = stock_data['close'].pct_change() * 100
        corr_cols = ['OPEN', 'HIGH', 'LOW', 'close', 'VOLUME', 'Daily_Return']
        corr_matrix = stock_data[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                    ax=axes[idx, 0], annot_kws={'size': 5}, cbar=False)  # Smaller annotations
        axes[idx, 0].set_title(f'{stock} - Correlation', fontsize=8)
        axes[idx, 0].tick_params(axis='both', labelsize=6)
    plt.suptitle('Correlation Heatmaps - Stocks 1-7', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()

    # Correlation Heatmaps - Group 2 (Reduced Size)
    fig, axes = plt.subplots(7, 1, figsize=(6, 12), squeeze=False)
    for idx, stock in enumerate(group2_stocks):
        stock_data = df[df['Source'] == stock].copy()
        stock_data['Daily_Return'] = stock_data['close'].pct_change() * 100
        corr_cols = ['OPEN', 'HIGH', 'LOW', 'close', 'VOLUME', 'Daily_Return']
        corr_matrix = stock_data[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                    ax=axes[idx, 0], annot_kws={'size': 5}, cbar=False)
        axes[idx, 0].set_title(f'{stock} - Correlation', fontsize=8)
        axes[idx, 0].tick_params(axis='both', labelsize=6)
    plt.suptitle('Correlation Heatmaps - Stocks 8-14', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()

    # Daily Trading Volume - Group 1 (Reduced Size)
    fig, axes = plt.subplots(7, 1, figsize=(6, 8), sharex=True)
    for idx, stock in enumerate(group1_stocks):
        stock_data = df[df['Source'] == stock]
        axes[idx].plot(stock_data['Date'], stock_data['VOLUME'], label=stock)
        axes[idx].set_title(f'{stock} - Daily Trading Volume', fontsize=8)
        axes[idx].set_ylabel('Volume', fontsize=6)
        axes[idx].legend(fontsize=6, loc='upper left')
        axes[idx].grid(True)
        axes[idx].tick_params(axis='both', labelsize=6)
    axes[-1].set_xlabel('Date', fontsize=6)
    plt.suptitle('Daily Trading Volume - Stocks 1-7', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()

    # Daily Trading Volume - Group 2 (Reduced Size)
    fig, axes = plt.subplots(7, 1, figsize=(6, 8), sharex=True)
    for idx, stock in enumerate(group2_stocks):
        stock_data = df[df['Source'] == stock]
        axes[idx].plot(stock_data['Date'], stock_data['VOLUME'], label=stock)
        axes[idx].set_title(f'{stock} - Daily Trading Volume', fontsize=8)
        axes[idx].set_ylabel('Volume', fontsize=6)
        axes[idx].legend(fontsize=6, loc='upper left')
        axes[idx].grid(True)
        axes[idx].tick_params(axis='both', labelsize=6)
    axes[-1].set_xlabel('Date', fontsize=6)
    plt.suptitle('Daily Trading Volume - Stocks 8-14', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.show()

    # 52-Week High and Low Prices - Group 1 (Reduced Size)
    plt.figure(figsize=(6, 4))  # Reduced from (10, 6)
    x = np.arange(len(group1_stocks))
    width = 0.35
    highs = [df[df['Source'] == stock]['HIGH'].max() for stock in group1_stocks]
    lows = [df[df['Source'] == stock]['LOW'].min() for stock in group1_stocks]
    plt.bar(x - width/2, highs, width, label='High', color='green', alpha=0.4)
    plt.bar(x + width/2, lows, width, label='Low', color='red', alpha=0.4)
    plt.xticks(x, group1_stocks, rotation=45, fontsize=6)
    plt.title('52-Week High and Low Prices - Stocks 1-7', fontsize=10)
    plt.xlabel('Stock', fontsize=6)
    plt.ylabel('Price', fontsize=6)
    plt.legend(fontsize=6)
    plt.tick_params(axis='y', labelsize=6)
    plt.tight_layout()
    plt.show()

    # 52-Week High and Low Prices - Group 2 (Reduced Size)
    plt.figure(figsize=(6, 4))
    x = np.arange(len(group2_stocks))
    highs = [df[df['Source'] == stock]['HIGH'].max() for stock in group2_stocks]
    lows = [df[df['Source'] == stock]['LOW'].min() for stock in group2_stocks]
    plt.bar(x - width/2, highs, width, label='High', color='green', alpha=0.4)
    plt.bar(x + width/2, lows, width, label='Low', color='red', alpha=0.4)
    plt.xticks(x, group2_stocks, rotation=45, fontsize=6)
    plt.title('52-Week High and Low Prices - Stocks 8-14', fontsize=10)
    plt.xlabel('Stock', fontsize=6)
    plt.ylabel('Price', fontsize=6)
    plt.legend(fontsize=6)
    plt.tick_params(axis='y', labelsize=6)
    plt.tight_layout()
    plt.show()

    recommendations = []
    for stock in stocks:
        roi = roi_results[stock]
        volatility = volatility_results[stock]['Volatility']
        beta = volatility_results[stock]['Beta']
        score = (roi / 100) - volatility - (beta if not np.isnan(beta) else 0)
        recommendations.append({
            'Stock': stock,
            'ROI': roi,
            'Volatility': volatility,
            'Beta': beta,
            'Score': score
        })
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df = recommendations_df.sort_values('Score', ascending=False)

    print("\nStock Recommendations:")
    print(recommendations_df)

    volume_stats = {}
    for stock in stocks:
        stock_data = df[df['Source'] == stock]
        avg_volume = stock_data['VOLUME'].mean()
        volume_stats[stock] = {
            'Average Volume': avg_volume,
            'Volume Std': stock_data['VOLUME'].std()
        }

    print("\nVolume Statistics:")
    for stock, stats in volume_stats.items():
        print(f"{stock}: Avg Volume={stats['Average Volume']:.0f}, Std={stats['Volume Std']:.0f}")

    print("\n52-Week High and Low Prices:")
    for stock in stocks:
        high = df[df['Source'] == stock]['HIGH'].max()
        low = df[df['Source'] == stock]['LOW'].min()
        price_ranges[stock] = {'52W High': high, '52W Low': low}
        print(f"{stock}: High={high:.2f}, Low={low:.2f}")
