from langchain_community.document_loaders import WebBaseLoader

stock_analysis_data_by_ticker = WebBaseLoader("https://www.tipranks.com/stocks/aapl").load()
analyst_forecast_data_by_ticker = WebBaseLoader("https://www.tipranks.com/stocks/aapl/forecast").load()
ta_data_by_ticker = WebBaseLoader("https://www.tipranks.com/stocks/aapl/technical-analysis").load()
options_data_by_ticker = WebBaseLoader("https://www.tipranks.com/stocks/aapl/options-chain").load()
earnings_data_by_ticker = WebBaseLoader("https://www.tipranks.com/stocks/aapl/earnings").load()
ownership_data_by_ticker = WebBaseLoader("https://www.tipranks.com/stocks/aapl/ownership").load()
financials_data_by_ticker = WebBaseLoader("https://www.tipranks.com/stocks/aapl/financials").load()
similar_stocks_data_by_ticker = WebBaseLoader("https://www.tipranks.com/stocks/aapl/similar-stocks").load()


print(stock_analysis_data_by_ticker[0].page_content)