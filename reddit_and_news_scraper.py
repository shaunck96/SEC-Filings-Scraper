#%pip install --quiet "pyautogen>=0.2.3"
#!pip install langchain==0.0.340
#!pip install openai==0.28
#!pip install langchain_community
#!pip install gnews
#!pip install tiktoken
#!pip install langchain_text_splitters
#!pip install langchain_openai
#!pip install praw

from langchain_community.document_loaders import WebBaseLoader
import tiktoken
import nltk
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
import openai
import ast
from typing import List
import json
from gnews import GNews
import pandas as pd
from typing import Optional
import praw
import json
import pandas as pd
import logging
from typing import Optional, List, Dict
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from os.path import isfile
import praw
import pandas as pd
from time import sleep
from typing import List, Dict
nltk.download('punkt')

with open('openai_config.json', 'r') as f:
    config = json.load(f)

openai.api_key = config["openai_key"]

# Configure logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MongoDBHandler:
    def __init__(self, uri):
        try:
            self.client = MongoClient(uri)
            logging.info("MongoDB connection established.")
        except errors.ConnectionFailure:
            logging.error("Failed to connect to MongoDB.")
            raise

    def insert_records(self, db_name, collection_name, records, avoid_duplicates=False):
        if not records:  # Validate records is not empty
            logging.warning("No records provided to insert.")
            return
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            if avoid_duplicates:
                inserted_count = 0
                for record in records:
                    if collection.count_documents({"unique_identifier": record.get("unique_identifier")}, limit=1) == 0:
                        collection.insert_one(record)
                        inserted_count += 1
                logging.info(f"{inserted_count} records inserted successfully into MongoDB.")
            else:
                result = collection.insert_many(records)
                logging.info(f"{len(result.inserted_ids)} records inserted successfully into MongoDB.")
        except Exception as e:
            logging.error(f"Failed to insert records: {e}")

    def count_records(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            count = collection.count_documents({})
            return count
        except Exception as e:
            logging.error(f"Error counting records: {e}")
            return None

    def get_last_upload_timestamp(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            last_record = collection.find_one(sort=[('_id', -1)])
            if last_record:
                return last_record.get("Accepted Date")
            return None
        except Exception as e:
            logging.error(f"Error retrieving last upload timestamp: {e}")
            return None

    def update_record(self, db_name, collection_name, filter_query, update_query):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            result = collection.update_one(filter_query, {"$set": update_query})
            if result.modified_count > 0:
                logging.info("Record updated successfully.")
            else:
                logging.warning("No records updated.")
        except Exception as e:
            logging.error(f"Failed to update record: {e}")

    def truncate_collection(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            collection.delete_many({})
            logging.info("Collection truncated successfully.")
        except Exception as e:
            logging.error(f"Failed to truncate collection: {e}")

    def read_collection(self, db_name, collection_name):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            return list(collection.find())
        except Exception as e:
            logging.error(f"Failed to read collection: {e}")
            return []

    def generate_descriptive_statistics(self, db_name, collection_name):
        # Implementation depends on the data and required statistics
        pass

    def drop_duplicates_and_rewrite(self, db_name, collection_name, unique_key):
        db = self.client[db_name]
        collection = db[collection_name]
        try:
            data = list(collection.find())
            unique_records = {}
            for record in data:
                key = record.get(unique_key)
                if key and key not in unique_records:
                    unique_records[key] = record

            if unique_records:
                collection.delete_many({})
                collection.insert_many(list(unique_records.values()))
                logging.info("Duplicates dropped and table rewritten successfully.")
            else:
                logging.warning("No unique records found to write.")
        except Exception as e:
            logging.error(f"Failed to drop duplicates and rewrite: {e}")

nltk.download('punkt')

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_completion(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Query your LLM model with your prompt.
    Parameters:
    prompt (str): The text prompt you want the LLM to respond to.
    model (str, optional): The model to be used for generating the response. Default is "gpt-3.5-turbo".
    Returns:
    str: The generated text completion from the specified model.
    """
    openai.api_key = config["openai_key"]
    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model= model,
        messages=messages,
        temperature=0.5
    )

    return response.choices[0].message["content"]

class SubredditScraper:

    def __init__(self, sub, sort='new', lim=900, mode='w'):
        self.sub = sub
        self.sort = sort
        self.lim = lim
        self.mode = mode
        print(
            f'SubredditScraper instance created with values '
            f'sub = {sub}, sort = {sort}, lim = {lim}, mode = {mode}')
        with open("reddit_creds.json") as f:
            client_creds = json.load(f)
        self.reddit = praw.Reddit(client_id=client_creds["client_id"],
                                  client_secret=client_creds['client_secret'],
                                  user_agent=client_creds['user_agent'],
                                  ratelimit_seconds=300)

    def set_sort(self):
        if self.sort == 'new':
            return self.sort, self.reddit.subreddit(self.sub).new(limit=self.lim)
        elif self.sort == 'top':
            return self.sort, self.reddit.subreddit(self.sub).top(limit=self.lim)
        elif self.sort == 'hot':
            return self.sort, self.reddit.subreddit(self.sub).hot(limit=self.lim)
        else:
            self.sort = 'hot'
            print('Sort method was not recognized, defaulting to hot.')
            return self.sort, self.reddit.subreddit(self.sub).hot(limit=self.lim)

    def get_posts(self):
        """Get unique posts from a specified subreddit."""

        sub_dict = {
            'selftext': [], 'title': [], 'id': [], 'sorted_by': [],
            'num_comments': [], 'score': [], 'ups': [], 'downs': []}
        csv = f'{self.sub}_posts.csv'

        # Attempt to specify a sorting method.
        sort, subreddit = self.set_sort()

        # Set csv_loaded to True if csv exists since you can't
        # evaluate the truth value of a DataFrame.
        df, csv_loaded = (pd.read_csv(csv), 1) if isfile(csv) else ('', 0)

        print(f'csv = {csv}')
        print(f'After set_sort(), sort = {sort} and sub = {self.sub}')
        print(f'csv_loaded = {csv_loaded}')

        print(f'Collecting information from r/{self.sub}.')

        for post in subreddit:

            # Check if post.id is in df and set to True if df is empty.
            # This way new posts are still added to dictionary when df = ''
            unique_id = post.id not in tuple(df.id) if csv_loaded else True

            # Save any unique posts to sub_dict.
            if unique_id:
                sub_dict['selftext'].append(post.selftext)
                sub_dict['title'].append(post.title)
                sub_dict['id'].append(post.id)
                sub_dict['sorted_by'].append(sort)
                sub_dict['num_comments'].append(post.num_comments)
                sub_dict['score'].append(post.score)
                sub_dict['ups'].append(post.ups)
                sub_dict['downs'].append(post.downs)
            sleep(0.1)

        new_df = pd.DataFrame(sub_dict)

        # Add new_df to df if df exists then save it to a csv.
        if 'DataFrame' in str(type(df)) and self.mode == 'w':
            pd.concat([df, new_df], axis=0, sort=0).to_csv(csv, index=False)
            print(
                f'{len(new_df)} new posts collected and added to {csv}')
        elif self.mode == 'w':
            new_df.to_csv(csv, index=False)
            print(f'{len(new_df)} posts collected and saved to {csv}')
        else:
            print(
                f'{len(new_df)} posts were collected but they were not '
                f'added to {csv} because mode was set to "{self.mode}"')

def prompt_selection(task='technical_analysis', input_to_llm='', stock='', resp='', req=''):

  class techAnalysis(BaseModel):
      current_price_and_range: str = Field(description="Current stock price and day's range")
      price_52_week_range: str = Field(description="52-week price range of the stock")
      analyst_consensus_and_target: str = Field(description="Analyst consensus and price target for the next 12 months")
      key_financial_ratios: str = Field(description="Key financial ratios like P/E, P/B, P/S, P/CF")
      earnings_report_summary: str = Field(description="Latest earnings report summary and next earnings report date")
      technical_analysis_consensus: str = Field(description="Technical analysis consensus")
      recent_news_headlines: str = Field(description="Most recent major news headlines related to the stock")
      dividend_information: str = Field(description="Dividend information including last dividend amount and yield")
      major_risks: str = Field(description="Major risks associated with the stock")
      web_traffic_data: str = Field(description="Web traffic and user interest data")

  class StockForecast(BaseModel):
      average_price_target: str = Field(description="Average price target for the stock over the next 12 months")
      highest_price_target: str = Field(description="Highest price target for the stock over the next 12 months")
      lowest_price_target: str = Field(description="Lowest price target for the stock over the next 12 months")
      analyst_rating_consensus: str = Field(description="Overall analyst rating consensus")
      number_of_analysts: str = Field(description="Number of analysts giving ratings")
      buy_ratings: str = Field(description="Number of buy ratings")
      hold_ratings: str = Field(description="Number of hold ratings")
      sell_ratings: str = Field(description="Number of sell ratings")
      next_quarters_earnings_estimate: str = Field(description="Next quarter's earnings estimate for the stock")
      sales_forecast: str = Field(description="Sales forecast for the next quarter")
      major_news_headlines: str = Field(description="Most recent major news headlines related to the stock")

  class StockTechnicalAnalysis(BaseModel):
      overall_consensus: str = Field(description="Overall consensus based on technical analysis")
      macd_indicator: str = Field(description="Moving Averages Convergence Divergence indicator value")
      rsi: str = Field(description="Relative Strength Index value")
      williams_r: str = Field(description="Williams %R value")
      cci: str = Field(description="Commodity Channel Index value")
      roc: str = Field(description="Price Rate of Change value")
      pivot_points: dict = Field(description="Pivot points including S3, S2, S1, Pivot Point, R1, R2, R3 values")
      moving_averages: dict = Field(description="Moving averages including simple and exponential values for different periods")
      implied_action: dict = Field(description="Implied actions based on technical indicators")

  class OptionDetail(BaseModel):
      strike_price: float = Field(description="The price at which the option can be exercised")
      last_price: float = Field(description="The last traded price of the option")
      change_percentage: float = Field(description="Percentage change in the option's price")
      volume: int = Field(description="Trading volume of the option")
      open_interest: int = Field(description="Open interest of the option")
      open_interest_change: int = Field(description="Change in open interest")
      last_trade_time: str = Field(description="Time of the last trade of the option")

  class OptionsChainData(BaseModel):
      next_earnings_date: str = Field(..., description="Next earnings date for the stock")
      call_options: List[OptionDetail] = Field(..., description="List of call options data")
      put_options: List[OptionDetail] = Field(..., description="List of put options data")

  class EarningsHistoryEntry(BaseModel):
      report_date: str = Field(..., description="Date of the earnings report")
      fiscal_quarter: str = Field(..., description="Fiscal quarter of the earnings report")
      forecast_eps: float = Field(..., description="Forecasted EPS for the quarter")
      actual_eps: float = Field(..., description="Actual EPS for the quarter")
      eps_yoy_change: str = Field(..., description="Year-over-year change in EPS")

  class EarningsPriceChangeEntry(BaseModel):
      report_date: str = Field(..., description="Date of the earnings report")
      price_before: float = Field(..., description="Stock price one day before the earnings report")
      price_after: float = Field(..., description="Stock price one day after the earnings report")
      percentage_change: float = Field(..., description="Percentage change in stock price due to the earnings report")

  class EarningsData(BaseModel):
      next_earnings_date: str = Field(..., description="Next scheduled earnings report date")
      period_ending: str = Field(..., description="The period ending for the next earnings report")
      consensus_eps_forecast: float = Field(..., description="Consensus EPS forecast for the next earnings report")
      last_year_eps: float = Field(..., description="EPS for the same quarter last year")
      analyst_consensus: str = Field(..., description="Overall analyst consensus rating")
      earnings_history: List[EarningsHistoryEntry] = Field(..., description="Historical earnings data")
      earnings_related_price_changes: List[EarningsPriceChangeEntry] = Field(..., description="Earnings-related price changes")

  class InsiderTradingActivity(BaseModel):
      date: str = Field(..., description="Date of the insider trading activity")
      name: str = Field(..., description="Name of the insider")
      position: str = Field(..., description="Position of the insider within the company")
      action: str = Field(..., description="Type of activity (bought or sold)")
      value: float = Field(..., description="Value of the traded shares")

  class HedgeFundTradingActivity(BaseModel):
      date: str = Field(..., description="Date of the hedge fund trading activity")
      firm: str = Field(..., description="Name of the hedge fund")
      action: str = Field(..., description="Type of activity (bought or sold)")
      value: float = Field(..., description="Value of the traded shares")

  class Shareholder(BaseModel):
      name: str = Field(..., description="Name of the shareholder")
      shares: float = Field(..., description="Number of shares held")
      type: str = Field(..., description="Type of shareholder (Institution, Individual, etc.)")
      holding_percentage: float = Field(..., description="Percentage of total shares held by the shareholder")
      value: float = Field(..., description="Market value of the shares held")

  class StockOwnership(BaseModel):
      insiders_percentage: float = Field(..., description="Percentage of stocks owned by insiders")
      mutual_funds_percentage: float = Field(..., description="Percentage of stocks owned by mutual funds")
      institutional_investors_percentage: float = Field(..., description="Percentage of stocks owned by other institutional investors")
      public_companies_individuals_percentage: float = Field(..., description="Percentage of stocks owned by public companies and individual investors")
      recent_insider_trading: List[InsiderTradingActivity] = Field(..., description="Recent insider trading activities")
      recent_hedge_fund_trading: List[HedgeFundTradingActivity] = Field(..., description="Recent hedge fund trading activities")
      top_shareholders: List[Shareholder] = Field(..., description="List of top shareholders")
      top_mutual_fund_holders: List[Shareholder] = Field(..., description="List of top mutual fund holders")
      top_etf_holders: List[Shareholder] = Field(..., description="List of top ETF holders")

  class IncomeStatement(BaseModel):
      total_revenue: float = Field(..., description="Total revenue of the company")
      gross_profit: float = Field(..., description="Gross profit of the company")
      ebit: float = Field(..., description="Earnings before interest and taxes (EBIT)")
      ebitda: float = Field(..., description="Earnings before interest, taxes, depreciation, and amortization (EBITDA)")
      net_income: float = Field(..., description="Net income available to common stockholders")

  class BalanceSheet(BaseModel):
      cash_and_equivalents: float = Field(..., description="Total cash, cash equivalents and short-term investments")
      total_assets: float = Field(..., description="Total assets of the company")
      total_debt: float = Field(..., description="Total debt of the company")
      net_debt: float = Field(..., description="Net debt of the company")
      total_liabilities: float = Field(..., description="Total liabilities of the company")
      stockholders_equity: float = Field(..., description="Total stockholders' equity")

  class CashFlow(BaseModel):
      free_cash_flow: float = Field(..., description="Free cash flow of the company")
      operating_cash_flow: float = Field(..., description="Operating cash flow")
      investing_cash_flow: float = Field(..., description="Investing cash flow")
      financing_cash_flow: float = Field(..., description="Financing cash flow")

  class Financials(BaseModel):
      market_cap: float = Field(..., description="Market capitalization")
      eps_ttm: float = Field(..., description="Earnings per share for the trailing twelve months")
      pe_ratio: float = Field(..., description="Price to earnings ratio")
      dividend_yield: float = Field(..., description="Dividend yield percentage")
      next_earnings_date: str = Field(..., description="Date of the next earnings report")
      income_statement: IncomeStatement = Field(..., description="Income statement details")
      balance_sheet: BalanceSheet = Field(..., description="Balance sheet details")
      cash_flow: CashFlow = Field(..., description="Cash flow details")

  class StockCompetitor(BaseModel):
      name: str = Field(..., description="Name of the competing company")
      price: float = Field(..., description="Current price of the company's stock")
      market_cap: str = Field(..., description="Market capitalization of the competing company")
      pe_ratio: float = Field(..., description="Price to earnings ratio of the competing company")
      yearly_gain: float = Field(..., description="Yearly gain percentage of the company's stock")
      analyst_consensus: str = Field(..., description="Overall analyst consensus on the stock (e.g., Buy, Hold, Sell)")
      analyst_price_target: str = Field(..., description="Analyst price target for the stock")
      top_analysts_price_target: str = Field(..., description="Price target given by top analysts")
      smart_score: int = Field(..., description="TipRanks Smart Score of the stock")

  class StockCompetitors(BaseModel):
      apple_details: StockCompetitor = Field(..., description="Details of stock")
      competitors: List[StockCompetitor] = Field(..., description="List of stock competitors")

  if task == 'technical_analysis':
    prompt = f"""
    Given the following webpage content from TipRanks about {stock} stock analysis:

    "{input_to_llm}"

    Extract and summarize the following information:

    1. Current {stock} stock price and day's range.
    2. 52-week price range of {stock} stock.
    3. Analyst consensus and price target for {stock} over the next 12 months.
    4. Key financial ratios: P/E, P/B, P/S, P/CF.
    5. Latest earnings report summary and next earnings report date.
    6. Technical analysis consensus (Bullish, Bearish, Neutral).
    7. Most recent major news headlines related to {stock}.
    8. Dividend information: last dividend amount and yield.
    9. Major risks associated with {stock} stock as identified.
    10. Web traffic and user interest data if available.

    Provide the information in a structured and concise format."""
    pydantic_object=techAnalysis
    return [prompt, pydantic_object]

  elif task == "forecast_data":
    prompt = f"""
    Given the following webpage content from TipRanks about {stock} stock forecast:

    "{input_to_llm}"

    Extract and summarize the following forecast data:

    1. The average price target for {stock} over the next 12 months.
    2. The highest price target for {stock} over the next 12 months.
    3. The lowest price target for {stock} over the next 12 months.
    4. The overall analyst rating consensus for {stock}.
    5. The total number of analysts giving ratings to {stock}.
    6. The number of buy, hold, and sell ratings.
    7. The next quarter's earnings estimate for {stock}.
    8. The sales forecast for the next quarter for {stock}.
    9. The most recent major news headlines related to {stock}.

    Provide the information in a structured and concise format.
    """
    pydantic_object = StockForecast
    return [prompt, pydantic_object]

  elif task == "ta":
    prompt = f"""
    Given the following webpage content from TipRanks about {stock} technical analysis:

    "{input_to_llm}"

    Extract and summarize the following technical analysis data:

    1. Overall technical analysis consensus (e.g., Sell, Neutral, Buy).
    2. Values and implied actions for key technical indicators:
        - MACD (Moving Average Convergence Divergence)
        - RSI (Relative Strength Index)
        - Williams %R
        - CCI (Commodity Channel Index)
        - ROC (Rate of Change)
    3. Pivot points including S3, S2, S1, central pivot point, R1, R2, R3.
    4. Moving averages for different periods (e.g., 5-day, 20-day, 50-day, etc.) and the associated market signals (Sell or Buy).

    Provide the information in a structured and concise format.
    """
    pydantic_object = StockTechnicalAnalysis
    return [prompt, pydantic_object]

  elif task == "OptionDetail":
    prompt = f"""
    Given the following content extracted from the TipRanks website about {stock} stock options chain and prices:

    "{input_to_llm}"

    Extract and summarize the options chain data including:

    1. Next earnings date for {stock}.
    2. Call options data, including strike price, last price, percentage change, volume, open interest, and last trade time for each available strike.
    3. Put options data, including strike price, last price, percentage change, volume, open interest, and last trade time for each available strike.

    Present the information in a clear and structured format suitable for further analysis.
    """
    pydantic_object = OptionsChainData
    return [prompt, pydantic_object]


  elif task == 'earnings_analysis':
    prompt = f"""
    Given the following webpage content from TipRanks about {stock} earnings dates and reports:

    "{input_to_llm}"

    Extract and summarize the following information:

    1. Next scheduled earnings report date for {stock}.
    2. Period ending and corresponding fiscal quarter for the next earnings report.
    3. Consensus EPS forecast and last year's EPS for the next scheduled earnings.
    4. Analyst consensus rating for {stock}.
    5. Historical earnings data, including report dates, fiscal quarters, forecasted EPS, actual EPS, and EPS year-over-year change.
    6. Stock price changes related to the earnings reports, including the price one day before and after the earnings release, and the percentage change.

    Present the information in a clear, structured, and concise format."""
    pydantic_object = EarningsData
    return [prompt, pydantic_object]

  elif task == 'ownership':
    prompt = f"""
    Given the following webpage content from TipRanks about {stock} stock ownership:

    "{input_to_llm}"

    Extract and summarize the following information:

    1. Percentage ownership breakdown of {stock} stock by Insiders, Mutual Funds, Other Institutional Investors, and Public Companies/Individual Investors.
    2. Recent insider trading activities including date, name, position, action, and value.
    3. Recent hedge fund trading activities including date, firm, action, and value.
    4. Details of top shareholders including name, number of shares, type, percentage holding, and value.
    5. Details of top mutual fund holders including name, number of shares, percentage holding, and value.
    6. Details of top ETF holders including name, number of shares, percentage holding, and value.

    Present the information in a clear, structured, and concise format."""
    pydantic_object = StockOwnership
    return [prompt, pydantic_object]

  elif task == 'financials':
    prompt = f"""
    Given the following webpage content from TipRanks about {stock} financial statements:

    "{input_to_llm}"

    Extract and summarize the following information:

    1. Market capitalization, EPS (TTM), P/E ratio, and dividend yield.
    2. Next earnings date.
    3. Detailed financials including:
        a. Income Statement: Total Revenue, Gross Profit, EBIT, EBITDA, and Net Income.
        b. Balance Sheet: Cash and Equivalents, Total Assets, Total Debt, Net Debt, Total Liabilities, and Stockholders Equity.
        c. Cash Flow: Free Cash Flow, Operating Cash Flow, Investing Cash Flow, and Financing Cash Flow.

    Present the information in a clear, structured, and concise format."""
    pydantic_object = Financials
    return [prompt, pydantic_object]

  elif task == 'competitors':
    prompt = f"""
    Given the following webpage content from TipRanks about {stock} Stock Competitors:

    "{input_to_llm}"

    Extract and summarize the following information:

    1. {stock} stock details including price, market cap, P/E ratio, yearly gain, analyst consensus, and Smart Score.
    2. Information on similar stocks including name, price, market cap, P/E ratio, yearly gain, analyst consensus, analyst price target, top analysts' price target, and Smart Score.

    Present the information in a clear, structured, and concise format."""
    pydantic_object = StockCompetitors
    return [prompt, pydantic_object]


def gpt_response(task='', input_to_llm='', stock = '', resp='', req=''):
  prompt_and_pyd_obj = prompt_selection(task, input_to_llm, resp, req)
  prompt = prompt_and_pyd_obj[0]
  pydantic_object = prompt_and_pyd_obj[1]
  pydantic_parser = PydanticOutputParser(pydantic_object=pydantic_object)
  format_instructions = pydantic_parser.get_format_instructions()
  print(format_instructions)
  query = prompt
  prompt = PromptTemplate(
      template="Answer the user query.\n{format_instructions}\n{query}\n",
      input_variables=["query"],
      partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
  )
  _input = prompt.format_prompt(query=query)
  answer = get_completion(_input.to_string())
  return answer

def gpt_trigger(input_to_llm, stock, task):
  ta = gpt_response(task,
                    input_to_llm=input_to_llm,
                    stock=stock)
  return ta

def stock_data(ticker):
  ticker = ticker
  stock_analysis_data_by_ticker = WebBaseLoader(f"https://www.tipranks.com/stocks/{ticker}").load()
  analyst_forecast_data_by_ticker = WebBaseLoader(f"https://www.tipranks.com/stocks/{ticker}/forecast").load()
  ta_data_by_ticker = WebBaseLoader(f"https://www.tipranks.com/stocks/{ticker}/technical-analysis").load()
  options_data_by_ticker = WebBaseLoader(f"https://www.tipranks.com/stocks/{ticker}/options-chain").load()
  earnings_data_by_ticker = WebBaseLoader(f"https://www.tipranks.com/stocks/{ticker}/earnings").load()
  ownership_data_by_ticker = WebBaseLoader(f"https://www.tipranks.com/stocks/{ticker}/ownership").load()
  financials_data_by_ticker = WebBaseLoader(f"https://www.tipranks.com/stocks/{ticker}/financials").load()
  similar_stocks_data_by_ticker = WebBaseLoader(f"https://www.tipranks.com/stocks/{ticker}/similar-stocks").load()

  results = {}

  # Web scraping and data extraction for technical analysis
  ta_data = gpt_trigger(stock_analysis_data_by_ticker[0].page_content, ticker.upper(), "technical_analysis")
  try:
      ta_data = ast.literal_eval(ta_data.split("```json\n")[1].replace("```",""))
      results['technical_analysis'] = ta_data
  except Exception as e:
      print("failed")
      results['technical_analysis'] = ta_data

  # Web scraping and data extraction for technical analysis details
  tech_data = gpt_trigger(ta_data_by_ticker[0].page_content, ticker.upper(), "ta")
  try:
      tech_data = ast.literal_eval(tech_data)#.split("```json\n")[1].replace("```",""))
      results['technical_analysis_details'] = tech_data
  except Exception as e:
      print("failed")
      results['technical_analysis_details'] = tech_data

  # Web scraping and data extraction for options chain
  options_data = gpt_trigger(options_data_by_ticker[0].page_content, ticker.upper(), "OptionDetail")
  try:
      options_data = ast.literal_eval(options_data)#.split("```json\n")[1].replace("```",""))
      results['options_chain'] = options_data
  except Exception as e:
      results['options_chain'] = options_data

  # Web scraping and data extraction for earnings analysis
  earnings_data = gpt_trigger(earnings_data_by_ticker[0].page_content, ticker.upper(), "earnings_analysis")
  try:
      earnings_data = ast.literal_eval(earnings_data)#.replace("```json\n","").replace("\n```",""))
      results['earnings_analysis'] = earnings_data
  except Exception as e:
      results['earnings_analysis'] = earnings_data

  # Web scraping and data extraction for ownership data
  ownership_data = gpt_trigger(ownership_data_by_ticker[0].page_content, ticker.upper(), "ownership")
  try:
      ownership_data = json.loads(ownership_data)#.replace("```json\n","").replace("\n```",""))
      results['ownership_data'] = ownership_data
  except Exception as e:
      print("failed")
      results['ownership_data'] = ownership_data

  # Web scraping and data extraction for financials
  financials_data = gpt_trigger(financials_data_by_ticker[0].page_content, ticker.upper(), "financials")
  try:
      financials_data = ast.literal_eval(financials_data)#.replace("```json\n","").replace("\n```",""))
      results['financials'] = financials_data
  except Exception as e:
      print("failed")
      results['financials'] = financials_data

  # Web scraping and data extraction for similar stocks
  similar_stocks_data = gpt_trigger(similar_stocks_data_by_ticker[0].page_content, ticker.upper(), "competitors")
  try:
      similar_stocks_data = ast.literal_eval(similar_stocks_data)#.replace("```json\n","").replace("\n```",""))
      results['similar_stocks'] = similar_stocks_data
  except Exception as e:
      print("failed")
      results['similar_stocks'] = similar_stocks_data

  return results

def google_news_scraper(tickr):
    google_news = GNews(language='en', country='US', period='7d')
    news = google_news.get_news(tickr)
    news_scrapper = pd.DataFrame(news)
    news_scrapper.sort_values(by=['published date'], ascending=False, inplace=True)
    return news_scrapper

def get_news_content(url):
    try:
        return WebBaseLoader(WebBaseLoader(url).load()[0].page_content.split("Google NewsOpening ")[1]).load()[0].page_content
    except Exception as e:
        print(f"Error retrieving content: {e}")
        return "Dummy content"

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reddit_scraper(subreddit_name, search_term):
    try:
        # Load Reddit credentials from a JSON file
        with open('reddit_creds.json', 'r') as f:
            client_creds = json.load(f)

        # Initialize PRAW with credentials
        reddit = praw.Reddit(client_id=client_creds["client_id"],
                             client_secret=client_creds['client_secret'],
                             user_agent=client_creds['user_agent'],
                             ratelimit_seconds=300)

        # Fetch posts from subreddit
        posts = []
        subreddit = reddit.subreddit(subreddit_name).search(search_term, time_filter='month', limit=100)
        for post in subreddit:
            posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

        if not posts:
            logging.warning("No posts found for the given search term.")
            return

        # Convert list of posts into a DataFrame
        posts_df = pd.DataFrame(posts, columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

        # Fetch and attach comments to each post
        for index, post_id in enumerate(posts_df['id']):
            try:
                submission = reddit.submission(id=post_id)
                submission.comments.replace_more(limit=0)
                comments = [comment.body for comment in submission.comments.list()]
                posts_df.at[index, 'comments'] = ', '.join(comments)
            except Exception as e:
                logging.error(f"Error fetching comments for post {post_id}: {e}")

        # Print the DataFrame or write it to a CSV file
        return(posts_df)
        # Uncomment the line below to write the result to a CSV file
        # posts_df.to_csv(f'top_reddit_posts_with_comments_{subreddit_name}_{search_term}.csv')
    except Exception as e:
        logging.error(f"An error occurred: {e}")

class StockNewsDetail(BaseModel):
    stock_name: str = Field(description="Name of the stock or company mentioned in the news.")
    recent_performance: Optional[str] = Field(None, description="Recent performance of the stock.")
    key_initiatives: Optional[str] = Field(None, description="Key initiatives or projects undertaken by the company.")
    market_trends: Optional[str] = Field(None, description="Current market trends affecting the stock.")
    future_outlook: Optional[str] = Field(None, description="Future outlook or predictions for the stock.")
    analyst_sentiment: Optional[str] = Field(None, description="Analyst sentiment or opinions regarding the stock.")
    news_sentiment: Optional[str] = Field(None, description="Overall Sentiment of the news relevant to the stock")

class GeneralStockNews(BaseModel):
    publication_date: Optional[str] = Field(None, description="Date the article was published.")
    author: Optional[str] = Field(None, description="Author of the article.")
    title: Optional[str] = Field(None, description="Title of the news article.")
    details: List[StockNewsDetail] = Field(description="Details extracted from the news article regarding the specified stock or stocks.")

class BlockCommentInsight(BaseModel):
    average_sentiment: str = Field(description="The average sentiment of the comments block (positive, negative, neutral).")
    average_sentiment_score: float = Field(description="A numeric score representing the average strength of the sentiment across comments.")
    common_themes: List[str] = Field(description="Common themes or topics identified across the comments.")
    sentiment_distribution: Dict[str, int] = Field(description="A distribution of sentiment types within the block of comments.")
    common_recommendations: List[str] = Field(description="Common recommendations or warnings derived from the block of comments.")
    common_trending_links: Optional[List[str]] = Field(description="List of frequently mentioned links for additional information or context.")
    common_highlighted_quotes: Optional[List[str]] = Field(description="Common notable quotes or statements from the block of comments.")
    associated_rumors: Optional[List[str]] = Field(description="Common rumors or news items mentioned within the block of comments.")
    market_impact_insights: Optional[str] = Field(description="Collective insights on potential market impacts derived from the block of comments.")

class RedditStockPost(BaseModel):
    sentiment: Optional[str] = Field(description="The overall sentiment of the post (bullish, bearish, neutral).")
    key_insights: Optional[str] = Field(description="Key insights related to share purchases, warnings, or stock performance.")
    rumors_and_news: Optional[str] = Field(description="Information on stock rumors and news mentioned in the post.")
    comments_summary: Optional[str] = Field(description="Summary of relevant opinions and insights from the comments.")
    block_comments_insights: Optional[List[BlockCommentInsight]] = Field(description="Summarized insights from blocks of comments, including average sentiment, key points, common themes, popularity, and potential market impact.")

class BlockCommentInsight(BaseModel):
    average_sentiment: str = Field(description="The average sentiment of the comments block (positive, negative, neutral).")
    average_sentiment_score: float = Field(description="A numeric score representing the average strength of the sentiment across comments.")
    common_themes: List[str] = Field(description="Common themes or topics identified across the comments.")
    sentiment_distribution: Dict[str, int] = Field(description="A distribution of sentiment types within the block of comments.")
    common_recommendations: List[str] = Field(description="Common recommendations or warnings derived from the block of comments.")
    common_trending_links: Optional[List[str]] = Field(description="List of frequently mentioned links for additional information or context.")
    common_highlighted_quotes: Optional[List[str]] = Field(description="Common notable quotes or statements from the block of comments.")
    associated_rumors: Optional[List[str]] = Field(description="Common rumors or news items mentioned within the block of comments.")
    market_impact_insights: Optional[str] = Field(description="Collective insights on potential market impacts derived from the block of comments.")

class RedditStockPost(BaseModel):
    sentiment: Optional[str] = Field(description="The overall sentiment of the post (bullish, bearish, neutral).")
    key_insights: Optional[str] = Field(description="Key insights related to share purchases, warnings, or stock performance.")
    rumors_and_news: Optional[str] = Field(description="Information on stock rumors and news mentioned in the post.")
    comments_summary: Optional[str] = Field(description="Summary of relevant opinions and insights from the comments.")
    block_comments_insights: Optional[List[BlockCommentInsight]] = Field(description="Summarized insights from blocks of comments, including average sentiment, key points, common themes, popularity, and potential market impact.")

def is_valid_dict_literal(s):
    try:
        parsed_value = ast.literal_eval(s)
        if isinstance(parsed_value, dict):
            return True
        else:
            return False
    except (ValueError, SyntaxError):
        return False

class TickerAnalysis(BaseModel):
    ticker: str = Field(description="Stock ticker symbol.")
    general_sentiment: str = Field(description="Overall sentiment regarding the ticker.")
    key_points: str = Field(description="Summary of key points mentioned in the posts.")
    market_sentiment: str = Field(description="Overall market sentiment reflected in the posts.")
    trends_patterns: str = Field(description="Trends or patterns noted for the ticker.")
    advice_given: str = Field(description="Advice or tips provided in relation to the ticker.")
    uncertainties: str = Field(description="Questions or uncertainties raised about the ticker.")
    actionable_steps: str = Field(description="Actionable steps or research tasks identified.")

class GroupTickerAnalysis(BaseModel):
    analyses: List[TickerAnalysis] = Field(description="List of analyses for each ticker mentioned.")
    
stock = "mrna"
index=20
stock_info = stock_data(stock)
stock_news = google_news_scraper(stock)
stock_news['content'] = stock_news['url'].apply(get_news_content)
stock_news = stock_news[stock_news['content']!='Dummy content']
stock_news_content = stock_news['content'].iloc[index]
print(stock_info)
print(stock_news['content'].iloc[index])
post_df = reddit_scraper("wallstreetbets", stock)

prompt_template = f"""
# System Instructions:
Given the text from a news article, the task is to extract and summarize key information relevant {stock} discussed in the article. The information should be extracted following a structured format based on a predefined Pydantic model. The output should adhere to the fields defined in the model: Name of the stock, Recent performance, Key initiatives, Market trends, Future outlook, and Analyst sentiment.

Ensure to capture significant details such as specific figures, percentages, descriptions of projects, current market impacts, predictions, and analyst opinions. The response should be clear, concise, and structured according to the fields of the Pydantic model, ensuring all relevant information from the article is accurately reflected.

# User Prompt:
Given the following content from a news article about the stock {stock}:

{stock_news_content}

Extract and summarize the key information relevant to {stock} discussed in the article:

1. Name of the stock or company.
2. Recent performance of the stock, including any specific figures or percentages.
3. Description of any key initiatives or projects undertaken by the company.
4. Mention of current market trends and how they are affecting the stock or company.
5. Predictions or future outlook provided for the stock.
6. Analyst sentiment or general opinions regarding the stock's future performance.

# Return Format:
The output should be organized following the structure of the Pydantic models 'StockNewsDetail' and 'GeneralStockNews'. This entails providing the structured information under respective fields like 'stock_name', 'recent_performance', 'key_initiatives', 'market_trends', 'future_outlook', and 'analyst_sentiment' for the 'StockNewsDetail' model; and 'publication_date', 'author', 'title', and 'details' for the 'GeneralStockNews' model.

Ensure the information is clearly separated and adequately formatted to match the descriptions and types specified in the Pydantic model fields.

"""
pydantic_object = GeneralStockNews

pydantic_parser = PydanticOutputParser(pydantic_object=pydantic_object)
format_instructions = pydantic_parser.get_format_instructions()
query = prompt_template
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
)
_input = prompt.format_prompt(query=query)

prompt_template = """
For the following group of Reddit posts about the stock {}:

{}

Follow the following instructions to generate insights about the stock {stock} from each of the above reddit posts:

Instructions:
1. For each of the Reddit post content about the stock {stock}, perform the following analysis. Extract and summarize the information into the structured format shown below. Only include information about the stock {stock} and ensure clarity on investment decisions and market perspectives. Use additional filters as necessary for accurate extraction.
2. Ensure the final output is structured according to the format provided. This structure is essential for the output to be directly usable in Python code, particularly with `ast.literal_eval` or `json.loads`.
3. After completing the analysis, double-check that the output format matches the structure shown below. Make sure all placeholders (like 'CLASSIFY_AS', 'SUMMARIZE_KEY_INSIGHTS', etc.) are replaced with actual data derived from the analysis. Ensure that strings are quoted, lists and dictionaries are properly formatted, and no trailing commas are left.
4. Verify that the final output does not contain any syntax errors and can be easily converted into a dictionary using `ast.literal_eval`. This can be done by copying the final output into a Python environment and attempting to parse it with `ast.literal_eval`. If no errors are raised, the format is correct.

1. Sentiment of the post with respect to {stock}: Classify as bullish, bearish, or neutral.
2. Key insights for share purchases with respect to {stock}: Summarize any specific recommendations, warnings, or insights.
3. Stock rumors and news with respect to {stock}: Identify and summarize any rumors or news that could influence stock prices.
4. Summary of comments with respect to {stock}: Extract and condense relevant opinions and insights from the comments.
5. Detailed comments insights with respect to {stock}: Provide detailed insights from individual comments, including the commenter's sentiment, key points, upvotes, and any relevant links mentioned. Summarize major themes or discussions emerging from these detailed insights.

Ensure the information is structured according to the defined fields, only about the stock {stock}  and provides clarity on investment decisions and market perspectives. Use additional filters as necessary for accurate extraction.
"""

pydantic_object = RedditStockPost

pydantic_parser = PydanticOutputParser(pydantic_object=pydantic_object)
format_instructions = pydantic_parser.get_format_instructions()
query = prompt_template
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
)
_input = prompt.format_prompt(query=query)
prompt_length = num_tokens_from_string(_input.to_string(), "gpt-3.5-turbo")

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

reddit_content = ""

post_df.sort_values(by=['score','created'], ascending=False, inplace=True)
post_df.reset_index(drop=True, inplace=True)
print(post_df.columns)
#post_df = post_df[post_df['score']>500]
post_df['content_length'] = post_df.apply(lambda row: num_tokens_from_string((getattr(row, 'body', '') + getattr(row, 'comments', '')), "gpt-3.5-turbo"), axis=1)
post_df['prompt_length'] = prompt_length
post_df['combined_content_length'] =  post_df['content_length'].cumsum()

# Initialize chunking variables
chunks = []
current_chunk = []
current_tokens = 0

# Iterate through the DataFrame
for index, row in post_df.iterrows():
    post_tokens = row['content_length']

    if post_tokens > 16000:  # Check if a single post exceeds the token limit
        print("ENTERED")  # For debugging, remove or replace with logging in production
        # Split the content and add each split part as a separate chunk
        post_content = row['body'] + row['comments']
        split_chunks = text_splitter.create_documents([post_content])
        chunks.extend(['\n\n'.join(split_chunk) for split_chunk in split_chunks])
    else:
        # Continue with regular chunk aggregation
        if current_tokens + post_tokens < 16000 - prompt_length:
            current_chunk.append(row['body'] + row['comments'])
            current_tokens += post_tokens
        else:
            # Add the current chunk to chunks and reset
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [row['body'] + row['comments']]
            current_tokens = post_tokens

# Add the last chunk if not empty
if current_chunk:
    chunks.append(' '.join(current_chunk))

insights = []

for chunk in chunks:
  reddit_content = chunk

  prompt_template = f"""
  For the following group of Reddit posts about the stock {stock}:

  {reddit_content}

  Follow the following instructions to generate insights about the stock {stock} from each of the above reddit posts:

  Instructions:
  1. For each of the Reddit post content about the stock {stock}, perform the following analysis. Extract and summarize the information into the structured format shown below. Only include information about the stock {stock} and ensure clarity on investment decisions and market perspectives. Use additional filters as necessary for accurate extraction.
  2. Ensure the final output is structured according to the format provided. This structure is essential for the output to be directly usable in Python code, particularly with `ast.literal_eval` or `json.loads`.
  3. After completing the analysis, double-check that the output format matches the structure shown below. Make sure all placeholders (like 'CLASSIFY_AS', 'SUMMARIZE_KEY_INSIGHTS', etc.) are replaced with actual data derived from the analysis. Ensure that strings are quoted, lists and dictionaries are properly formatted, and no trailing commas are left.
  4. Verify that the final output does not contain any syntax errors and can be easily converted into a dictionary using `ast.literal_eval`. This can be done by copying the final output into a Python environment and attempting to parse it with `ast.literal_eval`. If no errors are raised, the format is correct.

  1. Sentiment of the post with respect to {stock}: Classify as bullish, bearish, or neutral.
  2. Key insights for share purchases with respect to {stock}: Summarize any specific recommendations, warnings, or insights.
  3. Stock rumors and news with respect to {stock}: Identify and summarize any rumors or news that could influence stock prices.
  4. Summary of comments with respect to {stock}: Extract and condense relevant opinions and insights from the comments.
  5. Detailed comments insights with respect to {stock}: Provide detailed insights from individual comments, including the commenter's sentiment, key points, upvotes, and any relevant links mentioned. Summarize major themes or discussions emerging from these detailed insights.

  Ensure the information is structured according to the defined fields, only about the stock {stock}  and provides clarity on investment decisions and market perspectives. Use additional filters as necessary for accurate extraction.
  """

  pydantic_object = RedditStockPost

  pydantic_parser = PydanticOutputParser(pydantic_object=pydantic_object)
  format_instructions = pydantic_parser.get_format_instructions()
  query = prompt_template
  prompt = PromptTemplate(
      template="Answer the user query.\n{format_instructions}\n{query}\n",
      input_variables=["query"],
      partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
  )
  _input = prompt.format_prompt(query=query)

  answer = get_completion(_input.to_string())

  insights.append(answer)

SubredditScraper(
    'wallstreetbets',
        lim=50,
        mode='w',
        sort='top').get_posts()

for index in range(len(split_chunks)):
  prompt_template = f"""
  For the following subreddit posts discussing various stock tickers:

  {split_chunks[index]}

  Follow these instructions to generate insights about each mentioned ticker:

  Instructions:
  1. Identify each unique ticker mentioned in the group of posts. For each ticker, perform the following analysis. Summarize the information into the structured format shown below. Focus on details crucial for understanding the stock's sentiment, key points, market sentiment, trends, advice, uncertainties, and actionable steps derived from the discussion.
  2. Ensure the final output is structured according to the format provided. This structure is essential for the output to be directly usable in Python code, particularly with `ast.literal_eval` or `json.loads`.
  3. After completing the analysis for each ticker, compile them into a comprehensive report. Double-check that the final output matches the structure shown below. Replace all placeholders with actual data derived from the analysis. Ensure that strings are quoted, lists and dictionaries are properly formatted, and no trailing commas are left.
  4. Verify that the final output does not contain any syntax errors and can be easily converted into a Python dictionary using `ast.literal_eval`. This can be done by copying the final output into a Python environment and attempting to parse it with `ast.literal_eval`. If no errors are raised, the format is correct.

  Each ticker analysis should include:
  1. Ticker: The stock's ticker symbol.
  2. General Sentiment: Overall sentiment regarding the ticker.
  3. Key Points: Summary of key points mentioned in the posts.
  4. Market Sentiment: Overall market sentiment reflected in the posts.
  5. Trends/Patterns: Trends or patterns noted for the ticker.
  6. Advice Given: Advice or tips provided in relation to the ticker.
  7. Uncertainties: Questions or uncertainties raised about the ticker.
  8. Actionable Steps: Actionable steps or research tasks identified.

  Ensure the information is structured according to the defined fields and provides clarity on investment decisions and market perspectives. Use additional filters as necessary for accurate extraction.
  """

  validation_template = """
  Ensure the output of the analysis adheres to the GroupTickerAnalysis model structure. Follow these validation steps:

  Validation Instructions:
  1. Check that the output is a list of TickerAnalysis objects, each representing a unique stock ticker discussed in the subreddit posts.
  2. For each TickerAnalysis object in the list, ensure the following:
    a. 'ticker' is a non-empty string representing the stock's ticker symbol.
    b. 'general_sentiment' is a string and should be one of the following: 'Positive', 'Negative', or 'Neutral'.
    c. 'key_points' is a string summarizing the main points discussed about the ticker.
    d. 'market_sentiment' is a string reflecting the overall market sentiment towards the ticker.
    e. 'trends_patterns' is a string detailing any trends or patterns identified for the ticker.
    f. 'advice_given' is a string summarizing any advice or tips provided for the ticker.
    g. 'uncertainties' is a string listing any questions or uncertainties raised about the ticker.
    h. 'actionable_steps' is a string outlining actionable steps or research tasks derived from the posts.
  3. Verify that all strings are properly quoted and lists and dictionaries are correctly formatted within the output.
  4. Ensure there are no trailing commas and syntax errors. The output should be valid Python code that can be converted into a dictionary using `ast.literal_eval`.
  5. If the output passes all the above checks, it conforms to the expected format and is valid according to the GroupTickerAnalysis model.

  Please apply the above validation instructions to the following analysis output:

  {answer}
  """

  pydantic_object = GroupTickerAnalysis

  pydantic_parser = PydanticOutputParser(pydantic_object=pydantic_object)
  format_instructions = pydantic_parser.get_format_instructions()
  query = prompt_template
  prompt = PromptTemplate(
      template="Answer the user query.\n{format_instructions}\n{query}\n",
      input_variables=["query"],
      partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
  )
  _input = prompt.format_prompt(query=query)
  answer = get_completion(_input.to_string())
  if is_valid_dict_literal(answer):
      print(answer)
  else:
      fa = get_completion(validation_template.format(answer=answer))
      print(fa)

