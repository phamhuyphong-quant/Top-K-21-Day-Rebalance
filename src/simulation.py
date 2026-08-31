import pandas as pd
import numpy as np
from dataclasses import dataclass
import random


class DataEngine():
    def __init__(self, df):
        self.data = df
        self.idx = 0
        self.trading_dates = sorted(df['date'].unique())
        self.symbol_arrays = {}
        for symbol, group in df.groupby('Symbol'):
            group = group.sort_values('date')
            self.symbol_arrays[symbol] = {
                'dates': group['date'].values,
                'close': group['close'].values,
                'rows': group.reset_index(drop=True)  # keep for full-row access if needed
            }
        self.date_to_rows = {d: g for d, g in df.groupby('date')}  # replaces get_date's mask
    def reset(self):
        """Rewind the cursor for a fresh run without rebuilding symbol_arrays/date_to_rows."""
        self.idx = 0
    def update_idx(self):
        self.idx +=1
    def is_finished(self):
        return self.idx >= len(self.trading_dates)
    def get_upcoming(self,symbol):
        if self.is_finished():return False

        today = self.trading_dates[self.idx]
        
        mask = (self.data['date'] >= today) & (self.data['Symbol'] == symbol)
        rows = self.data[mask].sort_values(by='date')
        if rows.empty:return 0
        else: return rows.iloc[0]
    def get_last(self, symbol):
        if self.is_finished() or symbol not in self.symbol_arrays:
            return False
        today = self.trading_dates[self.idx]
        arr = self.symbol_arrays[symbol]
        pos = np.searchsorted(arr['dates'], np.datetime64(today), side='right') - 1
        if pos < 0:
            return False
        return arr['rows'].iloc[pos]
    def get_today(self):
        return self.trading_dates[self.idx]
    def get_date(self, date):
        return self.date_to_rows.get(date, self.data.iloc[0:0])
@dataclass 
class pack():
    amount:int
    T:int

class Stock():
    def __init__(self):
        self.available_shares = 0
        self.pending_shares = []
    def next_day(self):
        still_pending = []
        for p in self.pending_shares:
            p.T -=1
            if p.T<=0:
                self.available_shares+=p.amount
            else:
                still_pending.append(p)
        self.pending_shares = still_pending
    def add_pack(self,amount, T):
        if T <= 0:
           self.available_shares += amount
        else:
            self.pending_shares.append(pack(amount,T)) 
    def reduce_shares(self,amount):
        if (self.available_shares < amount):return False
        self.available_shares -= amount
        return True

class Portfolio():
    def __init__(self,initial):
        self.history = []
        self.positions = {}
        self.liquid_cash = initial
        self.illiquid_cash = []
        self.tax = 0.1/100
        self.fee = 0.1/100
        self.T = 3
        self.skipped_orders = []  # (date, side, symbol, reason) audit trail for silent failures
        self.holdings_history = []  # (date, {symbol: total_shares}) snapshot, appended each next_day
    def execute_buy(self,symbol,shares,prices,date=None):
        cost = (shares *prices)*(1+self.fee)
        if cost > self.liquid_cash:
            self.skipped_orders.append((date, "buy", symbol, "insufficient_cash"))
            return False
        #execution
        self.liquid_cash -= cost

        if symbol not in self.positions:
            self.positions[symbol] = Stock()
        self.positions[symbol].add_pack(shares,self.T)
        return True
    
    def execute_sell(self,symbol,shares, prices, delay= 0, date=None):
        if symbol not in self.positions.keys():
            self.skipped_orders.append((date, "sell", symbol, "no_position"))
            return False
        if self.positions[symbol].reduce_shares(shares) is False:
            self.skipped_orders.append((date, "sell", symbol, "insufficient_shares"))
            return False
        revenue = (shares*prices)*(1-self.tax - self.fee)
    
        self.illiquid_cash.append(pack(revenue,self.T+delay))
        return True

    def next_day(self, data_engine, date):
        # 1. Sum up cash
        nav = self.liquid_cash + sum(c.amount for c in self.illiquid_cash)
        
        # 2. Add market value of held positions
        holdings_snapshot = {}
        for symbol, stock in self.positions.items():
            stock.next_day()
            total_shares = stock.available_shares + sum(p.amount for p in stock.pending_shares)
            if total_shares > 0:
                holdings_snapshot[symbol] = total_shares
            # Fetch price for the specific symbol
            price_row = data_engine.get_last(symbol)
            if price_row is not False:
                # Assuming your dataframe has a 'close' column
                current_price = price_row['close'] 
                nav += total_shares * current_price
        self.holdings_history.append((date, holdings_snapshot))
                
        # 3. Update illiquid cash (settlement)
        still_illiquidate = []
        for cash in self.illiquid_cash:
            cash.T -= 1
            if cash.T <= 0:
                self.liquid_cash += cash.amount
            else:
                still_illiquidate.append(cash)
        self.illiquid_cash = still_illiquidate
        
        # 4. Record history
        self.history.append((date, nav))
class DataMismatch(Exception):
    "Exception raised when Dataframe doesn't have the same number of unique date, or first and last date doesn't match"
    pass



class StepFilter:
    """
    Always returns a fixed multiplier, regardless of row.
    Meant to be composed with ConditionalFilter when you want
    it to apply only under some condition.
    """
    def __init__(self, multiplier: float):
        self.multiplier = multiplier

    def compute(self, row: pd.Series) -> float:
        return self.multiplier
 
class RampFilter:
    """
    Numeric ramp filter on a single column.
 
    value <= floor      -> min_value
    value >= threshold   -> 1.0
    in between           -> linear interpolation from min_value to 1.0
 
    Missing column / NaN value -> no-op (1.0).
    Raises at construction time if floor >= threshold (config error).
    """
 
    def __init__(self, column: str, floor: float, threshold: float, min_value=0.0, max_value = 1):
        if floor >= threshold:
            raise ValueError(
                f"RampFilter requires floor < threshold, got floor={floor}, threshold={threshold}"
            )
        self.column = column
        self.floor = floor
        self.threshold = threshold
        self.min_value = min_value
        self.max_value = max_value
 
    def compute(self, row: pd.Series) -> float:
        if self.column not in row.index:
            return self.max_value
 
        x = row[self.column]
        if pd.isna(x):
            return self.max_value
 
        if x <= self.floor:
            return self.min_value
        if x >= self.threshold:
            return self.max_value
 
        frac = (x - self.floor) / (self.threshold - self.floor)
        return self.min_value + frac * (self.max_value - self.min_value)

class FilterGroup:
    def __init__(self,condition: str=None, transformations = None):
        self.condition = condition
        self.transformations = transformations if transformations is not None else []
    def compute(self, row):
        if self.condition is not None:
            try:
                matched = eval(self.condition, {}, row.to_dict())
            except Exception as e:
                raise ValueError(f"FilterGroup  condition failed: '{self.condition}' -> {e}") from e
            if not matched:
                return 1.0
        result = 1.0
        for t in self.transformations:
            result *= t.compute(row)
        return result


def build_data_engines(df_raw, df_predict, df_condition=None):
    """
    Do the (one-time) date-range filtering and cross-validation between
    df_raw / df_predict / df_condition, then build DataEngine objects.

    Call this ONCE per dataset and reuse the returned engines across many
    OrderManager/NullHypothesisTest runs via engine.reset() -- don't rebuild
    engines per trial.

    Returns: (engine_raw, engine_predict, engine_condition_or_None)
    """
    last_date = sorted(df_predict['date'].unique())[-1]
    first_date = sorted(df_predict['date'].unique())[0]
    df_raw = df_raw[(df_raw['date'] >= first_date) & (df_raw['date'] <= last_date)]

    raw_dates = sorted(df_raw['date'].unique())
    predict_dates = sorted(df_predict['date'].unique())
    if set(raw_dates) != set(predict_dates):
        raise DataMismatch()

    engine_raw = DataEngine(df_raw)
    engine_predict = DataEngine(df_predict)

    engine_condition = None
    if df_condition is not None:
        df_condition = df_condition[
            (df_condition['date'] >= first_date) & (df_condition['date'] <= last_date)
        ]
        condition_dates = sorted(df_condition['date'].unique())
        if set(raw_dates) != set(condition_dates):
            raise DataMismatch()
        engine_condition = DataEngine(df_condition)

    return engine_raw, engine_predict, engine_condition


class OrderManager():

    def __init__(self, initial, topk, df_raw: 'DataEngine', df_predict: 'DataEngine',
                 df_condition: 'DataEngine' = None, regime_filter: FilterGroup = None):
        """
        df_raw / df_predict / df_condition must already be DataEngine instances
        (build them once with build_data_engines()). Their cursors are rewound
        via reset() so the same engines can be reused cheaply across many runs.
        """
        self.initial = initial
        self.topk = topk
        self.regime_filter = regime_filter
        self.portfolio = None

        self.df_raw = df_raw
        self.df_predict = df_predict
        self.df_condition = df_condition

        self.df_raw.reset()
        self.df_predict.reset()
        if self.df_condition is not None:
            self.df_condition.reset()
    def _get_condition_row(self, symbol, curr_date):
        if self.df_condition is None:
            return None
        row = self.df_condition.get_last(symbol)
        if row is False or row is None:
            return None
        if row['date'] != curr_date:
            return None
        return row

    def get_holdings(self, start_date=None, end_date=None):
        """
        Return {date: {symbol: shares}} for every recorded day in
        [start_date, end_date] (inclusive). Pass None for either bound
        to leave it open-ended. Must be called after run_strategy().
        """
        if self.portfolio is None:
            raise RuntimeError("run_strategy() must be called before get_holdings()")
        result = {}
        for date, snapshot in self.portfolio.holdings_history:
            if start_date is not None and date < start_date:
                continue
            if end_date is not None and date > end_date:
                continue
            result[date] = dict(snapshot)
        return result

    def get_symbols_held(self, start_date=None, end_date=None):
        """
        Return the sorted union of all symbols held at any point during
        [start_date, end_date]. Convenience wrapper around get_holdings().
        """
        symbols = set()
        for snapshot in self.get_holdings(start_date, end_date).values():
            symbols.update(snapshot.keys())
        return sorted(symbols)

    def run_strategy(self, allocation_strategy="equal"):
        prt = Portfolio(self.initial)
        self.portfolio = prt
        cnt = 0
        best_stocks = []
        symbols = []
        target_weights = pd.Series(dtype=float)
        while not self.df_raw.is_finished():
            curr_date = self.df_raw.get_today()
            prt.next_day(self.df_raw, curr_date)

            if cnt % 21 == 0:
                best_stocks = self._get_best_stocks()
                symbols = best_stocks["Symbol"].tolist()
                if allocation_strategy == "equal":
                    weights = {sym: 1.0 / len(symbols) for sym in symbols}

                    target_weights = pd.Series(weights)
                elif allocation_strategy == "rank_weighted":
                    best_stocks = best_stocks.sort_values(by="pred_score", ascending=True)
                    scores = best_stocks["pred_score"].values.astype(float)
                    # Shift so the minimum score is 0 -- guards against mixed-sign
                    # pred_scores producing negative or runaway weights below.
                    shifted = scores - scores.min()
                    total = shifted.sum()
                    if total <= 0:
                        # All scores tied (or only one stock): fall back to equal weight
                        weights = np.full(len(shifted), 1.0 / len(shifted))
                    else:
                        weights = shifted / total
                    symbols = best_stocks["Symbol"].tolist()
                    target_weights = pd.Series(weights, index=symbols)
                self._selling_phase(prt,target_weights, curr_date, lot_size=100)
            if (cnt - 3)%21 == 0:
                self._buying_phase(prt,target_weights,curr_date,lot_size=100)
            self.df_raw.update_idx()
            self.df_predict.update_idx()
            if self.df_condition is not None:
                self.df_condition.update_idx()
            cnt += 1

    def _get_best_stocks(self):
        today = self.df_predict.get_today()
        dashboard = self.df_predict.get_date(today).sort_values(
            by="pred_score"
        )
        return dashboard.iloc[-self.topk :]

    def _selling_phase(self, prt, target_weights, curr_date, lot_size=100):
        best_stocks = target_weights.index.tolist()
        # NAV as of today (prt.next_day already ran earlier this loop iteration
        # and appended today's mark-to-market value to history).
        nav = prt.history[-1][1] if prt.history else prt.liquid_cash

        # Iterate over a list of keys so we can safely delete from the dict during the loop
        for symbol in list(prt.positions.keys()):
            stock = prt.positions[symbol]

            # Check if there are any shares at all (available or pending)
            total_shares = stock.available_shares + sum(p.amount for p in stock.pending_shares)

            if total_shares > 0:
                # Use get_last to prevent look-ahead bias
                price_row = self.df_raw.get_last(symbol)
                if price_row is not False and price_row['date'] == curr_date:
                    price = price_row["close"]

                    if symbol not in best_stocks:
                        # Full exit: no longer part of the target portfolio at all.
                        # You will need to decide how to handle pending_shares here.
                        # Currently, this only sells what is strictly available.
                        if stock.available_shares > 0:
                            prt.execute_sell(
                                symbol,
                                stock.available_shares,
                                price,
                                delay=0,
                                date=curr_date,
                            )
                    else:
                        # Still in the target portfolio, but may be overweight
                        # (e.g. it was held last cycle and got bought again, or
                        # its weight simply shrank). Trim down to target value
                        # instead of blindly re-buying/holding as-is.
                        weight = target_weights[symbol]
                        current_value = total_shares * price
                        target_value = nav * weight
                        excess_value = current_value - target_value
                        if excess_value > 0 and stock.available_shares > 0:
                            shares_to_sell = int(excess_value / price)
                            # Respect lot size, and never sell more than what's
                            # actually available (pending shares aren't liquid yet).
                            shares_to_sell = (shares_to_sell // lot_size) * lot_size
                            shares_to_sell = min(shares_to_sell, stock.available_shares)
                            if shares_to_sell >= lot_size:
                                prt.execute_sell(
                                    symbol,
                                    shares_to_sell,
                                    price,
                                    delay=0,
                                    date=curr_date,
                                )

            # Clean up empty positions to keep the dictionary lean
            if stock.available_shares == 0 and not stock.pending_shares:
                del prt.positions[symbol]

    def _buying_phase(self, prt, target_weights, curr_date, lot_size=100):
        # NAV as of today, same basis _selling_phase used 3 days ago when it
        # computed target values -- this keeps buy/sell sizing consistent
        # with "target_weights * NAV", not "target_weights * cash-on-hand".
        nav = prt.history[-1][1] if prt.history else prt.liquid_cash

        for symbol, weight in target_weights.items():
            price_row = self.df_raw.get_last(symbol)
            # Require a price actually dated today, same look-ahead-bias guard
            # used in _selling_phase -- otherwise we could buy on a stale price
            # from days ago with no indication anything was off.
            if price_row is not False and price_row['date'] == curr_date:
                price = price_row["close"]

                # Only buy the shortfall vs. target value -- a symbol already
                # holding its full (or more than its) target weight from a
                # prior cycle should not be bought again from scratch.
                stock = prt.positions.get(symbol)
                current_shares = 0
                if stock is not None:
                    current_shares = stock.available_shares + sum(p.amount for p in stock.pending_shares)
                current_value = current_shares * price
                target_value = nav * weight
                shortfall_value = target_value - current_value
                if shortfall_value <= 0:
                    continue

                # If any of the configured queries is satisfied for this
                # symbol today, only buy a fraction of the shortfall
                # instead of going all the way to target weight.
                if self.regime_filter is not None:
                    row = self._get_condition_row(symbol, curr_date)
                    if row is not None:
                        shortfall_value *= self.regime_filter.compute(row)

                max_shares = shortfall_value / price
                shares_to_buy = (int(max_shares / lot_size)) * lot_size
                if shares_to_buy >= lot_size:
                    prt.execute_buy(symbol, shares_to_buy, price, date=curr_date)


class NullHypothesisTest(OrderManager):
    """
    Subclass dành riêng cho việc giả lập Monte Carlo.
    Ghi đè hàm chọn cổ phiếu để chọn hoàn toàn NGẪU NHIÊN topk mã.
    """
    def reset(self):
        self.df_raw.reset()
        self.df_predict.reset()
        if self.df_condition is not None:
            self.df_condition.reset()
    def _get_best_stocks(self):
        today = self.df_predict.get_today()
        # Lấy toàn bộ danh sách các cổ phiếu có dữ liệu vào ngày hôm nay
        all_available_stocks = self.df_predict.get_date(today)
        
        # Đảm bảo không lấy trùng và số lượng cổ phiếu trong rổ đủ lớn hơn topk
        pool = all_available_stocks["Symbol"].unique()
        k = min(self.topk, len(pool))
        
        # Chọn ngẫu nhiên k cổ phiếu từ pool
        random_symbols = random.sample(list(pool), k)
        
        # Lọc ra dataframe chứa các cổ phiếu ngẫu nhiên này
        random_selection = all_available_stocks[all_available_stocks["Symbol"].isin(random_symbols)].copy()
        
        # Nếu dùng chiến lược 'rank_weighted', ta gán ngẫu nhiên điểm số giả lập cho chúng
        # để đảm bảo tính phân bổ trọng số vẫn hoạt động bình thường
        random_selection["pred_score"] = np.random.uniform(0.1, 1.0, size=len(random_selection))
        
        return random_selection