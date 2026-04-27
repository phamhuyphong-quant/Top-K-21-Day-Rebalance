import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer
from scipy.stats import spearmanr
# --- CÁC TOÁN TỬ WORLDQUANT CƠ BẢN ---
# --- CÁC TOÁN TỬ WORLDQUANT CƠ BẢN (Bản cập nhật) ---
# --- CẬP NHẬT TRONG alpha_mining.py ---

import numpy as np
import pandas as pd

def to_series(df, col):
    if isinstance(col, str):
        return df[col]
    if isinstance(col, pd.Series):
        return col
    # Đảm bảo chuyển đổi mảng numpy về Series khớp với index của df
    return pd.Series(col, index=df.index)

def ts_delay(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).shift(d)

def ts_delta(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).diff(d)

def ts_mean(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window=d, min_periods=1).mean()
    )

def ts_min(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window=d, min_periods=1).min()
    )

def ts_rank(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window=d, min_periods=1).rank(pct=True)
    )

def ts_argmax(df, col, d):
    data = to_series(df, col)
    return data.groupby(df['Symbol']).transform(
        lambda x: x.rolling(window=d, min_periods=1).apply(np.argmax) + 1
    )
class WorldQuantAlphas:
    def __init__(self, df):
        self.df = df.copy()
        # Đảm bảo dữ liệu đã được sắp xếp đúng
        self.df = self.df.sort_values(by=['Symbol', 'date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.df[col] = self.df.groupby('Symbol')[col].shift(1)
        # Now, all calculations below will automatically use T-1 data

    def get_alpha_012(self):
        # Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))
        # Logic: Mua khi giá rớt nhưng volume tăng (Cạn cung, bắt đáy)
        delta_vol = ts_delta(self.df, 'volume', 1)
        delta_close = ts_delta(self.df, 'close', 1)
        return np.sign(delta_vol) * (-1 * delta_close)

    def get_alpha_041(self):
        # Alpha#41: (((high * low)^0.5) - vwap)
        # Tạm thay vwap (Volume Weighted Avg Price) bằng giá close nếu bạn chưa có vwap
        vwap = self.df['close'] 
        return np.sqrt(self.df['high'] * self.df['low']) - vwap

    def get_alpha_054(self):
        # Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        # Logic: Sự bất thường giữa giá đóng/mở và biên độ trong ngày
        numerator = -1 * (self.df['low'] - self.df['close']) * (self.df['open'] ** 5)
        denominator = (self.df['low'] - self.df['high']) * (self.df['close'] ** 5)
        # Tránh chia cho 0
        return np.where(denominator == 0, 0, numerator / denominator)

    def get_alpha_101(self):
        # Alpha#101: ((close - open) / ((high - low) + 0.001))
        # Logic: Đo lường sức mạnh đóng cửa (Intraday momentum)
        return (self.df['close'] - self.df['open']) / ((self.df['high'] - self.df['low']) + 0.001)

    def get_alpha_006(self):
        g = self.df.groupby('Symbol')
        return -1 * g['open'].transform(
        lambda x: x.rolling(10).corr(self.df.loc[x.index, 'volume'])
    ).fillna(0)
    def get_alpha_024(self):
        # Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) <= 0.05) ...
        
        # Bước 1: Tính Mean 100 ngày (tránh lồng hàm trực tiếp gây lỗi Series)
        mean_100 = ts_mean(self.df, 'close', 100)
        
        # Bước 2: Tính Delta của cái Mean đó
        delta_mean = ts_delta(self.df, mean_100, 100)
        
        # Bước 3: Tính Delay của giá close
        low_delay = ts_delay(self.df, 'close', 100)
        
        # Bước 4: Kiểm tra điều kiện (Thêm 0.001 để tránh chia cho 0)
        cond = (delta_mean / (low_delay + 0.001)) <= 0.05
        
        # Bước 5: Trả về kết quả dựa trên điều kiện
        return np.where(
            cond, 
            -1 * (self.df['close'] - ts_min(self.df, 'close', 100)), 
            -1 * ts_delta(self.df, 'close', 3)
        )
    def get_alpha_028(self):
        # Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
        adv20 = ts_mean(self.df, 'volume', 20)
        corr = self.df.groupby('Symbol')['volume'].transform(
    lambda x: x.rolling(5).corr(self.df.loc[x.index, 'low'])
)
        return corr + ((self.df['high'] + self.df['low']) / 2) - self.df['close']

    def get_alpha_053(self):
        # Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        inner = ((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])) / (self.df['close'] - self.df['low'] + 0.0001)
        # Gán tạm vào df để dùng hàm ts_delta
        result = -1 * ts_delta(self.df, inner, 9)
        return result

    def get_alpha_060(self):
        # Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
        # Bản rút gọn tập trung vào dòng tiền định hướng (Money Flow)
        return (((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])) / (self.df['high'] - self.df['low'] + 0.001)) * self.df['volume']

    def generate_all(self):
        """Tự động tính toán và trả về DataFrame chứa tất cả Alpha"""
        print("Đang tạo WorldQuant Alphas...")
        self.df['WQ_Alpha_012'] = self.get_alpha_012()
        self.df['WQ_Alpha_041'] = self.get_alpha_041()
        self.df['WQ_Alpha_054'] = self.get_alpha_054()
        self.df['WQ_Alpha_101'] = self.get_alpha_101()
        self.df['WQ_Alpha_006'] = self.get_alpha_006()
        self.df['WQ_Alpha_024'] = self.get_alpha_024()
        self.df['WQ_Alpha_028'] = self.get_alpha_028()
        self.df['WQ_Alpha_053'] = self.get_alpha_053()
        self.df['WQ_Alpha_060'] = self.get_alpha_060()
        
        # Điền NaN bằng 0 và trả về các cột mới
        alpha_cols = [c for c in self.df.columns if 'WQ_Alpha_' in c]
        self.df[alpha_cols] = self.df[alpha_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.df[alpha_cols]
    


def add_and_filter_alphas(gp_model, original_df, input_features):
    df_result = original_df.copy()
    
    # 1. Chuẩn bị dữ liệu để tính toán (điền 0 cho các giá trị NaN tạm thời)
    X_full = df_result[input_features].values
    X_filled = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Tính toán 10 Alpha
    alpha_values = gp_model.transform(X_filled)
    
    alpha_values = gp_model.transform(X_filled)

    _, idx = np.unique(alpha_values, axis=1, return_index=True)
    alpha_values_unique = alpha_values[:, np.sort(idx)]

    alpha_df_unique = pd.DataFrame(
    alpha_values_unique,
    columns=[f'Mega_Alpha_{i}' for i in range(alpha_values_unique.shape[1])])
    
    # 4. Ghép các cột Unique vào DataFrame gốc
    new_cols = []
    for col in alpha_df_unique.columns:
        df_result[col] = alpha_df_unique[col].values
        new_cols.append(col)
        
    print(f"Từ 10 công thức ban đầu, đã lọc ra {len(new_cols)} Mega-Alphas độc lập: {new_cols}")
    return df_result, new_cols



def rank_ic_fitness(y_true, y_pred):
    """
    Hàm đánh giá độ hiệu quả của công thức bằng Rank IC.
    y_true: Lợi nhuận thực tế (next_1m_ret trong file của bạn)
    y_pred: Giá trị mà công thức Alpha vừa tạo ra
    """
    # Tính tương quan hạng Spearman
    rho, _ = spearmanr(y_true, y_pred)
    return rho
